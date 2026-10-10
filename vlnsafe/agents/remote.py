"""Subprocess worker client: spawn a worker in its own conda env, talk JSON-lines, tear
it down. Knows nothing about any specific agent — only the protocol and a config block.

What: WorkerHandle owns one long-lived worker process (loading a checkpoint is the
expensive part). WorkerActionProvider encodes the observation + pose for a `decide`
request, decodes the returned primitive list into a local queue that rollout drains one
at a time, and exposes the worker's per-step external-predictor reading. Any divergence
(the executed action not matching what was queued) clears the queue, so the next
propose_action requests a fresh decision from the actual post-divergence pose.

Who reads it: vlnsafe.agents.registry, for any `family: *_worker` agent.

Must not import: habitat, torch. It serialises arrays and shells out.

Ported from the pre-release harness: encode_observation_for_etpnav_worker /
agent_pose_for_etpnav (360-388), ETPNavPolicyActionProvider (391-529), ETPNavWorkerHandle
/ get_or_create_etpnav_worker / close_all (532-608).
"""
import base64
import json
import os
import random
import subprocess
import zlib
from collections import deque

import numpy as np

from vlnsafe.geometry import project_depth_to_occupancy_points_m
from vlnsafe.records.schema import UNAVAILABLE
from vlnsafe.simulator.pose import agent_pose

STOP = 0  # habitat HabitatSimActions.STOP; MOVE_FORWARD 1, TURN_LEFT 2, TURN_RIGHT 3

# The sensor uuids the ETPNav-family worker consumes (12x RGB + 12x depth).
WORKER_OBSERVATION_KEYS = ["rgb", "depth"] + [
    "{}_{}".format(s, d) for s in ("rgb", "depth") for d in range(30, 360, 30)
]
# What a navigation VLA worker (StreamVLN / NaVILA) reads: one forward RGB + DEPTH + pose.
VLA_OBSERVATION_KEYS = ["rgb", "depth", "gps", "compass"]
# OmniNav reads its tri-view RGB only (vlnsafe.simulator.sensors.OMNINAV_VIEWS).
OMNINAV_OBSERVATION_KEYS = ["rgb_front", "rgb_left", "rgb_right"]


def encode_observation(observation, keys=WORKER_OBSERVATION_KEYS):
    """Inverse of the worker's decode_observation: numpy sensor arrays -> base64 bytes +
    shape/dtype; 'instruction' passed through (already plain python types). `keys` is the
    subset of sensor uuids this worker actually consumes."""
    payload = {}
    for key, value in observation.items():
        if key == "instruction":
            payload[key] = {
                "text": value.get("text", ""),
                "tokens": list(value.get("tokens", [])),
                "trajectory_id": value.get("trajectory_id", ""),
            }
            continue
        if key not in keys:
            continue
        arr = np.asarray(value)
        payload[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
    return payload


class WorkerHandle(object):
    def __init__(self, argv, cwd):
        # VLNSAFE_WORKER_GPU puts the model worker on its own GPU while the simulator keeps
        # CUDA_VISIBLE_DEVICES: an 8B VLA worker can outgrow a card it shares with the
        # simulator (M3 shard 2, CUDA OOM twice). Unset = the worker inherits the harness's GPU.
        env = None
        if os.environ.get("VLNSAFE_WORKER_GPU"):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ["VLNSAFE_WORKER_GPU"])
        self.proc = subprocess.Popen(
            argv, cwd=cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            universal_newlines=True, bufsize=1,
        )

    def request(self, payload):
        payload = dict(payload)
        payload["cmd"] = "decide"
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(
                "worker exited unexpectedly (returncode={}); see its stderr".format(
                    self.proc.poll()))
        response = json.loads(line)
        if not response.get("ok", False):
            raise RuntimeError("worker error: {}".format(response.get("error")))
        return response

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            self.proc.terminate()


class WorkerActionProvider(object):
    name = "worker"
    tryout = False   # Enable_tryout() sets it per instance

    def __init__(self, env, handle, turn_unit_deg, forward_unit_m, agent_radius_m,
                 depth_hfov_deg, depth_camera_height_m,
                 depth_min_depth_m=0.0, depth_max_depth_m=10.0, depth_sensor_keys=None,
                 observation_keys=None, send_occupancy=True, stream_observations=False,
                 exempt_guard_redecisions=False):
        self.env = env
        self.handle = handle
        self.turn_unit_deg = float(turn_unit_deg)
        self.forward_unit_m = float(forward_unit_m)
        self.agent_radius_m = float(agent_radius_m)
        self.depth_hfov_deg = float(depth_hfov_deg)
        self.depth_camera_height_m = float(depth_camera_height_m)
        self.depth_min_depth_m = float(depth_min_depth_m)
        self.depth_max_depth_m = float(depth_max_depth_m)
        self.depth_sensor_keys = depth_sensor_keys or (
            ["depth"] + ["depth_{}".format(30 * k) for k in range(1, 12)]
        )
        # which sensor uuids to serialise, and whether to project depth -> occupancy for
        # the worker's external clearance predictor (ETPNav-family only; a VLA worker
        # gets neither).
        self.observation_keys = observation_keys or WORKER_OBSERVATION_KEYS
        self.send_occupancy = bool(send_occupancy)
        # A streaming VLA consumes one frame per *executed* step, not one per decision:
        # StreamVLN's eval loop appends the observation to its SlowFast/KV history every
        # iteration and only calls generate() when its action chunk runs dry
        # (streamvln_eval.py:247-344 @e48f6ff). So we buffer the frames seen since the
        # last decision and ship them alongside it; a worker that decides per step (the
        # ETPNav family) leaves this off and the request shape is unchanged.
        self.stream_observations = bool(stream_observations)
        self._frames = []
        self._queue = deque()
        self._clearance_queue = deque()
        self._last_obs = None
        self._reset_pending = True
        self._decision_index = 0
        # True once the worker has returned a decision ending in STOP. A divergence after
        # that (vlnsafe.guard vetoing a step on the way to the stop node) must not ask for
        # another decision: an ETPNav-family worker cannot re-decide after a stop. The
        # guard executes the STOP instead; this makes any other route fail loudly.
        self._stop_committed = False
        # When set, the first decision after a guard manoeuvre is sent
        # with `budget_exempt`, and an ETPNav-family worker does not count it against its
        # max_len budget. Off by default; only `vlnsafe run --exempt-guard-redecisions`
        # turns it on.
        self.exempt_guard_redecisions = bool(exempt_guard_redecisions)
        self._guard_overridden = False
        # Set by _request_decision (queue refill); rollout reads then clears
        # pending_decision_signals, so it is non-None only on a fresh-decision step.
        self.pending_decision_signals = None
        self.pending_step_predicted_clearance_m = None
        # The ETP-lineage Tryout controller (vlnsafe/agents/tryout.py).
        # Off by default; only `vlnsafe run --tryout` turns it on (enable_tryout). When off,
        # none of the tryout attributes is read and the provider is unchanged.
        self.tryout = False
        self._tryout_seed = None
        self._tryout_rng = None
        self._tryout_chunk = None
        self._tryout_clearances = []
        self._tryout_escapes_before = 0   # escapes of this episode's finished chunks
        self.pending_tryout_phase = None
        # A GuardLayer whose check_forward vetoes Tryout's forwards
        # (`run --tryout --guard`). It lives here, not in rollout, because the escape is
        # the agent's own; rollout logs pending_guard_row and the guard's counters.
        self.tryout_guard = None
        self.pending_guard_row = None
        self._guard_checks = []

    def enable_tryout(self, seed, actions, guard=None):
        """`actions`: HabitatSimActions (int-able STOP / MOVE_FORWARD / TURN_LEFT /
        TURN_RIGHT). Needs the worker's `hops` in each response (etpnav_worker).
        `guard`: an optional vlnsafe.guard.GuardLayer (memory_frames 1) that checks every
        forward before it executes (vlnsafe/agents/tryout.py)."""
        self.tryout = True
        self._tryout_seed = int(seed)
        self._tryout_actions = actions
        self.tryout_guard = guard

    # --- ActionProvider contract -----------------------------------------------------
    def reset(self, observation):
        self._last_obs = observation
        self._queue.clear()
        self._clearance_queue.clear()
        self._reset_pending = True
        self._decision_index = 0
        self._frames = []
        self._stop_committed = False
        self._guard_overridden = False
        self.pending_decision_signals = None
        self.pending_step_predicted_clearance_m = None
        if self.tryout:
            self._tryout_chunk = None
            self._tryout_escapes_before = 0
            self.pending_tryout_phase = None
            # per episode, so a shard reproduces regardless of which episodes precede it
            key = "{}:{}".format(self._tryout_seed, self.env.current_episode.episode_id)
            self._tryout_rng = random.Random(zlib.crc32(key.encode("utf-8")))
            self.pending_guard_row = None
            self._guard_checks = []
            if self.tryout_guard is not None:
                self.tryout_guard.reset()

    def act(self, observation, pose):
        if self.tryout:
            return self._tryout_act()
        if self.stream_observations:
            self._frames.append(observation)
        if not self._queue:
            self._request_decision()
        self.pending_step_predicted_clearance_m = (
            self._clearance_queue[0] if self._clearance_queue else None
        )
        return self._queue[0]

    def planned_chunk(self):
        if self.tryout:
            return list(self._tryout_chunk.plan) if self._tryout_chunk is not None else []
        return list(self._queue)

    def update_after_step(self, observation, executed_action, done):
        self._last_obs = observation
        if self.tryout:
            chunk = self._tryout_chunk
            if chunk is not None and chunk.current() == executed_action:
                chunk.executed()
            else:
                self._drop_tryout_chunk()
            return
        if self._queue and self._queue[0] == executed_action:
            self._queue.popleft()
            if self._clearance_queue:
                self._clearance_queue.popleft()
        else:
            self._queue.clear()
            self._clearance_queue.clear()

    def override_chunk(self, primitives):
        """vlnsafe.guard: replace the queued plan with the guard's manoeuvre. It drains
        through act() like a decided chunk but without a model call, and the next decision
        is requested when it runs out. Its steps carry no external-predictor reading."""
        self._queue = deque(int(a) for a in primitives)
        self._clearance_queue = deque([None] * len(self._queue))
        self._guard_overridden = True

    # --- internals -----------------------------------------------------------------
    def _tryout_act(self):
        chunk = self._tryout_chunk
        action = chunk.current() if chunk is not None else None
        if action is None:
            self._drop_tryout_chunk()
            self._request_decision()
            chunk = self._tryout_chunk
            action = chunk.current()
        self.pending_tryout_phase = chunk.phase
        if self.tryout_guard is not None:
            self.pending_guard_row = self._guard_row(action)
        clearances = self._tryout_clearances
        self.pending_step_predicted_clearance_m = (
            clearances[chunk.worker_index] if chunk.worker_index is not None else None)
        return action

    def _guard_veto(self, policy_step):
        """TryoutChunk's veto_fn: the guard's verdict on a forward from the current pose,
        on the current (pre-step) frame."""
        verdict, clearance = self.tryout_guard.check_forward(
            self._last_obs, agent_pose(self.env), policy_step)
        self._guard_checks.append((verdict, clearance))
        return verdict in ("abort", "stall")

    def _guard_row(self, action):
        """This step's guard_* columns. A veto has no step of its own (the vetoed forward
        never executes), so it is logged on the next executed step, as vlnsafe.guard logs
        a veto on the manoeuvre's first step: flagged 1, action abort / stall (the last
        veto since the previous step). An executed forward carries its own check (pass /
        yield); turns and STOP are pass with no clearance."""
        checks, self._guard_checks = self._guard_checks, []
        vetoes = [c for c in checks if c[0] in ("abort", "stall")]
        if vetoes:
            flagged, (label, clearance) = 1, vetoes[-1]
        elif action == int(self._tryout_actions.MOVE_FORWARD) and checks:
            label, clearance = checks[-1]
            flagged = int(label == "yield")
        else:
            flagged, label, clearance = 0, "pass", None
        return {
            "guard_flagged": flagged,
            "guard_action": label,
            "guard_clearance_m": UNAVAILABLE if clearance is None else clearance,
            "guard_consecutive_aborts": self.tryout_guard._aborts,
        }

    def tryout_escape_count(self):
        """Tryout escapes so far this episode (steps.csv phase `escape` starts)."""
        chunk = self._tryout_chunk
        return self._tryout_escapes_before + (chunk.escapes if chunk is not None else 0)

    def _drop_tryout_chunk(self):
        if self._tryout_chunk is not None:
            self._tryout_escapes_before += self._tryout_chunk.escapes
        self._tryout_chunk = None

    def _start_tryout_chunk(self, response, position, rotation, actions, clearances):
        """Build this decision's TryoutChunk and check that, collision-free, it is exactly
        the worker's primitive list (i.e. the Tryout-off chunk)."""
        from .tryout import TryoutChunk

        if "hops" not in response:
            raise RuntimeError("--tryout needs a worker that returns `hops` (etpnav_worker)")
        hops = response["hops"]
        final_stop = self._stop_committed
        if not response["primitive_actions"]:
            # the off path turns an empty chunk into STOP (above); so does this one
            hops, final_stop = [], True
        chunk = TryoutChunk(
            hops, final_stop, position, rotation, self.turn_unit_deg, self.forward_unit_m,
            self._tryout_actions,
            pose_fn=lambda: agent_pose(self.env),
            collided_fn=lambda: bool(self.env.sim.previous_step_collided),
            rng=self._tryout_rng,
            veto_fn=self._guard_veto if self.tryout_guard is not None else None)
        expected = chunk.open_loop_primitives(hops, position, rotation, final_stop)
        if expected != actions:
            raise RuntimeError("tryout: hop plan {} != worker primitives {}".format(
                expected, actions))
        self._tryout_chunk = chunk
        self._tryout_clearances = clearances

    def _request_decision(self):
        if self._stop_committed:
            raise RuntimeError(
                "decision requested after the policy committed to STOP (decision {})".format(
                    self._decision_index))
        position, rotation = agent_pose(self.env)
        request = {
            "episode_reset": self._reset_pending,
            "decision_index": self._decision_index,
            "agent_position": position,
            "agent_rotation": rotation,
            "turn_unit_deg": self.turn_unit_deg,
            "forward_unit_m": self.forward_unit_m,
            "observation": encode_observation(self._last_obs, self.observation_keys),
            "agent_radius_m": self.agent_radius_m,
        }
        if self.exempt_guard_redecisions and self._guard_overridden:
            request["budget_exempt"] = True
        self._guard_overridden = False
        if self.stream_observations:
            # every frame since the last decision, oldest first; the last element is the
            # one `observation` above also carries, and is the frame generate() runs on.
            request["history_observations"] = [
                encode_observation(f, self.observation_keys) for f in self._frames[:-1]]
            self._frames = []
        if self.send_occupancy:
            request["occupancy_points_m"] = project_depth_to_occupancy_points_m(
                self._last_obs,
                min_depth_m=self.depth_min_depth_m,
                max_depth_m=self.depth_max_depth_m,
                sensor_keys=self.depth_sensor_keys,
                hfov_deg=self.depth_hfov_deg,
                camera_height_m=self.depth_camera_height_m,
            )
        response = self.handle.request(request)
        self._reset_pending = False
        self._decision_index += 1
        actions = [int(a) for a in response["primitive_actions"]]
        clearances = response.get("predicted_clearance_by_step")
        if not actions:
            actions = [STOP]
            clearances = None
        self._stop_committed = bool(response.get("stop")) or actions[-1] == STOP
        if not clearances or len(clearances) != len(actions):
            clearances = [None] * len(actions)
        self._queue = deque(actions)
        self._clearance_queue = deque(clearances)
        self.pending_decision_signals = response.get("risk_signals")
        if self.tryout:
            self._start_tryout_chunk(response, position, rotation, actions, clearances)
