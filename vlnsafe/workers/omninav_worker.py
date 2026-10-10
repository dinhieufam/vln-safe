#!/usr/bin/env python3
"""OmniNav worker (runs under vlnsafe-omninav, py3.9 / torch 2.6, no habitat).

What: OmniNav (Qwen2.5-VL-3B + flow-matching waypoint head, the released `OmniNav_Flow`
checkpoint) as a harness agent. One `decide` request per OmniNav step: the worker adds the
tri-view frame (front / left / right RGB) to OmniNav's own history, runs one forward pass,
and turns the first predicted waypoint into harness primitives.

Upstream executes that waypoint with its own `GO_TOWARD_POINT` task action -- rotate by
`recover_angle`, then one straight `step_filter` move of `r` metres (0.1-0.4 m) -- which
has no engine-collision flag and no forward/turn steps. The harness measures contact on
habitat's primitive actions for every agent, so the waypoint is walked out instead as
TURN_LEFT/RIGHT to the nearest turn unit, then MOVE_FORWARD steps. The forward count is
rounded, with the rounding remainder carried to the next decision of the episode, so the
distance walked stays unbiased over an episode (a 0.3 m waypoint is one 0.25 m step, and
the 0.05 m left over is spent later). A decision that would otherwise be empty (no turn,
no forward, no arrival) walks one forward step -- an empty chunk reads as STOP to the
provider. docs/agent_notes/omninav.md.

Upstream's eval loop also STOPs after 150 steps, and after 25 consecutive steps at which
distance-to-goal did not change (the agent did not move). The worker applies both, per
decision; the second is keyed on the agent's own position, not the privileged distance.

Who talks to it: vlnsafe.agents.remote (WorkerActionProvider, no occupancy payload,
OMNINAV_OBSERVATION_KEYS, one frame per decision -- OmniNav's history holds only the
frames it decided on).

Ported: the agent is external/OmniNav @ e8b4859 infer_r2r_rxr/agent/waypoint_agent.py,
imported, not copied; `_QuietAgent.act` is its `act()` minus the image/video writes.
"""
import argparse
import base64
import json
import math
import os
import sys
import zlib

import numpy as np

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
# rgb uuid (harness rig, vlnsafe.simulator.sensors.OMNINAV_VIEWS) -> OmniNav's key
_VIEWS = (("rgb_front", "front"), ("rgb_left", "left"), ("rgb_right", "right"))


class _Actions(object):
    STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT


def waypoint_to_primitives(r_m, theta_rad, turn_unit_deg, forward_unit_m, residual_m=0.0):
    """One OmniNav waypoint (move `r_m` along heading + `theta_rad`, positive = left, as
    GO_TOWARD_POINT's rtheta_to_global_coordinates) -> (primitives, residual_m).

    Turns: theta snapped to the nearest turn unit (vlnsafe.geometry.turn_actions_for_angle,
    the ETPNav hop rule). Forwards: round((r + residual) / unit), the remainder returned
    for the next decision. Never empty: a waypoint that rounds to no motion at all walks
    one forward step (and owes it back through the residual)."""
    from vlnsafe.geometry import turn_actions_for_angle

    turns, _applied = turn_actions_for_angle(float(theta_rad), turn_unit_deg, _Actions)
    want = float(r_m) + float(residual_m)
    steps = max(0, int(math.floor(want / forward_unit_m + 0.5)))
    if not turns and steps == 0:
        steps = 1
    return list(turns) + [MOVE_FORWARD] * steps, want - steps * forward_unit_m


def arrived(arrive_values, threshold=0.5):
    """Upstream's flow-matching stop rule (waypoint_agent.py act(), flow_match branch):
    STOP only when every one of the 5 per-waypoint arrive values exceeds 0.5."""
    values = [float(v) for v in np.asarray(arrive_values, dtype=np.float64).reshape(-1)]
    return bool(values) and all(v > threshold for v in values)


class EarlyStop(object):
    """Upstream's two eval-loop cut-offs (waypoint_agent.py evaluate_agent; r2r_pano.yaml
    EVAL.EARLY_STOP_STEPS 150, EARLY_STOP_ROTATION 25). `step` is called once per decision
    with the agent's position and returns True when the loop would issue STOP: after more
    than `max_steps` decisions, or more than `max_still` consecutive decisions at which the
    agent had not moved since the previous one (upstream: distance-to-goal unchanged)."""

    def __init__(self, max_steps=150, max_still=25, tol_m=1e-4):
        self.max_steps, self.max_still, self.tol_m = int(max_steps), int(max_still), tol_m
        self.reset()

    def reset(self):
        self.decisions = 0
        self.still = 0
        self._last = None

    def step(self, position):
        position = np.asarray(position, dtype=np.float64)
        if self._last is not None and np.linalg.norm(position - self._last) < self.tol_m:
            self.still += 1
        else:
            self.still = 0
        self._last = position
        stop = self.still > self.max_still or self.decisions > self.max_steps
        self.decisions += 1
        return stop


def decision_seed(seed, instruction, decision_index):
    """Flow matching starts from torch.randn noise. Seeding it per (run seed, episode,
    decision) makes an episode's actions independent of the shard it runs in."""
    key = "{}:{}:{}".format(int(seed), instruction, int(decision_index))
    return zlib.crc32(key.encode("utf-8"))


def _decode_observation(payload):
    obs = {}
    for key, value in payload.items():
        if key == "instruction":
            obs[key] = value
            continue
        raw = base64.b64decode(value["data_b64"])
        obs[key] = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(
            tuple(value["shape"])).copy()
    return obs


class _RedirectStdoutToDevnull(object):
    """fd 1 -> /dev/null for a block: upstream prints on every step, and any stray line
    desyncs the caller's readline() protocol (same reason as the other workers)."""

    def __enter__(self):
        sys.stdout.flush()
        self._saved = os.dup(1)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self._saved, 1)
        os.close(self._null)
        os.close(self._saved)


def _load_agent_class(code_root):
    """external/OmniNav/infer_r2r_rxr: import its Waypoint_Agent with the flow-matching
    branch on and every disk write off."""
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from agent import waypoint_agent as upstream

    upstream.flow_match = True       # run_infer.py --flow-match true (eval_r2r.sh default)
    upstream.SAVE_RANDER_IMG = False

    class _QuietAgent(upstream.Waypoint_Agent):
        """Waypoint_Agent.act minus the per-frame PNG and top-down-map writes; returns the
        raw arrive values too, for the risk read-out."""

        def act(self, observations):
            torch = upstream.torch
            self.add_frame([observations["left"], observations["right"],
                            observations["front"]], observations["pose"])
            messages = self.generate_infer_prompt(observations["instruction"]["text"])
            wp_pred, arrive_raw, sin_angle, cos_angle = self.model.qwen_infer(messages)
            wp = wp_pred.cpu().type(torch.float32).numpy().squeeze()
            recover = torch.atan2(sin_angle, cos_angle).detach().cpu().type(
                torch.float32).numpy().squeeze()
            arrive = arrive_raw.detach().cpu().type(torch.float32).numpy().squeeze()
            return {"waypoints": np.atleast_2d(wp), "recover_angle": np.atleast_1d(recover),
                    "arrive": np.atleast_1d(arrive)}

    return upstream, _QuietAgent


class OmniNavEngine(object):
    def __init__(self, code_root, checkpoint_path, seed, model_cfg=None):
        model_cfg = model_cfg or {}
        os.chdir(code_root)
        self.upstream, agent_cls = _load_agent_class(code_root)
        self.torch = self.upstream.torch
        # result_path=None: no output directories; require_map=False: no top-down map
        self.agent = agent_cls(checkpoint_path, None, require_map=False)
        self.seed = int(seed)
        self.early_stop = EarlyStop(model_cfg.get("early_stop_steps", 150),
                                    model_cfg.get("early_stop_still", 25))
        self.waypoint_index = int(model_cfg.get("execute_waypoint", 0))
        self._residual_m = 0.0
        self._instruction = ""

    def reset_episode(self, observation):
        instr = observation.get("instruction", {})
        self._instruction = instr.get("text", "") if isinstance(instr, dict) else str(instr)
        self.agent.reset()
        self.early_stop.reset()
        self._residual_m = 0.0

    def decide(self, observation, position, rotation, decision_index,
               turn_unit_deg, forward_unit_m):
        # the upstream loop checks its cut-offs before calling the agent
        if self.early_stop.step(position):
            return {"primitive_actions": [STOP], "stop": True, "risk_signals": {},
                    "meta": {"early_stop": True, "decision": decision_index}}
        qx, qy, qz, qw = rotation
        frames = dict((ours, observation[ours]) for ours, _ in _VIEWS)
        obs = dict((theirs, np.asarray(frames[ours])[..., :3].astype(np.uint8))
                   for ours, theirs in _VIEWS)
        obs["pose"] = {"position": list(position), "rotation": [qw, qx, qy, qz]}
        obs["instruction"] = {"text": self._instruction}
        self.torch.manual_seed(decision_seed(self.seed, self._instruction, decision_index))
        with self.torch.no_grad():   # run_infer.py wraps evaluate_agent in no_grad
            out = self.agent.act(obs)
        arrive = out["arrive"]
        signals = {"stop_prob": float(np.min(arrive)),
                   "arrive_by_waypoint": [float(a) for a in arrive]}
        if arrived(arrive):
            return {"primitive_actions": [STOP], "stop": True, "risk_signals": signals,
                    "meta": {"arrive": True, "decision": decision_index}}
        k = self.waypoint_index
        waypoint = out["waypoints"][k]
        r_m = float(np.linalg.norm(waypoint))
        theta = float(out["recover_angle"][k])
        primitives, self._residual_m = waypoint_to_primitives(
            r_m, theta, turn_unit_deg, forward_unit_m, self._residual_m)
        return {"primitive_actions": primitives, "stop": False, "risk_signals": signals,
                "meta": {"r_m": r_m, "theta_rad": theta, "residual_m": self._residual_m,
                         "decision": decision_index}}


def handle_request(engine, request):
    if request.get("cmd") != "decide":
        return {"ok": False, "error": "unknown cmd {!r}".format(request.get("cmd"))}
    observation = _decode_observation(request["observation"])
    if request.get("episode_reset"):
        engine.reset_episode(observation)
    decision = engine.decide(
        observation, request["agent_position"], request["agent_rotation"],
        int(request.get("decision_index", 0)),
        float(request.get("turn_unit_deg", 15.0)), float(request.get("forward_unit_m", 0.25)))
    return {
        "ok": True,
        "primitive_actions": [int(a) for a in decision["primitive_actions"]],
        "stop": bool(decision["stop"]),
        "risk_signals": decision["risk_signals"],
        "predicted_clearance_by_step": None,
        "meta": decision["meta"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp-config", default=None)
    parser.add_argument("--precision", default="bf16", choices=["bf16"],
                        help="upstream loads torch_dtype='auto' (the checkpoint's bf16)")
    parser.add_argument("--model-config", default="{}")
    parser.add_argument("--hfov-deg", type=float, default=110.0)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    with _RedirectStdoutToDevnull():
        engine = OmniNavEngine(args.code_root, args.checkpoint, args.seed,
                               json.loads(args.model_config))
    print("VLNSAFE_OMNINAV_WORKER_READY", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": "bad json: {}".format(exc)}), flush=True)
            continue
        if request.get("cmd") == "shutdown":
            break
        try:
            with _RedirectStdoutToDevnull():
                response = handle_request(engine, request)
        except Exception as exc:  # noqa: BLE001
            import traceback
            response = {"ok": False, "error": "{}: {}".format(exc, traceback.format_exc())}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
