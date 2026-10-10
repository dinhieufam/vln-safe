"""Training-free collision guard: veto a forward primitive whose predicted clearance is
below a metric margin, and replace it with a short escape manoeuvre.

What: one GuardLayer per shard, reset per episode. On EVERY executed step -- not only
decision steps, since ETP-R1's contacts happen inside a waypoint's primitive chunk --
rollout hands it the pre-step frame, the agent pose and the primitive the policy proposed.
The guard projects the frame to an occupancy cloud, merges it with the last
`memory_frames` frames' clouds re-expressed in the current ego frame by odometry, and
checks the clearance at the pose a forward step ends in. The memory is there because a
low obstacle leaves the depth band inside ~0.9 m:
a per-frame check sees it only while it is still too far away to matter. Only
MOVE_FORWARD can produce an engine collision; turns and STOP always pass.

A forward step whose clearance is below `threshold_m` (default 0.05 m, one navmesh voxel)
is aborted and replaced by a manoeuvre: the smallest turn, up to `max_escape_deg` either
side, whose `lookahead_steps`-step ray clears the threshold, followed by one forward step
along it. If no heading clears, the manoeuvre is only the turns toward the freest heading.
Then control returns to the policy. A provider that queues primitives (the worker
providers) gets the manoeuvre installed in its queue through `override_chunk`, so it
drains without a model call and the policy re-decides once, from the pose the manoeuvre
reached. A per-step policy (CMA/Seq2Seq) is still called every step, and its proposals
are overridden until the manoeuvre ends. A proposed STOP always wins, and a veto inside a
chunk that ends in the policy's STOP executes that STOP at once ("stop"): the policy has
already chosen to end the episode, and an ETPNav-family worker cannot re-decide after a
stop decision -- its graph loses the link to the previous node (prev_vp = None) and its
budget check (stepk == max_len - 1) never fires again. N10's first launch did exactly
that: 62 of 720 episodes ran past 15 decisions, up to 75, and one shard crashed on the
disconnected graph.

`max_consecutive_aborts` bounds a livelock (target behind the obstacle, and the policy
keeps re-entering it). Aborts are counted until a policy-proposed forward step executes
unflagged; once the count reaches the cap, flagged forwards pass through, logged "yield",
until that happens. That does not bound an abort -> manoeuvre -> re-decide -> re-approach
cycle, because one clean policy step resets the count: M1b left 7.5 % of
episodes looping to the 500-step limit. `max_episode_aborts` is the episode-level bound:
after that many aborts the guard yields for the rest of the episode. None means unbounded.

`stall_m` (None = off) adds a stall veto. Depth cannot see every obstacle:
contact against a navmesh edge with no rendered surface reads as clear, and an agent can press
forward into it for hundreds of steps (the N11 smoke: CMA, 0.23 m of "clearance" on every one).
The simulator is deterministic, so a forward that moved less than `stall_m` from some pose
will fail again from the same pose. The guard remembers, per episode, each pose (position
within `stall_m`, heading within half a turn unit) from which an executed forward stalled,
and vetoes a forward from it ("stall"). On the landed arms, a forward right after a stalled
forward collided every time (CMA 543,935 of 543,935, ETP-R1 27,085 of 27,085), and such
forwards are 98.7 % / 81.2 % of their collision steps. A stall veto takes the same
manoeuvre as an abort, but it neither counts toward nor obeys the two abort caps: those
yield a flagged forward because depth may be wrong, and a forward known to fail never
gains anything by being let through. It uses only the odometry the guard already reads.

`check_forward` is the same veto without the manoeuvre, for the agent's
own Tryout escape (vlnsafe/agents/tryout.py): with `run --tryout --guard`, the provider
asks it before each of Tryout's forward steps and Tryout escapes instead of the guard.

`eps` (score-margin reselection among a T2 agent's candidates)
is fixed at 0 here: the manoeuvre is the eps=0 branch. eps>0 needs the worker to
return its alternatives' primitive sequences, and is refused rather than silently ignored.

Who reads it: vlnsafe.cli (builds it from configs/guard/<name>.yaml), vlnsafe.rollout.

Pure numpy; must not import habitat or torch. Python 3.6 compatible.
"""
import math
from collections import deque

import numpy as np

from .geometry import _quat_rotate_vector, _quat_wxyz_from_coeff
from .records.schema import UNAVAILABLE

# The step-row columns this layer owns, at their value in a run without --guard.
GUARD_OFF_ROW = {
    "guard_flagged": UNAVAILABLE,
    "guard_action": UNAVAILABLE,
    "guard_clearance_m": UNAVAILABLE,
    "guard_consecutive_aborts": UNAVAILABLE,
}

_CONFIG_KEYS = ("threshold_m", "max_consecutive_aborts", "max_episode_aborts",
                "memory_frames", "lookahead_steps", "max_escape_deg", "eps", "stall_m",
                "depth_veto", "oracle")
ORACLE_BLOCKED_M = -1.0   # clearance an oracle guard logs for a forward the navmesh would shorten
ORACLE_CLEAR_M = 1.0      # ... and for one it would not (both only ever compared with threshold_m)


def _planar_axes(rotation):
    """World (x, z) unit vectors of the agent's forward (-z) and left (-x) axes, from a
    habitat [x, y, z, w] rotation."""
    q = _quat_wxyz_from_coeff(rotation)
    fwd = _quat_rotate_vector(q, np.array([0.0, 0.0, -1.0]))[[0, 2]]
    left = _quat_rotate_vector(q, np.array([-1.0, 0.0, 0.0]))[[0, 2]]
    return fwd / np.linalg.norm(fwd), left / np.linalg.norm(left)


def ego_to_world(points, position, rotation):
    """[ego_forward_m, ego_left_m] rows -> world (x, z) rows, at this pose."""
    fwd, left = _planar_axes(rotation)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    origin = np.array([position[0], position[2]], dtype=np.float64)
    return origin + pts[:, :1] * fwd + pts[:, 1:] * left


def world_to_ego(world, position, rotation):
    """Inverse of ego_to_world."""
    fwd, left = _planar_axes(rotation)
    d = np.asarray(world, dtype=np.float64).reshape(-1, 2) - np.array(
        [position[0], position[2]], dtype=np.float64)
    # elementwise, not np.dot: the harness's BLAS dot costs ~18 ms a call on this loaded
    # 64-core box (thread contention), which made the guard ~10x slower than its projection
    return np.stack([d[:, 0] * fwd[0] + d[:, 1] * fwd[1],
                     d[:, 0] * left[0] + d[:, 1] * left[1]], axis=1)


def _heading(direction):
    """World-plane heading (rad) of a world (x, z) direction."""
    return math.atan2(direction[1], direction[0])


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GuardLayer(object):
    def __init__(self, project, forward_unit_m, turn_unit_deg, agent_radius_m, actions,
                 threshold_m=0.05, max_consecutive_aborts=3, max_episode_aborts=None,
                 memory_frames=8, lookahead_steps=2, max_escape_deg=90.0, eps=0.0,
                 stall_m=None, depth_veto=True, oracle=False):
        if eps:
            raise NotImplementedError(
                "guard eps={} (candidate reselection) needs worker support; only eps=0 "
                "is implemented".format(eps))
        self.project = project    # obs -> ego-frame occupancy points (list of [fwd, left])
        self.forward_unit_m = float(forward_unit_m)
        self.turn_unit_deg = float(turn_unit_deg)
        self.agent_radius_m = float(agent_radius_m)
        self.threshold_m = float(threshold_m)
        self.max_consecutive_aborts = int(max_consecutive_aborts)
        self.max_episode_aborts = (None if max_episode_aborts is None
                                   else int(max_episode_aborts))
        self.memory_frames = int(memory_frames)
        self.lookahead_steps = int(lookahead_steps)
        self.max_escape_deg = float(max_escape_deg)
        self.stall_m = None if stall_m is None else float(stall_m)
        # False = the stall-only ablation: depth never vetoes a forward, but the
        # clearance is still logged and the escape still picks its heading from depth.
        self.depth_veto = bool(depth_veto)
        # True = the oracle guard: the navmesh, not depth, says whether a displacement
        # would be shortened -- for the pending forward and for every escape heading. The
        # caller installs `oracle_fn` (see navmesh_oracle) once the simulator exists.
        self.oracle = bool(oracle)
        self.oracle_fn = None
        self._pose = None
        self.stop = int(actions.STOP)
        self.forward = int(actions.MOVE_FORWARD)
        self.left = int(actions.TURN_LEFT)
        self.right = int(actions.TURN_RIGHT)
        self.reset()

    @classmethod
    def from_config(cls, guard_cfg, predictor):
        """`predictor` is the vlnsafe.clearance_predictor.PrimitiveClearancePredictor for
        this rig: the guard reuses its projection and its rig constants, so the guard and
        the logged predicted_clearance_m column can never disagree on the camera."""
        unknown = sorted(set(guard_cfg) - set(_CONFIG_KEYS))
        if unknown:
            raise ValueError("unknown guard config key(s): " + ", ".join(unknown))
        return cls(predictor.points, predictor.forward_unit_m, predictor.turn_unit_deg,
                   predictor.agent_radius_m, predictor.actions, **guard_cfg)

    def reset(self):
        self._memory = deque(maxlen=self.memory_frames)   # world (x, z) arrays, per frame
        self._maneuver = deque()
        self._aborts = 0
        self.abort_total = 0    # -> episodes.csv guard_abort_count (depth aborts only)
        self._policy_aborts = 0  # check_forward only: depth vetoes on policy forwards
        self.yield_total = 0    # -> episodes.csv guard_yield_count
        self.stall_total = 0    # stall vetoes; in steps.csv as guard_action == "stall"
        self._blocked = []      # (world x, z, heading rad) an executed forward stalled from
        self._forward_from = None

    def step(self, obs, proposed, pose, provider=None):
        """(frame, proposed primitive, (position, rotation)) -> (primitive to execute,
        this step's guard_* columns). `obs` is the pre-step frame the proposal was made
        on."""
        position, rotation = pose
        self._pose = pose
        self._axes = _planar_axes(rotation)
        here = (float(position[0]), float(position[2]), _heading(self._axes[0]))
        self._note_stall(here)
        self._memory.append(ego_to_world(self.project(obs), position, rotation))
        ego = world_to_ego(np.concatenate(list(self._memory)), position, rotation)

        if proposed == self.stop:
            self._maneuver.clear()
            return proposed, self._row(0, "pass", None)
        in_maneuver = bool(self._maneuver)
        candidate = self._maneuver.popleft() if in_maneuver else proposed
        label = "maneuver" if in_maneuver else "pass"
        if candidate != self.forward:
            return candidate, self._row(0, label, None)

        clearance = self._clearance(ego, self.forward_unit_m, 0.0)
        if self._is_blocked(here):
            self.stall_total += 1
            return self._veto(ego, provider, here, "stall", clearance)
        if not self.depth_veto or clearance is None or clearance >= self.threshold_m:
            if not in_maneuver:
                self._aborts = 0
            return self._go(candidate, here, self._row(0, label, clearance))
        if self._aborts >= self.max_consecutive_aborts or (
                self.max_episode_aborts is not None
                and self.abort_total >= self.max_episode_aborts):
            self.yield_total += 1
            return self._go(candidate, here, self._row(1, "yield", clearance))

        self._aborts += 1
        self.abort_total += 1
        return self._veto(ego, provider, here, "abort", clearance)

    def check_forward(self, obs, pose, policy_step=True):
        """The verdict step() reaches on a proposed forward, without the manoeuvre, for a
        controller that runs its own escape: the agent's Tryout with --guard
        (vlnsafe/agents/tryout.py). Returns (verdict, clearance):
        "pass" / "yield" mean execute the forward now, "abort" / "stall" mean do not.
        Counters and the stall memory update as in step(). Stall bookkeeping needs the
        caller to execute the forward right after a pass or yield. Only memory_frames = 1:
        frames are seen only at checks, not at every step.

        `policy_step` False marks an escape probe or resume step. A pass there does not
        reset the consecutive-abort count. A veto there is never yielded and does not
        count toward either cap: the caps bound an abort -> re-decide loop on the policy's
        route, and Tryout's probes are bounded by construction (six per escape). With the
        caps counting them, one escape's vetoed probes exhausted the consecutive cap, and
        28 of the 29 forwards then yielded collided (N34 smoke). abort_total still counts
        every depth veto."""
        if self.memory_frames != 1:
            raise ValueError("check_forward needs memory_frames = 1")
        position, rotation = pose
        self._pose = pose
        self._axes = _planar_axes(rotation)
        here = (float(position[0]), float(position[2]), _heading(self._axes[0]))
        self._note_stall(here)
        self._memory.append(ego_to_world(self.project(obs), position, rotation))
        ego = world_to_ego(np.concatenate(list(self._memory)), position, rotation)
        clearance = self._clearance(ego, self.forward_unit_m, 0.0)
        if self._is_blocked(here):
            self.stall_total += 1
            return "stall", clearance
        if not self.depth_veto or clearance is None or clearance >= self.threshold_m:
            if policy_step:
                self._aborts = 0
            self._forward_from = here
            return "pass", clearance
        if not policy_step:
            self.abort_total += 1
            return "abort", clearance
        if self._aborts >= self.max_consecutive_aborts or (
                self.max_episode_aborts is not None
                and self._policy_aborts >= self.max_episode_aborts):
            self.yield_total += 1
            self._forward_from = here
            return "yield", clearance
        self._aborts += 1
        self._policy_aborts += 1
        self.abort_total += 1
        return "abort", clearance

    # --- internals -----------------------------------------------------------------
    def _veto(self, ego, provider, here, label, clearance):
        if self.stop in (provider.planned_chunk() if hasattr(provider, "planned_chunk") else []):
            self._maneuver.clear()
            return self.stop, self._row(1, "stop", clearance)
        plan = self._escape(ego, here)
        self._maneuver = deque(plan[1:])
        if hasattr(provider, "override_chunk"):
            provider.override_chunk(plan)
        return self._go(plan[0], here, self._row(1, label, clearance))

    def _go(self, action, here, row):
        """Remember where an executed forward started, so the next step can tell whether
        it moved."""
        self._forward_from = here if action == self.forward else None
        return action, row

    def _note_stall(self, here):
        start, self._forward_from = self._forward_from, None
        if self.stall_m is None or start is None:
            return
        if math.hypot(here[0] - start[0], here[1] - start[1]) < self.stall_m:
            self._blocked.append(start)

    def _is_blocked(self, here):
        if self.stall_m is None:
            return False
        tol = math.radians(self.turn_unit_deg) / 2.0
        for x, z, h in self._blocked:
            if (math.hypot(here[0] - x, here[1] - z) < self.stall_m
                    and abs(_wrap(here[2] - h)) < tol):
                return True
        return False

    def _clearance(self, ego, x_fwd, y_left):
        if self.oracle:
            if self.oracle_fn is None:
                raise RuntimeError("oracle guard without an oracle_fn (see navmesh_oracle)")
            target = ego_to_world([[x_fwd, y_left]], *self._pose)[0]
            return ORACLE_BLOCKED_M if self.oracle_fn(self._pose[0], target) else ORACLE_CLEAR_M
        if ego.shape[0] == 0:
            return None
        return float(np.hypot(ego[:, 0] - x_fwd, ego[:, 1] - y_left).min()) - self.agent_radius_m

    def _ray_clearance(self, ego, theta_rad):
        step = self.forward_unit_m
        return min(
            self._clearance(ego, j * step * math.cos(theta_rad), j * step * math.sin(theta_rad))
            for j in range(1, self.lookahead_steps + 1))

    def _escape(self, ego, here):
        """Smallest safe turn + one forward step, else turns toward the freest heading.
        Always starts with a turn: the straight-ahead heading is the one just flagged. A
        heading a forward already stalled on from this pose is never safe."""
        fwd, left = self._axes
        freest = None   # (ray clearance, k, turn action)
        for k in range(1, int(self.max_escape_deg // self.turn_unit_deg) + 1):
            safe = []
            for sign, turn in ((1, self.left), (-1, self.right)):
                theta = math.radians(sign * k * self.turn_unit_deg)
                ray = self._ray_clearance(ego, theta)
                if freest is None or ray > freest[0]:
                    freest = (ray, k, turn)
                heading = _heading(math.cos(theta) * fwd + math.sin(theta) * left)
                if ray >= self.threshold_m and not self._is_blocked(here[:2] + (heading,)):
                    safe.append((ray, turn))
            if safe:
                turn = max(safe, key=lambda s: s[0])[1]
                return [turn] * k + [self.forward]
        return [freest[2]] * freest[1]

    def _row(self, flagged, action, clearance):
        return {
            "guard_flagged": flagged,
            "guard_action": action,
            "guard_clearance_m": UNAVAILABLE if clearance is None else clearance,
            "guard_consecutive_aborts": self._aborts,
        }


def navmesh_oracle(sim, sliding):
    """(position [x, y, z], world target (x, z)) -> True if the navmesh would shorten the
    straight move -- the same filter and the same test as the engine contact flag
    (paper Sec. 3.1: |p' - p|^2 + 1e-5 < |p* - p|^2). `sliding` picks the regime's filter.
    Takes the simulator, not its pathfinder: the navmesh changes with every scene load, so
    it is looked up per query (binding the first pathfinder queried the wrong scene --
    N41 smoke, 143 of 148 flagged forwards did not collide)."""
    def blocked(position, target):
        pathfinder = sim.pathfinder
        step = pathfinder.try_step if sliding else pathfinder.try_step_no_sliding
        start = np.asarray(position, dtype=np.float32)
        end = np.array([target[0], position[1], target[1]], dtype=np.float32)
        got = np.asarray(step(start, end), dtype=np.float64)
        return (float(((got - start) ** 2).sum()) + 1e-5
                < float(((end - start) ** 2).sum()))
    return blocked
