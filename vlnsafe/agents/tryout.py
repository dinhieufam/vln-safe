"""ETP-lineage Tryout: the collision-escape heuristic ETPNav and ETP-R1 ship in their own
controller, ported so it can run as a baseline of its own.

What: the released controller walks each hop of a decision with single_step_control
(external/ETP-R1/vlnce_baselines/common/environments.py:384-442): turn toward the hop,
then k forward steps. With Tryout on, a forward step that collides ends that run of
forwards, and the hop's remaining steps are retried sideways: turn 90 deg to a random
side, then probe six headings (90, 60, 30, -30, -60, -90 deg from the hop's heading,
starting on the chosen side) with one forward step each. The first probe that changes
the agent's position is followed by the turn back to the hop's heading and the hop's
remaining forward steps, which stop at the next collision. If no probe moves the agent,
the hop is abandoned facing the last heading probed. Every hop of every decision gets
this, including the walk back to a stop node. The released code enables it exactly when
sliding is off (ss_trainer_ETP_R1.py:1009, `tryout: True` in both released
iter_train.yaml), so the `block` regime with Tryout is the agent as released.

Two differences from the released controller, both so the only change against the
landed Tryout-off arm is Tryout itself:
  - Hops are planned open-loop from the decision pose, as the landed arms do
    (vlnsafe.geometry.decompose_hops_to_primitives); the released controller re-aims each
    hop from the actual pose. A chunk in which no forward collides is therefore the
    landed chunk, primitive for primitive. After an escape the planned chain no longer
    matches the pose, so the hops still to come are re-planned from the actual pose,
    which is what the released controller does at every hop.
  - The side is drawn from an RNG the provider seeds per episode from (seed, episode_id),
    not from the global `random`, so a shard reproduces.

With a guard (`veto_fn`: `vlnsafe run --tryout --guard <name>`), every
forward step is checked before it executes, and a vetoed one is treated as blocked
without touching the obstacle. A vetoed forward on the policy's route starts the escape
as a collision would. A vetoed probe is skipped: that heading counts as "did not move".
A vetoed resume step ends the resume. Collisions the veto misses still trigger Tryout as
before. The guard's own manoeuvre and the policy re-decision are not used: the escape is
the agent's own. Without `veto_fn` the controller is exactly the N28 one.

Who reads it: vlnsafe.agents.remote (WorkerActionProvider with tryout on).

Must not import: habitat, torch. Python 3.6 (runs in the harness env).
"""
import math

from vlnsafe.geometry import hop_segments, turn_actions_for_angle

# (head turn, tail turn) in degrees, per probe, verbatim from single_step_control. The head
# turns accumulate from the 90 deg side turn; each tail turn restores the hop's heading.
PROBES_FROM_LEFT = ((0, 270), (330, 300), (330, 330), (300, 30), (330, 60), (330, 90))
PROBES_FROM_RIGHT = ((0, 90), (30, 60), (30, 30), (60, 330), (30, 300), (30, 270))

# step phases, logged per step in steps.csv `tryout_phase`
PLAN = "plan"        # the decision's open-loop chunk, as in the Tryout-off harness
ESCAPE = "escape"    # side turn, probe turns and probe forward steps
RESUME = "resume"    # turn back to the hop's heading and the hop's remaining forwards
REPLAN = "replan"    # hops re-planned from the actual pose after an escape


class TryoutChunk(object):
    """One decision's primitives, produced one at a time with Tryout applied.

    The caller alternates current() -> execute that action in the simulator ->
    executed(). The controller reads the post-step pose and collision flag lazily, on the
    next current(), so it always sees the state the executed action produced.

    `pose_fn()` -> (position [x, y, z], rotation [qx, qy, qz, qw]); `collided_fn()` ->
    the simulator's previous_step_collided. `hops` and the decision pose are the ones the
    worker decomposed, so without a collision this yields exactly the worker's list.
    """

    def __init__(self, hops, final_stop, position, rotation, turn_unit_deg,
                 forward_unit_m, actions, pose_fn, collided_fn, rng, veto_fn=None):
        self.turn_unit_deg = float(turn_unit_deg)
        self.forward_unit_m = float(forward_unit_m)
        self.actions = actions
        self.forward = int(actions.MOVE_FORWARD)
        self.stop = int(actions.STOP)
        self.pose_fn = pose_fn
        self.collided_fn = collided_fn
        self.rng = rng
        # veto_fn(policy_step) -> True = do not execute this forward (vlnsafe.guard
        # check_forward via the provider); policy_step is False for probe/resume forwards
        self.veto_fn = veto_fn
        self.escapes = 0
        # set with each action: its phase, the known plan from it on (for planned_chunk),
        # and its index in the worker's primitive list while still on that list, else None
        self.phase = None
        self.plan = []
        self.worker_index = None
        self._current = None
        self._finished = False
        self._gen = self._run(list(hops), bool(final_stop), position, rotation)

    def open_loop_primitives(self, hops, position, rotation, final_stop):
        segs = self._segments(hops, position, rotation)
        return self._flatten(segs) + ([self.stop] if final_stop else [])

    def current(self):
        """The next primitive, or None once the decision is fully executed."""
        if self._current is None and not self._finished:
            try:
                self._current = next(self._gen)
            except StopIteration:
                self._finished = True
        if self._current is None:
            return None
        action, self.phase, rest, self.worker_index = self._current
        self.plan = [action] + rest
        return action

    def executed(self):
        self._current = None

    # --- internals -------------------------------------------------------------------
    def _segments(self, hops, position, rotation):
        return hop_segments(hops, position, rotation, self.turn_unit_deg,
                            self.forward_unit_m, self.actions)

    def _flatten(self, segs):
        return [a for turns, k in segs for a in list(turns) + [self.forward] * k]

    def _turn(self, degrees):
        return turn_actions_for_angle(
            math.radians(degrees), self.turn_unit_deg, self.actions)[0]

    def _position(self):
        return list(self.pose_fn()[0])

    def _vetoed(self, policy_step):
        return self.veto_fn is not None and bool(self.veto_fn(policy_step))

    def _run(self, hops, final_stop, position, rotation):
        segs = self._segments(hops, position, rotation)
        tail_stop = [self.stop] if final_stop else []
        phase = PLAN
        index = 0          # position in the worker's list while phase == PLAN

        def emit(action, rest):
            return (action, phase, rest, index if phase == PLAN else None)

        i = 0
        while i < len(segs):
            turns, k = segs[i]
            tail = self._flatten(segs[i + 1:]) + tail_stop
            for j, a in enumerate(turns):
                yield emit(a, list(turns[j + 1:]) + [self.forward] * k + tail)
                index += 1
            moved = 0
            for j in range(k):
                if self._vetoed(True):
                    break
                yield emit(self.forward, [self.forward] * (k - j - 1) + tail)
                index += 1
                if self.collided_fn():
                    break
                moved += 1
            if k - moved > 0:
                for item in self._escape(k - moved):
                    yield item
                position, rotation = self.pose_fn()
                segs = segs[:i + 1] + self._segments(hops[i + 1:], position, rotation)
                phase = REPLAN
            i += 1
        if final_stop:
            yield emit(self.stop, [])

    def _escape(self, remaining):
        """single_step_control's `else` branch after the forward run broke on a collision,
        with `remaining` = the hop's forward steps not yet taken (the colliding one
        included)."""
        self.escapes += 1
        side = self.rng.choice((90, 270))
        probes = PROBES_FROM_LEFT if side == 90 else PROBES_FROM_RIGHT
        for a in self._turn(side):
            yield (a, ESCAPE, [], None)
        for head, tail in probes:
            for a in self._turn(head):
                yield (a, ESCAPE, [], None)
            if self._vetoed(False):
                continue
            before = self._position()
            yield (self.forward, ESCAPE, [], None)
            if self._position() != before:
                back = self._turn(tail)
                for j, a in enumerate(back):
                    yield (a, RESUME, list(back[j + 1:]) + [self.forward] * remaining, None)
                for j in range(remaining):
                    if self._vetoed(False):
                        break
                    yield (self.forward, RESUME, [self.forward] * (remaining - j - 1), None)
                    if self.collided_fn():
                        break
                return
