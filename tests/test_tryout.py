"""vlnsafe/agents/tryout.py: the ETP-lineage Tryout controller, checked
against single_step_control's behaviour in a small no-sliding world, plus its wiring into
the worker provider and the CLI. No habitat."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
from geometry_inputs import ACTIONS  # noqa: E402

from vlnsafe import geometry  # noqa: E402
from vlnsafe.agents import tryout as T  # noqa: E402
from vlnsafe.records import schema  # noqa: E402

FWD, LEFT, RIGHT, STOP = (int(ACTIONS.MOVE_FORWARD), int(ACTIONS.TURN_LEFT),
                          int(ACTIONS.TURN_RIGHT), int(ACTIONS.STOP))


class _Quat(object):
    def __init__(self, xyzw):
        self.imag = np.array(xyzw[:3])
        self.real = float(xyzw[3])


class _State(object):
    def __init__(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = _Quat(rotation)


class World(object):
    """No-sliding habitat stand-in on the x/z plane: heading 0 faces -z, TURN_LEFT adds
    `turn_deg`, a forward step moves 0.25 m unless `blocked(x, z)` holds for its end point,
    in which case the agent stays put and previous_step_collided is set."""

    def __init__(self, blocked=lambda x, z: False, turn_deg=15.0, x=0.0, z=0.0, yaw_deg=0.0):
        self.blocked = blocked
        self.turn_deg = turn_deg
        self.x, self.z, self.yaw = x, z, math.radians(yaw_deg)
        self.previous_step_collided = False
        self.log = []

    def pose(self):
        h = self.yaw / 2.0
        return [self.x, 0.0, self.z], [0.0, math.sin(h), 0.0, math.cos(h)]

    def step(self, action):
        self.log.append(action)
        self.previous_step_collided = False
        if action == LEFT:
            self.yaw += math.radians(self.turn_deg)
        elif action == RIGHT:
            self.yaw -= math.radians(self.turn_deg)
        elif action == FWD:
            nx = self.x - 0.25 * math.sin(self.yaw)
            nz = self.z - 0.25 * math.cos(self.yaw)
            if self.blocked(nx, nz):
                self.previous_step_collided = True
            else:
                self.x, self.z = nx, nz

    # the bits of a habitat env the provider reads
    @property
    def sim(self):
        return self

    def get_agent_state(self):
        position, rotation = self.pose()
        return _State(position, rotation)


class _Side(object):
    """rng stand-in: always picks `side` (90 = left first, 270 = right first)."""

    def __init__(self, side):
        self.side = side

    def choice(self, options):
        assert tuple(options) == (90, 270)
        return self.side


def _chunk(world, hops, final_stop=False, side=90):
    position, rotation = world.pose()
    return T.TryoutChunk(hops, final_stop, position, rotation, world.turn_deg, 0.25, ACTIONS,
                         pose_fn=world.pose, collided_fn=lambda: world.previous_step_collided,
                         rng=_Side(side))


def _drive(world, chunk, limit=200):
    """Execute the chunk to its end; returns [(action, phase, worker_index), ...]."""
    out = []
    for _ in range(limit):
        a = chunk.current()
        if a is None:
            return out
        out.append((a, chunk.phase, chunk.worker_index))
        assert chunk.plan[0] == a
        world.step(a)
        chunk.executed()
    raise AssertionError("chunk did not end")


def test_world_heading_matches_the_geometry_convention():
    w = World(yaw_deg=90)
    assert math.isclose(geometry.heading_from_quaternion(np.array(w.pose()[1])),
                        math.pi / 2, abs_tol=1e-9)
    w.step(FWD)                                    # heading 90 deg = -x, i.e. to the left
    assert math.isclose(w.x, -0.25) and math.isclose(w.z, 0.0, abs_tol=1e-12)


def test_without_a_collision_the_chunk_is_the_worker_list():
    w = World()
    hops = [[0.3, 0.0, -1.0], [0.3, 0.0, -2.1], [-0.8, 0.0, -2.1]]
    position, rotation = w.pose()
    worker = geometry.decompose_hops_to_primitives(hops, position, rotation, 15, 0.25,
                                                   ACTIONS) + [STOP]
    chunk = _chunk(w, hops, final_stop=True)
    assert chunk.open_loop_primitives(hops, position, rotation, True) == worker
    steps = _drive(w, chunk)
    assert [a for a, _, _ in steps] == worker
    assert {p for _, p, _ in steps} == {T.PLAN}
    assert [i for _, _, i in steps] == list(range(len(worker)))
    assert chunk.escapes == 0


def test_a_collision_escapes_sideways_resumes_and_replans_the_next_hop():
    """A pillar straight ahead at z < -0.6: the third forward of hop 0 collides, the left
    probe at 90 deg moves, the tail turn restores the heading, the two remaining forwards
    pass the pillar, and hop 1 is re-aimed from where the agent actually is."""
    w = World(blocked=lambda x, z: abs(x) < 0.1 and -1.4 < z < -0.6)
    hops = [[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]
    steps = _drive(w, _chunk(w, hops, side=90))
    expected = (
        [(FWD, T.PLAN)] * 3                        # third one collides; 2 steps remain
        + [(LEFT, T.ESCAPE)] * 6                   # side turn 90 deg left
        + [(FWD, T.ESCAPE)]                        # probe (0, 270): head turn 0, moves
        + [(RIGHT, T.RESUME)] * 6                  # tail 270 = -90: back to the hop heading
        + [(FWD, T.RESUME)] * 2                    # the hop's remaining forwards
        + [(RIGHT, T.REPLAN)] + [(FWD, T.REPLAN)] * 4  # hop 1 from (-0.25, -1.0): -14 deg
    )
    assert [(a, p) for a, p, _ in steps] == expected
    assert [i for _, _, i in steps[:3]] == [0, 1, 2]
    assert all(i is None for _, _, i in steps[3:])


def test_probes_sweep_from_the_chosen_side_to_the_other():
    """The hop's first forward collides and every probe but the last is blocked: the
    probes walk 90, 60, 30, -30, -60, -90 deg from the hop heading (left first), and the
    tail turn from -90 is 90 deg left, followed by the hop's four remaining forwards."""
    tried = []

    def blocked(x, z):
        angle = round(math.degrees(math.atan2(-x, -z)))
        if abs(x) < 1e-9:
            return True                            # straight ahead: the hop is blocked
        if len(tried) < 6:
            tried.append(angle)
            return len(tried) < 6
        return False

    w = World(blocked=blocked)
    chunk = _chunk(w, [[0.0, 0.0, -1.0]], side=90)
    steps = _drive(w, chunk)
    assert tried == [90, 60, 30, -30, -60, -90]
    assert [a for a, p, _ in steps if p == T.RESUME] == [LEFT] * 6 + [FWD] * 4
    assert chunk.escapes == 1


def test_right_first_probes_mirror_the_left_first_ones():
    for (lh, lt), (rh, rt) in zip(T.PROBES_FROM_LEFT, T.PROBES_FROM_RIGHT):
        assert (lh + rh) % 360 == 0 and (lt + rt) % 360 == 0
    # every tail turn restores the hop heading
    for side, probes in ((90, T.PROBES_FROM_LEFT), (270, T.PROBES_FROM_RIGHT)):
        heading = side
        for head, tail in probes:
            heading += head
            assert (heading + tail) % 360 == 0


def test_when_no_probe_moves_the_hop_is_abandoned_and_the_rest_replanned():
    w = World(blocked=lambda x, z: not (x == 0.0 and z == -0.25))  # one step, then walls
    hops = [[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]
    chunk = _chunk(w, hops, final_stop=True, side=270)
    steps = _drive(w, chunk)
    escape = [a for a, p, _ in steps if p == T.ESCAPE]
    # side turn right 90, then six probes: heads 0, 30, 30, 60, 30, 30 deg left
    assert escape[:24] == ([RIGHT] * 6 + [FWD] + [LEFT] * 2 + [FWD] + [LEFT] * 2 + [FWD]
                      + [LEFT] * 4 + [FWD] + [LEFT] * 2 + [FWD] + [LEFT] * 2 + [FWD])
    assert not any(p == T.RESUME for _, p, _ in steps)
    # facing +90 now; hop 1 is re-aimed from there (90 deg right), its first forward is
    # blocked too, so it escapes again, and the chunk still ends in the policy's STOP
    replan = [a for a, p, _ in steps if p == T.REPLAN]
    assert replan == [RIGHT] * 6 + [FWD, STOP]
    assert chunk.escapes == 2 and len(escape) == 48


def test_a_stop_chunk_ends_in_stop_after_an_escape():
    w = World(blocked=lambda x, z: abs(x) < 0.1 and -1.4 < z < -0.6)
    steps = _drive(w, _chunk(w, [[0.0, 0.0, -1.0]], final_stop=True))
    assert steps[-1][0] == STOP


# --- provider wiring ----------------------------------------------------------------------

class _Episode(object):
    episode_id = "42"


class _Env(object):
    def __init__(self, world):
        self.sim = world
        self.current_episode = _Episode()


class _Handle(object):
    def __init__(self, world, hops, stop=True):
        self.world, self.hops, self.stop = world, hops, stop
        self.requests = []

    def request(self, payload):
        self.requests.append(payload)
        prims = geometry.decompose_hops_to_primitives(
            self.hops, payload["agent_position"], payload["agent_rotation"], 15, 0.25,
            ACTIONS) + ([STOP] if self.stop else [])
        return {"ok": True, "stop": self.stop, "primitive_actions": prims,
                "risk_signals": {}, "hops": self.hops,
                "predicted_clearance_by_step": [float(i) for i in range(len(prims))]}


def _provider(world, handle, tryout):
    from vlnsafe.agents.remote import WorkerActionProvider

    p = WorkerActionProvider(_Env(world), handle, turn_unit_deg=15, forward_unit_m=0.25,
                             agent_radius_m=0.1, depth_hfov_deg=90, depth_camera_height_m=1.25,
                             observation_keys=["rgb"], send_occupancy=False)
    if tryout:
        p.enable_tryout(20260704, ACTIONS)
    p.reset({})
    return p


def _run(world, provider, limit=200):
    out = []
    for _ in range(limit):
        a = provider.act({}, None)
        out.append((a, provider.pending_tryout_phase, provider.pending_step_predicted_clearance_m,
                    list(provider.planned_chunk())))
        world.step(a)
        provider.update_after_step({}, a, a == STOP)
        if a == STOP:
            return out
    raise AssertionError("no stop")


def test_provider_with_tryout_matches_the_off_provider_when_nothing_collides():
    hops = [[0.3, 0.0, -1.0], [0.3, 0.0, -2.1]]
    w_off, w_on = World(), World()
    off = _run(w_off, _provider(w_off, _Handle(w_off, hops), tryout=False))
    on = _run(w_on, _provider(w_on, _Handle(w_on, hops), tryout=True))
    assert [(a, c, ch) for a, _, c, ch in off] == [(a, c, ch) for a, _, c, ch in on]
    assert all(ph is None for _, ph, _, _ in off) and {ph for _, ph, _, _ in on} == {T.PLAN}


def test_provider_escapes_logs_phases_and_counts_escapes():
    w = World(blocked=lambda x, z: abs(x) < 0.1 and -1.4 < z < -0.6)
    handle = _Handle(w, [[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
    p = _provider(w, handle, tryout=True)
    steps = _run(w, p)
    phases = [ph for _, ph, _, _ in steps]
    assert T.ESCAPE in phases and T.RESUME in phases and phases[-1] == T.REPLAN
    assert [c for _, _, c, _ in steps[:3]] == [0.0, 1.0, 2.0]
    assert all(c is None for _, ph, c, _ in steps if ph != T.PLAN)
    assert p.tryout_escape_count() == 1 and len(handle.requests) == 1
    assert (w.x, w.z) != (0.0, -0.5)                   # got past the pillar


def test_provider_refuses_a_worker_without_hops():
    w = World()

    class _NoHops(_Handle):
        def request(self, payload):
            r = _Handle.request(self, payload)
            del r["hops"]
            return r

    p = _provider(w, _NoHops(w, [[0.0, 0.0, -1.0]]), tryout=True)
    with pytest.raises(RuntimeError):
        p.act({}, None)


def test_side_draws_are_reproducible_per_episode():
    w = World()
    a = _provider(w, _Handle(w, []), tryout=True)
    b = _provider(w, _Handle(w, []), tryout=True)
    assert [a._tryout_rng.choice((90, 270)) for _ in range(20)] == [
        b._tryout_rng.choice((90, 270)) for _ in range(20)]


def test_tryout_columns_are_in_the_schema():
    assert "tryout_phase" in schema.STEP_COLUMNS
    assert "tryout_escape_count" in schema.EPISODE_COLUMNS


@pytest.mark.parametrize("extra", [
    ["--regime", "slide"],
    ["--guard", "stall"],                                     # 8-frame memory
    ["--guard", "stall-mem1", "--exempt-guard-redecisions"],  # nothing to exempt
])
def test_cli_refuses_tryout_with_sliding_or_an_unsupported_guard(extra):
    from vlnsafe import cli

    argv = ["run", "--agent", "etpr1", "--benchmark", "r2r", "--tryout"]
    argv += extra if "--regime" in extra else ["--regime", "block"] + extra
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_cli_refuses_tryout_for_an_agent_outside_the_etp_lineage():
    from vlnsafe import cli

    with pytest.raises(SystemExit):
        cli.main(["run", "--agent", "cma", "--benchmark", "r2r", "--regime", "block",
                  "--tryout"])


# --- the guard vetoes Tryout's forward steps -----------------------------------------

def _veto_chunk(world, hops, veto, side=90, final_stop=False):
    position, rotation = world.pose()
    calls = []

    def veto_fn(policy_step):
        calls.append(policy_step)
        return veto(world.x, world.z, world.yaw, policy_step)

    chunk = T.TryoutChunk(hops, final_stop, position, rotation, world.turn_deg, 0.25,
                          ACTIONS, pose_fn=world.pose,
                          collided_fn=lambda: world.previous_step_collided,
                          rng=_Side(side), veto_fn=veto_fn)
    return chunk, calls


def _ahead(world_x, world_z, yaw, d=0.25):
    return world_x - d * math.sin(yaw), world_z - d * math.cos(yaw)


def test_a_veto_that_never_fires_changes_nothing():
    pillar = lambda x, z: abs(x) < 0.1 and -1.4 < z < -0.6
    hops = [[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]
    w0 = World(blocked=pillar)
    plain = _drive(w0, _chunk(w0, hops))
    w1 = World(blocked=pillar)
    chunk, calls = _veto_chunk(w1, hops, lambda x, z, yaw, p: False)
    assert _drive(w1, chunk) == plain
    assert len(calls) == sum(1 for a, _, _ in plain if a == FWD)


def test_a_perfect_veto_escapes_the_pillar_without_touching_it():
    """Same pillar as the collision test: the veto sees it coming, so the third forward is
    never executed, the escape starts from where the agent stands, and nothing collides."""
    pillar = lambda x, z: abs(x) < 0.1 and -1.4 < z < -0.6
    w = World(blocked=pillar)
    collisions = []
    step = w.step

    def counting_step(a):
        step(a)
        collisions.append(w.previous_step_collided)
    w.step = counting_step
    chunk, calls = _veto_chunk(
        w, [[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]],
        lambda x, z, yaw, p: pillar(*_ahead(x, z, yaw)))
    steps = _drive(w, chunk)
    assert not any(collisions)
    assert [(a, p) for a, p, _ in steps] == (
        [(FWD, T.PLAN)] * 2                        # the third is vetoed, not executed
        + [(LEFT, T.ESCAPE)] * 6 + [(FWD, T.ESCAPE)]
        + [(RIGHT, T.RESUME)] * 6 + [(FWD, T.RESUME)] * 2
        + [(RIGHT, T.REPLAN)] + [(FWD, T.REPLAN)] * 4)
    assert chunk.escapes == 1
    assert calls[:3] == [True, True, True] and calls[3] is False   # probe = not policy


def test_a_vetoed_probe_is_skipped_not_walked_into():
    """Probes at 90 and 60 deg are vetoed: they are never executed, the sweep moves on,
    and the 30 deg probe is the first forward of the escape."""
    def veto(x, z, yaw, policy_step):
        if policy_step:
            return abs(x) < 1e-9 and z < -0.2         # the hop is blocked after one step
        return round(math.degrees(yaw)) % 360 in (90, 60)
    w = World()
    chunk, _ = _veto_chunk(w, [[0.0, 0.0, -1.0]], veto)
    steps = _drive(w, chunk)
    escape = [a for a, p, _ in steps if p == T.ESCAPE]
    assert escape == [LEFT] * 6 + [RIGHT] * 2 + [RIGHT] * 2 + [FWD]
    assert [a for a, p, _ in steps if p == T.RESUME] == [RIGHT] * 2 + [FWD] * 3


def test_check_forward_matches_step_verdicts():
    """GuardLayer.check_forward reaches step()'s verdict on one policy forward, without
    installing a manoeuvre, and applies the same consecutive-abort cap."""
    from vlnsafe.guard import GuardLayer

    pose = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    def guard(*frames):
        it = iter(frames)
        return GuardLayer(lambda obs: next(it), 0.25, 15, 0.1, ACTIONS, memory_frames=1,
                          max_consecutive_aborts=3)
    for frame, verdict in (([[0.3, 0.0]], "abort"), ([[5.0, 0.0]], "pass")):
        a, b = guard(frame), guard(frame)
        assert a.step({}, FWD, pose)[1]["guard_action"] == verdict
        assert b.check_forward({}, pose)[0] == verdict and not b._maneuver
        assert (a.abort_total, a._aborts) == (b.abort_total, b._aborts)
    near = [[0.3, 0.0]]
    g = guard(near, near, near, near, [[5.0, 0.0]], near)
    assert [g.check_forward({}, pose)[0] for _ in range(6)] == [
        "abort", "abort", "abort", "yield", "pass", "abort"]
    assert (g.abort_total, g.yield_total) == (4, 1)


def test_probe_vetoes_never_yield_and_leave_the_caps_alone():
    from vlnsafe.guard import GuardLayer

    near = [[0.3, 0.0]]
    g = GuardLayer(lambda obs: near, 0.25, 15, 0.1, ACTIONS, memory_frames=1,
                   max_consecutive_aborts=3, max_episode_aborts=12)
    pose = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    assert [g.check_forward({}, pose, policy_step=False)[0] for _ in range(20)] == ["abort"] * 20
    assert (g.abort_total, g.yield_total, g._aborts) == (20, 0, 0)
    assert [g.check_forward({}, pose)[0] for _ in range(4)] == ["abort"] * 3 + ["yield"]


def test_check_forward_refuses_depth_memory():
    from vlnsafe.guard import GuardLayer

    g = GuardLayer(lambda obs: [], 0.25, 15, 0.1, ACTIONS, memory_frames=8)
    with pytest.raises(ValueError):
        g.check_forward({}, ([0, 0, 0], [0, 0, 0, 1]))


def test_provider_logs_a_veto_on_the_next_executed_step():
    pillar = lambda x, z: abs(x) < 0.1 and -1.4 < z < -0.6

    class _Guard(object):
        abort_total = yield_total = _aborts = 0

        def __init__(self, world):
            self.world = world

        def reset(self):
            pass

        def check_forward(self, obs, pose, policy_step=True):
            if pillar(*_ahead(self.world.x, self.world.z, self.world.yaw)):
                self.abort_total += 1
                return "abort", 0.01
            return "pass", 1.0

    w = World(blocked=pillar)
    p = _provider(w, _Handle(w, [[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]), tryout=False)
    p.enable_tryout(20260704, ACTIONS, guard=_Guard(w))
    p.reset({})
    rows = []
    for _ in range(100):
        a = p.act({}, None)
        rows.append((a, dict(p.pending_guard_row)))
        w.step(a)
        assert not w.previous_step_collided
        p.update_after_step({}, a, a == STOP)
        if a == STOP:
            break
    flagged = [(a, r["guard_action"]) for a, r in rows if r["guard_flagged"]]
    # the escape's first turn (side drawn per episode) carries the veto
    assert len(flagged) == 1 and flagged[0][0] in (LEFT, RIGHT) and flagged[0][1] == "abort"
    assert rows[0] == (FWD, {"guard_flagged": 0, "guard_action": "pass",
                             "guard_clearance_m": 1.0, "guard_consecutive_aborts": 0})
    assert p.tryout_escape_count() == 1
