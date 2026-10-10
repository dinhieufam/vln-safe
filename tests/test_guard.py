"""vlnsafe/guard.py: the odometry-registered memory, the veto, the escape manoeuvre, the
livelock bound, and the provider hook it uses. Pure numpy; no habitat."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
from geometry_inputs import ACTIONS  # noqa: E402

from vlnsafe.clearance_predictor import PrimitiveClearancePredictor  # noqa: E402
from vlnsafe.guard import GUARD_OFF_ROW, GuardLayer, ego_to_world, world_to_ego  # noqa: E402
from vlnsafe.records import schema  # noqa: E402

FWD, LEFT, RIGHT, STOP = (ACTIONS.MOVE_FORWARD, ACTIONS.TURN_LEFT, ACTIONS.TURN_RIGHT,
                          ACTIONS.STOP)


def _pose(x=0.0, z=0.0, yaw_deg=0.0):
    """habitat pose with yaw about +y (positive = turn left); rotation as [x, y, z, w]."""
    h = math.radians(yaw_deg) / 2.0
    return [x, 0.0, z], [0.0, math.sin(h), 0.0, math.cos(h)]


def _wall(distance_m, half_width_m=1.0, n=41):
    """A wall across the heading `distance_m` ahead, as ego points."""
    return [[distance_m, y] for y in np.linspace(-half_width_m, half_width_m, n)]


class _Frames(object):
    """A `project` stand-in that returns queued ego point lists, one per call."""

    def __init__(self, *frames):
        self.frames = list(frames)

    def __call__(self, obs):
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]


def _guard(project, **kw):
    return GuardLayer(project, forward_unit_m=0.25, turn_unit_deg=15, agent_radius_m=0.1,
                      actions=ACTIONS, **kw)


def test_ego_world_round_trip_and_axes():
    position, rotation = _pose(1.0, 2.0, 0.0)
    # identity rotation: forward is world -z, left is world -x
    w = ego_to_world([[1.0, 0.0], [0.0, 1.0]], position, rotation)
    assert np.allclose(w, [[1.0, 1.0], [0.0, 2.0]])
    rng = np.random.RandomState(0)
    pts = rng.uniform(-3, 3, (50, 2))
    for yaw in (0.0, 30.0, 137.0, -95.0):
        position, rotation = _pose(0.3, -1.2, yaw)
        back = world_to_ego(ego_to_world(pts, position, rotation), position, rotation)
        assert np.allclose(back, pts, atol=1e-9)
    # a 90-degree left turn: a point that was ahead is now on the right (negative left)
    world = ego_to_world([[1.0, 0.0]], *_pose(yaw_deg=0.0))
    assert np.allclose(world_to_ego(world, *_pose(yaw_deg=90.0)), [[0.0, -1.0]], atol=1e-9)


def test_single_frame_clearance_matches_the_logged_predictor():
    """memory_frames=1: the guard's number is predicted_clearance_m's number."""
    points = _wall(0.4)
    predictor = PrimitiveClearancePredictor(
        turn_unit_deg=15, forward_unit_m=0.25, agent_radius_m=0.1, hfov_deg=90,
        camera_height_m=1.25, sensor_keys=["depth"], min_depth_m=0.0, max_depth_m=10.0,
        actions=ACTIONS)
    predictor.points = lambda obs: points
    guard = GuardLayer.from_config({"memory_frames": 1, "threshold_m": -1.0}, predictor)
    _, row = guard.step({}, FWD, _pose(yaw_deg=40.0))
    assert abs(row["guard_clearance_m"] - predictor({}, FWD)) < 1e-9
    assert row["guard_flagged"] == 0 and row["guard_action"] == "pass"


def test_open_space_passes_every_primitive_unchanged():
    guard = _guard(_Frames(_wall(5.0)))
    for a in (FWD, LEFT, RIGHT, FWD, STOP):
        executed, row = guard.step({}, a, _pose())
        assert executed == a and row["guard_flagged"] == 0
    assert guard.abort_total == 0 and guard.yield_total == 0


def test_blocked_forward_is_replaced_by_a_turn_then_a_forward_step():
    # wall 0.3 m ahead: a 0.25 m step ends 0.05 m from it, minus the 0.1 m body -> -0.05
    guard = _guard(_Frames(_wall(0.3, half_width_m=0.3)))
    executed, row = guard.step({}, FWD, _pose())
    assert executed in (LEFT, RIGHT)
    assert row["guard_flagged"] == 1 and row["guard_action"] == "abort"
    assert abs(row["guard_clearance_m"] - (-0.05)) < 1e-6
    assert row["guard_consecutive_aborts"] == 1 and guard.abort_total == 1
    plan = [executed] + list(guard._maneuver)
    assert plan[-1] == FWD and set(plan[:-1]) == {executed}


def test_escape_turns_toward_the_open_side():
    # obstacle ahead and to the left; the right is open
    pts = _wall(0.3, half_width_m=0.05) + [[x, y] for x in np.linspace(0.0, 1.0, 21)
                                           for y in (0.3,)]
    executed, _ = _guard(_Frames(pts)).step({}, FWD, _pose())
    assert executed == RIGHT


def test_memory_keeps_an_obstacle_that_left_the_view():
    """Frame 1 sees a box 0.55 m ahead; after one forward step the camera no longer
    does (the N8 blind zone). With memory the guard still vetoes the next step."""
    near_box = [[0.55, y] for y in np.linspace(-0.2, 0.2, 9)]
    for frames, expect_flag in ((8, 1), (1, 0)):
        guard = _guard(_Frames(near_box, []), memory_frames=frames)
        executed, row = guard.step({}, FWD, _pose(z=0.0))   # 0.55 - 0.25 - 0.1 = 0.20: pass
        assert executed == FWD and row["guard_flagged"] == 0
        executed, row = guard.step({}, FWD, _pose(z=-0.25))  # moved 0.25 m forward (-z)
        assert row["guard_flagged"] == expect_flag
        if expect_flag:
            assert executed in (LEFT, RIGHT)
            assert abs(row["guard_clearance_m"] - (0.30 - 0.25 - 0.1)) < 1e-6


def test_livelock_bound_yields_until_an_unflagged_policy_forward():
    close = _wall(0.3, half_width_m=3.0)   # the pose never changes, so it stays blocked
    guard = _guard(_Frames(close), max_consecutive_aborts=2)
    actions = []
    for _ in range(12):
        executed, row = guard.step({}, FWD, _pose())
        actions.append(row["guard_action"])
    assert actions.count("abort") == 2
    assert actions[-1] == "yield" and guard.yield_total >= 1
    # the policy then gets an unflagged forward through: the count resets
    guard.project = _Frames(_wall(5.0))
    guard._maneuver.clear()
    guard._memory.clear()
    _, row = guard.step({}, FWD, _pose())
    assert row["guard_action"] == "pass" and row["guard_consecutive_aborts"] == 0


def test_stop_wins_over_a_pending_maneuver():
    guard = _guard(_Frames(_wall(0.3, half_width_m=0.3)))
    guard.step({}, FWD, _pose())
    assert guard._maneuver
    executed, row = guard.step({}, STOP, _pose())
    assert executed == STOP and row["guard_action"] == "pass" and not guard._maneuver


def test_queueing_provider_gets_the_whole_maneuver():
    class _Queue(object):
        chunk = None

        def override_chunk(self, primitives):
            self.chunk = list(primitives)

    provider = _Queue()
    guard = _guard(_Frames(_wall(0.3, half_width_m=0.3)))
    executed, _ = guard.step({}, FWD, _pose(), provider)
    assert provider.chunk[0] == executed and provider.chunk[1:] == list(guard._maneuver)


def test_worker_provider_override_drains_without_a_decision():
    from collections import deque
    from vlnsafe.agents.remote import WorkerActionProvider

    p = WorkerActionProvider.__new__(WorkerActionProvider)
    p._queue, p._clearance_queue = deque([FWD, FWD]), deque([0.2, 0.1])
    p.stream_observations = False
    p._request_decision = lambda: pytest.fail("a manoeuvre step asked the model")
    p.override_chunk([LEFT, FWD])
    p.update_after_step({}, LEFT, False)
    assert p.act({}, None) == FWD and p.pending_step_predicted_clearance_m is None


def test_config_is_checked():
    predictor = PrimitiveClearancePredictor(15, 0.25, 0.1, 90, 1.25, ["depth"], 0, 10, ACTIONS)
    with pytest.raises(ValueError):
        GuardLayer.from_config({"threshold": 0.05}, predictor)
    with pytest.raises(NotImplementedError):
        GuardLayer.from_config({"eps": 0.1}, predictor)


def test_default_config_loads_and_guard_columns_are_in_the_schema():
    from vlnsafe import config as cfg

    name, guard_cfg = cfg.load_guard("default")
    assert name == "default" and guard_cfg["threshold_m"] == 0.05
    predictor = PrimitiveClearancePredictor(15, 0.25, 0.1, 90, 1.25, ["depth"], 0, 10, ACTIONS)
    GuardLayer.from_config(guard_cfg, predictor)
    assert set(GUARD_OFF_ROW).issubset(schema.STEP_COLUMNS)
    assert {"guard_abort_count", "guard_yield_count"}.issubset(schema.EPISODE_COLUMNS)


def test_writer_refuses_to_append_under_an_old_header(tmp_path):
    from vlnsafe.records.writers import StepWriter

    path = tmp_path / "steps.csv"
    path.write_text("episode_id,step_index\n1,0\n")
    with pytest.raises(ValueError):
        StepWriter(str(path), append=True)


def test_a_veto_on_the_way_to_the_policys_stop_executes_the_stop():
    """An ETPNav-family worker cannot re-decide after a stop decision (N10's first launch:
    decisions past the 15 cap, a disconnected graph); the guard ends the episode instead."""
    class _Chunk(object):
        chunk = None

        def planned_chunk(self):
            return [FWD, FWD, STOP]

        def override_chunk(self, primitives):
            self.chunk = list(primitives)

    provider = _Chunk()
    guard = _guard(_Frames(_wall(0.3, half_width_m=0.3)))
    executed, row = guard.step({}, FWD, _pose(), provider)
    assert executed == STOP and row["guard_action"] == "stop" and row["guard_flagged"] == 1
    assert provider.chunk is None and not guard._maneuver and guard.abort_total == 1


def test_worker_provider_refuses_a_decision_after_a_stop_decision(monkeypatch):
    import vlnsafe.agents.remote as remote

    class _Handle(object):
        def request(self, payload):
            return {"ok": True, "stop": True, "primitive_actions": [FWD, STOP]}

    monkeypatch.setattr(remote, "agent_pose", lambda env: ([0, 0, 0], [0, 0, 0, 1]))
    p = _bare_worker_provider(_Handle(), exempt=False)
    assert p.act({}, None) == FWD
    p.update_after_step({}, LEFT, False)       # the guard diverged from the stop chunk
    with pytest.raises(RuntimeError):
        p.act({}, None)


def _bare_worker_provider(handle, exempt):
    from collections import deque
    from vlnsafe.agents.remote import WorkerActionProvider

    p = WorkerActionProvider.__new__(WorkerActionProvider)
    p.env, p.handle = None, handle
    p.turn_unit_deg, p.forward_unit_m, p.agent_radius_m = 15.0, 0.25, 0.1
    p.observation_keys, p.send_occupancy, p.stream_observations = [], False, False
    p._frames, p._queue, p._clearance_queue = [], deque(), deque()
    p._last_obs, p._reset_pending, p._decision_index = {}, True, 0
    p._stop_committed, p._guard_overridden = False, False
    p.exempt_guard_redecisions = exempt
    return p


@pytest.mark.parametrize("exempt", [False, True])
def test_only_the_decision_after_a_manoeuvre_is_marked_budget_exempt(exempt, monkeypatch):
    import vlnsafe.agents.remote as remote

    class _Handle(object):
        requests = []

        def request(self, payload):
            self.requests.append(payload)
            return {"ok": True, "stop": False, "primitive_actions": [FWD]}

    monkeypatch.setattr(remote, "agent_pose", lambda env: ([0, 0, 0], [0, 0, 0, 1]))
    handle = _Handle()
    handle.requests = []
    p = _bare_worker_provider(handle, exempt)
    p.act({}, None)                          # decision 1: policy
    p.override_chunk([LEFT, FWD])            # guard manoeuvre replaces the chunk
    p.update_after_step({}, LEFT, False)
    p.act({}, None)
    p.update_after_step({}, FWD, False)
    p.act({}, None)                          # decision 2: forced by the manoeuvre
    p.update_after_step({}, FWD, False)
    p.act({}, None)                          # decision 3: policy again
    marks = [r.get("budget_exempt", False) for r in handle.requests]
    assert marks == ([False, True, False] if exempt else [False, False, False])


def test_worker_extends_its_force_stop_by_one_per_exempt_decision(monkeypatch):
    from vlnsafe.workers import etpnav_worker

    class _Engine(object):
        HabitatSimActions = ACTIONS
        stepk, budget_extension = 3, 0

        def reset_episode(self, observation):
            self.budget_extension = 0

        def decide(self, observation, position, rotation):
            return {"act": 4, "cur_vp": "0", "target_vp": "g0", "hops": [],
                    "final_stop": False, "risk_signals": {}, "candidates": []}

    monkeypatch.setattr(etpnav_worker, "decode_observation", lambda payload: {})
    engine = _Engine()
    base = {"cmd": "decide", "observation": {}, "agent_position": [0, 0, 0],
            "agent_rotation": [0, 0, 0, 1]}
    etpnav_worker.handle_request(engine, dict(base))
    assert engine.budget_extension == 0
    etpnav_worker.handle_request(engine, dict(base, budget_exempt=True))
    etpnav_worker.handle_request(engine, dict(base, budget_exempt=True))
    assert engine.budget_extension == 2
    etpnav_worker.handle_request(engine, dict(base, episode_reset=True))
    assert engine.budget_extension == 0


def test_episode_abort_budget_ends_a_livelock_that_the_consecutive_cap_misses():
    """M1b's tail: abort, then one clean policy step (resets the consecutive count), then
    abort again, forever. The episode budget makes the guard yield for good."""
    blocked, clear = _wall(0.3, half_width_m=3.0), _wall(5.0)
    guard = _guard(_Frames(clear), max_consecutive_aborts=3, max_episode_aborts=4,
                   memory_frames=1)
    labels = []
    for _ in range(10):
        guard.project = _Frames(blocked)
        _, row = guard.step({}, FWD, _pose())
        labels.append(row["guard_action"])
        guard._maneuver.clear()
        guard.project = _Frames(clear)
        guard.step({}, FWD, _pose())            # clean policy forward: resets the streak
    assert labels == ["abort"] * 4 + ["yield"] * 6
    assert guard.abort_total == 4 and guard.yield_total == 6
    guard.reset()
    guard.project = _Frames(blocked)
    assert guard.step({}, FWD, _pose())[1]["guard_action"] == "abort"   # per episode


def test_without_an_episode_budget_the_same_cycle_never_yields():
    blocked, clear = _wall(0.3, half_width_m=3.0), _wall(5.0)
    guard = _guard(_Frames(clear), max_consecutive_aborts=3, memory_frames=1)
    for _ in range(10):
        guard.project = _Frames(blocked)
        assert guard.step({}, FWD, _pose())[1]["guard_action"] == "abort"
        guard._maneuver.clear()
        guard.project = _Frames(clear)
        guard.step({}, FWD, _pose())


def test_stall_veto_blocks_a_forward_from_the_pose_where_one_already_stalled():
    """Depth reads clear (a navmesh edge with no surface, the N11 smoke), but the forward
    did not move: the same forward from the same pose is vetoed, and only that one."""
    guard = _guard(_Frames(_wall(5.0)), stall_m=0.01)
    assert guard.step({}, FWD, _pose())[1]["guard_action"] == "pass"
    executed, row = guard.step({}, FWD, _pose())             # did not move
    assert row["guard_action"] == "stall" and row["guard_flagged"] == 1
    assert executed in (LEFT, RIGHT)
    assert guard.stall_total == 1 and guard.abort_total == 0
    guard._maneuver.clear()
    assert guard.step({}, FWD, _pose(yaw_deg=30.0))[1]["guard_action"] == "pass"
    assert guard.step({}, FWD, _pose(z=-0.25))[1]["guard_action"] == "pass"   # it moved
    assert guard.step({}, FWD, _pose())[1]["guard_action"] == "stall"          # back again
    guard.reset()
    assert guard.step({}, FWD, _pose())[1]["guard_action"] == "pass"           # per episode


def test_stall_veto_ignores_the_abort_caps():
    """The caps yield because depth may be wrong; a forward known to stall never gains."""
    guard = _guard(_Frames(_wall(5.0)), stall_m=0.01, max_consecutive_aborts=0,
                   max_episode_aborts=0)
    guard.step({}, FWD, _pose())
    for _ in range(3):
        assert guard.step({}, FWD, _pose())[1]["guard_action"] == "stall"
        guard._maneuver.clear()
    assert guard.yield_total == 0


def test_without_stall_m_a_repeated_stalled_forward_passes():
    guard = _guard(_Frames(_wall(5.0)))
    for _ in range(3):
        assert guard.step({}, FWD, _pose())[1]["guard_action"] == "pass"


def test_escape_skips_a_heading_already_stalled_from_this_pose():
    guard = _guard(_Frames(_wall(5.0)), stall_m=0.01)
    guard.step({}, FWD, _pose(yaw_deg=15.0))       # left of 0 by one turn unit: stalls
    guard.step({}, RIGHT, _pose(yaw_deg=15.0))
    guard.step({}, FWD, _pose())                    # heading 0: stalls too
    executed, row = guard.step({}, FWD, _pose())
    assert row["guard_action"] == "stall"
    assert executed == RIGHT                        # left at k = 1 is the stalled heading
    fresh = _guard(_Frames(_wall(5.0)), stall_m=0.01)
    fresh.step({}, FWD, _pose())
    assert fresh.step({}, FWD, _pose())[0] == LEFT  # symmetric depth: the tie goes left


def test_stall_config_loads():
    from vlnsafe import config as cfg

    name, guard_cfg = cfg.load_guard("stall")
    _, default_cfg = cfg.load_guard("default")
    assert guard_cfg.pop("stall_m") == 0.01 and guard_cfg == default_cfg
    predictor = PrimitiveClearancePredictor(15, 0.25, 0.1, 90, 1.25, ["depth"], 0, 10, ACTIONS)
    assert GuardLayer.from_config(dict(guard_cfg, stall_m=0.01), predictor).stall_m == 0.01


def test_stall_mem1_config_is_stall_without_memory():
    from vlnsafe import config as cfg

    _, mem1 = cfg.load_guard("stall-mem1")
    _, stall = cfg.load_guard("stall")
    assert mem1.pop("memory_frames") == 1 and stall.pop("memory_frames") == 8
    assert mem1 == stall


def test_without_the_depth_veto_a_flagged_forward_passes_but_a_stall_is_still_vetoed():
    """depth_veto False: clearance is still logged, depth never vetoes, and the
    stall veto still fires from a pose where a forward already stalled."""
    guard = _guard(_Frames(_wall(0.3, half_width_m=0.3)), stall_m=0.01, depth_veto=False)
    executed, row = guard.step({}, FWD, _pose())
    assert executed == FWD and row["guard_action"] == "pass" and row["guard_flagged"] == 0
    assert abs(row["guard_clearance_m"] - (-0.05)) < 1e-6
    executed, row = guard.step({}, FWD, _pose())             # did not move
    assert row["guard_action"] == "stall" and executed in (LEFT, RIGHT)
    assert guard.abort_total == 0 and guard.yield_total == 0


def test_stall_only_config_is_stall_mem1_without_the_depth_veto():
    from vlnsafe import config as cfg

    _, only = cfg.load_guard("stall-only")
    _, mem1 = cfg.load_guard("stall-mem1")
    assert only.pop("depth_veto") is False and only == mem1
    predictor = PrimitiveClearancePredictor(15, 0.25, 0.1, 90, 1.25, ["depth"], 0, 10, ACTIONS)
    assert GuardLayer.from_config(mem1, predictor).depth_veto is True


def test_depth_mem1_config_is_stall_mem1_without_the_stall_veto():
    from vlnsafe import config as cfg

    _, depth = cfg.load_guard("depth-mem1")
    _, mem1 = cfg.load_guard("stall-mem1")
    assert mem1.pop("stall_m") == 0.01 and depth == mem1
    predictor = PrimitiveClearancePredictor(15, 0.25, 0.1, 90, 1.25, ["depth"], 0, 10, ACTIONS)
    assert GuardLayer.from_config(depth, predictor).stall_m is None


class _WallPathfinder(object):
    """Navmesh stand-in: a wall at world z = -0.2 (the agent faces -z at yaw 0); no-sliding
    stops a move at the wall."""

    def try_step_no_sliding(self, start, end):
        end = np.array(end, dtype=np.float64)
        if end[2] < -0.2:
            end[2] = -0.2
        return end

    try_step = try_step_no_sliding


class _Sim(object):
    def __init__(self, pathfinder):
        self.pathfinder = pathfinder


def test_oracle_reads_the_current_scenes_navmesh():
    """A scene load replaces sim.pathfinder; the oracle must query the new one."""
    from vlnsafe.guard import navmesh_oracle

    class _Open(object):
        def try_step_no_sliding(self, start, end):
            return np.array(end, dtype=np.float64)

    sim = _Sim(_Open())
    blocked = navmesh_oracle(sim, sliding=False)
    assert not blocked([0.0, 0.0, 0.0], (0.0, -0.25))
    sim.pathfinder = _WallPathfinder()
    assert blocked([0.0, 0.0, 0.0], (0.0, -0.25))


def test_oracle_vetoes_exactly_the_forwards_the_navmesh_would_shorten():
    from vlnsafe.guard import navmesh_oracle

    # depth reads open space, so any veto comes from the oracle
    guard = _guard(_Frames(_wall(5.0)), stall_m=0.01, oracle=True)
    guard.oracle_fn = navmesh_oracle(_Sim(_WallPathfinder()), sliding=False)
    executed, row = guard.step({}, FWD, _pose())            # 0.25 m ahead crosses the wall
    assert row["guard_action"] == "abort" and executed in (LEFT, RIGHT)
    assert row["guard_clearance_m"] < 0
    guard.reset()
    guard.oracle_fn = navmesh_oracle(_Sim(_WallPathfinder()), sliding=False)
    executed, row = guard.step({}, FWD, _pose(z=0.5))       # 0.25 m ahead stays clear
    assert executed == FWD and row["guard_action"] == "pass"


def test_oracle_guard_refuses_to_run_without_an_oracle():
    guard = _guard(_Frames(_wall(5.0)), oracle=True)
    with pytest.raises(RuntimeError):
        guard.step({}, FWD, _pose())


def test_oracle_config_is_stall_mem1_with_the_oracle():
    from vlnsafe import config as cfg

    _, oracle = cfg.load_guard("oracle-mem1")
    _, mem1 = cfg.load_guard("stall-mem1")
    assert oracle.pop("oracle") is True and oracle == mem1
