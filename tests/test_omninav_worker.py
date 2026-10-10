"""OmniNav wiring: the waypoint -> primitive walk-out, upstream's stop and early-stop
rules, the tri-view rig, and the registry branch. No habitat, no torch, no model."""
import math

import pytest

from vlnsafe import config
from vlnsafe.agents import registry
from vlnsafe.workers.omninav_worker import (
    MOVE_FORWARD, STOP, TURN_LEFT, TURN_RIGHT, EarlyStop, arrived, decision_seed,
    waypoint_to_primitives)


def test_straight_waypoint_rounds_and_carries_the_remainder():
    prims, res = waypoint_to_primitives(0.30, 0.0, 15.0, 0.25)
    assert prims == [MOVE_FORWARD] and res == pytest.approx(0.05)
    # the carried 0.05 m tips the next 0.30 m waypoint over to two steps
    prims, res = waypoint_to_primitives(0.30, 0.0, 15.0, 0.25, 0.10)
    assert prims == [MOVE_FORWARD, MOVE_FORWARD] and res == pytest.approx(-0.10)


def test_distance_is_unbiased_over_an_episode():
    res, walked = 0.0, 0
    for _ in range(40):
        prims, res = waypoint_to_primitives(0.30, 0.0, 15.0, 0.25, res)
        walked += prims.count(MOVE_FORWARD)
    assert abs(walked * 0.25 - 40 * 0.30) <= 0.125 + 1e-9


def test_positive_theta_turns_left_to_the_nearest_unit():
    prims, _ = waypoint_to_primitives(0.25, math.radians(40), 15.0, 0.25)
    assert prims == [TURN_LEFT, TURN_LEFT, TURN_LEFT, MOVE_FORWARD]   # 40 -> 45 deg
    prims, _ = waypoint_to_primitives(0.25, math.radians(-20), 15.0, 0.25)
    assert prims == [TURN_RIGHT, MOVE_FORWARD]                          # -20 -> -15 deg
    prims, _ = waypoint_to_primitives(0.0, math.radians(90), 30.0, 0.25)
    assert prims == [TURN_LEFT] * 3                                     # RxR's 30 deg unit


def test_a_decision_is_never_empty():
    """An empty chunk would read as STOP to the provider."""
    prims, res = waypoint_to_primitives(0.05, math.radians(3), 15.0, 0.25)
    assert prims == [MOVE_FORWARD] and res == pytest.approx(-0.20)


def test_arrive_needs_all_five_above_half():
    assert arrived([0.9, 0.8, 0.7, 0.6, 0.51])
    assert not arrived([0.9, 0.8, 0.7, 0.6, 0.5])
    assert not arrived([])


def test_early_stop_matches_upstream_counters():
    es = EarlyStop(max_steps=150, max_still=25)
    stops = [es.step([float(i), 0.0, 0.0]) for i in range(152)]
    assert stops.index(True) == 151                  # iter_step > 150
    es = EarlyStop(max_steps=150, max_still=25)
    stops = [es.step([0.0, 0.0, 0.0]) for _ in range(30)]
    assert stops.index(True) == 26                   # still count 26 > 25
    es.reset()
    assert es.step([1.0, 0.0, 0.0]) is False


def test_decision_seed_is_per_episode_and_decision():
    assert decision_seed(1, "go left", 3) == decision_seed(1, "go left", 3)
    assert decision_seed(1, "go left", 3) != decision_seed(1, "go left", 4)
    assert decision_seed(1, "go left", 3) != decision_seed(1, "go right", 3)


class _Obj(object):
    pass


def _task_config():
    tc = _Obj()
    tc.SIMULATOR = _Obj()
    tc.SIMULATOR.RGB_SENSOR = _Obj()
    tc.SIMULATOR.RGB_SENSOR.WIDTH = 224
    tc.SIMULATOR.RGB_SENSOR.HEIGHT = 224
    tc.SIMULATOR.RGB_SENSOR.HFOV = 90
    tc.SIMULATOR.RGB_SENSOR.POSITION = [0, 1.25, 0]
    tc.SIMULATOR.AGENT_0 = _Obj()
    tc.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    tc.TASK = _Obj()
    tc.TASK.SENSORS = ["INSTRUCTION_SENSOR", "SHORTEST_PATH_SENSOR"]
    return tc


def test_omninav_rig_is_the_upstream_tri_view():
    from vlnsafe.agents.remote import OMNINAV_OBSERVATION_KEYS
    from vlnsafe.simulator.sensors import build_omninav_rig
    tc = _task_config()
    build_omninav_rig(tc, config.load_agent("omninav")["rig"])
    sim = tc.SIMULATOR
    assert sim.AGENT_0.SENSORS == ["RGB_FRONT", "RGB_LEFT", "RGB_RIGHT", "DEPTH_SENSOR"]
    for name, yaw in (("RGB_FRONT", 0.0), ("RGB_LEFT", 1.5707963), ("RGB_RIGHT", -1.5707963)):
        cam = getattr(sim, name)
        assert (cam.WIDTH, cam.HEIGHT, cam.HFOV) == (720, 640, 110)
        assert cam.ORIENTATION == [0.0, yaw, 0.0] and cam.POSITION[1] == 1.25
        assert cam.UUID in OMNINAV_OBSERVATION_KEYS
    assert sim.RGB_SENSOR.WIDTH == 224               # the base sensor is left alone
    assert tc.TASK.SENSORS == ["INSTRUCTION_SENSOR"]


def test_omninav_config_and_argv():
    agent = config.load_agent("omninav")
    assert agent["family"] == "omninav_worker" in config.AGENT_FAMILIES
    assert agent["published"]["r2r"]["sr"] == 69.5
    argv = registry.vla_worker_argv(agent, "/py", "/code", None, "/ck")
    assert argv[2] == "vlnsafe.workers.omninav_worker"
    assert argv[argv.index("--hfov-deg") + 1] == "110.0"
    assert '"early_stop_steps": 150' in argv[argv.index("--model-config") + 1]


def test_registry_omninav_fails_loudly_without_checkpoint():
    agent = dict(config.load_agent("omninav"), checkpoint={"r2r": "/nonexistent/ck"})
    with pytest.raises(FileNotFoundError):
        registry.build_provider({"agent": agent, "defaults": {"seed": 1}}, None, None, "r2r")
