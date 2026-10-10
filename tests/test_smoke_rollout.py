"""End-to-end rollout on a couple of episodes with a scripted provider. Marked slow:
needs habitat-sim, VLN-CE on the path and the MP3D scenes. Runs
for real when those are present, skips cleanly otherwise.
"""
import os

import pytest

pytestmark = pytest.mark.slow

_VLNCE_ROOT = os.path.join(os.path.dirname(__file__), "..", "external", "VLN-CE")


def _have_habitat():
    try:
        import habitat  # noqa: F401
        return os.path.isdir(_VLNCE_ROOT)
    except Exception:
        return False


@pytest.mark.skipif(not _have_habitat(), reason="habitat-sim / VLN-CE not available")
def test_two_episode_scripted_rollout(tmp_path):
    import sys

    sys.path.insert(0, os.path.abspath(_VLNCE_ROOT))
    os.chdir(os.path.abspath(_VLNCE_ROOT))

    from vlnsafe import config as cfg
    from vlnsafe.simulator import environment as envmod
    from vlnsafe.agents.scripted import StraightProvider
    from vlnsafe.records.writers import EpisodeWriter, StepWriter
    from vlnsafe.rollout import run_episode

    bench = cfg.load_benchmark("r2r")
    official = bench["dataset"]["official"]
    if not os.path.isdir(official):
        pytest.skip("R2R official split not on this machine")

    metas = envmod.select_episodes(official, bench["split"], max_episodes=2)
    habitat_config = envmod.build_r2r_config(
        bench["habitat_base_config"], bench["split"], official,
        bench["dataset"]["scenes"], official, 20260704, "block", "vlnce_recurrent",
    )
    env = envmod.build_env(habitat_config)
    envmod.filter_env_episodes_ordered(env, metas)

    ep_path = tmp_path / "episodes.csv"
    with EpisodeWriter(str(ep_path)) as ew, StepWriter(str(tmp_path / "steps.csv")) as sw:
        try:
            for meta in metas:
                run_episode(env, meta, StraightProvider(), ew, sw,
                            agent="straight", benchmark="r2r", regime="block",
                            seed=20260704, shard_index=0)
        finally:
            env.close()

    lines = ep_path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 episodes
    assert lines[0].startswith("episode_id,scene_id,")


@pytest.mark.skipif(not _have_habitat(), reason="habitat-sim / VLN-CE not available")
def test_guard_stops_a_straight_walker_hitting_walls(tmp_path):
    """The known-stuck case: a
    straight-line walker drives into walls; with the default guard it must collide
    less, and the guard must have acted. Without --guard its columns are UNAVAILABLE."""
    import csv
    import sys

    sys.path.insert(0, os.path.abspath(_VLNCE_ROOT))
    os.chdir(os.path.abspath(_VLNCE_ROOT))

    from vlnsafe import config as cfg
    from vlnsafe.simulator import environment as envmod
    from vlnsafe.agents.scripted import StraightProvider
    from vlnsafe.clearance_predictor import PrimitiveClearancePredictor
    from vlnsafe.geometry import CLEARANCE_DEPTH_SENSOR_KEYS
    from vlnsafe.guard import GuardLayer
    from vlnsafe.records.writers import EpisodeWriter, StepWriter
    from vlnsafe.rollout import run_episode

    bench = cfg.load_benchmark("r2r")
    official = bench["dataset"]["official"]
    if not os.path.isdir(official):
        pytest.skip("R2R official split not on this machine")
    metas = envmod.select_episodes(official, bench["split"], max_episodes=3)
    habitat_config = envmod.build_r2r_config(
        bench["habitat_base_config"], bench["split"], official,
        bench["dataset"]["scenes"], official, 20260704, "block", "vlnce_recurrent",
    )
    predictor = PrimitiveClearancePredictor.from_habitat_config(
        habitat_config, sensor_keys=CLEARANCE_DEPTH_SENSOR_KEYS,
        sensor_name="CLEARANCE_DEPTH_0")
    guard = GuardLayer.from_config(cfg.load_guard("default")[1], predictor)

    env = envmod.build_env(habitat_config)
    results = {}
    try:
        for label, g in (("off", None), ("on", guard)):
            envmod.filter_env_episodes_ordered(env, metas)
            out = tmp_path / label
            with EpisodeWriter(str(out / "episodes.csv")) as ew, \
                    StepWriter(str(out / "steps.csv")) as sw:
                for meta in metas:
                    run_episode(env, meta, StraightProvider(max_forward=200), ew, sw,
                                agent="straight", benchmark="r2r", regime="block",
                                seed=20260704, shard_index=0,
                                clearance_predictor=predictor, guard=g)
            with open(str(out / "episodes.csv")) as f:
                results[label] = list(csv.DictReader(f))
    finally:
        env.close()

    collisions = {k: sum(int(r["collision_count"]) for r in v) for k, v in results.items()}
    assert collisions["off"] > 0, "the straight walker never hit anything; bad fixture"
    assert all(r["guard_abort_count"] == "UNAVAILABLE" for r in results["off"])
    assert sum(int(r["guard_abort_count"]) for r in results["on"]) > 0
    assert collisions["on"] < collisions["off"], collisions
