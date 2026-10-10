"""Phase D1 / D2 wiring: the ETP-R1 worker command line, the generalised instruction-file
resolver, and the in-process vlnce_policy branch of the registry. No habitat, no torch.
"""
import os
import sys

import pytest

from vlnsafe import config
from vlnsafe.agents import registry
from vlnsafe.simulator import environment


# --- D1: ETP-R1 is a config swap on the ETPNav worker ---------------------------------

def _argv(agent_id, benchmark):
    agent = config.load_agent(agent_id)
    return registry.etpnav_worker_argv(
        agent, "/py", "/code", "run/eval.yaml", "/ckpt.pth", "/pre.pt", benchmark)


def test_etpr1_worker_argv_has_task_type_and_trainer_module():
    r2r = _argv("etpr1", "r2r")
    assert "--trainer-module" in r2r
    assert r2r[r2r.index("--trainer-module") + 1] == "vlnce_baselines.ss_trainer_ETP_R1"
    assert r2r[r2r.index("--task-type") + 1] == "1"
    assert _argv("etpr1", "rxr")[_argv("etpr1", "rxr").index("--task-type") + 1] == "2"


def test_etpnav_worker_argv_unchanged_shape():
    """ETPNav keeps ETPNav's trainer and never gets --task-type (its call sites are the
    unchanged signatures)."""
    argv = _argv("etpnav", "r2r")
    assert "--task-type" not in argv
    assert argv[argv.index("--trainer-module") + 1] == "vlnce_baselines.ss_trainer_ETP"


def test_worker_module_defaults_to_etpnav_worker():
    """ETPNav / ETP-R1 / BEVBert keep the worker that produced the landed arms."""
    for agent_id in ("etpnav", "etpr1", "bevbert"):
        argv = _argv(agent_id, "r2r")
        assert argv[argv.index("-m") + 1] == "vlnsafe.workers.etpnav_worker"


def test_g3dlf_uses_its_own_worker_module_without_task_type():
    argv = _argv("g3dlf", "r2r")
    assert argv[argv.index("-m") + 1] == "vlnsafe.workers.g3dlf_worker"
    assert "--task-type" not in argv
    assert argv[argv.index("--trainer-module") + 1] == "vlnce_baselines.ss_trainer_ETP"


class _Obj(object):
    pass


def _task_config(width=224, height=224):
    tc = _Obj()
    tc.SIMULATOR = _Obj()
    tc.SIMULATOR.RGB_SENSOR = _Obj()
    tc.SIMULATOR.RGB_SENSOR.WIDTH = width
    tc.SIMULATOR.RGB_SENSOR.HEIGHT = height
    return tc


def test_apply_rgb_size_is_noop_without_rig():
    from vlnsafe.simulator.sensors import apply_rgb_size
    for rig in (None, {}, {"depth": {"width": 128}}):
        tc = _task_config()
        apply_rgb_size(tc, rig)
        assert (tc.SIMULATOR.RGB_SENSOR.WIDTH, tc.SIMULATOR.RGB_SENSOR.HEIGHT) == (224, 224)
    for agent_id in ("etpnav", "etpr1", "bevbert"):
        assert config.load_agent(agent_id).get("rig") is None


def test_apply_rgb_size_takes_g3dlf_rig():
    from vlnsafe.simulator.sensors import apply_rgb_size
    tc = _task_config()
    apply_rgb_size(tc, config.load_agent("g3dlf")["rig"])
    assert (tc.SIMULATOR.RGB_SENSOR.WIDTH, tc.SIMULATOR.RGB_SENSOR.HEIGHT) == (336, 336)


def test_instruction_data_path_probes_layout(tmp_path):
    root = tmp_path / "ds"
    (root / "val_unseen").mkdir(parents=True)
    # xlmr / official layout: {split}.json.gz
    (root / "val_unseen" / "val_unseen.json.gz").write_bytes(b"x")
    tmpl = environment._instruction_data_path(str(root), "val_unseen")
    assert tmpl.endswith(os.path.join("{split}", "{split}.json.gz"))

    # a bertidx dump wins when present (checked first)
    (root / "val_unseen" / "val_unseen_bertidx.json.gz").write_bytes(b"x")
    assert environment._instruction_data_path(str(root), "val_unseen").endswith(
        "{split}_bertidx.json.gz")


def test_instruction_data_path_raises_on_missing(tmp_path):
    (tmp_path / "val_unseen").mkdir()
    with pytest.raises(ValueError):
        environment._instruction_data_path(str(tmp_path), "val_unseen")


# --- D2: CMA / Seq2Seq run in-process via family: vlnce_policy ------------------------

def test_vlnce_policy_module_imports_without_torch():
    import vlnsafe.agents.vlnce_policy as vp

    for method in ("reset", "act", "planned_chunk", "update_after_step"):
        assert callable(getattr(vp.VlncePolicyProvider, method))
    assert vp._PROB_KEYS == ("prob_stop", "prob_forward", "prob_left", "prob_right")


def test_registry_vlnce_policy_rejects_missing_benchmark_checkpoint():
    # seq2seq has no RxR checkpoint (none is public) -> a clear ValueError, not a
    # KeyError, and before any torch import.
    run_cfg = {"agent": config.load_agent("seq2seq")}
    with pytest.raises(ValueError):
        registry.build_provider(run_cfg, None, None, "rxr")


def test_cma_has_the_public_rxr_checkpoint():
    """The RxR CMA arm was once slated to become an \\nmark for lack of a checkpoint. The RxR-Habitat monolingual CMA baseline is public
    (VLN-CE README), so the arm is real."""
    cma = config.load_agent("cma")
    assert cma["checkpoint"]["rxr"].endswith("rxr_cma_en/ckpt.pth")
    assert cma["exp_config"]["rxr"].endswith("rxr_cma_en.yaml")


def test_cma_config_is_vlnce_recurrent_official():
    cma = config.load_agent("cma")
    assert cma["family"] == "vlnce_policy"
    assert cma["sensor_rig"] == "vlnce_recurrent"
    assert cma["policy_name"]
    assert config.agent_dataset_key(cma, "r2r") == "official"


# --- D4 / D5: navigation VLA workers (scaffold) --------------------------------------

def test_streamvln_action_string_parser():
    """StreamVLN's own vocabulary is OrderedDict({'STOP':0, '↑':1, '←':2, '→':3})
    (streamvln_agent.py:actions2idx @e48f6ff); ours is a tolerant superset."""
    from vlnsafe.workers.streamvln_worker import parse_action_string

    assert parse_action_string("↑ ↑ → ↑") == [1, 1, 3, 1]
    assert parse_action_string("↑←→STOP") == [1, 2, 3, 0]          # no separators
    assert parse_action_string("forward, turn left, forward, STOP, forward") == [1, 2, 1, 0]
    assert parse_action_string("move forward move_forward TURN_RIGHT") == [1, 1, 3]
    assert parse_action_string("blah") == []


def test_navila_sentence_parser_matches_upstream():
    """Reproduces NaVILA's own eval-loop parse (navila_trainer.py:222-282 @76b98f2):
    one action type per generation, its magnitude snapped to NaVILA's output grid."""
    from vlnsafe.workers.navila_worker import parse_navila_sentence as p

    # R2R step units (0.25 m / 15 deg) -> upstream's own decomposition, exactly
    assert p("The next action is move forward 75 cm.", 0.25, 15.0) == [1, 1, 1]
    assert p("The next action is turn left 30 degrees.", 0.25, 15.0) == [2, 2]
    # a magnitude off NaVILA's grid snaps to the nearest allowed one (40 -> 45 deg)
    assert p("The next action is turn right 40 degrees.", 0.25, 15.0) == [3, 3, 3]
    # bare verb -> upstream defaults: 25 cm / 15 deg, i.e. a single primitive
    assert p("The next action is turn right.", 0.25, 15.0) == [3]
    assert p("The next action is move forward.", 0.25, 15.0) == [1]
    # "stop" is checked first, so it wins wherever it appears
    assert p("You have arrived. stop", 0.25, 15.0) == [0]
    assert p("nothing parseable here", 0.25, 15.0) == [0]
    # RxR turns are 30 deg in the simulator: the same sentence is ONE primitive there.
    # Decomposing by a hard-coded 15 would double every RxR turn.
    assert p("The next action is turn left 30 degrees.", 0.25, 30.0) == [2]


def test_vla_worker_argv():
    for fam in ("streamvln_worker", "navila_worker"):
        assert fam in config.AGENT_FAMILIES
    sv = config.load_agent("streamvln")
    argv = registry.vla_worker_argv(sv, "/py", "/code", "config/vln_r2r.yaml", "/ckpt")
    assert argv[:3] == ["/py", "-m", "vlnsafe.workers.streamvln_worker"]
    # the repo loads bfloat16, not the fp16 originally assumed
    assert argv[argv.index("--precision") + 1] == "bf16"
    assert argv[argv.index("--exp-config") + 1] == "config/vln_r2r.yaml"
    nv = config.load_agent("navila")
    assert registry.vla_worker_argv(nv, "/py", "/c", None, "/ck")[2] == \
        "vlnsafe.workers.navila_worker"


def test_vla_rig_constants_come_from_the_agent_config():
    """The camera a VLA was trained at lives in its config, never in the harness."""
    sv = config.load_agent("streamvln")["rig"]
    assert (sv["rgb"]["width"], sv["rgb"]["height"], sv["rgb"]["hfov_deg"]) == (640, 480, 79)
    nv = config.load_agent("navila")["rig"]
    # NaVILA sizes RGB and depth differently -- 512 vs 256
    assert nv["rgb"]["width"] == 512 and nv["depth"]["width"] == 256


def test_registry_scripted_branch_is_wired():
    run_cfg = {"agent": {"id": "x", "family": "scripted"}, "defaults": {"seed": 1}}
    prov, handle = registry.build_provider(run_cfg, None, None, "r2r")
    assert handle is None
    for m in ("reset", "act", "planned_chunk", "update_after_step"):
        assert hasattr(prov, m)


def test_check_reproduction_gate(tmp_path):
    import subprocess
    arm = tmp_path / "results" / "etpnav" / "r2r" / "slide"
    arm.mkdir(parents=True)
    (arm / "episodes.csv").write_text(
        "success,spl\n1,0.5\n1,0.6\n0,0.0\n1,0.55\n")  # SR 75.0 vs published 57.0 -> FAIL
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "check_reproduction.py")
    rc = subprocess.call([sys.executable, script, str(arm)])
    assert rc == 1
    # an arm with no published number is a skip (rc 0)
    arm2 = tmp_path / "results" / "etpnav" / "rxr" / "block"
    arm2.mkdir(parents=True)
    (arm2 / "episodes.csv").write_text("success,spl\n1,0.5\n0,0.0\n")
    assert subprocess.call([sys.executable, script, str(arm2)]) == 0


def _streaming_provider(monkeypatch, handle, **kw):
    from vlnsafe.agents import remote
    monkeypatch.setattr(remote, "agent_pose", lambda env: ([0.0, 0.0, 0.0], [0, 0, 0, 1]))
    return remote.WorkerActionProvider(
        None, handle, turn_unit_deg=15, forward_unit_m=0.25, agent_radius_m=0.1,
        depth_hfov_deg=79, depth_camera_height_m=1.25,
        observation_keys=["rgb"], send_occupancy=False, **kw)


class _RecordingHandle(object):
    def __init__(self, chunk):
        self.requests = []
        self._chunk = chunk

    def request(self, payload):
        self.requests.append(payload)
        return {"primitive_actions": list(self._chunk), "risk_signals": {}}


def _obs(i):
    return {"rgb": [[i]],
            "instruction": {"text": "go", "tokens": [], "trajectory_id": ""}}


def test_streaming_provider_ships_one_frame_per_executed_step(monkeypatch):
    """A streaming VLA pushes a frame into its video/KV history on every executed step and
    generates only when its chunk runs dry (StreamVLN streamvln_eval.py:247-344). Across a
    3-primitive chunk the worker must therefore receive 3 distinct frames: two as
    `history_observations` plus the current one."""
    handle = _RecordingHandle([1, 1, 2])
    p = _streaming_provider(monkeypatch, handle, stream_observations=True)
    p.reset(_obs(0))
    for i, executed in enumerate([1, 1, 2]):
        p.act(_obs(i), None)
        p.update_after_step(_obs(i), executed, False)
    p.act(_obs(3), None)          # chunk empty -> second decision

    assert len(handle.requests) == 2
    first, second = handle.requests
    assert first["history_observations"] == []           # nothing executed yet
    assert len(second["history_observations"]) == 2      # obs1, obs2 ...
    assert second["observation"]["rgb"]["shape"] == [1, 1]   # ... and obs3 is `observation`
    assert second["decision_index"] == 1 and first["episode_reset"] is True


def test_non_streaming_provider_request_shape_is_unchanged(monkeypatch):
    """The ETPNav family decides per step; it must not gain a history payload."""
    handle = _RecordingHandle([1])
    p = _streaming_provider(monkeypatch, handle)      # stream_observations defaults False
    p.reset(_obs(0))
    p.act(_obs(0), None)
    assert "history_observations" not in handle.requests[0]


def test_navila_frame_sampling_matches_upstream():
    """_sample_and_pad_images reimplements NaVILA's own helper (inlined because importing
    its trainer drags in habitat). Left-pad to num_frames, then num_frames-1 evenly spaced
    plus the latest."""
    from vlnsafe.workers.navila_worker import _sample_and_pad_images

    class _Img(object):
        def __init__(self, tag):
            self.tag = tag

        @staticmethod
        def new(mode, size, color):
            return _Img("pad")

    # fewer frames than the window -> padded on the left, latest kept last
    out = _sample_and_pad_images([_Img("a"), _Img("b")], 8, _Img)
    assert len(out) == 8
    assert out[-1].tag == "b"
    assert out[0].tag == "pad"

    # more frames than the window -> strided sample, latest still last
    frames = [_Img(str(i)) for i in range(20)]
    out = _sample_and_pad_images(frames, 8, _Img)
    assert len(out) == 8
    assert out[-1].tag == "19"
    assert [f.tag for f in out[:-1]] == ["0", "2", "5", "8", "10", "13", "16"]
