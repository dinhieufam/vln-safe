"""Build the Habitat env for a (benchmark, regime), and select + order its episodes.

What: one path for R2R and one for RxR, differing only in constants read from
configs/benchmarks/<id>.yaml (HFOV, camera height, turn angle, agent radius) — never
hard-coded, because a hard-coded R2R value silently corrupts every RxR geometry number.
Regime maps to the literal habitat ALLOW_SLIDING value: block -> False, slide -> True.

Who reads it: vlnsafe.cli, once per shard.

Imports habitat + vlnce_baselines (VLN-CE's) lazily. Must not import torch.

Ported from the pre-release harness + the autopsy's
GROUP_BY_SCENE override; the pre-release harness; the episode
helpers select_official_split_episodes / load_split / scene_id_from_episode /
filter_env_episodes_ordered / physics_manifest_fields from the pre-release harness
and the pre-release harness.
"""
import gzip
import json
import os

from .sensors import (
    apply_rgb_size, build_clearance_ring, build_etpnav_rig, build_omninav_rig, build_vla_rig)


def load_split(dataset_root, split):
    path = os.path.join(dataset_root, split, "{}.json.gz".format(split))
    with gzip.open(path, "rt") as f:
        return json.load(f)["episodes"]


def scene_id_from_episode(episode):
    scene_path = episode["scene_id"]
    if scene_path.endswith(".glb"):
        return os.path.basename(os.path.dirname(scene_path))
    return scene_path


def select_episodes(dataset_root, split, max_episodes=None):
    """One meta dict per episode, in the split file's own order (the order the archived
    runs used). meta: uid, episode_id, scene_id, source_split."""
    episodes = load_split(dataset_root, split)
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]
    if not episodes:
        raise ValueError("split {!r} selected zero episodes".format(split))
    return [
        {
            "uid": "{}:{}".format(split, ep["episode_id"]),
            "episode_id": str(ep["episode_id"]),
            "scene_id": scene_id_from_episode(ep),
            "source_split": split,
        }
        for ep in episodes
    ]


def shard(metas, num_shards, shard_index):
    """Round-robin: shard i takes metas[i::num_shards] (matches the archived runners)."""
    if num_shards <= 1:
        return metas
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index {} out of range for {} shards".format(shard_index, num_shards))
    out = metas[shard_index::num_shards]
    if not out:
        raise ValueError("shard selected zero episodes")
    return out


def filter_env_episodes_ordered(env, episode_metas):
    """Restrict env to exactly episode_metas, in that order."""
    desired = [str(m["episode_id"]) for m in episode_metas]
    by_id = {str(ep.episode_id): ep for ep in env.episodes}
    missing = [eid for eid in desired if eid not in by_id]
    if missing:
        raise RuntimeError("episodes missing from the Habitat env: {}".format(missing[:20]))
    ordered = [by_id[eid] for eid in desired]
    env._episodes = ordered
    env._dataset.episodes = ordered
    iter_opts = {k.lower(): v for k, v in env._config.ENVIRONMENT.ITERATOR_OPTIONS.items()}
    iter_opts["seed"] = env._config.SEED
    env._episode_iterator = env._dataset.get_episode_iterator(**iter_opts)
    env.number_of_episodes = len(ordered)


def physics_manifest_fields(config):
    import habitat
    import habitat_sim

    sim_v0 = config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0
    agent_0 = config.TASK_CONFIG.SIMULATOR.AGENT_0
    return {
        "habitat_sim_version": habitat_sim.__version__,
        "habitat_lab_version": habitat.__version__,
        "allow_sliding": bool(sim_v0.ALLOW_SLIDING),
        "agent_radius_m": agent_0.RADIUS,
        "agent_height_m": agent_0.HEIGHT,
        "turn_angle_deg": config.TASK_CONFIG.SIMULATOR.TURN_ANGLE,
        "forward_step_m": config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE,
    }


# regime -> the literal habitat ALLOW_SLIDING value.
_ALLOW_SLIDING = {"block": False, "slide": True}

# The instruction file inside a resolved dataset root: the BERT-tokenised pass
# (R2R_VLNCE_v1-2_preprocessed_BERTidx) suffixes the split name; the XLM-R pass (ETP-R1)
# and the plain official pass do not. Probed against the actual root so a newly promoted
# tokenised dump drops in without a code change (docs/agent_notes/etpr1.md, Phase D1).
_INSTRUCTION_FILE_SUFFIXES = ("_bertidx", "")


def _instruction_data_path(instr_data_root, split):
    """The habitat DATA_PATH template (the '{split}' placeholder is left literal for
    habitat-lab to fill) for whichever instruction file is actually present under
    instr_data_root/<split>/. Raises if none of the known layouts match."""
    for suffix in _INSTRUCTION_FILE_SUFFIXES:
        filename = "{}{}.json.gz".format(split, suffix)
        if os.path.exists(os.path.join(instr_data_root, split, filename)):
            return os.path.join(
                instr_data_root, "{split}", "{split}" + suffix + ".json.gz")
    raise ValueError(
        "no instruction file for split {!r} under {} (tried {})".format(
            split, instr_data_root,
            ["{{split}}{}.json.gz".format(s) for s in _INSTRUCTION_FILE_SUFFIXES]))


def _register_xlmr_dataset():
    """Register `VLN-CE-v2`: VLN-CE's own `VLNCEDatasetV1` (habitat_extensions/task.py)
    reads a top-level `instruction_vocab` block that the XLM-R-tokenised pass ETP-R1 uses
    (R2R_VLNCE_v1-3_preprocessed_xlmr) does not carry, and KeyErrors.
    ETP-R1's own fork registers a `VLN-CE-v2` loader for exactly this file, but it lives in
    ETP-R1's own conda env/process (the worker subprocess), not this harness process, and
    it additionally depends on a `DATASET.SUFFIX` config field this harness's config schema
    does not define. This is V1's loader with the `instruction_vocab` line removed and
    nothing else changed -- the xlmr episode schema is otherwise identical to V1's (checked
    against R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz)."""
    import gzip
    import json

    from habitat.core.dataset import ALL_SCENES_MASK, Dataset
    from habitat.core.registry import registry
    from habitat.tasks.nav.nav import NavigationGoal
    from habitat.tasks.vln.vln import InstructionData

    # reuse V1's episode dataclass and constants verbatim -- the xlmr episode schema is
    # identical to V1's, so nothing about episode/goal parsing needs to differ.
    from habitat_extensions.task import (
        ALL_EPISODES_MASK, DEFAULT_SCENE_PATH_PREFIX, VLNExtendedEpisode,
    )

    @registry.register_dataset(name="VLN-CE-v2")
    class _XlmrVLNCEDataset(Dataset):
        def __init__(self, config=None):
            self.episodes = []
            if config is None:
                return
            with gzip.open(config.DATA_PATH.format(split=config.SPLIT), "rt") as f:
                deserialized = json.load(f)
            for episode in deserialized["episodes"]:
                episode["episode_id"] = str(episode["episode_id"])
                episode["trajectory_id"] = str(episode["trajectory_id"])
                episode = VLNExtendedEpisode(**episode)
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]
                episode.scene_id = os.path.join(config.SCENES_DIR, episode.scene_id)
                episode.instruction = InstructionData(**episode.instruction)
                if episode.goals is not None:
                    for i, goal in enumerate(episode.goals):
                        episode.goals[i] = NavigationGoal(**goal)
                self.episodes.append(episode)
            if ALL_SCENES_MASK not in config.CONTENT_SCENES:
                scenes_to_load = set(config.CONTENT_SCENES)
                self.episodes = [
                    e for e in self.episodes
                    if os.path.basename(os.path.dirname(e.scene_id)) in scenes_to_load]
            if ALL_EPISODES_MASK not in config.EPISODES_ALLOWED:
                allowed = set(config.EPISODES_ALLOWED)
                self.episodes = [e for e in self.episodes if e.episode_id in allowed]

        @staticmethod
        def check_config_paths_exist(config):
            return (os.path.exists(config.DATA_PATH.format(split=config.SPLIT))
                    and os.path.exists(config.SCENES_DIR))

        @classmethod
        def get_scenes_to_load(cls, config):
            assert cls.check_config_paths_exist(config)
            dataset = cls(config)
            return sorted({
                os.path.basename(os.path.dirname(e.scene_id)) for e in dataset.episodes})


def build_r2r_config(exp_config, split, dataset_root, scenes_dir, instr_data_root,
                     seed, regime, sensor_rig, agent_rig=None, dataset_type="VLN-CE-v1"):
    """Verbatim logic of the pre-release harness for
    baseline_policy=etpnav plus the autopsy's GROUP_BY_SCENE=False, with the ETPNav
    12-view rig from vlnsafe.simulator.sensors. `instr_data_root` is the per-agent
    instruction source resolved by cli (bert / xlmr / official); `dataset_root` (the
    official split) still supplies the episode list, order and NDTW/SDTW GT.
    `dataset_type` is the matching habitat TASK_CONFIG.DATASET.TYPE: the
    v1 loader expects an `instruction_vocab` block that the xlmr (ETP-R1) pass lacks."""
    from vlnce_baselines.config.default import get_config

    if dataset_type == "VLN-CE-v2":
        _register_xlmr_dataset()

    data_path = os.path.join(dataset_root, "{split}", "{split}.json.gz")
    gt_path = os.path.join(dataset_root, "{split}", "{split}_gt.json.gz")
    opts = [
        "EVAL.SPLIT", split,
        "TASK_CONFIG.SEED", str(seed),
        "TASK_CONFIG.DATASET.TYPE", dataset_type,
        "TASK_CONFIG.DATASET.SPLIT", split,
        "TASK_CONFIG.DATASET.DATA_PATH", data_path,
        "TASK_CONFIG.DATASET.SCENES_DIR", scenes_dir,
        "TASK_CONFIG.TASK.NDTW.GT_PATH", gt_path,
    ]
    config = get_config(exp_config, opts)
    config.defrost()
    # the benchmark's own depth camera, before an agent rig (a VLA's 640x480 HFOV 79) overrides it
    ring_base_sensor = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.clone()
    config.TASK_CONFIG.DATASET.TYPE = dataset_type
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.DATASET.DATA_PATH = data_path
    config.TASK_CONFIG.DATASET.SCENES_DIR = scenes_dir
    config.TASK_CONFIG.TASK.NDTW.GT_PATH = gt_path
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    if "SDTW" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("SDTW")
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    config.TORCH_GPU_ID = 0
    config.SIMULATOR_GPU_IDS = [0]
    if not _ALLOW_SLIDING[regime]:
        config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
    if sensor_rig == "etpnav_12view":
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
        config.TASK_CONFIG.TASK.SENSORS = ["INSTRUCTION_SENSOR"]
        # g3D-LF renders its 12 RGB views at 336 (CLIP ViT-L/14@336); an agent without a
        # `rig:` block (ETPNav, ETP-R1, BEVBert) keeps the benchmark's 224 untouched.
        apply_rgb_size(config.TASK_CONFIG, agent_rig)
        build_etpnav_rig(config.TASK_CONFIG)
        # ETPNav reads BERT-tokenised instructions, ETP-R1 the XLM-R pass; both are
        # different preprocessing passes over the same official episodes. NDTW.GT_PATH /
        # SCENES_DIR keep pointing at dataset_root (the official split) so scoring matches
        # every other agent; run_episode's episode_id assertion is the safety net if the
        # passes ever disagree on coverage.
        config.TASK_CONFIG.DATASET.DATA_PATH = _instruction_data_path(instr_data_root, split)
    elif sensor_rig == "vlnce_recurrent":
        # CMA / Seq2Seq keep VLN-CE's own single RGB+DEPTH rig from the base exp_config
        # untouched (the archived autopsy ran with keep_sensors=True). Only the
        # instruction source is pinned to the resolved root (official, untokenised — the
        # recurrent policy tokenises internally).
        config.TASK_CONFIG.DATASET.DATA_PATH = _instruction_data_path(instr_data_root, split)
    elif sensor_rig in ("streamvln", "navila"):
        # navigation VLA: one forward RGB+DEPTH + GPS/COMPASS at the camera the model was
        # trained with (agent config's `rig:` block). Raw text instructions from the
        # untokenised root -- the VLA tokenises internally.
        build_vla_rig(config.TASK_CONFIG, agent_rig)
        config.TASK_CONFIG.DATASET.DATA_PATH = _instruction_data_path(instr_data_root, split)
    elif sensor_rig == "omninav":
        # OmniNav's tri-view RGB (docs/agent_notes/omninav.md); raw text instructions.
        build_omninav_rig(config.TASK_CONFIG, agent_rig)
        config.TASK_CONFIG.DATASET.DATA_PATH = _instruction_data_path(instr_data_root, split)
    else:
        raise NotImplementedError("sensor_rig {!r}".format(sensor_rig))
    if sensor_rig != "etpnav_12view":
        # T1 agents see one forward camera (HFOV 79-90), so a wall the agent is hugging
        # sideways is outside their depth. The external clearance predictor reads this
        # passive 12-camera ring instead -- the same sensing the T2 rig has -- so the
        # geometry column means the same thing for every agent (measured 2026-09-21:
        # a forward-camera-only predictor put collided StreamVLN steps at a median +1.49 m).
        # Built from the benchmark's depth camera (256x256, HFOV 90, as T2's rig), not the
        # agent's: a VLA's narrower vertical FOV leaves a blind zone below the camera.
        build_clearance_ring(config.TASK_CONFIG, min_depth_m=0.0, max_depth_m=10.0,
                             base_sensor=ring_base_sensor)
    config.freeze()
    return config


def build_env(config):
    from habitat import Env

    return Env(config=config.TASK_CONFIG)


def _register_rxr_passthrough_sensor():
    """VLN-CE's own RxRInstructionSensor reads precomputed BERT .npz text features that
    were never downloaded; ETPNav's checkpoint consumes the XLM-R tokens baked into the
    dataset directly. Register a pass-through under the same name (a later registration
    for a name wins with no error). Verbatim from the pre-release harness."""
    from gym import spaces
    from habitat.core.registry import registry
    from habitat.core.simulator import Sensor, SensorTypes

    @registry.register_sensor(name="RxRInstructionSensor")
    class _PassthroughRxRInstructionSensor(Sensor):
        cls_uuid = "rxr_instruction"

        def __init__(self, *args, **kwargs):
            self.uuid = "instruction"
            self.observation_space = spaces.Discrete(0)

        def _get_uuid(self, *args, **kwargs):
            return self.cls_uuid

        def _get_sensor_type(self, *args, **kwargs):
            return SensorTypes.MEASUREMENT

        def get_observation(self, *args, episode, **kwargs):
            return {
                "text": episode.instruction.instruction_text,
                "tokens": episode.instruction.instruction_tokens,
                "trajectory_id": episode.trajectory_id,
            }


def build_rxr_config(exp_config, split, dataset_root, scenes_dir, seed, regime, sensor_rig,
                     agent_rig=None, turn_angle_deg=None):
    """Verbatim logic of the pre-release harness, with the rig from
    vlnsafe.simulator.sensors and the ALLOW_SLIDING value from `regime`.

    `turn_angle_deg` overrides the benchmark's 30 degrees. It exists only for the StreamVLN
    RxR turn-angle gate (docs/agent_notes/streamvln.md): the model is prompted for 15 degree
    turns. The applied value is echoed into manifest.json by physics_manifest_fields."""
    from vlnce_baselines.config.default import get_config
    import habitat_extensions  # noqa: F401  -- registers VLN-CE's own sensors first...

    _register_rxr_passthrough_sensor()  # ...then override RxRInstructionSensor

    config = get_config(exp_config, [])
    config.defrost()
    config.EVAL.SPLIT = split
    config.TASK_CONFIG.SEED = seed

    env_cfg = config.TASK_CONFIG.ENVIRONMENT
    env_cfg.MAX_EPISODE_STEPS = 5000
    env_cfg.ITERATOR_OPTIONS.SHUFFLE = False
    env_cfg.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    env_cfg.ITERATOR_OPTIONS.GROUP_BY_SCENE = False

    sim = config.TASK_CONFIG.SIMULATOR
    sim.TURN_ANGLE = 30 if turn_angle_deg is None else int(turn_angle_deg)
    sim.TILT_ANGLE = 30
    sim.ACTION_SPACE_CONFIG = "v0"
    sim.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    sim.AGENT_0.HEIGHT = 0.88
    sim.AGENT_0.RADIUS = 0.18
    sim.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    sim.HABITAT_SIM_V0.ALLOW_SLIDING = _ALLOW_SLIDING[regime]
    sim.RGB_SENSOR.WIDTH = 224
    sim.RGB_SENSOR.HEIGHT = 224
    sim.RGB_SENSOR.HFOV = 63
    sim.RGB_SENSOR.POSITION = [0, 0.88, 0]
    sim.DEPTH_SENSOR.WIDTH = 256
    sim.DEPTH_SENSOR.HEIGHT = 256
    sim.DEPTH_SENSOR.HFOV = 63
    sim.DEPTH_SENSOR.MIN_DEPTH = 0.5
    sim.DEPTH_SENSOR.MAX_DEPTH = 5.0
    sim.DEPTH_SENSOR.POSITION = [0, 0.88, 0]

    task = config.TASK_CONFIG.TASK
    task.TYPE = "VLN-v0"
    task.SUCCESS_DISTANCE = 3.0
    task.SENSORS = ["RXR_INSTRUCTION_SENSOR"]
    task.INSTRUCTION_SENSOR_UUID = "instruction"
    task.MEASUREMENTS = [
        "DISTANCE_TO_GOAL", "SUCCESS", "SPL", "NDTW", "PATH_LENGTH",
        "ORACLE_SUCCESS", "STEPS_TAKEN",
    ]
    if "SDTW" not in task.MEASUREMENTS:
        task.MEASUREMENTS.append("SDTW")
    task.SUCCESS.SUCCESS_DISTANCE = 3.0
    task.SPL.SUCCESS_DISTANCE = 3.0
    task.ORACLE_SUCCESS.SUCCESS_DISTANCE = 3.0
    task.NDTW.SUCCESS_DISTANCE = 3.0
    task.NDTW.GT_PATH = os.path.join(dataset_root, "{split}", "{split}_guide_gt.json.gz")
    task.NDTW.SPLIT = split

    ds = config.TASK_CONFIG.DATASET
    ds.TYPE = "RxR-VLN-CE-v1"
    ds.SPLIT = split
    ds.ROLES = ["guide"]
    ds.LANGUAGES = ["*"]
    ds.DATA_PATH = os.path.join(dataset_root, "{split}", "{split}_guide.json.gz")
    ds.SCENES_DIR = scenes_dir

    config.TORCH_GPU_ID = 0
    config.SIMULATOR_GPU_IDS = [0]

    if sensor_rig == "etpnav_12view":
        sim.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
        build_etpnav_rig(config.TASK_CONFIG)
    elif sensor_rig == "vlnce_recurrent":
        # CMA / Seq2Seq on RxR (conditional on a public checkpoint;
        # otherwise \nmark). The RxR body/rig set above stands; keep VLN-CE's own single
        # RGB+DEPTH. Unverified until a checkpoint exists.
        sim.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    elif sensor_rig in ("streamvln", "navila"):
        # the RxR body/rig set above stands unless the agent's own `rig:` block overrides it.
        build_vla_rig(config.TASK_CONFIG, agent_rig)
    elif sensor_rig == "omninav":
        build_omninav_rig(config.TASK_CONFIG, agent_rig)
    else:
        raise NotImplementedError("sensor_rig {!r} for rxr".format(sensor_rig))
    # the clearance-only ring is rig-independent and RxR needs it (the policy depth input
    # is floored at 0.5 m, so it cannot double as the clearance sensor); every rig gets it.
    build_clearance_ring(config.TASK_CONFIG, min_depth_m=0.0, max_depth_m=10.0)

    config.freeze()
    return config


def select_rxr_metas(env, stride, offset, max_episodes, subsample=1):
    """Post-Env stride sample (the pre-release harness): one trajectory
    carries ~9 language variants in consecutive rows, so a stride spreads the sample."""
    eps = env.episodes[offset:]
    if stride > 1:
        eps = eps[::stride]
    if subsample and subsample > 1:  # every k-th of the stride sample: spread over scenes
        eps = eps[::subsample]
    if max_episodes:
        eps = eps[:max_episodes]
    return [
        {
            "uid": "rxr:{}".format(ep.episode_id),
            "episode_id": str(ep.episode_id),
            "scene_id": os.path.splitext(os.path.basename(ep.scene_id))[0],
            "instruction_id": getattr(ep.instruction, "instruction_id", "") or "",
        }
        for ep in eps
    ]
