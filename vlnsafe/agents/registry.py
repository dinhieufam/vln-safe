"""Agent id -> provider instance, from configs/agents/<id>.yaml.

What: reads an agent config's `family` and builds the matching provider — etpnav_worker /
streamvln_worker / navila_worker / omninav_worker (subprocess via remote.py), vlnce_policy (in-process), or
scripted. Rig constants passed to a worker provider are read from the merged habitat
config, not the yaml, so they can never drift from what the simulator was actually built
with.

Who reads it: vlnsafe.cli.

Must not import: habitat, torch. The provider class and WorkerHandle are imported lazily.
"""
import os

from ..config import AGENT_FAMILIES as FAMILIES


def _abs(root, path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))


def _pick(spec, benchmark):
    """A config field that is either a scalar or a per-benchmark dict."""
    if isinstance(spec, dict):
        return spec.get(benchmark)
    return spec


def etpnav_worker_argv(agent, worker_python, code_root, exp_config, checkpoint,
                       pretrained, benchmark_id):
    """The `python -m vlnsafe.workers.etpnav_worker ...` command line for one agent/
    benchmark. Pure (no filesystem, no spawn) so tests can assert the wiring: ETP-R1 gets
    --trainer-module and --task-type, ETPNav neither, and `pretrained` may be None
    (BEVBert's eval config sets MODEL.pretrained_path: null). g3D-LF names its own
    `worker_module` (a subclass of the ETPNav engine with the lookahead step)."""
    argv = [
        worker_python, "-m", agent.get("worker_module", "vlnsafe.workers.etpnav_worker"),
        "--code-root", code_root,
        "--exp-config", exp_config,
        "--checkpoint", checkpoint,
        "--back-algo", agent.get("back_algo", "control"),
    ]
    if pretrained:
        argv += ["--pretrained-path", pretrained]
    trainer_module = agent.get("worker_trainer_module")
    if trainer_module:
        argv += ["--trainer-module", trainer_module]
    # ETP-R1's one signature change vs ETPNav: a task_type tag (1=r2r, 2=rxr) threaded
    # through extract_instruction_tokens / _nav_gmap_variable. Config field, per
    # benchmark; absent for ETPNav, whose call sites stay unchanged.
    task_type = (agent.get("task_type") or {}).get(benchmark_id)
    if task_type is not None:
        argv += ["--task-type", str(int(task_type))]
    return argv


def vla_worker_argv(agent, worker_python, code_root, exp_config, checkpoint):
    """The command line for a navigation-VLA worker (StreamVLN / NaVILA). Pure, so tests
    can assert the wiring without spawning an 8B model. The `model:` block (frame counts,
    conversation template) and the rig HFOV travel as JSON rather than as a flag each, so
    adding a model knob is a config edit and not a harness edit."""
    import json as _json

    rig = (agent.get("rig") or {}).get("rgb") or {}
    argv = [
        worker_python, "-m", "vlnsafe.workers.{}".format(agent["family"]),
        "--code-root", code_root,
        "--checkpoint", checkpoint,
        "--precision", agent.get("precision", "bf16"),
        "--model-config", _json.dumps(agent.get("model") or {}, sort_keys=True),
        "--hfov-deg", str(float(rig.get("hfov_deg", 90))),
    ]
    if exp_config:
        argv += ["--exp-config", exp_config]
    return argv


def build_provider(run_cfg, habitat_config, env, benchmark_id, clearance_keys=None):
    """Returns (provider, closeable) — `closeable` is a WorkerHandle to close at shard
    teardown, or None for an in-process provider. `clearance_keys` are the depth sensor
    uuids the worker's occupancy projection reads (RxR uses the unfloored clearance ring,
    not the policy depth sensors)."""
    from .remote import (
        OMNINAV_OBSERVATION_KEYS, VLA_OBSERVATION_KEYS, WorkerActionProvider, WorkerHandle)

    agent = run_cfg["agent"]
    family = agent.get("family")
    if family not in FAMILIES:
        raise ValueError("agent {!r} has unknown family {!r}".format(agent.get("id"), family))

    if family == "etpnav_worker":
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        code_root = _abs(repo_root, agent["code_root"])
        worker_python = os.path.expandvars(agent["worker_python"])
        exp_config = agent["exp_config"][benchmark_id]
        checkpoint = os.path.expandvars(agent["checkpoint"][benchmark_id])
        # `pretrained` is the ETP pretrain the policy is built on top of. BEVBert's eval
        # config sets MODEL.pretrained_path: null and needs none, so it is optional.
        pretrained = _pick(agent.get("pretrained"), benchmark_id)
        pretrained = os.path.expandvars(pretrained) if pretrained else None
        checks = [("checkpoint", checkpoint)]
        if pretrained is not None:
            checks.append(("pretrained", pretrained))
        for label, path in checks:
            if not os.path.exists(path):
                raise FileNotFoundError("{} not found: {}".format(label, path))
        argv = etpnav_worker_argv(
            agent, worker_python, code_root, exp_config, checkpoint, pretrained,
            benchmark_id)
        handle = WorkerHandle(argv, cwd=code_root)
        sim = habitat_config.TASK_CONFIG.SIMULATOR
        provider = WorkerActionProvider(
            env, handle,
            turn_unit_deg=sim.TURN_ANGLE,
            forward_unit_m=sim.FORWARD_STEP_SIZE,
            agent_radius_m=sim.AGENT_0.RADIUS,
            depth_hfov_deg=sim.DEPTH_SENSOR.HFOV,
            depth_camera_height_m=sim.DEPTH_SENSOR.POSITION[1],
            depth_sensor_keys=clearance_keys,
        )
        return provider, handle

    if family in ("streamvln_worker", "navila_worker"):
        # A navigation VLA in its own conda env: one generate() per decision returns a
        # short primitive chunk and the first-action-token read-out. Same JSON-lines
        # client as the ETPNav worker, minus the occupancy payload and the panoramic
        # sensor set. Request shape verified against each repo's own eval loop
        # (docs/agent_notes/{streamvln,navila}.md).
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        code_root = _abs(repo_root, agent["code_root"])
        worker_python = os.path.expandvars(agent["worker_python"])
        exp_config = (agent.get("exp_config") or {}).get(benchmark_id)
        checkpoint = (agent.get("checkpoint") or {}).get(benchmark_id)
        if checkpoint is None:
            raise ValueError("agent {!r} has no checkpoint for benchmark {!r}".format(
                agent.get("id"), benchmark_id))
        checkpoint = os.path.expandvars(checkpoint)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError("checkpoint not found: {}".format(checkpoint))
        argv = vla_worker_argv(agent, worker_python, code_root, exp_config, checkpoint)
        handle = WorkerHandle(argv, cwd=code_root)
        sim = habitat_config.TASK_CONFIG.SIMULATOR
        provider = WorkerActionProvider(
            env, handle,
            turn_unit_deg=sim.TURN_ANGLE,
            forward_unit_m=sim.FORWARD_STEP_SIZE,
            agent_radius_m=sim.AGENT_0.RADIUS,
            depth_hfov_deg=sim.DEPTH_SENSOR.HFOV,
            depth_camera_height_m=sim.DEPTH_SENSOR.POSITION[1],
            depth_sensor_keys=clearance_keys,
            observation_keys=VLA_OBSERVATION_KEYS,
            send_occupancy=False,
            # both VLAs push a frame into their video/KV history on every executed step,
            # not only when they generate (StreamVLN streamvln_eval.py, NaVILA
            # navila_trainer.py past_rgbs) -- see docs/agent_notes/streamvln.md.
            stream_observations=True,
        )
        return provider, handle

    if family == "omninav_worker":
        # OmniNav (docs/agent_notes/omninav.md): one forward pass per decision, whose first
        # waypoint the worker walks out as primitives. Its history holds only the frames it
        # decided on (waypoint_agent.py add_frame is called from act()), so no streaming.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        code_root = _abs(repo_root, agent["code_root"])
        worker_python = os.path.expandvars(agent["worker_python"])
        checkpoint = os.path.expandvars((agent.get("checkpoint") or {}).get(benchmark_id) or "")
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError("checkpoint not found for {!r} on {!r}: {!r}".format(
                agent.get("id"), benchmark_id, checkpoint))
        seed = run_cfg.get("seed", run_cfg.get("defaults", {}).get("seed", 0))
        argv = vla_worker_argv(agent, worker_python, code_root, None, checkpoint)
        argv += ["--seed", str(int(seed))]
        handle = WorkerHandle(argv, cwd=code_root)
        sim = habitat_config.TASK_CONFIG.SIMULATOR
        provider = WorkerActionProvider(
            env, handle,
            turn_unit_deg=sim.TURN_ANGLE,
            forward_unit_m=sim.FORWARD_STEP_SIZE,
            agent_radius_m=sim.AGENT_0.RADIUS,
            depth_hfov_deg=sim.DEPTH_SENSOR.HFOV,
            depth_camera_height_m=sim.DEPTH_SENSOR.POSITION[1],
            depth_sensor_keys=clearance_keys,
            observation_keys=OMNINAV_OBSERVATION_KEYS,
            send_occupancy=False,
            stream_observations=False,
        )
        return provider, handle

    if family == "scripted":
        from .scripted import RandomProvider, StraightProvider

        policy = agent.get("policy", "straight")
        seed = run_cfg.get("defaults", {}).get("seed", 0)
        provider = RandomProvider(seed) if policy == "random" else StraightProvider()
        return provider, None

    if family == "vlnce_policy":
        from .vlnce_policy import VlncePolicyProvider

        checkpoint = (agent.get("checkpoint") or {}).get(benchmark_id)
        if checkpoint is None:
            raise ValueError("agent {!r} has no checkpoint for benchmark {!r}".format(
                agent.get("id"), benchmark_id))
        checkpoint = os.path.expandvars(checkpoint)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError("checkpoint not found: {}".format(checkpoint))
        provider = VlncePolicyProvider(
            habitat_config, env, checkpoint, agent["policy_name"])
        return provider, None

    raise NotImplementedError("family {!r}".format(family))
