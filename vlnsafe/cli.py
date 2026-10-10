"""Command-line entry point: `python -m vlnsafe run|smoke|merge|describe`.

`run` parses (agent, benchmark, regime, shard) into a run, builds the Habitat env and the
action provider from configs/, and calls vlnsafe.rollout per episode. `smoke` is `run` on
a handful of episodes into scratch. `merge` wraps scripts/merge_shards.py. `describe`
resolves an agent config and prints its paths without touching habitat.

Ported from the pre-release harness + run_e1_safety_autopsy.py::main, stripped to
the run/merge/describe flags. Every other knob lives in a config file.

Must not import habitat/torch at module level; `run` imports the simulator lazily.
"""
import argparse
import csv
import json
import os
import sys

from . import config as cfg


_R2R_CLEARANCE_KEYS = ["depth"] + ["depth_{}".format(30 * k) for k in range(1, 12)]
_RXR_CLEARANCE_KEYS = ["clearance_depth_0"] + [
    "clearance_depth_{}".format(30 * k) for k in range(1, 12)]


def _path_field(value, resolved=None):
    """{value, exists} for a path; `resolved` is the absolute form actually checked."""
    check = resolved if resolved is not None else value
    return {"value": value, "exists": bool(check) and os.path.exists(check)}


def _pick(spec, benchmark):
    """A config field that is either a scalar or a per-benchmark dict."""
    if isinstance(spec, dict):
        return spec.get(benchmark)
    return spec


def cmd_describe(args):
    """Resolve an agent config for one benchmark and report every path it points at, with
    an existence flag. Exit 0 when every path exists (or the config is marked
    assets_pending); exit 1 otherwise. Touches no habitat."""
    bench_id = args.benchmark
    run_cfg = cfg.load_run(args.agent, bench_id, "block")
    agent = run_cfg["agent"]
    benchmark = run_cfg["benchmark"]
    repo = cfg.repo_root()
    family = agent["family"]
    is_worker = family.endswith("_worker")
    pending = bool(agent.get("assets_pending"))

    # An agent has an arm for a benchmark iff it names a checkpoint for it. Agents that
    # genuinely have none (BEVBert released R2R-CE only; no public RxR-CE Seq2Seq) are
    # reported as such instead of resolving paths that were never meant to exist -- their
    # table cells are \nmark, declared in the config's `unavailable:` block.
    has_arm = _pick(agent.get("checkpoint"), bench_id) is not None
    if not has_arm:
        print(json.dumps({
            "agent": agent["id"],
            "display_name": agent["display_name"],
            "tier": agent["tier"],
            "family": family,
            "benchmark": bench_id,
            "sensor_rig": agent["sensor_rig"],
            "arm_available": False,
            "unavailable": agent.get("unavailable", {}),
            "assets_pending": pending,
            "missing_paths": [],
            "ok": True,
        }, indent=2))
        return 0

    paths = {}
    code_root_abs = None
    if is_worker:
        code_root_abs = os.path.normpath(os.path.join(repo, agent["code_root"]))
        paths["worker_python"] = _path_field(os.path.expandvars(agent["worker_python"]))
        paths["code_root"] = _path_field(agent["code_root"], code_root_abs)

    exp_config = _pick(agent.get("exp_config", {}), bench_id)
    if exp_config is not None:
        base = code_root_abs if is_worker else os.path.join(repo, "external", "VLN-CE")
        paths["exp_config"] = _path_field(exp_config, os.path.join(base, exp_config))

    checkpoint = _pick(agent.get("checkpoint"), bench_id)
    if checkpoint is not None:
        paths["checkpoint"] = _path_field(os.path.expandvars(checkpoint))

    pretrained = _pick(agent.get("pretrained"), bench_id)
    if pretrained is not None:
        paths["pretrained"] = _path_field(os.path.expandvars(pretrained))

    ds_key = cfg.agent_dataset_key(agent, bench_id)
    ds_root = cfg.benchmark_dataset_root(benchmark, ds_key)
    paths["dataset"] = _path_field(ds_root)
    paths["dataset"]["key"] = ds_key

    missing = sorted(k for k, v in paths.items() if not v["exists"])
    ok = pending or not missing
    print(json.dumps({
        "agent": agent["id"],
        "display_name": agent["display_name"],
        "tier": agent["tier"],
        "family": family,
        "benchmark": bench_id,
        "sensor_rig": agent["sensor_rig"],
        "arm_available": True,
        "back_algo": agent.get("back_algo"),
        "reproduction_tolerance_sr": agent.get("reproduction_tolerance_sr"),
        "published": _pick(agent.get("published", {}), bench_id),
        "episode_subset": _pick(agent.get("episode_subset"), bench_id),
        "paths": paths,
        "unavailable": agent.get("unavailable", {}),
        "assets_pending": pending,
        "missing_paths": missing,
        "ok": ok,
    }, indent=2))
    return 0 if ok else 1


def cmd_run(args):
    run_cfg = cfg.load_run(args.agent, args.benchmark, args.regime)
    bench = run_cfg["benchmark"]
    agent = run_cfg["agent"]
    seed = args.seed if args.seed is not None else run_cfg["defaults"]["seed"]
    # --split overrides the benchmark split (held-out calibration on R2R val_seen)
    split = getattr(args, "split", None) or bench["split"]
    sensor_rig = agent["sensor_rig"]
    vlnce_root = os.path.join(cfg.repo_root(), "external", "VLN-CE")

    guard_name, guard_cfg = (cfg.load_guard(args.guard) if getattr(args, "guard", None)
                             else (None, None))
    # a guarded run never defaults into the landed arm's directory (resume would append to it)
    exempt = bool(getattr(args, "exempt_guard_redecisions", False))
    if exempt and guard_name is None:
        raise SystemExit("--exempt-guard-redecisions needs --guard")
    arm_dir = args.regime if guard_name is None else "{}_guard-{}{}".format(
        args.regime, guard_name, "-exempt" if exempt else "")
    # The agent's own Tryout controller, as its own baseline arm. The
    # released code enables it only without sliding. With --guard the guard checks
    # each of Tryout's forward steps instead of running its own manoeuvre.
    tryout = bool(getattr(args, "tryout", False))
    if tryout:
        if exempt:
            raise SystemExit("--tryout --guard makes no re-decisions to exempt")
        if guard_name is not None and guard_cfg.get("memory_frames", 8) != 1:
            raise SystemExit("--tryout --guard needs a guard with memory_frames 1 "
                             "(e.g. stall-mem1)")
        if args.regime != "block":
            raise SystemExit("--tryout: the released code enables Tryout only without "
                             "sliding (regime block)")
        if agent.get("family") != "etpnav_worker":
            raise SystemExit("--tryout: agent {!r} is not ETPNav-lineage".format(args.agent))
        arm_dir = "{}_tryout".format(args.regime) + (
            "" if guard_name is None else "_guard-{}".format(guard_name))
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(
        cfg.repo_root(), "results", args.agent, args.benchmark, arm_dir)
    os.makedirs(out_dir, exist_ok=True)

    # VLN-CE's get_config resolves exp_config relative to cwd; the archived runs ran from
    # the VLN-CE root, so we do the same. Output paths are already absolute.
    sys.path.insert(0, vlnce_root)
    os.chdir(vlnce_root)

    from .simulator import environment as envmod
    from .agents.registry import build_provider
    from .clearance_predictor import PrimitiveClearancePredictor
    from .geometry import CLEARANCE_DEPTH_SENSOR_KEYS
    from .guard import GuardLayer
    from .records.writers import EpisodeWriter, StepWriter, write_manifest
    from .rollout import run_episode

    # which dataset root supplies this agent's instructions (bert / xlmr / official); the
    # untokenised `official` split still provides the episode list, order and NDTW GT.
    ds_key = cfg.agent_dataset_key(agent, args.benchmark)
    instr_root = cfg.benchmark_dataset_root(bench, ds_key)
    instr_type = cfg.benchmark_dataset_type(bench, ds_key)

    # Resume: if this shard's episodes.csv already has rows (an earlier attempt was
    # killed mid-run -- the harness's background-task tracking doesn't survive a session
    # restart), skip episode_ids already written instead of redoing them. episodes.csv is
    # the resume record, not steps.csv, since only completed episodes write an episode row.
    suffix = "" if args.shards <= 1 else ".shard{}".format(args.shard_index)
    ep_path = os.path.join(out_dir, "episodes{}.csv".format(suffix))
    already_done = set()
    if os.path.exists(ep_path) and os.path.getsize(ep_path) > 0:
        with open(ep_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and header[0] == "episode_id":
                already_done = {row[0] for row in reader if row}

    # A recurrent VLN-CE policy is built from the same yaml that carries its MODEL block, so
    # its own exp_config is the base (Seq2Seq's differs from the benchmark's CMA default:
    # unidirectional instruction encoder). Other families keep the benchmark's base config.
    base_config = bench["habitat_base_config"]
    if agent.get("family") == "vlnce_policy":
        base_config = _pick(agent.get("exp_config"), args.benchmark) or base_config

    if args.benchmark == "r2r":
        clearance_keys = _R2R_CLEARANCE_KEYS
        metas = envmod.select_episodes(bench["dataset"]["official"], split, args.episodes)
        metas = envmod.shard(metas, args.shards, args.shard_index)
        total_assigned = len(metas)
        metas = [m for m in metas if m["episode_id"] not in already_done]
        habitat_config = envmod.build_r2r_config(
            base_config, split, bench["dataset"]["official"],
            bench["dataset"]["scenes"], instr_root,
            seed, args.regime, sensor_rig, agent.get("rig"),
            dataset_type=instr_type,
        )
        env = envmod.build_env(habitat_config)
        envmod.filter_env_episodes_ordered(env, metas)
    else:  # rxr: stride-sampled from env.episodes under its own seed
        clearance_keys = _RXR_CLEARANCE_KEYS
        sel = bench.get("select", {})
        rxr_seed = args.seed if args.seed is not None else sel.get("seed", seed)
        seed = rxr_seed
        habitat_config = envmod.build_rxr_config(
            base_config, split, instr_root,
            bench["dataset"]["scenes"], rxr_seed, args.regime, sensor_rig,
            agent.get("rig"), turn_angle_deg=getattr(args, "turn_angle_deg", None),
        )
        env = envmod.build_env(habitat_config)
        metas = envmod.select_rxr_metas(
            env, sel.get("stride", 1), sel.get("offset", 0), args.episodes,
            subsample=getattr(args, "subsample", None) or 1)
        metas = envmod.shard(metas, args.shards, args.shard_index)
        total_assigned = len(metas)
        metas = [m for m in metas if m["episode_id"] not in already_done]
        envmod.filter_env_episodes_ordered(env, metas)

    if already_done:
        print("resume: {} episodes already done, {} remaining".format(
            len(already_done), len(metas)))
    ep_writer = EpisodeWriter(ep_path, append=True)
    step_writer = StepWriter(os.path.join(out_dir, "steps{}.csv".format(suffix)), append=True)
    run_cfg["seed"] = seed   # the run's seed, for workers that sample (OmniNav's flow head)
    provider = handle = None
    try:
        provider, handle = build_provider(
            run_cfg, habitat_config, env, args.benchmark, clearance_keys)
        # A provider that sends an occupancy cloud to a worker (T2) already reports its own
        # predicted clearance; every other agent gets the same predictor applied to its
        # own depth frame, so the geometry column exists for all of them.
        predictor = None
        if not getattr(provider, "send_occupancy", False):
            predictor = PrimitiveClearancePredictor.from_habitat_config(
                habitat_config, sensor_keys=CLEARANCE_DEPTH_SENSOR_KEYS,
                sensor_name="CLEARANCE_DEPTH_0")
        # The guard projects with the same rig as the clearance column: T1 reuses the
        # predictor above (ring, cached per frame); a T2 worker provider's projection
        # reads the policy rig's depth cameras, so the guard gets that rig too.
        guard = tryout_guard = timing_writer = None
        if guard_cfg is not None:
            # constants from the sensor that renders `clearance_keys`: on RxR that is the
            # 0-10 m clearance ring, not the 0.5-5 m policy camera (N43 smoke: read with the
            # policy camera's range, a 0.3 m wall decoded as 0.64 m and no depth veto fired)
            ring = clearance_keys[0].startswith("clearance_depth")
            guard_predictor = predictor or PrimitiveClearancePredictor.from_habitat_config(
                habitat_config, sensor_keys=clearance_keys,
                sensor_name="CLEARANCE_DEPTH_0" if ring else "DEPTH_SENSOR")
            guard = GuardLayer.from_config(guard_cfg, guard_predictor)
            if guard.oracle:   # The navmesh answers instead of depth (upper bound)
                from .guard import navmesh_oracle
                guard.oracle_fn = navmesh_oracle(env.sim, args.regime == "slide")
            if tryout:   # the guard runs inside the provider's Tryout, not in rollout
                guard, tryout_guard = None, guard
        if exempt:
            if not getattr(provider, "send_occupancy", False):   # ETPNav-family worker
                raise SystemExit("--exempt-guard-redecisions: agent {!r} has no decision "
                                 "budget to exempt".format(args.agent))
            provider.exempt_guard_redecisions = True
        if tryout:
            from habitat.sims.habitat_simulator.actions import HabitatSimActions
            provider.enable_tryout(seed, HabitatSimActions, guard=tryout_guard)
        timing_writer = None
        if getattr(args, "timing", False):   # Per-step wall clock, sidecar file
            timing_fh = open(os.path.join(out_dir, "timing{}.csv".format(suffix)), "w",
                             newline="")
            timing_writer = csv.writer(timing_fh)
            timing_writer.writerow(["episode_id", "step_index", "executed_action",
                                    "guard_action", "policy_s", "predictor_s", "guard_s",
                                    "sim_s", "step_s"])
        for meta in metas:
            run_episode(env, meta, provider, ep_writer, step_writer,
                        agent=args.agent, benchmark=args.benchmark, regime=args.regime,
                        seed=seed, shard_index=args.shard_index,
                        clearance_sensor_keys=clearance_keys,
                        clearance_predictor=predictor, guard=guard,
                        timing_writer=timing_writer)
            if timing_writer is not None:
                timing_fh.flush()
    finally:
        if timing_writer is not None:
            timing_fh.close()
        ep_writer.close()
        step_writer.close()
        if handle is not None:
            handle.close()
        env.close()

    ckpt_spec = agent.get("checkpoint") or {}
    ckpt = ckpt_spec.get(args.benchmark) if isinstance(ckpt_spec, dict) else ckpt_spec
    write_manifest(os.path.join(out_dir, "manifest{}.json".format(suffix)), {
        "agent": args.agent, "benchmark": args.benchmark, "regime": args.regime,
        "split": split, "seed": seed,
        "episode_count": total_assigned,
        "dataset_key": cfg.agent_dataset_key(agent, args.benchmark),
        # a partial split draws the dagger in the tables; None once the full-split arm lands
        "episode_subset": _pick(agent.get("episode_subset"), args.benchmark),
        "full_split": (args.episodes is None and not getattr(args, "subsample", None)
                       and _pick(agent.get("episode_subset"), args.benchmark) is None),
        "subsample": getattr(args, "subsample", None),
        "shards": args.shards, "shard_index": args.shard_index,
        "physics": envmod.physics_manifest_fields(habitat_config),
        "checkpoint": os.path.expandvars(ckpt) if ckpt else None,
        "command": " ".join(sys.argv),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "guard": None if guard_cfg is None else {
            "name": guard_name, "config": guard_cfg,
            "exempt_guard_redecisions": exempt,
            "mode": "tryout_veto" if tryout else "manoeuvre"},
        "tryout": tryout,
    })
    print("wrote {}/episodes{}.csv ({} new episodes, {} total)".format(
        out_dir, suffix, len(metas), total_assigned))
    return 0


def cmd_smoke(args):
    """`make smoke`: a handful of episodes for one agent, into scratch — the ten-minute
    'does my setup work' check."""
    out = args.out or os.path.join(
        os.path.expandvars("$VLNSAFE_SCRATCH"), "smoke", args.agent, args.benchmark)
    ns = argparse.Namespace(
        agent=args.agent, benchmark=args.benchmark, regime=args.regime,
        shards=1, shard_index=0, episodes=args.episodes, seed=args.seed, out=out,
        guard=args.guard, tryout=args.tryout)
    rc = cmd_run(ns)
    print("smoke: {} {} {} -> {} (rc={})".format(
        args.agent, args.benchmark, args.regime, out, rc))
    return rc


def cmd_merge(args):
    """Thin wrapper over scripts/merge_shards.py so the documented interface is
    `python -m vlnsafe merge <arm_dir>`."""
    import subprocess
    script = os.path.join(cfg.repo_root(), "scripts", "merge_shards.py")
    return subprocess.call([sys.executable, script, args.arm_dir])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vlnsafe")
    sub = parser.add_subparsers(dest="cmd")

    r = sub.add_parser("run")
    r.add_argument("--agent", required=True)
    r.add_argument("--benchmark", required=True, choices=["r2r", "rxr"])
    r.add_argument("--regime", required=True, choices=["block", "slide"])
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--episodes", type=int, default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--split", default=None,
                   help="dataset split (default: the benchmark config's, val_unseen)")
    r.add_argument("--turn-angle-deg", type=int, default=None,
                   help="rxr only: override the benchmark's 30 degree turn (gate runs)")
    r.add_argument("--subsample", type=int, default=None,
                   help="rxr only: every k-th episode of the stride sample (gate runs)")
    r.add_argument("--guard", default=None,
                   help="guard operating point: configs/guard/<name>.yaml or a yaml path "
                        "(vlnsafe/guard.py); default out becomes <regime>_guard-<name>")
    r.add_argument("--exempt-guard-redecisions", action="store_true",
                   help="with --guard: a re-decision forced by a guard manoeuvre does not "
                        "count against an ETPNav-family worker's max_len budget")
    r.add_argument("--timing", action="store_true",
                   help="also write timing.csv: per-step wall clock of policy, predictor, "
                        "guard and simulator; rows are unchanged")
    r.add_argument("--tryout", action="store_true",
                   help="ETPNav-family, block only: run the agent's own Tryout collision "
                        "escape (vlnsafe/agents/tryout.py); default out <regime>_tryout. "
                        "With --guard the guard vetoes Tryout's forward steps; out "
                        "<regime>_tryout_guard-<name>")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("smoke")
    s.add_argument("--agent", required=True)
    s.add_argument("--benchmark", default="r2r", choices=["r2r", "rxr"])
    s.add_argument("--regime", default="block", choices=["block", "slide"])
    s.add_argument("--episodes", type=int, default=5)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--guard", default=None)
    s.add_argument("--tryout", action="store_true")
    s.set_defaults(func=cmd_smoke)

    m = sub.add_parser("merge")
    m.add_argument("arm_dir")
    m.set_defaults(func=cmd_merge)

    d = sub.add_parser("describe")
    d.add_argument("--agent", required=True)
    d.add_argument("--benchmark", default="r2r", choices=["r2r", "rxr"])
    d.set_defaults(func=cmd_describe)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
