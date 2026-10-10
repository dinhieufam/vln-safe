"""Config loading: configs/defaults.yaml + configs/benchmarks/<id>.yaml +
configs/agents/<id>.yaml, merged, with $VLNSAFE_* and ${...} expanded.

What: turns the three YAML files into one dict per run. Expands environment variables in
every string value (both $VAR and ${VAR}). Fails loudly on an unknown agent/benchmark id
or a missing required key.

Who reads it: vlnsafe.cli, vlnsafe.simulator.environment (rig constants), vlnsafe.agents.
registry, analysis/make_tables.py (the `unavailable:` block -> \\nmark).

Must not import: habitat, torch.

Ported from the config plumbing scattered through the pre-release harness (read_json,
the $VLNSAFE_* storage roots).
"""
import os

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(_ROOT, "configs")

# $VLNSAFE_* name the bulk-storage roots and the worker conda envs. When they are not
# already exported, fall back to the git-ignored data/ directory (environment/setup.sh can
# symlink data/datasets, data/models, ... to a large disk). This keeps every machine path
# out of the tracked code while letting `make arm` / `describe` / the tests resolve configs
# without a manual `export` step.
for _var, _kind in (("VLNSAFE_DATASETS", "datasets"), ("VLNSAFE_MODELS", "models"),
                    ("VLNSAFE_CHECKPOINTS", "checkpoints"), ("VLNSAFE_SCRATCH", "scratch"),
                    ("VLNSAFE_ENVS", "envs")):
    os.environ.setdefault(_var, os.path.join(_ROOT, "data", _kind))

# the provider class an agent config's `family:` selects (vlnsafe.agents.registry builds it).
AGENT_FAMILIES = (
    "etpnav_worker",     # ETPNav / ETP-R1 / BEVBert, subprocess via agents/remote.py
    "vlnce_policy",      # CMA / Seq2Seq, in-process
    "streamvln_worker",  # StreamVLN, subprocess, no habitat
    "navila_worker",     # NaVILA, subprocess, no habitat
    "omninav_worker",    # OmniNav, subprocess, no habitat (waypoint -> primitives)
    "scripted",          # random / straight-line, for smoke tests
)
AGENT_TIERS = ("T1", "T2")
_REQUIRED_AGENT_KEYS = ("id", "display_name", "tier", "family", "sensor_rig")


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _load_yaml(path):
    with open(path) as f:
        return _expand(yaml.safe_load(f))


def repo_root():
    return _ROOT


def load_defaults():
    return _load_yaml(os.path.join(CONFIGS, "defaults.yaml"))


def load_benchmark(benchmark):
    path = os.path.join(CONFIGS, "benchmarks", "{}.yaml".format(benchmark))
    if not os.path.exists(path):
        raise ValueError("no benchmark config for {!r} ({})".format(benchmark, path))
    return _load_yaml(path)


def load_agent(agent):
    path = os.path.join(CONFIGS, "agents", "{}.yaml".format(agent))
    if not os.path.exists(path):
        raise ValueError("no agent config for {!r} ({})".format(agent, path))
    cfg = _load_yaml(path)
    if cfg.get("id") != agent:
        raise ValueError("agent config {} declares id {!r}, expected {!r}".format(
            path, cfg.get("id"), agent))
    missing = [k for k in _REQUIRED_AGENT_KEYS if k not in cfg]
    if missing:
        raise ValueError("agent config {} is missing required key(s): {}".format(
            path, ", ".join(missing)))
    if cfg["tier"] not in AGENT_TIERS:
        raise ValueError("agent config {} has tier {!r}, expected one of {}".format(
            path, cfg["tier"], AGENT_TIERS))
    if cfg["family"] not in AGENT_FAMILIES:
        raise ValueError("agent config {} has family {!r}, expected one of {}".format(
            path, cfg["family"], AGENT_FAMILIES))
    return cfg


def agent_dataset_key(agent_cfg, benchmark_id):
    """Which key in the benchmark config's `dataset:` map this agent's instructions come
    from. Defaults to `official` for agents that read the untokenised split (VLAs, and any
    config without a `dataset:` block)."""
    return agent_cfg.get("dataset", {}).get(benchmark_id, "official")


def benchmark_dataset_root(benchmark_cfg, key):
    """Resolve a `dataset:` key (see agent_dataset_key) to its root path, failing loudly on
    an unknown key so a typo in an agent config never silently falls back."""
    roots = benchmark_cfg.get("dataset", {})
    if key not in roots:
        raise ValueError("benchmark {!r} has no dataset root {!r}; known: {}".format(
            benchmark_cfg.get("id"), key, sorted(roots)))
    return roots[key]


def benchmark_dataset_type(benchmark_cfg, key):
    """Resolve a `dataset:` key to its habitat TASK_CONFIG.DATASET.TYPE:
    the v1 loader expects an `instruction_vocab` block that the xlmr (ETP-R1) pass does not
    carry. Defaults to VLN-CE-v1, the value every pass except xlmr needs."""
    return benchmark_cfg.get("dataset_type", {}).get(key, "VLN-CE-v1")


def load_guard(name_or_path):
    """A guard operating point: configs/guard/<name>.yaml, or a path to a yaml file.
    Returns (name, dict); the name labels the default output directory."""
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(CONFIGS, "guard", "{}.yaml".format(name_or_path))
    if not os.path.exists(path):
        raise ValueError("no guard config {!r} ({})".format(name_or_path, path))
    name = os.path.splitext(os.path.basename(path))[0]
    return name, _load_yaml(path) or {}


def load_run(agent, benchmark, regime):
    if regime not in ("block", "slide"):
        raise ValueError("regime must be block or slide, got {!r}".format(regime))
    return {
        "defaults": load_defaults(),
        "benchmark": load_benchmark(benchmark),
        "agent": load_agent(agent),
        "regime": regime,
    }
