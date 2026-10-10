# VLN-Safe: Successful but Unsafe

Code, configs and episode-level results for

> **Successful but Unsafe: An Analytical Benchmark for Revealing Collisions in
> Vision-and-Language Navigation**
> TODO: authors. arXiv:TODO, 2026. [[paper]](https://arxiv.org/abs/TODO)

Standard VLN-CE metrics (SR, SPL, nDTW) say whether an agent reached the goal, not whether it
hit anything on the way. VLN-Safe re-runs published, frozen VLN-CE agents in Habitat and
measures contact alongside success:

- **P1: do successful episodes avoid collisions?** Under the standard *sliding* regime,
  38.5–56.8% of successful R2R-CE episodes contain at least one collision step, across six
  agents.
- **P2: how much do evaluation choices matter?** Turning sliding off (*blocking*) lowers
  success rate by 11.7–36.1 points on R2R-CE. The displacement proxy that some codebases use
  instead of the simulator flag misses 77.7–87.6% of episodes that had contact.
- **Collision-prediction signals.** Depth geometry and movement history predict imminent
  contact; the policies' own scores do not reliably predict it.
- **A training-free collision guard.** It vetoes risky forward steps, makes a short escape,
  and hands control back to the unchanged policy. Under blocking it removes 62–98% of
  collision steps and raises SR by 6.9–29.3 points across five agents. Under sliding it
  leaves SR unchanged and raises collision-free success by 4.1–12.4 points.

## Repository layout

| path | contents |
|---|---|
| `vlnsafe/` | the harness: simulator setup, rollout loop, collision guard (`guard.py`), depth geometry, agent providers and subprocess workers, log writers |
| `configs/agents/` | one YAML per agent: checkpoint, sensor rig, worker env, published reference numbers |
| `configs/benchmarks/` | R2R-CE and RxR-CE rig and dataset settings |
| `configs/guard/` | guard configurations; see [Guard configs](#guard-configs) |
| `analysis/` | results → `results/summary/*.json` → LaTeX tables and figures in `outputs/` |
| `scripts/` | launch, merge, replay, setup and checkpoint-download helpers |
| `results/` | released per-episode logs (`episodes.csv` + `manifest.json` per arm) and the summary JSONs |
| `survey/` | the literature audit behind the "only 3 papers state the regime" count |
| `docs/schema.md` | the log format, column by column |
| `docs/agent_notes/` | per-agent integration notes |
| `environment/` | conda env records and `setup.sh` |
| `tests/` | unit tests (no simulator needed) and one slow rollout test |

## Reproduce the paper's tables and figures from the released logs

No simulator or GPU is needed. Use Python 3.9+ with `pandas`, `numpy`, `scipy`,
`matplotlib` and `pyyaml`:

```bash
python analysis/collect.py          # results/ -> results/summary/paper_numbers.json
make tables figures                 # -> outputs/tables/, outputs/figures/
python analysis/supp_tables.py      # supplementary tables -> outputs/tables/supp/
python analysis/audit_survey.py     # literature-audit tables -> outputs/tables/generated/
```

Each experimental condition ("arm") is the triple `(agent, benchmark, regime)`. Its logs are
in `results/<agent>/<benchmark>/<regime>[_guard-<config>]/`. The per-step logs
(`steps.csv.gz`, several GB in total) are not in this repository. The summaries that need
them (signal AUROCs, headroom, first-contact position) are released precomputed in
`results/summary/`.

## Run the benchmark

### Setup

The harness runs in Python 3.6 with habitat-sim 0.1.7 and habitat-lab v0.1.7, the stack
VLN-CE and ETPNav are released on. VLA agents (StreamVLN, OmniNav) run as subprocess
workers in their own envs, without habitat.

```bash
export VLNSAFE_ROOT=/path/to/large/disk   # datasets, checkpoints, envs (~100s of GB)
environment/setup.sh                       # envs, data/ symlinks, external checkouts, checkpoints
```

`setup.sh` runs these steps, which can also be run individually:

- `scripts/setup_external.sh` clones every third-party agent repository at its pinned
  commit into `external/`. Nothing third-party is redistributed here.
- `scripts/download_checkpoints.sh --fetch` downloads every released checkpoint and checks
  its sha256.
- `scripts/link_external_data.sh` wires each checkout's `data/` directory to the stores.

Storage roots are read from `$VLNSAFE_DATASETS`, `$VLNSAFE_MODELS`, `$VLNSAFE_CHECKPOINTS`,
`$VLNSAFE_SCRATCH` and `$VLNSAFE_ENVS`, which default to `data/<kind>`. The datasets are
Matterport3D scenes (`$VLNSAFE_DATASETS/mp3d/`) and the R2R-CE / RxR-CE episode files
named in `configs/benchmarks/*.yaml`. Both require accepting their own licenses.

Check that an agent's paths resolve:

```bash
python -m vlnsafe describe --agent etpr1 --benchmark r2r
```

### Run an arm

```bash
# unguarded, blocking regime
python -m vlnsafe run --agent etpr1 --benchmark r2r --regime block
# the same with the collision guard
python -m vlnsafe run --agent etpr1 --benchmark r2r --regime block --guard stall-mem1
# sharded over GPUs, then merged
scripts/launch_arm.sh etpr1 r2r block 2
python scripts/merge_shards.py results/etpr1/r2r/block
```

`--regime slide` uses Habitat's sliding (`ALLOW_SLIDING: True`); `block` disables it. Agents
evaluated in the paper: `etpr1`, `etpnav`, `omninav`, `streamvln`, `cma`, `seq2seq`. The
configs for `bevbert`, `navila` and `g3dlf` are included but were not part of the paper's
evaluation.

### Guard configs

| config | paper |
|---|---|
| `stall-mem1` | **the guard** (Tables 3–5): stall veto + single-frame depth veto, τ = 0.05 m |
| `stall-only` | ablation: stall veto only (also the simplest reactive baseline) |
| `depth-mem1` | ablation: depth veto only |
| `stall` | ablation: 8-frame depth memory |
| `stall-mem1-tau{0.02,0.10,0.20}` | sensitivity to τ |
| `oracle-mem1` | oracle bound: the navigation mesh replaces both predictors |
| `default`, `uncapped`, `stall-mem1-uncapped` | earlier development variants, not in the paper |

The guard needs only panoramic depth, odometry and access to the controller's queued
primitives. It reads no policy scores or weights, so the same config wraps every agent.

### Tests

```bash
make test        # pure tests, no habitat
```

## Citation

```bibtex
@article{vlnsafe2026,
  title   = {Successful but Unsafe: An Analytical Benchmark for Revealing Collisions in
             Vision-and-Language Navigation},
  author  = {TODO},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2026}
}
```

## License

The code in this repository is released under the MIT License (`LICENSE`). The evaluated
agents, their checkpoints, Habitat, Matterport3D and the R2R-CE / RxR-CE datasets keep their
own licenses and are not redistributed here.
