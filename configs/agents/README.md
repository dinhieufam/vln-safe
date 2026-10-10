# Agent configs

One `<id>.yaml` per agent — `seq2seq`, `cma`, `navila`, `streamvln`, `bevbert`, `etpnav`,
`etpr1`. The id is the file's basename and appears nowhere else as a literal
.

## Keys

| key | meaning |
|---|---|
| `id` / `display_name` / `tier` (`T1`/`T2`) | identity; `display_name` is what the paper prints |
| `family` | the provider class: `etpnav_worker` (ETPNav / ETP-R1 / BEVBert subprocess), `vlnce_policy` (CMA / Seq2Seq in-process), `streamvln_worker`, `navila_worker` |
| `sensor_rig` | a named rig `simulator/sensors.py` builds (`etpnav_12view`, `vlnce_recurrent`, `streamvln`, `navila`) |
| `worker_python` / `code_root` | worker families only: the env interpreter and the repo it `chdir`s into |
| `precision` | `streamvln_worker` / `navila_worker`: `fp16` / `bf16` / `fp32` for the model load |
| `policy_name` | `vlnce_policy`: the VLN-CE registered policy (`CMAPolicy` / `Seq2SeqPolicy`) |
| `worker_trainer_module` | `etpnav_worker` only: the SS trainer class to import (`vlnce_baselines.ss_trainer_ETP` for ETPNav/BEVBert, `…_ETP_R1` for ETP-R1) |
| `task_type` | `etpnav_worker` only, ETP-R1: `{benchmark: 1|2}` — the tag threaded through `extract_instruction_tokens` / `_nav_gmap_variable`. Absent = ETPNav's unchanged signatures |
| `exp_config` | `{benchmark: path}` — the upstream config the worker/policy loads (relative to `code_root`, or to `external/VLN-CE` for `vlnce_policy`) |
| `dataset` | `{benchmark: key}` — which root in the benchmark config's `dataset:` map supplies instructions. Omit to use `official` (the untokenised split; VLAs read raw text). Resolved by `config.agent_dataset_key` / `config.benchmark_dataset_root` |
| `checkpoint` / `pretrained` | `{benchmark: path}` (or a scalar for a shared `pretrained`), `$VLNSAFE_*`-expanded |
| `published` | `{benchmark: {sr, spl, regime, source}}` — the reproduction-gate target; `sr: null` where no comparable public number exists |
| `reproduction_tolerance_sr` | 1.0 for ETPNav-family and the recurrent baselines; 2.0 for the VLAs (upstream habitat-sim 0.2.4 vs our 0.1.7) |
| `episode_subset` | `{benchmark: n}` — a partial split, echoed into `manifest.json` and drawn as the `†` in the tables (CMA's 700) |
| `unavailable` | `{column: reason}` — every `\nmark` the paper draws, declared with a greppable reason. `analysis/make_tables.py` reads it |
| `assets_pending: true` | checkpoints / env not downloaded yet. `describe` still resolves and reports the paths but does not fail on the missing ones |

## Gate C

```
python -m vlnsafe describe --agent <id> --benchmark {r2r,rxr}
```

prints the resolved paths (with an `exists` flag each), the tier, the sensor rig and the
declared `unavailable:` block, and exits non-zero if a path is missing unless the config
is `assets_pending`. For `etpr1` every path resolves and exists.
