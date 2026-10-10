# NaVILA — integration notes (Phase D5, 2026-09-11)

NaVILA is an 8B Llama-3 VILA navigation VLA (`github.com/AnjieCheng/NaVILA`). Runs in
`vlnsafe-navila` (py3.10, **no habitat**) as a subprocess. `family: navila_worker`.
Its P3/P4 belong to the released per-agent report, not the paper tables — the gate and the
P1/P2 rows are what matter.

## What is already wired

- **Worker IPC** (`vlnsafe/workers/navila_worker.py`): same JSON-lines shape as the
  StreamVLN worker.
- **Sentence parser** `parse_navila_sentence(text, forward_unit_m, turn_unit_deg)`,
  rewritten against `evaluation/vlnce_baselines/navila_trainer.py:222-282` @ `76b98f2`.
  The scaffold had guessed wrong on three points:

  | | scaffold guessed | upstream actually does |
  |---|---|---|
  | clauses | many actions per sentence | **exactly one action type** per generation |
  | selection | verb match | `stop` \| `is move forward` \| `is turn left` \| `is turn right`, **first match wins** |
  | defaults | 0.75 m / 30° | **25 cm / 15°** (i.e. a single primitive) |
  | magnitudes | any value | snapped to NaVILA's grid: 25/50/75 cm, 15/30/45° |

  The one deliberate generalisation: upstream divides by a hard-coded 25 cm / 15°, which
  is right only because that *is* R2R's step unit. We snap to NaVILA's output grid and
  then divide by the **simulator's** unit from the merged config — identical on R2R, and
  correct on RxR where a turn is 30° (a hard-coded 15 would double every RxR turn).
  Locked by `tests/test_agent_wiring.py`.
- **Client / rig**: `WorkerActionProvider` (VLA variant, `stream_observations=True` —
  NaVILA appends to `past_rgbs` every step too), `sensor_rig: navila` → `build_vla_rig`
  with the agent config's own `rig:` block: **RGB 512×512 HFOV 90, depth 256×256**
  (`habitat_extensions/config/vlnce_task.yaml`). RGB and depth genuinely differ in size,
  which is why the rig block sizes them separately. NaVILA's eval runs on VLN-CE /
  habitat-lab 0.1.x — the same generation as our harness — so these transfer directly.
- `exp_config` corrected to the paths that exist:
  `evaluation/vlnce_baselines/config/{r2r,rxr}_baselines/navila.yaml` (an RxR config
  *does* exist, unlike StreamVLN).

## `_NavilaModel` — bound and smoke-tested (2026-09-11)

Reproduces the prompt, frame sampling and `generate()` of
`navila_trainer.py::_eval_checkpoint` — the loop the published numbers come from — plus
`output_scores=True`. `load_pretrained_model(ckpt, name)` → `(tokenizer, model,
image_processor, ctx)`, `conv_mode llama_3`, 8-frame video history, `max_new_tokens=32`
with `KeywordsStoppingCriteria`.

`sample_and_pad_images` is **inlined** rather than imported: NaVILA's
`vlnce_baselines/__init__.py` imports every trainer in that package, which needs habitat
and lmdb — and this worker's whole point is that it runs in a habitat-free env. The
reimplementation is locked to upstream's behaviour by `tests/test_agent_wiring.py`.

### The P3 grouping choice

NaVILA emits a *sentence*, so no single token's distribution is the four-way action
categorical. The rule we use:

> Take the distribution at the first **verb** token over `{stop, move, turn}` — NaVILA's
> own action selection is exactly that three-way choice — then split the `turn` mass using
> the `{left, right}` distribution at the **following** position.

That is the chain rule over the two tokens that actually determine the action, and it
yields a proper 4-way distribution. All five words are single tokens in this tokenizer
(ids `stop 3009, move 3351, turn 2543, left 2163, right 1314`); if any were multi-token or
collided, the read-out returns `None` and the columns are `UNAVAILABLE`.

**Smoke test on GPU (random RGB — the plumbing is what is being checked):**

```
LOADED in 92s    verb ids [3009, 3351, 2543]  dir ids [2163, 1314]  readout_ok True
DECIDE 1  "The next action is turn left 45 degree."  -> [2, 2, 2]
  prob_stop 0.000  prob_forward 0.154  prob_left 0.597  prob_right 0.249
DECIDE 2  "The next action is move forward 75 cm."   -> [1, 1, 1]
  prob_stop 0.000  prob_forward 0.562  prob_left 0.196  prob_right 0.242
```

45°/15° = 3 turns and 75 cm/25 cm = 3 forwards: the parser's grid snapping and step-unit
decomposition are right on genuine model output, and the read-out's argmax agrees with the
emitted action in both cases.

## Env (`scripts/build_vla_envs.sh navila`)

py3.10 / **torch 2.3.0** / torchvision 0.18.0 / transformers 4.37.2 / numpy 1.26.0, from
`pyproject.toml`'s `dependencies` — *not* our guess. Two steps are easy to miss:

- `environment_setup.sh` copies `llava/train/transformers_replace/*` over the installed
  transformers. NaVILA **patches** transformers; the model will not load without it.
- The FlashAttention2 wheel is built for exactly `cu122torch2.3` + `cp310`. An env with a
  different torch imports fine and then dies at model load, which is why
  `scripts/check_vla_envs.py` asserts the versions rather than just "it imports".

Verify with `python scripts/check_vla_envs.py`.

## Gate

`reproduction_tolerance_sr: 2.0` (habitat-sim 0.2.4 upstream). Published R2R-CE val-unseen
SR 54.0 / SPL 49.0. One whole A5000 per worker; never run alongside StreamVLN (3-GPU cap).
