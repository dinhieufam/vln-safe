# StreamVLN — integration notes (Phase D4, 2026-09-11)

StreamVLN is an 8B Qwen video-LM navigation VLA (`github.com/InternRobotics/StreamVLN`).
It runs in `vlnsafe-streamvln` (py3.9 / torch 2.1.2, **no habitat**) as a subprocess; the
harness process owns the simulator. `family: streamvln_worker`.

## What is already wired (no repo / env needed)

- **Worker IPC** (`vlnsafe/workers/streamvln_worker.py`): JSON-lines loop, observation
  decode, `_RedirectStdoutToDevnull`, `VLNSAFE_STREAMVLN_WORKER_READY` handshake — same
  shape as the ETPNav worker.
- **Action-text parser** `parse_action_string()` — arrows (`↑ ← →`), words
  (`forward / left / right / stop`), commas, repeats → habitat primitive ids, truncated at
  the first STOP. Locked by `tests/test_agent_wiring.py`.
- **Client** (`agents/remote.py`): `WorkerActionProvider` with `observation_keys =
  VLA_OBSERVATION_KEYS` (`rgb, depth, gps, compass`) and `send_occupancy=False` — no
  occupancy payload, no panoramic sensor set. `registry.build_provider` builds it for
  `family in (streamvln_worker, navila_worker)`.
- **Rig** `sensor_rig: streamvln` → `sensors.build_vla_rig` (forward RGB+DEPTH + GPS +
  COMPASS). `build_r2r_config` / `build_rxr_config` both have the branch.
- The first-action-token read-out flows through `risk_signals = {prob_stop, prob_forward,
  prob_left, prob_right}` → `rollout.py` writes the T1 columns.

## Read from the repo (`external/StreamVLN` @ `e48f6ff`) — corrections to the plan

| | originally assumed | the repo actually does |
|---|---|---|
| precision | fp16 | **bf16** + `flash_attention_2` (`streamvln_agent.py::__main__`) |
| camera | — | **640×480, HFOV 79**, depth 0–10 m (`config/vln_r2r.yaml`) |
| action vocab | `STOP ↑ ← →` | confirmed: `OrderedDict({'STOP':0,'↑':1,'←':2,'→':3})` |
| history | "keeps history across requests" | **one frame per *executed* step**, not per decision |
| RxR config | `config/vln_rxr.yaml` | does not exist; the repo ships R2R only |

The history finding is the load-bearing one. `streamvln_eval.py:247-344` appends the
observation to `rgb_list`/`time_ids` on **every** loop iteration and calls `generate()`
only when the action chunk runs dry; the KV cache is reset every `num_frames` (32) steps
and a strided slice of the history is prepended then. Sending only the decision-step frame
would silently corrupt that history. So `WorkerActionProvider` gained
`stream_observations=True`: it buffers the frames seen since the last decision and ships
them as `history_observations` (oldest first) alongside the current one, and the worker
replays them through `observe()` before generating. `tests/test_agent_wiring.py`
pins the count — three executed primitives ⇒ two history frames plus the current.

## `_StreamVlnModel` — bound and smoke-tested (2026-09-11)

It wraps the repo's own `VLNEvaluator` rather than reimplementing `step()`: that method
carries the SlowFast/KV-cache bookkeeping the published numbers depend on (when to reset
the cache, when to prepend a strided history slice), so it is reused verbatim.
`observe()` is `step(run_model=False)`, `generate()` is `step(run_model=True)`.

For the P3 read-out, `model.generate` is **wrapped** to force `output_scores=True` and
stash the scores — the original "generate() needs output_scores=True added", done
without editing upstream. The probabilities are the softmax over the four action-symbol
token ids at the *first generated token that is one of them*. If the tokenizer gives a
multi-token or colliding id for any symbol the read-out returns `None`, so the columns
become `UNAVAILABLE` rather than a fabricated number.

**Smoke test on GPU (random RGB, so the content is meaningless — the plumbing is not):**

```
LOADED in ~90s
DECIDE 1 in 3.8s   text '→→→→'   primitives [3,3,3,3]
  prob_stop 0.0000  prob_forward 0.0001  prob_left 0.4687  prob_right 0.5311
DECIDE 2 in 0.42s  (KV cache hit)        primitives [3,3,3,3]
  prob_stop 0.0000  prob_forward 0.0001  prob_left 0.0001  prob_right 0.9998
```

Four primitives per generation matches `num_future_steps: 4`; the 3.8s → 0.42s drop is the
KV cache working across requests; and the first decision's near-tie is exactly the kind of
signal P3/P4 consume.

**Two assets no README mentions, both found by loading the model** (now in
`scripts/download_checkpoints.sh --fetch`):

1. The `_v1_3` HF repo ships **no tokenizer files**, though the eval script calls
   `AutoTokenizer.from_pretrained(model_path)`. They are copied from the same author's
   previous checkpoint, after asserting `vocab_size`, `image_token_index`, `hidden_size`,
   layer count, bos/eos and vision tower all match.
2. `config.json` names `mm_vision_tower: google/siglip-so400m-patch14-384` (3.3 GB),
   fetched from the Hub at load time — pre-cached so a run is not one network blip from
   dying.

The env is verified by `scripts/check_vla_envs.py`: py3.9, torch 2.1.2+cu121,
transformers 4.45.1, numpy 1.26.1, flash_attn 2.5.8, and the repo's model code imports.

## Gate

habitat-sim differs (upstream 0.2.4 vs our 0.1.7) → `reproduction_tolerance_sr: 2.0`, and
the version difference is reported beside the published SR 56.4 in Table 2's notes.
GPU: one whole A5000 per model worker; at most 3 StreamVLN shards at once under the cap.

## RxR: two open questions found Sep 20 (before any RxR arm)

1. **Published number exists.** The README reports RxR NE 5.65 / SR 54.4 / SPL 45.4 / nDTW
   63.7 for this checkpoint; `configs/agents/streamvln.yaml` had `sr: null`
   ("test only"), now 54.4. The README names neither split nor collision regime.
2. **Turn angle mismatch.** The model's prompt (`streamvln_eval.py:103`) says "TURN LEFT
   or TURN RIGHT by 15 degrees" and `config/vln_r2r.yaml` uses `turn_angle: 15`; the repo
   ships no RxR eval config. Our RxR env (`build_rxr_config`) applies the benchmark's
   `TURN_ANGLE = 30`, and `build_vla_rig` deliberately does not override turn angle.
   Unresolved which one upstream used for RxR.
