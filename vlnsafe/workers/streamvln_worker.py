#!/usr/bin/env python3
"""StreamVLN worker (runs under vlnsafe-streamvln, py3.9 / torch 2.1.2, no habitat).

What: a bf16 load of the benchmark-reproduction checkpoint (never `_real_world`; the repo
loads with torch_dtype=bfloat16 + flash_attention_2, not the fp16 originally
assumed). One `decide` request per *chunk*: `history_observations` carries the frames
executed since the last decision -- StreamVLN pushes one frame per executed step and only
generates when its chunk runs dry -- then the worker runs one `generate()`, parses the emitted
action text into a short primitive chunk (STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT),
and — from `output_scores=True` — reads the logits at the *first* action token restricted
to the four action symbols to publish prob_stop / prob_forward / prob_left / prob_right.
The SlowFast / KV-cache history is kept across requests within an episode and cleared on
`episode_reset`, exactly as StreamVLN's own eval loop does.

Who talks to it: vlnsafe.agents.remote (WorkerActionProvider, VLA variant — no occupancy
payload, VLA_OBSERVATION_KEYS only).

Ported: none — new in this harness. The IPC layer and the action-text parser below are
final; `_StreamVlnModel` is the seam that binds to `external/StreamVLN` once that repo is
a pinned submodule and `vlnsafe-streamvln` exists. Constants and the streaming contract
below are read from external/StreamVLN @ e48f6ff (streamvln/streamvln_agent.py,
streamvln/streamvln_eval.py, config/vln_r2r.yaml).
"""
import argparse
import base64
import json
import math
import os
import re
import sys

import numpy as np

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3

# StreamVLN's own vocabulary is OrderedDict({'STOP': 0, '↑': 1, '←': 2, '→': 3}) and it
# parses with a regex alternation over those literals (streamvln_agent.py::parse_actions
# @e48f6ff) -- note it scans, so the arrows need no separators. We keep that behaviour and
# add the word forms the prompt also licenses ("MOVE FORWARD (↑)", "TURN LEFT (←)").
_ACTION_TOKENS = (
    ("<stop>", STOP), ("stop", STOP),
    ("move forward", MOVE_FORWARD), ("forward", MOVE_FORWARD), ("↑", MOVE_FORWARD),
    ("turn left", TURN_LEFT), ("left", TURN_LEFT), ("←", TURN_LEFT),
    ("turn right", TURN_RIGHT), ("right", TURN_RIGHT), ("→", TURN_RIGHT),
)
# longest surface form first, so "move forward" wins over "forward" at the same position
_ACTION_RE = re.compile(
    "|".join(re.escape(s) for s, _ in sorted(_ACTION_TOKENS, key=lambda kv: -len(kv[0]))),
    re.IGNORECASE)
_ACTION_LOOKUP = dict((s, a) for s, a in _ACTION_TOKENS)
_PROB_KEYS = ("prob_stop", "prob_forward", "prob_left", "prob_right")


def parse_action_string(text):
    """StreamVLN's generated action text -> list of habitat primitive ids, truncated at
    the first STOP (inclusive; nothing after a STOP is executable). Underscores and
    hyphens are normalised so `move_forward` / `TURN_RIGHT` parse too."""
    raw = text.replace("->", "→").replace("<-", "←")
    raw = raw.replace("_", " ").replace("-", " ")
    actions = []
    for match in _ACTION_RE.finditer(raw):
        primitive = _ACTION_LOOKUP[match.group(0).lower()]
        actions.append(primitive)
        if primitive == STOP:
            break
    return actions


def _decode_observation(payload):
    obs = {}
    for key, value in payload.items():
        if key == "instruction":
            obs[key] = value
            continue
        shape = tuple(value["shape"])
        dtype = np.dtype(value["dtype"])
        raw = base64.b64decode(value["data_b64"])
        obs[key] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return obs


class _RedirectStdoutToDevnull(object):
    """fd 1 -> /dev/null for a block; model-load chatter must not desync the caller's
    readline() protocol (same reason as the ETPNav worker)."""

    def __enter__(self):
        sys.stdout.flush()
        self._saved = os.dup(1)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self._saved, 1)
        os.close(self._null)
        os.close(self._saved)


_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _camera_intrinsic(height, width, hfov_deg):
    """4x4 pinhole intrinsic for the harness's rig. StreamVLN's VLNEvaluator wants one;
    at present it only reaches tensors the model ignores (the agent fakes depth and pose
    -- "the model does not use voxel pooling. So we only use the rgb"), but it is built
    from the real rig rather than left at the repo's 384x384 demo value."""
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([
        [focal, 0.0, (width - 1) / 2.0, 0.0],
        [0.0, focal, (height - 1) / 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


class _StreamVlnModel(object):
    """Binds to external/StreamVLN @ e48f6ff.

    Wraps the repo's own `VLNEvaluator` rather than reimplementing its step(): that method
    carries the SlowFast/KV-cache bookkeeping (when to reset the cache, when to prepend a
    strided history slice) that the published numbers depend on, so it is reused verbatim.
    The one thing it does not do is expose logits, so `model.generate` is wrapped to force
    `output_scores=True` and stash the scores -- the original "generate() needs
    output_scores=True added", done without touching upstream code.
    """

    def __init__(self, code_root, checkpoint_path, precision, model_cfg=None,
                 hfov_deg=79.0):
        model_cfg = model_cfg or {}
        self.code_root = code_root
        self.hfov_deg = float(hfov_deg)
        os.chdir(code_root)
        for path in (code_root, os.path.join(code_root, "streamvln")):
            if path not in sys.path:
                sys.path.insert(0, path)

        import torch
        import transformers
        from model.stream_video_vln import StreamVLNForCausalLM
        from streamvln.streamvln_agent import VLNEvaluator

        self.torch = torch
        dtype = getattr(torch, _DTYPES[precision])
        max_len = int(model_cfg.get("model_max_length", 4096))

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint_path, model_max_length=max_len, padding_side="right")
        config = transformers.AutoConfig.from_pretrained(checkpoint_path)
        model = StreamVLNForCausalLM.from_pretrained(
            checkpoint_path, attn_implementation="flash_attention_2",
            torch_dtype=dtype, config=config, low_cpu_mem_usage=False)
        model.model.num_history = int(model_cfg.get("num_history", 8))
        model.reset(1)
        model.requires_grad_(False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()

        class _Args(object):
            num_frames = int(model_cfg.get("num_frames", 32))
            num_future_steps = int(model_cfg.get("num_future_steps", 4))
            num_history = int(model_cfg.get("num_history", 8))

        # intrinsic is filled in on the first frame, once the real image size is known
        self._sensor_cfg = {"rgb_height": 1.25, "camera_intrinsic": _camera_intrinsic(
            480, 640, self.hfov_deg)}
        self.evaluator = VLNEvaluator(
            self._sensor_cfg, model=model, tokenizer=tokenizer, args=_Args())
        self.model = model
        self.tokenizer = tokenizer
        self._instruction = ""
        self._scores = None
        self._prompt_len = 0
        self._action_ids = self._resolve_action_token_ids()
        self._wrap_generate()

    # --- P3 read-out -----------------------------------------------------------------
    def _resolve_action_token_ids(self):
        """Token id per action surface form, in habitat order [STOP, forward, left,
        right]. Returns None (-> the read-out is UNAVAILABLE, never a fabricated number)
        if any is multi-token or two share a first token, since then the logits at one
        position do not name the four actions."""
        ids = []
        for surface in ("STOP", "↑", "←", "→"):
            encoded = self.tokenizer.encode(surface, add_special_tokens=False)
            if not encoded:
                return None
            ids.append(encoded[0])
        return ids if len(set(ids)) == 4 else None

    def _wrap_generate(self):
        real_generate = self.model.generate

        def generate(*args, **kwargs):
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True
            out = real_generate(*args, **kwargs)
            self._scores = getattr(out, "scores", None)
            return out

        self.model.generate = generate

    def _first_action_logits(self, sequences):
        """Logits over the four action symbols at the FIRST generated token that is one of
        them -- the paper's "internal signals are read at the first action token"."""
        if self._action_ids is None or not self._scores:
            return None
        generated = sequences[0][-len(self._scores):].tolist()
        wanted = set(self._action_ids)
        for position, token_id in enumerate(generated):
            if token_id in wanted:
                logits = self._scores[position][0]
                return [float(logits[i].item()) for i in self._action_ids]
        return None

    # --- contract used by StreamVlnEngine ---------------------------------------------
    def reset_episode(self, instruction):
        self._instruction = instruction or ""
        self._scores = None
        self.evaluator.reset_memory()

    def _rgb(self, observation):
        frame = np.asarray(observation["rgb"])
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame[..., :3]

    def observe(self, observation):
        """A frame the agent executed but did not decide on: pushed into the history with
        run_model=False, exactly as StreamVLN's eval loop does every iteration."""
        self.evaluator.step(0, self._rgb(observation), self._instruction, run_model=False)

    def generate(self, observation, step_index):
        """-> (action_text, logits over [STOP, forward, left, right] | None)."""
        del step_index  # the evaluator keeps its own step counter
        self._scores = None
        _actions, _elapsed, text = self.evaluator.step(
            0, self._rgb(observation), self._instruction, run_model=True)
        logits = None
        if self.evaluator.output_ids is not None:
            logits = self._first_action_logits(self.evaluator.output_ids)
        return text or "", logits


def _softmax4(logits):
    a = np.asarray(logits, dtype=np.float64)
    a = a - a.max()
    e = np.exp(a)
    return (e / e.sum()).tolist()


class StreamVlnEngine(object):
    def __init__(self, code_root, checkpoint_path, precision, model_cfg=None, hfov_deg=79.0):
        self.model = _StreamVlnModel(
            code_root, checkpoint_path, precision, model_cfg, hfov_deg)

    def reset_episode(self, observation):
        instr = observation.get("instruction", {})
        self.model.reset_episode(instr.get("text", "") if isinstance(instr, dict) else instr)

    def observe(self, observation):
        """Push a frame into the SlowFast/KV history without generating -- the
        `run_model=False` half of StreamVLN's own step() (streamvln_agent.py:186)."""
        self.model.observe(observation)

    def decide(self, observation, step_index):
        action_text, first_logits = self.model.generate(observation, step_index)
        primitives = parse_action_string(action_text) or [STOP]
        risk_signals = {}
        if first_logits is not None:
            risk_signals = dict(zip(_PROB_KEYS, _softmax4(first_logits)))
        return {
            "primitive_actions": primitives,
            "stop": primitives[-1] == STOP,
            "risk_signals": risk_signals,
            "meta": {"action_text": action_text, "step": step_index},
        }


def handle_request(engine, request):
    if request.get("cmd") == "shutdown":
        return None
    if request.get("cmd") != "decide":
        return {"ok": False, "error": "unknown cmd {!r}".format(request.get("cmd"))}
    observation = _decode_observation(request["observation"])
    if request.get("episode_reset"):
        engine.reset_episode(observation)
    # frames executed since the last decision, oldest first: history only, no generate.
    for frame in request.get("history_observations") or []:
        engine.observe(_decode_observation(frame))
    decision = engine.decide(observation, int(request.get("decision_index", 0)))
    return {
        "ok": True,
        "primitive_actions": [int(a) for a in decision["primitive_actions"]],
        "stop": bool(decision["stop"]),
        "risk_signals": decision["risk_signals"],
        "predicted_clearance_by_step": None,
        "meta": decision["meta"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp-config", default=None)
    parser.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--model-config", default="{}",
                        help="JSON of the agent config's `model:` block "
                             "(num_frames / num_future_steps / num_history / model_max_length)")
    parser.add_argument("--hfov-deg", type=float, default=79.0)
    args = parser.parse_args()

    with _RedirectStdoutToDevnull():
        engine = StreamVlnEngine(args.code_root, args.checkpoint, args.precision,
                                 json.loads(args.model_config), args.hfov_deg)
    print("VLNSAFE_STREAMVLN_WORKER_READY", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": "bad json: {}".format(exc)}), flush=True)
            continue
        if request.get("cmd") == "shutdown":
            break
        try:
            with _RedirectStdoutToDevnull():
                response = handle_request(engine, request)
        except Exception as exc:  # noqa: BLE001
            import traceback
            response = {"ok": False, "error": "{}: {}".format(exc, traceback.format_exc())}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
