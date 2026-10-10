#!/usr/bin/env python3
"""NaVILA worker (runs under vlnsafe-navila, py3.10, no habitat).

What: the model emits a sentence ("The next action is move forward 75 cm").
`parse_navila_sentence` reproduces NaVILA's own eval-loop parse -- one action type per
generation, magnitude snapped to its 25/50/75 cm / 15/30/45 deg output grid -- and
decomposes it into however many primitives the simulator's step units need. The P3
categorical is the distribution at the first action-verb token over {stop, move,
turn-left, turn-right}; how the verb tokens are grouped is a modelling choice recorded in
docs/agent_notes/navila.md. NaVILA's P3/P4 belong to the released per-agent report, not
the paper tables, so the gate and the P1/P2 rows are what matter.

Who talks to it: vlnsafe.agents.remote (WorkerActionProvider, VLA variant).

Ported: none — new in this harness. The prompt, frame sampling and generate() call in
`_NavilaModel` reproduce evaluation/vlnce_baselines/navila_trainer.py::_eval_checkpoint
@ 76b98f2, plus `output_scores=True` for the P3 read-out.
"""
import argparse
import base64
import json
import os
import re
import sys

import numpy as np

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
_PROB_KEYS = ("prob_stop", "prob_forward", "prob_left", "prob_right")

# Verbatim from NaVILA's own eval loop (evaluation/vlnce_baselines/navila_trainer.py:222-282
# @76b98f2): ONE action type per generation, selected by the first pattern that matches in
# this order -- so a sentence mentioning "stop" anywhere is a stop.
_ACTION_PATTERNS = (
    (STOP, re.compile(r"\bstop\b", re.I)),
    (MOVE_FORWARD, re.compile(r"\bis move forward\b", re.I)),
    (TURN_LEFT, re.compile(r"\bis turn left\b", re.I)),
    (TURN_RIGHT, re.compile(r"\bis turn right\b", re.I)),
)
_DISTANCE_RE = re.compile(r"move forward (\d+) cm", re.I)
_LEFT_DEG_RE = re.compile(r"turn left (\d+) degree", re.I)
_RIGHT_DEG_RE = re.compile(r"turn right (\d+) degree", re.I)

# NaVILA's *output* vocabulary: it was trained to emit these magnitudes, and its eval loop
# snaps anything else to the nearest one. Independent of the simulator's step units.
_DISTANCE_GRID_CM = (25, 50, 75)
_TURN_GRID_DEG = (15, 30, 45)
_DEFAULT_DISTANCE_CM = 25
_DEFAULT_TURN_DEG = 15


def _snap(value, grid):
    return value if value % grid[0] == 0 else min(grid, key=lambda g: abs(g - value))


def parse_navila_sentence(text, forward_unit_m=0.25, turn_unit_deg=15.0):
    """NaVILA sentence -> list of habitat primitive ids.

    Faithful to NaVILA's eval loop: exactly one action *type* per generation, its
    magnitude snapped to NaVILA's own output grid (25/50/75 cm, 15/30/45 deg), then
    decomposed into however many primitives the *simulator's* step units need. On R2R
    (0.25 m / 15 deg) that reproduces upstream exactly -- 'move forward 75 cm' -> 3x
    MOVE_FORWARD, 'turn left 30 degrees' -> 2x TURN_LEFT. On RxR the turn unit is 30 deg,
    so the same sentence is 1x TURN_LEFT; decomposing by a hard-coded 15 there would
    double every turn.

    A sentence matching nothing is treated as STOP (upstream would step `None`)."""
    action = STOP
    for candidate, pattern in _ACTION_PATTERNS:
        if pattern.search(text):
            action = candidate
            break

    if action == MOVE_FORWARD:
        m = _DISTANCE_RE.search(text)
        cm = _snap(int(m.group(1)), _DISTANCE_GRID_CM) if m else _DEFAULT_DISTANCE_CM
        n = max(1, int(round((cm / 100.0) / forward_unit_m)))
        return [MOVE_FORWARD] * n
    if action in (TURN_LEFT, TURN_RIGHT):
        regex = _LEFT_DEG_RE if action == TURN_LEFT else _RIGHT_DEG_RE
        m = regex.search(text)
        deg = _snap(int(m.group(1)), _TURN_GRID_DEG) if m else _DEFAULT_TURN_DEG
        n = max(1, int(round(deg / float(turn_unit_deg))))
        return [action] * n
    return [STOP]


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


def _sample_and_pad_images(images, num_frames, image_cls, width=512, height=512):
    """Verbatim reimplementation of NaVILA's `sample_and_pad_images`
    (evaluation/vlnce_baselines/navila_trainer.py:38-53 @76b98f2): left-pad with black
    frames until there are `num_frames`, then take `num_frames - 1` evenly spaced frames
    plus the latest one.

    Inlined rather than imported because `vlnce_baselines/__init__.py` imports every
    trainer in that package -- which needs habitat and lmdb. This worker's whole point is
    that it runs in a habitat-free env, so it must not import that tree.
    """
    frames = list(images)
    while len(frames) < num_frames:
        frames.insert(0, image_cls.new("RGB", (width, height), color=(0, 0, 0)))
    latest = frames[-1]
    indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    return [frames[i] for i in indices] + [latest]


def _single_token_id(tokenizer, surfaces):
    """First surface form in `surfaces` that encodes to exactly one token -> its id.
    Llama-3's BPE usually wants the leading-space variant mid-sentence."""
    for surface in surfaces:
        encoded = tokenizer.encode(surface, add_special_tokens=False)
        if len(encoded) == 1:
            return encoded[0]
    return None


class _NavilaModel(object):
    """Binds to external/NaVILA @ 76b98f2.

    Reproduces the prompt, frame sampling and generate() call of
    evaluation/vlnce_baselines/navila_trainer.py::_eval_checkpoint, which is the loop the
    published numbers come from. `generate` additionally asks for scores so the P3
    categorical can be read (see _verb_logits for the grouping choice).
    """

    def __init__(self, code_root, checkpoint_path, precision, model_cfg=None):
        model_cfg = model_cfg or {}
        self.code_root = code_root
        os.chdir(code_root)
        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        import torch
        from PIL import Image
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            KeywordsStoppingCriteria, process_images, tokenizer_image_token)
        from llava.model.builder import load_pretrained_model

        self.torch = torch
        self.Image = Image
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token

        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        tokenizer, model, image_processor, _ctx = load_pretrained_model(
            checkpoint_path, model_name)
        self.tokenizer = tokenizer
        self.model = model.cuda().eval()
        self.image_processor = image_processor
        self.conv_mode = model_cfg.get("conv_type", "llama_3")
        self.num_video_frames = int(
            model_cfg.get("num_frames", getattr(model.config, "num_video_frames", 8)))
        self.dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        # P3 read-out vocabulary (see _verb_logits)
        self._verb_ids = [
            _single_token_id(tokenizer, (" stop", "stop")),
            _single_token_id(tokenizer, (" move", "move")),
            _single_token_id(tokenizer, (" turn", "turn")),
        ]
        self._dir_ids = [
            _single_token_id(tokenizer, (" left", "left")),
            _single_token_id(tokenizer, (" right", "right")),
        ]
        self._readout_ok = (all(i is not None for i in self._verb_ids)
                            and all(i is not None for i in self._dir_ids)
                            and len(set(self._verb_ids)) == 3)
        self._instruction = ""
        self._past_rgbs = []

    # --- P3 read-out -----------------------------------------------------------------
    def _verb_logits(self, scores, generated):
        """Logits over [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT].

        NaVILA emits a sentence, so there is no single token whose distribution is the
        four-way action categorical -- the grouping is a modelling choice. Ours: take the distribution at the first *verb* token over {stop, move, turn}
        (its own action selection is exactly that three-way choice), then split the `turn`
        mass with the {left, right} distribution at the following position. That is the
        chain rule over the two tokens that actually determine the action, and it yields a
        proper 4-way distribution. Recorded in docs/agent_notes/navila.md.

        Returns None -- so the columns stay UNAVAILABLE rather than fabricated -- if the
        tokenizer does not give single tokens for those words, or no verb was generated.
        """
        if not self._readout_ok or not scores:
            return None
        verbs = set(self._verb_ids)
        for position, token_id in enumerate(generated):
            if token_id not in verbs:
                continue
            verb_logits = scores[position][0]
            stop_l, move_l, turn_l = (float(verb_logits[i].item()) for i in self._verb_ids)
            if position + 1 >= len(scores):
                return None
            dir_logits = scores[position + 1][0]
            left_l, right_l = (float(dir_logits[i].item()) for i in self._dir_ids)
            # p(turn) is split by p(left|turn), p(right|turn); work in log space so the
            # four numbers come back as logits the caller can softmax like any other.
            import math as _math
            shift = max(left_l, right_l)
            total = _math.log(_math.exp(left_l - shift) + _math.exp(right_l - shift)) + shift
            return [stop_l, move_l, turn_l + (left_l - total), turn_l + (right_l - total)]
        return None

    # --- contract used by NavilaEngine -------------------------------------------------
    def reset_episode(self, instruction):
        self._instruction = instruction or ""
        self._past_rgbs = []

    def _pil(self, observation):
        frame = np.asarray(observation["rgb"])
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return self.Image.fromarray(frame[..., :3]).convert("RGB")

    def observe(self, observation):
        """A frame the agent executed from a queued chunk: history only, no generate --
        upstream appends to past_rgbs on every executed step."""
        self._past_rgbs.append(self._pil(observation))

    def generate(self, observation, step_index):
        """-> (sentence, logits over [STOP, forward, left, right] | None)."""
        del step_index
        torch = self.torch
        current = self._pil(observation)
        frames = _sample_and_pad_images(
            self._past_rgbs + [current], self.num_video_frames, self.Image)

        interleaved = "<image>\n" * (len(frames) - 1)
        question = (
            "Imagine you are a robot programmed for navigation tasks. You have been given "
            "a video of historical observations {}, and current observation <image>\n. "
            "Your assigned task is: \"{}\" Analyze this series of images to decide your "
            "next action, which could be turning left or right by a specific degree, "
            "moving forward a certain distance, or stop if the task is completed."
        ).format(interleaved, self._instruction)

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = self.process_images(
            frames, self.image_processor, self.model.config).to(
                self.model.device, dtype=torch.float16)
        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX,
            return_tensors="pt").unsqueeze(0).cuda()
        stop_str = (conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2)
        stopping = self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            out = self.model.generate(
                input_ids, images=images_tensor.half().cuda(), do_sample=False,
                temperature=0.0, max_new_tokens=32, use_cache=True,
                stopping_criteria=[stopping], pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True, return_dict_in_generate=True)
        sequences = out.sequences if hasattr(out, "sequences") else out
        scores = list(getattr(out, "scores", None) or [])

        sentence = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        if sentence.endswith(stop_str):
            sentence = sentence[: -len(stop_str)].strip()

        generated = sequences[0][-len(scores):].tolist() if scores else []
        logits = self._verb_logits(scores, generated)
        # the decision frame enters the history too, as it does upstream
        self._past_rgbs.append(current)
        return sentence, logits


def _softmax4(logits):
    a = np.asarray(logits, dtype=np.float64)
    a = a - a.max()
    e = np.exp(a)
    return (e / e.sum()).tolist()


class NavilaEngine(object):
    def __init__(self, code_root, checkpoint_path, precision, forward_unit_m, turn_unit_deg,
                 model_cfg=None):
        self.model = _NavilaModel(code_root, checkpoint_path, precision, model_cfg)
        self.forward_unit_m = forward_unit_m
        self.turn_unit_deg = turn_unit_deg

    def reset_episode(self, observation):
        instr = observation.get("instruction", {})
        self.model.reset_episode(instr.get("text", "") if isinstance(instr, dict) else instr)

    def observe(self, observation):
        self.model.observe(observation)

    def decide(self, observation, step_index):
        sentence, first_logits = self.model.generate(observation, step_index)
        primitives = parse_navila_sentence(
            sentence, self.forward_unit_m, self.turn_unit_deg) or [STOP]
        risk_signals = {}
        if first_logits is not None:
            risk_signals = dict(zip(_PROB_KEYS, _softmax4(first_logits)))
        return {
            "primitive_actions": primitives,
            "stop": primitives[-1] == STOP,
            "risk_signals": risk_signals,
            "meta": {"sentence": sentence, "step": step_index},
        }


def handle_request(engine, request):
    if request.get("cmd") == "shutdown":
        return None
    if request.get("cmd") != "decide":
        return {"ok": False, "error": "unknown cmd {!r}".format(request.get("cmd"))}
    observation = _decode_observation(request["observation"])
    if request.get("episode_reset"):
        engine.reset_episode(observation)
    for frame in request.get("history_observations") or []:
        engine.observe(_decode_observation(frame))
    engine.forward_unit_m = float(request.get("forward_unit_m", engine.forward_unit_m))
    engine.turn_unit_deg = float(request.get("turn_unit_deg", engine.turn_unit_deg))
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
                        help="JSON of the agent config's `model:` block (num_frames / conv_type)")
    parser.add_argument("--hfov-deg", type=float, default=90.0)  # accepted, unused here
    args = parser.parse_args()

    with _RedirectStdoutToDevnull():
        engine = NavilaEngine(args.code_root, args.checkpoint, args.precision, 0.25, 15.0,
                              json.loads(args.model_config))
    print("VLNSAFE_NAVILA_WORKER_READY", file=sys.stderr, flush=True)

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
