#!/usr/bin/env python3
"""Verify the VLA worker envs match their upstream pins and can actually load.

    python scripts/check_vla_envs.py

Each env is probed with its OWN interpreter (they are py3.9 / py3.10 with different torch
builds, so nothing here can be imported in-process). Checks the versions that matter --
the ones a mismatch would let you discover only at model-load time: torch, numpy (its C
ABI has to match torch's), transformers, and the prebuilt flash-attn wheel both repos'
`attn_implementation="flash_attention_2"` needs. Exits non-zero on any mismatch.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = os.environ.get("VLNSAFE_ENVS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "envs"))

# env -> {package: expected version}, from each repo's own pins.
EXPECTED = {
    "vlnsafe-streamvln": {          # external/StreamVLN/requirements.txt
        "python": "3.9", "torch": "2.1.2", "torchvision": "0.16.2",
        "transformers": "4.45.1", "numpy": "1.26.1", "flash_attn": "2.5.8",
    },
    "vlnsafe-navila": {             # external/NaVILA/pyproject.toml
        "python": "3.10", "torch": "2.3.0", "torchvision": "0.18.0",
        "transformers": "4.37.2", "numpy": "1.26.0", "flash_attn": "2.5.8",
    },
    "vlnsafe-omninav": {            # external/OmniNav/requirements-docker.txt (torch: see
        "python": "3.9", "torch": "2.6.0", "torchvision": "0.21.0",   # build_vla_envs.sh)
        "transformers": "4.51.3", "numpy": "1.23.5", "flash_attn": "2.7.4.post1",
    },
}

PROBE = r"""
import json, sys
out = {"python": ".".join(str(v) for v in sys.version_info[:2])}
for mod in ("torch", "torchvision", "transformers", "numpy", "flash_attn"):
    try:
        out[mod] = __import__(mod).__version__
    except Exception as exc:
        out[mod] = "MISSING:%s" % type(exc).__name__
try:
    import numpy, torch
    torch.from_numpy(numpy.zeros((2, 2), dtype=numpy.float32))
    out["numpy_torch_bridge"] = "ok"
except Exception as exc:
    out["numpy_torch_bridge"] = "BROKEN:%s" % exc
try:
    import vlnsafe  # noqa: F401
    out["vlnsafe"] = "ok"
except Exception as exc:
    out["vlnsafe"] = "MISSING:%s" % type(exc).__name__
# The real test: can the repo's own model code be imported? Version pins that look right
# still fail here (NaVILA's llava.model.builder reaches deepspeed even for inference).
import os, sys as _s
root, probe = _s.argv[1], _s.argv[2]
try:
    os.chdir(root)
    for extra in (root, os.path.join(root, os.path.basename(root).lower())):
        if extra not in _s.path:
            _s.path.insert(0, extra)
    exec(probe)
    out["model_code"] = "ok"
except Exception as exc:
    out["model_code"] = "BROKEN:%s: %s" % (type(exc).__name__, str(exc)[:80])
print(json.dumps(out))
"""

# repo root + the import that proves the model code is reachable in that env
MODEL_PROBE = {
    "vlnsafe-streamvln": (
        "external/StreamVLN",
        "import sys, os;"
        "sys.path.insert(0, os.path.join(os.getcwd(), 'streamvln'));"
        "__import__('model.stream_video_vln', fromlist=['StreamVLNForCausalLM'])"),
    "vlnsafe-navila": (
        "external/NaVILA",
        "__import__('llava.model.builder', fromlist=['load_pretrained_model'])"),
    # the vendored transformers must be the one imported: stock 4.51.3 has no action heads
    "vlnsafe-omninav": (
        "external/OmniNav/infer_r2r_rxr",
        "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m;"
        "assert hasattr(m, 'FlowmatchingActionHead'), 'stock transformers, not OmniNav';"
        "__import__('agent.waypoint_agent', fromlist=['Waypoint_Agent'])"),
}


def main():
    failures = 0
    for env, expected in EXPECTED.items():
        python = os.path.join(ENVS, env, "bin", "python")
        print("== {}".format(env))
        if not os.path.exists(python):
            print("   NOT BUILT ({}) -- run scripts/build_vla_envs.sh".format(python))
            failures += 1
            continue
        rel, probe = MODEL_PROBE[env]
        try:
            raw = subprocess.check_output(
                [python, "-c", PROBE, os.path.join(REPO, rel), probe],
                stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            print("   probe failed: {}".format(exc.stderr.decode()[-300:]))
            failures += 1
            continue
        got = json.loads(raw.decode().strip().splitlines()[-1])
        for key, want in sorted(expected.items()):
            have = got.get(key, "?")
            # torch reports 2.1.2+cu121; compare on the release prefix only
            ok = have.split("+")[0] == want
            print("   {:<14} {:<18} {}".format(
                key, have, "ok" if ok else "EXPECTED " + want))
            failures += 0 if ok else 1
        for key in ("numpy_torch_bridge", "vlnsafe", "model_code"):
            have = got.get(key, "?")
            print("   {:<14} {}".format(key, have))
            failures += 0 if have == "ok" else 1
    print("\n{} problem(s)".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
