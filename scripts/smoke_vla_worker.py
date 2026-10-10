#!/usr/bin/env python3
"""End-to-end smoke test of a navigation-VLA worker, through the real protocol.

    python scripts/smoke_vla_worker.py streamvln [--gpu 7]
    python scripts/smoke_vla_worker.py navila    [--gpu 3]
    python scripts/smoke_vla_worker.py omninav   [--gpu 3]

Spawns the worker exactly as `vlnsafe.agents.registry` would -- same argv, same
WorkerHandle, same JSON-lines exchange -- and sends two `decide` requests with synthetic
frames, the second carrying `history_observations` so the streaming path is exercised too.
Everything except habitat is covered: config -> argv -> subprocess -> model load ->
generate -> action parse -> the prob_* read-out.

Run this before an arm. It is the difference between "the env imports" and "the worker
answers", and it costs ~2 minutes and one GPU instead of a failed 5-hour run.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vlnsafe import config as cfg                      # noqa: E402
from vlnsafe.agents import registry                    # noqa: E402
from vlnsafe.agents.remote import (                    # noqa: E402
    OMNINAV_OBSERVATION_KEYS, VLA_OBSERVATION_KEYS, WorkerHandle, encode_observation)

INSTRUCTION = "Walk forward past the sofa and turn left at the doorway."


def _frame(agent_cfg, seed):
    rig = (agent_cfg.get("rig") or {}).get("rgb") or {}
    height = int(rig.get("height", 480))
    width = int(rig.get("width", 640))
    rng = np.random.RandomState(seed)
    frame = {"instruction": {"text": INSTRUCTION, "tokens": [], "trajectory_id": ""}}
    views = OMNINAV_OBSERVATION_KEYS if agent_cfg["family"] == "omninav_worker" else ["rgb"]
    for key in views:
        frame[key] = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=["streamvln", "navila", "omninav"])
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for the worker")
    ap.add_argument("--benchmark", default="r2r", choices=["r2r", "rxr"])
    args = ap.parse_args()

    agent = cfg.load_agent(args.agent)
    repo = cfg.repo_root()
    code_root = os.path.normpath(os.path.join(repo, agent["code_root"]))
    checkpoint = os.path.expandvars(agent["checkpoint"][args.benchmark])
    exp_config = (agent.get("exp_config") or {}).get(args.benchmark)
    argv = registry.vla_worker_argv(
        agent, os.path.expandvars(agent["worker_python"]), code_root, exp_config, checkpoint)
    omninav = agent["family"] == "omninav_worker"
    if omninav:
        argv += ["--seed", "20260704"]
    print("argv:", " ".join(argv[:3]), "...")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HOME", os.path.expandvars("$VLNSAFE_MODELS/hf"))

    handle = WorkerHandle(argv, cwd=code_root)
    try:
        frames = [_frame(agent, i) for i in range(4)]
        keys = OMNINAV_OBSERVATION_KEYS if omninav else VLA_OBSERVATION_KEYS

        print("-- decide 1 (episode_reset, no history)")
        first = handle.request({
            "episode_reset": True, "decision_index": 0,
            "agent_position": [0.0, 0.0, 0.0], "agent_rotation": [0.0, 0.0, 0.0, 1.0],
            "turn_unit_deg": 15.0, "forward_unit_m": 0.25, "agent_radius_m": 0.1,
            "observation": encode_observation(frames[0], keys),
        })
        _report(first)

        print("-- decide 2 (two executed frames as history)")
        second = handle.request({
            "episode_reset": False, "decision_index": 1,
            "agent_position": [0.0, 0.0, 0.0], "agent_rotation": [0.0, 0.0, 0.0, 1.0],
            "turn_unit_deg": 15.0, "forward_unit_m": 0.25, "agent_radius_m": 0.1,
            "observation": encode_observation(frames[3], keys),
            "history_observations": ([] if omninav else
                                     [encode_observation(f, keys) for f in frames[1:3]]),
        })
        _report(second)
    finally:
        handle.close()

    print("SMOKE OK -- {} answers through the worker protocol".format(args.agent))
    return 0


def _report(response):
    actions = response["primitive_actions"]
    signals = response.get("risk_signals") or {}
    assert isinstance(actions, list) and actions, "worker returned no primitives"
    assert all(a in (0, 1, 2, 3) for a in actions), "bad primitive id: %r" % actions
    print("   primitives:", actions, "| stop:", response.get("stop"))
    if "stop_prob" in signals:   # OmniNav: arrive values, no action categorical
        print("   arrive     :", signals.get("arrive_by_waypoint"), "| meta:",
              {k: v for k, v in (response.get("meta") or {}).items() if k != "decision"})
        return
    if signals:
        total = sum(signals.values())
        print("   prob_*     :", {k: round(v, 4) for k, v in sorted(signals.items())},
              "(sums to %.4f)" % total)
        assert abs(total - 1.0) < 1e-3, "prob_* must be a distribution, got %r" % total
    else:
        print("   prob_*     : UNAVAILABLE (read-out not resolvable for this tokenizer)")
    meta = response.get("meta") or {}
    text = meta.get("action_text") or meta.get("sentence") or ""
    print("   raw output :", repr(text)[:120])


if __name__ == "__main__":
    sys.exit(main())
