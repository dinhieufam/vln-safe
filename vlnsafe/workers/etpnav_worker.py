#!/usr/bin/env python3
"""ETPNav / ETP-R1 / BEVBert worker (same lineage; the agent config's code_root picks the
checkout). Runs under vlnsafe-etpnav as a subprocess, hosts the frozen topological
decision engine, and returns each decision already decomposed into primitive actions.

What: one iteration of ss_trainer_ETP.rollout()'s per-step body (waypoint predict, pano
encode, graph update, nav head) picks "stop at node X" / "go to ghost Y";
vlnsafe.geometry.decompose_hops_to_primitives walks it out. Reads
nav_outs["global_logits"], gmap.node_stop_scores, gmap.ghost_aug_pos, MODEL.consume_ghost.

Who talks to it: vlnsafe.agents.remote, one JSON object per line on stdin/stdout.

Imports (deferred, in ETPNavDecisionEngine.__init__): torch, vlnce_baselines (ETPNav's),
habitat_extensions. From vlnsafe/ it imports only vlnsafe.geometry.

Ported near-verbatim from the pre-release etpnav_policy_worker.py; the
geometry helpers moved to vlnsafe/geometry.py (byte-locked by tests/test_geometry.py).
The stdout-redirect and stub-env shims stay here: habitat-sim's Magnum renderer and
ETPNav's setup print straight to fd 1, which is this worker's IPC channel.
"""
import argparse
import base64
import json
import math
import os
import sys

import numpy as np

from vlnsafe.geometry import (
    bin_clearance_toward_point,
    decompose_hops_to_primitives,
    predicted_clearance_by_primitive_step,
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


class _RedirectStdoutToDevnull(object):
    """fd 1 -> /dev/null for the duration of a block. Habitat-sim's GL context creation
    and some of ETPNav's setup print to the real stdout file descriptor, not just
    sys.stdout; any of it would desync the caller's readline()-based protocol."""

    def __enter__(self):
        sys.stdout.flush()
        self._saved_fd = os.dup(1)
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull_fd, 1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        os.dup2(self._saved_fd, 1)
        os.close(self._devnull_fd)
        os.close(self._saved_fd)


ETPNAV_OBSERVATION_KEYS = ["rgb", "depth"] + [
    "{}_{}".format(sensor, deg)
    for sensor in ("rgb", "depth")
    for deg in range(30, 360, 30)
]


class _StubEnvs(object):
    """Only .num_envs is read by the RLTrainer helpers this worker reuses unmodified
    (_vp_feature_variable, _nav_gmap_variable); the caller's real Habitat env is
    authoritative."""

    def __init__(self, num_envs):
        self.num_envs = num_envs


def decode_observation(payload):
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


class ETPNavDecisionEngine(object):
    def __init__(self, exp_config, opts, checkpoint_path, pretrained_path=None,
                 trainer_module="vlnce_baselines.ss_trainer_ETP", task_type=None):
        # `trainer_module` picks the SS trainer class: ETPNav's ss_trainer_ETP or
        # ETP-R1's ss_trainer_ETP_R1 (registered SS-ETP-R1). `task_type` (1=r2r, 2=rxr) is
        # ETP-R1's one extra kwarg on extract_instruction_tokens / _nav_gmap_variable;
        # None means the ETPNav-classic signatures (docs/agent_notes/etpr1.md Phase D1).
        # Deferred so merely importing this file (e.g. for --help) needs no torch / the
        # vlnsafe-etpnav env.
        import importlib
        import torch
        from vlnce_baselines.config.default import get_config
        RLTrainer = importlib.import_module(trainer_module).RLTrainer
        from habitat_baselines.common.obs_transformers import (
            apply_obs_transforms_batch,
            get_active_obs_transforms,
        )
        from habitat_baselines.utils.common import batch_obs
        from vlnce_baselines.common.utils import extract_instruction_tokens
        from vlnce_baselines.models.graph_utils import GraphMap
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        import habitat_extensions  # noqa: F401  (registers Sim-v1, HIGHTOLOW*, sensors)
        import vlnce_baselines  # noqa: F401  (registers SS-ETP trainer, ETP policy)

        self.torch = torch
        self.apply_obs_transforms_batch = apply_obs_transforms_batch
        self.batch_obs = batch_obs
        self.extract_instruction_tokens = extract_instruction_tokens
        self.GraphMap = GraphMap
        self.HabitatSimActions = HabitatSimActions

        full_opts = list(opts) + [
            "IL.ckpt_to_load", checkpoint_path,
            "IL.load_from_ckpt", "True",
            "IL.is_requeue", "False",
            "GPU_NUMBERS", "1",
            "NUM_ENVIRONMENTS", "1",
            # This worker never reads episode data from its throwaway env (closed right
            # after building the policy); any split present on disk works. val_unseen is
            # always there.
            "EVAL.SPLIT", "val_unseen",
            "TASK_CONFIG.DATASET.SPLIT", "val_unseen",
        ]
        if pretrained_path:
            full_opts += ["MODEL.pretrained_path", pretrained_path]
        config = get_config(exp_config, full_opts)
        config.defrost()
        config.local_rank = 0
        config.freeze()

        self.trainer = RLTrainer(config)
        self.trainer._set_config()
        observation_space, action_space = self.trainer._init_envs()
        self.trainer.envs.close()
        self.trainer.envs = _StubEnvs(1)
        self.trainer._initialize_policy(
            self.trainer.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.trainer.policy.eval()
        self.trainer.waypoint_predictor.eval()
        self.trainer.obs_transforms = get_active_obs_transforms(self.trainer.config)

        self.device = self.trainer.device
        self.instr_max_len = self.trainer.config.IL.max_text_len
        self.task_type = int(task_type) if task_type is not None else None
        if self.task_type is not None:
            # ETP-R1 reads XLM-R tokens; its rollout() pins pad_id=1 for both benchmarks
            # (ss_trainer_ETP_R1.py:875), unlike ETPNav's task-dependent pad id.
            self.instr_pad_id = 1
        else:
            self.instr_pad_id = 1 if self.trainer.config.MODEL.task_type == "rxr" else 0
        self.instruction_uuid = self.trainer.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        self.max_len = self.trainer.max_len

        self._reset_episode_state()

    def _reset_episode_state(self):
        self._txt_masks = None
        self._txt_embeds = None
        self.gmap = None
        self.prev_vp = None
        self.stepk = 0
        # decisions the harness marked `budget_exempt` (a re-decision forced by
        # vlnsafe.guard's manoeuvre): each moves the max_len force-stop
        # one step later, so the policy keeps max_len decisions of its own. Always 0 in a
        # run without the guard or without --exempt-guard-redecisions.
        self.budget_extension = 0

    def _build_batch(self, observation):
        extra = {} if self.task_type is None else {"task_type": self.task_type}
        observations = self.extract_instruction_tokens(
            [observation], self.instruction_uuid,
            max_length=self.instr_max_len, pad_id=self.instr_pad_id, **extra
        )
        batch = self.batch_obs(observations, self.device)
        return self.apply_obs_transforms_batch(batch, self.trainer.obs_transforms)

    def reset_episode(self, observation):
        self._reset_episode_state()
        batch = self._build_batch(observation)
        all_txt_ids = batch["instruction"]
        self._txt_masks = (all_txt_ids != self.instr_pad_id)
        # ETP-R1's embeddings forward takes a required task_type encoding alongside the
        # token ids (ETP_R1_vilmodel_cmt.py forward_txt); extract_instruction_tokens
        # already wrote it into the batch when self.task_type is set (ss_trainer_ETP_R1.py
        # rollout() reads the same key). ETPNav/BEVBert never set task_type, so the key is
        # absent and the kwarg is omitted, matching their embeddings signature.
        net_kwargs = {}
        if self.task_type is not None:
            net_kwargs["txt_task_encoding"] = batch["txt_task_encoding"]
        with self.torch.no_grad():
            self._txt_embeds = self.trainer.policy.net(
                mode="language", txt_ids=all_txt_ids, txt_masks=self._txt_masks, **net_kwargs
            )
        has_real_pos = False
        self.gmap = self.GraphMap(
            has_real_pos,
            self.trainer.config.IL.loc_noise,
            self.trainer.config.MODEL.merge_ghost,
            0,  # ghost_aug: 0 at eval, matches rollout()'s `mode != 'train'` branch
        )
        self.prev_vp = None

    def decide(self, observation, agent_position, agent_rotation):
        """One high-level ETPNav decision. Near-verbatim port of one iteration of
        ss_trainer_ETP.py::RLTrainer.rollout()'s per-stepk body (mode='eval'/'argmax'),
        specialised to a single episode with an externally supplied observation/pose."""
        F = self.torch.nn.functional
        batch = self._build_batch(observation)
        cur_pos = [np.array(agent_position, dtype=np.float64)]
        cur_ori = [np.array(agent_rotation, dtype=np.float64)]

        wp_outputs = self.trainer.policy.net(
            mode="waypoint",
            waypoint_predictor=self.trainer.waypoint_predictor,
            observations=batch,
            in_train=False,
        )
        vp_inputs = self.trainer._vp_feature_variable(wp_outputs)
        vp_inputs.update({"mode": "panorama"})
        pano_embeds, pano_masks = self.trainer.policy.net(**vp_inputs)
        avg_pano_embeds = self.torch.sum(
            pano_embeds * pano_masks.unsqueeze(2), 1
        ) / self.torch.sum(pano_masks, 1, keepdim=True)

        self.trainer.gmaps = [self.gmap]  # _nav_gmap_variable reads self.trainer.gmaps
        cur_vp, cand_vp, cand_pos = self.gmap.identify_node(
            cur_pos[0], cur_ori[0], wp_outputs["cand_angles"][0], wp_outputs["cand_distances"][0]
        )
        cur_embeds = avg_pano_embeds[0]
        cand_embeds = pano_embeds[0][vp_inputs["nav_types"][0] == 1]
        self.gmap.update_graph(
            self.prev_vp, self.stepk + 1,
            cur_vp, cur_pos[0], cur_embeds,
            cand_vp, cand_pos, cand_embeds,
            None,
        )

        if self.task_type is None:
            nav_inputs = self.trainer._nav_gmap_variable([cur_vp], cur_pos, cur_ori)
        else:  # ETP-R1's _nav_gmap_variable takes task_type as a 4th positional
            nav_inputs = self.trainer._nav_gmap_variable(
                [cur_vp], cur_pos, cur_ori, self.task_type)
        nav_inputs.update({
            "mode": "navigation",
            "txt_embeds": self._txt_embeds,
            "txt_masks": self._txt_masks,
        })
        no_vp_left = nav_inputs.pop("no_vp_left")
        nav_outs = self.trainer.policy.net(**nav_inputs)
        nav_logits = nav_outs["global_logits"]
        nav_probs = F.softmax(nav_logits, 1)
        self.gmap.node_stop_scores[cur_vp] = nav_probs[0, 0].data.item()

        a_t = int(nav_logits.argmax(dim=-1)[0].item())
        force_stop = (
            self.stepk == self.max_len - 1 + self.budget_extension) or bool(no_vp_left[0])

        valid_idx = self.torch.isfinite(nav_logits[0]).nonzero(as_tuple=True)[0].tolist()
        gmap_vp_ids_0 = nav_inputs["gmap_vp_ids"][0]
        candidates = []
        for i in valid_idx:
            vp = gmap_vp_ids_0[i]
            if i == 0:
                pos = None
            else:
                pos = self.gmap.node_pos.get(vp)
                if pos is None:
                    pos = self.gmap.ghost_aug_pos.get(vp)
            candidates.append({
                "prob": float(nav_probs[0, i].item()),
                "is_chosen": bool(i == a_t),
                "position": np.asarray(pos).tolist() if pos is not None else None,
            })
        probs_sorted = sorted((c["prob"] for c in candidates), reverse=True)
        top2_margin_prob = (probs_sorted[0] - probs_sorted[1]) if len(probs_sorted) >= 2 else 1.0
        candidate_entropy = -sum(p * math.log(p) for p in probs_sorted if p > 1e-12)
        candidate_entropy_normalized = (
            candidate_entropy / math.log(len(probs_sorted)) if len(probs_sorted) > 1 else 0.0
        )
        risk_signals = {
            "stop_prob": float(nav_probs[0, 0].item()),
            "candidate_entropy": candidate_entropy,
            "candidate_entropy_normalized": candidate_entropy_normalized,
            "top2_margin_prob": top2_margin_prob,
            "num_valid_candidates": len(candidates),
        }

        if a_t == 0 or force_stop:
            vp_stop_scores = list(self.gmap.node_stop_scores.items())
            stop_scores = [s[1] for s in vp_stop_scores]
            stop_vp = vp_stop_scores[int(np.argmax(stop_scores))][0]
            hops = [
                self.gmap.node_pos[vp]
                for vp in self.gmap.shortest_path[cur_vp][stop_vp]
            ][1:]
            self.prev_vp = None
            self.stepk += 1
            return {
                "act": 0, "cur_vp": cur_vp, "target_vp": stop_vp,
                "hops": [np.asarray(h).tolist() for h in hops],
                "final_stop": True,
                "risk_signals": risk_signals,
                "candidates": candidates,
            }

        ghost_vp = nav_inputs["gmap_vp_ids"][0][a_t]
        ghost_pos = self.gmap.ghost_aug_pos[ghost_vp]
        _, front_vp = self.gmap.front_to_ghost_dist(ghost_vp)
        hops = [
            self.gmap.node_pos[vp]
            for vp in self.gmap.shortest_path[cur_vp][front_vp]
        ][1:] + [ghost_pos]
        if self.trainer.config.MODEL.consume_ghost:
            self.gmap.delete_ghost(ghost_vp)
        self.prev_vp = front_vp
        self.stepk += 1
        return {
            "act": 4, "cur_vp": cur_vp, "target_vp": ghost_vp,
            "hops": [np.asarray(h).tolist() for h in hops],
            "final_stop": False,
            "risk_signals": risk_signals,
            "candidates": candidates,
        }


def handle_request(engine, request):
    cmd = request.get("cmd")
    if cmd == "shutdown":
        return None
    if cmd != "decide":
        return {"ok": False, "error": "unknown cmd {!r}".format(cmd)}

    observation = decode_observation(request["observation"])
    agent_position = request["agent_position"]
    agent_rotation = request["agent_rotation"]
    turn_unit_deg = float(request.get("turn_unit_deg", 15.0))
    forward_unit_m = float(request.get("forward_unit_m", 0.25))

    if request.get("episode_reset"):
        engine.reset_episode(observation)
    if request.get("budget_exempt"):
        engine.budget_extension += 1

    decision = engine.decide(observation, agent_position, agent_rotation)
    primitive_actions = decompose_hops_to_primitives(
        decision["hops"], agent_position, agent_rotation,
        turn_unit_deg, forward_unit_m, engine.HabitatSimActions,
    )
    if decision["final_stop"]:
        primitive_actions = primitive_actions + [int(engine.HabitatSimActions.STOP)]

    occupancy_points_m = request.get("occupancy_points_m")
    agent_radius_m = float(request.get("agent_radius_m", 0.1))
    risk_signals = dict(decision.get("risk_signals") or {})
    predicted_clearance_by_step = headroom_alt_exists = min_eps_for_safe_alt = None
    if occupancy_points_m:
        predicted_clearance_by_step = predicted_clearance_by_primitive_step(
            primitive_actions, turn_unit_deg, forward_unit_m, occupancy_points_m,
            engine.HabitatSimActions, agent_radius_m=agent_radius_m,
        )
        candidates = decision.get("candidates") or []
        chosen = next((c for c in candidates if c["is_chosen"]), None)
        if chosen is not None:
            safe_margin_m = 3.0 * forward_unit_m
            best_eps = None
            for cand in candidates:
                if cand["is_chosen"] or cand["position"] is None:
                    continue
                alt_clearance = bin_clearance_toward_point(
                    cand["position"], agent_position, agent_rotation, occupancy_points_m,
                    agent_radius_m=agent_radius_m,
                )
                if alt_clearance is None or alt_clearance <= safe_margin_m:
                    continue
                margin = abs(cand["prob"] - chosen["prob"])
                if best_eps is None or margin < best_eps:
                    best_eps = margin
            min_eps_for_safe_alt = best_eps
            headroom_alt_exists = int(best_eps is not None and best_eps <= 0.10)
    risk_signals["external_predicted_min_clearance_m"] = (
        min(predicted_clearance_by_step) if predicted_clearance_by_step else None
    )
    risk_signals["headroom_alt_exists_eps10pct"] = headroom_alt_exists
    risk_signals["min_eps_for_safe_alt"] = min_eps_for_safe_alt

    return {
        "ok": True,
        "stop": bool(decision["final_stop"]),
        "primitive_actions": primitive_actions,
        "risk_signals": risk_signals,
        "predicted_clearance_by_step": predicted_clearance_by_step,
        # the hop targets primitive_actions was decomposed from; the harness's Tryout
        # controller (vlnsafe/agents/tryout.py) walks them hop by hop
        "hops": decision["hops"],
        "meta": {
            "act": decision["act"], "cur_vp": decision["cur_vp"],
            "target_vp": decision["target_vp"], "stepk": engine.stepk,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True,
                        help="the ETPNav/ETP-R1 checkout; worker chdirs here")
    parser.add_argument("--exp-config", default="run_r2r/iter_train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-path", default=None,
                        help="ETP pretrain to build the policy on; BEVBert's eval\n                             config sets pretrained_path: null and needs none")
    parser.add_argument("--back-algo", default="control", choices=["control", "teleport"])
    parser.add_argument("--trainer-module", default="vlnce_baselines.ss_trainer_ETP",
                        help="SS trainer to import; ETP-R1 uses "
                             "vlnce_baselines.ss_trainer_ETP_R1")
    parser.add_argument("--task-type", type=int, default=None,
                        help="ETP-R1 only: 1=r2r, 2=rxr (omit for ETPNav)")
    args = parser.parse_args()

    os.chdir(args.code_root)
    if args.code_root not in sys.path:
        sys.path.insert(0, args.code_root)

    opts = ["IL.back_algo", args.back_algo]
    with _RedirectStdoutToDevnull():
        engine = ETPNavDecisionEngine(
            args.exp_config, opts, args.checkpoint, args.pretrained_path,
            trainer_module=args.trainer_module, task_type=args.task_type,
        )
    eprint("VLNSAFE_ETPNAV_WORKER_READY")

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
