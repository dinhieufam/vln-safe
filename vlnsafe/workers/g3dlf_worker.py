#!/usr/bin/env python3
"""g3D-LF worker (docs/agent_notes/g3dlf.md): ETPNav's decision engine plus g3D-LF's
lookahead step. Runs under vlnsafe-g3dlf as a subprocess; same JSON-lines protocol and
request handling as vlnsafe.workers.etpnav_worker, which this module reuses unchanged.

What differs from ETPNav, each mirrored from external/g3D-LF/Lookahead_3DFF/
vlnce_baselines/ss_trainer_ETP.py::RLTrainer.rollout() (mode='eval', argmax):
  - the waypoint pass also writes this step's 12 views into the 3D feature field
    (`feature_fields`, `batch_positions`, `batch_headings`);
  - after the navigation head, the two most probable candidates get panoramic features
    rendered at their positions from the field and are re-scored by the
    `lookahead_navigation` head; their logits are overwritten in place (`fused`), and
    the argmax of the fused logits is the decision;
  - stop scores stay the pre-lookahead navigation probabilities, as upstream (the
    upstream tensor is aliased, but node_stop_scores is read before the overwrite);
  - everything runs under torch.no_grad + torch.cuda.amp.autocast, as upstream eval().

Kept a separate module (a subclass, not an edit) so the ETPNav / ETP-R1 / BEVBert worker
that produced the landed arms stays byte-identical.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from vlnsafe.workers.etpnav_worker import (
    ETPNavDecisionEngine,
    _RedirectStdoutToDevnull,
    eprint,
    handle_request,
)

LOOKAHEAD_CANDIDATES = 2  # ss_trainer_ETP.py: selected_cand_num = 2 in eval
# R2R camera HFOV (ss_trainer_ETP.py rollout, stepk == 0); RxR's 79 is out of scope
# (g3D-LF releases no RxR-CE lookahead checkpoint).
FIELD_HFOV_DEG = {"r2r": 90.0}


class G3DLFDecisionEngine(ETPNavDecisionEngine):

    def __init__(self, *args, **kwargs):
        super(G3DLFDecisionEngine, self).__init__(*args, **kwargs)
        from torch.cuda.amp import autocast
        from habitat.tasks.utils import cartesian_to_polar
        from habitat.utils.geometry_utils import (
            quaternion_from_coeff,
            quaternion_rotate_vector,
        )
        self.autocast = autocast
        self._cartesian_to_polar = cartesian_to_polar
        self._quaternion_from_coeff = quaternion_from_coeff
        self._quaternion_rotate_vector = quaternion_rotate_vector
        self.feature_fields = self.trainer.feature_fields
        task = self.trainer.config.MODEL.task_type
        if task not in FIELD_HFOV_DEG:
            raise ValueError("g3D-LF worker: task_type {!r} unsupported".format(task))
        self.field_hfov = FIELD_HFOV_DEG[task]

    def _heading(self, agent_rotation):
        """environments.py::get_agent_info's heading (no mod 2*pi), from [x, y, z, w]."""
        quat = self._quaternion_from_coeff(np.asarray(agent_rotation, dtype=np.float64))
        heading_vector = self._quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
        return self._cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

    def reset_episode(self, observation):
        # rollout(): feature_fields.reset(mode='habitat') before the step loop, then at
        # stepk == 0 the camera settings and a second reset.
        self.feature_fields.reset(batch_size=1, mode="habitat")
        self.feature_fields.initialize_camera_setting(hfov=self.field_hfov, vfov=self.field_hfov)
        self.feature_fields.initialize_novel_view_setting(
            hfov=self.field_hfov, vfov=self.field_hfov)
        self.feature_fields.reset(1)
        with self.torch.no_grad(), self.autocast():
            super(G3DLFDecisionEngine, self).reset_episode(observation)

    def decide(self, observation, agent_position, agent_rotation):
        # upstream eval() is @torch.no_grad() and wraps rollout() in autocast()
        with self.torch.no_grad(), self.autocast():
            return self._decide(observation, agent_position, agent_rotation)

    def _decide(self, observation, agent_position, agent_rotation):
        torch = self.torch
        F = torch.nn.functional
        batch = self._build_batch(observation)
        cur_pos = [np.array(agent_position, dtype=np.float64)]
        cur_ori = [np.array(agent_rotation, dtype=np.float64)]

        wp_outputs = self.trainer.policy.net(
            mode="waypoint",
            waypoint_predictor=self.trainer.waypoint_predictor,
            observations=batch,
            feature_fields=self.feature_fields,
            batch_positions=[list(agent_position)],
            batch_headings=[self._heading(agent_rotation)],
            in_train=False,
        )
        vp_inputs = self.trainer._vp_feature_variable(wp_outputs)
        vp_inputs.update({"mode": "panorama"})
        pano_embeds, pano_masks = self.trainer.policy.net(**vp_inputs)
        avg_pano_embeds = torch.sum(
            pano_embeds * pano_masks.unsqueeze(2), 1
        ) / torch.sum(pano_masks, 1, keepdim=True)

        self.trainer.gmaps = [self.gmap]
        cur_vp, cand_vp, cand_pos = self.gmap.identify_node(
            cur_pos[0], cur_ori[0], wp_outputs["cand_angles"][0], wp_outputs["cand_distances"][0]
        )
        self.gmap.update_graph(
            self.prev_vp, self.stepk + 1,
            cur_vp, cur_pos[0], avg_pano_embeds[0],
            cand_vp, cand_pos, pano_embeds[0][vp_inputs["nav_types"][0] == 1],
            None,
        )

        nav_inputs = self.trainer._nav_gmap_variable([cur_vp], cur_pos, cur_ori)
        nav_inputs.update({
            "mode": "navigation",
            "txt_embeds": self._txt_embeds,
            "txt_masks": self._txt_masks,
            "lookahead_loc_fts": vp_inputs["lookahead_loc_fts"],
        })
        gmap_cand_pos = nav_inputs.pop("batch_gmap_cand_pos")[0]
        no_vp_left = nav_inputs.pop("no_vp_left")
        nav_outs = self.trainer.policy.net(**nav_inputs)
        nav_logits = nav_outs["global_logits"]
        nav_probs = F.softmax(nav_logits, 1)
        self.gmap.node_stop_scores[cur_vp] = nav_probs[0, 0].data.item()

        # --- lookahead: re-score the two most probable candidates ---
        k = LOOKAHEAD_CANDIDATES
        n_cand = len(gmap_cand_pos[2])
        _, cand_ids = torch.topk(nav_probs[0], min(k, n_cand), largest=True)
        cand_ids = cand_ids.cpu()
        positions = gmap_cand_pos[0][cand_ids]
        headings = gmap_cand_pos[1][cand_ids]
        if len(cand_ids) < k:  # upstream pads with candidate 0 (the stop slot)
            pad = k - len(cand_ids)
            positions = np.concatenate([positions] + [gmap_cand_pos[0][0:1]] * pad, axis=0)
            headings = np.concatenate([headings] + [gmap_cand_pos[1][0:1]] * pad, axis=0)
            cand_ids = torch.cat([cand_ids] + [torch.tensor([0])] * pad, dim=0)
        cand_fts = []
        with torch.no_grad():
            for ci in range(k):
                fts = self.feature_fields.run_panorama_encode(
                    [positions[ci]], [headings[ci]], panorama_encode=False)
                cand_fts.append(fts.unsqueeze(0))
        nav_inputs["cand_panorama_fts"] = torch.cat(cand_fts, dim=0).to(
            self.device).permute(1, 0, 2, 3)
        nav_inputs["selected_cand_ids"] = cand_ids.unsqueeze(0).to(self.device)
        nav_inputs.update({"mode": "lookahead_navigation"})
        lookahead_logits = self.trainer.policy.net(**nav_inputs)
        fused_logits = nav_logits.clone()
        fused_logits[0][nav_inputs["selected_cand_ids"][0].to(nav_logits.device)] = \
            lookahead_logits[0]
        fused_probs = F.softmax(fused_logits.float(), 1)

        a_t = int(fused_logits.argmax(dim=-1)[0].item())
        force_stop = (
            self.stepk == self.max_len - 1 + self.budget_extension) or bool(no_vp_left[0])

        # risk signals / candidates describe the distribution the decision was taken from
        valid_idx = torch.isfinite(fused_logits[0]).nonzero(as_tuple=True)[0].tolist()
        gmap_vp_ids_0 = nav_inputs["gmap_vp_ids"][0]
        candidates = []
        for i in valid_idx:
            vp = gmap_vp_ids_0[i]
            pos = None
            if i != 0:
                pos = self.gmap.node_pos.get(vp)
                if pos is None:
                    pos = self.gmap.ghost_aug_pos.get(vp)
            candidates.append({
                "prob": float(fused_probs[0, i].item()),
                "is_chosen": bool(i == a_t),
                "position": np.asarray(pos).tolist() if pos is not None else None,
            })
        probs_sorted = sorted((c["prob"] for c in candidates), reverse=True)
        entropy = -sum(p * math.log(p) for p in probs_sorted if p > 1e-12)
        risk_signals = {
            "stop_prob": float(fused_probs[0, 0].item()),
            "candidate_entropy": entropy,
            "candidate_entropy_normalized": (
                entropy / math.log(len(probs_sorted)) if len(probs_sorted) > 1 else 0.0),
            "top2_margin_prob": (
                probs_sorted[0] - probs_sorted[1] if len(probs_sorted) >= 2 else 1.0),
            "num_valid_candidates": len(candidates),
        }

        if a_t == 0 or force_stop:
            vp_stop_scores = list(self.gmap.node_stop_scores.items())
            stop_vp = vp_stop_scores[int(np.argmax([s[1] for s in vp_stop_scores]))][0]
            hops = [self.gmap.node_pos[vp]
                    for vp in self.gmap.shortest_path[cur_vp][stop_vp]][1:]
            self.prev_vp = None
            self.stepk += 1
            return {
                "act": 0, "cur_vp": cur_vp, "target_vp": stop_vp,
                "hops": [np.asarray(h).tolist() for h in hops],
                "final_stop": True, "risk_signals": risk_signals, "candidates": candidates,
            }

        ghost_vp = gmap_vp_ids_0[a_t]
        ghost_pos = self.gmap.ghost_aug_pos[ghost_vp]
        _, front_vp = self.gmap.front_to_ghost_dist(ghost_vp)
        hops = [self.gmap.node_pos[vp]
                for vp in self.gmap.shortest_path[cur_vp][front_vp]][1:] + [ghost_pos]
        if self.trainer.config.MODEL.consume_ghost:
            self.gmap.delete_ghost(ghost_vp)
        self.prev_vp = front_vp
        self.stepk += 1
        return {
            "act": 4, "cur_vp": cur_vp, "target_vp": ghost_vp,
            "hops": [np.asarray(h).tolist() for h in hops],
            "final_stop": False, "risk_signals": risk_signals, "candidates": candidates,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True,
                        help="external/g3D-LF/Lookahead_3DFF; worker chdirs here")
    parser.add_argument("--exp-config", default="run_r2r/iter_train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--back-algo", default="control", choices=["control", "teleport"])
    parser.add_argument("--trainer-module", default="vlnce_baselines.ss_trainer_ETP")
    parser.add_argument("--task-type", type=int, default=None,
                        help="unused (ETP-R1 only); accepted for argv compatibility")
    args = parser.parse_args()

    os.chdir(args.code_root)
    if args.code_root not in sys.path:
        sys.path.insert(0, args.code_root)

    with _RedirectStdoutToDevnull():
        engine = G3DLFDecisionEngine(
            args.exp_config, ["IL.back_algo", args.back_algo], args.checkpoint,
            args.pretrained_path, trainer_module=args.trainer_module, task_type=None,
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
