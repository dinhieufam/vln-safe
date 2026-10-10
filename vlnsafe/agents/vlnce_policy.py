"""In-process CMA / Seq2Seq provider, with the action categorical read out.

What: loads a VLN-CE recurrent policy checkpoint and runs it in the harness process (no
subprocess — VLN-CE's vlnce_baselines is already on the path). Each step is one decision
(horizon one), so on every step it inlines ILPolicy.act(deterministic=True) — one RNN
forward that yields both the chosen primitive and the 4-way categorical — and publishes
prob_stop / prob_forward / prob_left / prob_right. the pre-release CMA provider never
populated those, which is why every internal_* column of the archived CMA log is empty
.

Who reads it: vlnsafe.agents.registry, for agents with `family: vlnce_policy`.

Imports torch and vlnce_baselines. Must not import ETPNav's same-named package (this runs
in vlnsafe-harness, where only VLN-CE's vlnce_baselines is importable).

Ported from the pre-release harness; the deterministic
action is byte-identical to the archived `policy.act(..., deterministic=True)` call, so the
Phase E CMA reproduction is unaffected — the read-out is additive.
"""

# habitat discrete action ids; CMA's CategoricalNet has dim_actions == 4 in this order.
_PROB_KEYS = ("prob_stop", "prob_forward", "prob_left", "prob_right")


class VlncePolicyProvider(object):
    name = "vlnce_policy"

    def __init__(self, config, env, checkpoint_path, policy_name):
        import torch
        from habitat_baselines.common.baseline_registry import baseline_registry
        from habitat_baselines.common.obs_transformers import (
            apply_obs_transforms_batch,
            apply_obs_transforms_obs_space,
            get_active_obs_transforms,
        )
        from habitat_baselines.utils.common import batch_obs
        from vlnce_baselines.common.utils import extract_instruction_tokens
        import vlnce_baselines.models  # noqa: F401  (registers CMAPolicy / Seq2SeqPolicy)

        self.torch = torch
        self.batch_obs = batch_obs
        self.extract_instruction_tokens = extract_instruction_tokens
        self.apply_obs_transforms_batch = apply_obs_transforms_batch
        self.instruction_uuid = config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.config = config.clone()
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.MODEL.policy_name = policy_name
        self.config.TORCH_GPU_ID = 0
        self.config.SIMULATOR_GPU_IDS = [0]
        self.config.freeze()

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            env.observation_space, self.obs_transforms
        )
        policy_cls = baseline_registry.get_policy(policy_name)
        self.policy = policy_cls.from_config(
            config=self.config,
            observation_space=observation_space,
            action_space=env.action_space,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["state_dict"])
        self.policy.to(self.device)
        self.policy.eval()

        self.rnn_states = None
        self.prev_actions = None
        self.not_done_masks = None
        self.last_observation = None
        self._next_rnn_states = None
        # rollout reads then clears these; a fresh decision every step for a T1 policy.
        self.pending_decision_signals = None
        self.pending_step_predicted_clearance_m = None  # no external predictor in-process

    # --- ActionProvider contract ---------------------------------------------------
    def reset(self, observation):
        self.last_observation = observation
        self.rnn_states = self.torch.zeros(
            1,
            self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        self.prev_actions = self.torch.zeros(1, 1, device=self.device, dtype=self.torch.long)
        self.not_done_masks = self.torch.zeros(
            1, 1, dtype=self.torch.uint8, device=self.device
        )
        self.pending_decision_signals = None
        self.pending_step_predicted_clearance_m = None

    def act(self, observation, pose):
        del pose  # a recurrent policy conditions on its own hidden state, not the pose
        batch = self._batch()
        with self.torch.no_grad():
            # ILPolicy.act(deterministic=True), inlined so the categorical is captured in
            # the same forward pass that produces the action.
            features, next_rnn_states = self.policy.net(
                batch, self.rnn_states, self.prev_actions, self.not_done_masks
            )
            distribution = self.policy.action_distribution(features)
            action = int(distribution.mode()[0][0].item())
            probs = distribution.probs[0].detach().cpu().tolist()
        self._next_rnn_states = next_rnn_states
        self.pending_decision_signals = dict(zip(_PROB_KEYS, (float(p) for p in probs)))
        return action

    def planned_chunk(self):
        return []  # T1: one primitive per decision, nothing queued behind it

    def update_after_step(self, observation, executed_action, done):
        self.last_observation = observation
        self.rnn_states = self._next_rnn_states
        self.prev_actions.fill_(int(executed_action))
        self.not_done_masks = self.torch.tensor(
            [[0] if done else [1]], dtype=self.torch.uint8, device=self.device
        )

    # --- internals ---------------------------------------------------------------
    def _batch(self):
        observations = self.extract_instruction_tokens(
            [self.last_observation], self.instruction_uuid
        )
        batch = self.batch_obs(observations, self.device)
        return self.apply_obs_transforms_batch(batch, self.obs_transforms)
