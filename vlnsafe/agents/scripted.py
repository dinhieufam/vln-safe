"""Scripted providers for smoke tests and CI: a seeded random walker and a straight-line
walker. No checkpoint, no habitat model code — these exist so `make smoke` and
tests/test_smoke_rollout.py can exercise the rollout loop without a real agent.

Who reads it: vlnsafe.agents.registry (ids `random`, `straight`), tests.

Must not import: habitat, torch.

Ported from the pre-release harness. HandcraftedAction
Provider (215-257) is intentionally dropped.
"""

import numpy as np

STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


class RandomProvider(object):
    """Uniform random primitive each step, from a seeded RandomState."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)

    def reset(self, episode_observation):
        pass

    def act(self, observation, pose):
        return int(self._rng.randint(0, 4))

    def planned_chunk(self):
        return []

    def update_after_step(self, observation, executed_action, done):
        pass

    def signals(self):
        return None


class StraightProvider(object):
    """MOVE_FORWARD until a fixed step budget, then STOP. Deterministic."""

    def __init__(self, max_forward=40):
        self._max_forward = max_forward
        self._n = 0

    def reset(self, episode_observation):
        self._n = 0

    def act(self, observation, pose):
        self._n += 1
        return MOVE_FORWARD if self._n <= self._max_forward else STOP

    def planned_chunk(self):
        return []

    def update_after_step(self, observation, executed_action, done):
        pass

    def signals(self):
        return None
