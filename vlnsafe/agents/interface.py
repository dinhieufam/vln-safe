"""The ActionProvider contract — the file a reader opens to add an agent.

An action provider turns observations into a stream of primitive Habitat actions. It may
plan a chunk of primitives per decision (VLAs and ETPNav hops do); rollout pulls them one
at a time and asks for more when the chunk is empty.

Contract (this docstring is the specification; the ABC below just enforces the shape):

  reset(episode_observation) -> None
      Called once at episode start, before the first act().

  act(observation, pose) -> int
      Return the next primitive action id (STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT).
      May be served from an internal chunk without recomputing.

  planned_chunk() -> list[int]
      The primitives currently queued behind the one act() just returned, newest plan
      only. Logged verbatim into steps.csv `planned_chunk`. Empty for T1 providers that
      decide one primitive at a time.

  signals() -> dict | None
      On a step where a fresh decision was made, the internal columns for that decision
      row: prob_stop/forward/left/right for T1; candidate_entropy, top2_margin, stop_prob,
      num_candidates, value_estimate for T2. `UNAVAILABLE` (the string, never a blank,
      never 0) for a field the agent cannot supply. None on a non-decision step.

Ported from the implicit contract in the pre-release harness (propose_action /
preview_chunk / update_after_step / pending_decision_signals on the *ActionProvider
classes), made explicit here.
"""

try:
    from abc import ABC, abstractmethod
except ImportError:  # pragma: no cover - py<3.4 only
    ABC = object

    def abstractmethod(fn):
        return fn


UNAVAILABLE = "UNAVAILABLE"


class ActionProvider(ABC):
    @abstractmethod
    def reset(self, episode_observation):
        raise NotImplementedError

    @abstractmethod
    def act(self, observation, pose):
        raise NotImplementedError

    def planned_chunk(self):
        return []

    def signals(self):
        return None
