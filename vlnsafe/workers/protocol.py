"""The worker JSON-lines protocol: request/response shapes and light validators.

One request per line on the worker's stdin, one response per line on its stdout. This
module holds only the field names and shape checks; it imports nothing heavy and is used
by both sides (vlnsafe.agents.remote and vlnsafe/workers/*_worker.py).

Request (cmd="decide"):
    episode_reset   bool     -- reset the worker's per-episode state before deciding
    agent_position  [x,y,z]
    agent_rotation  [qx,qy,qz,qw]
    turn_unit_deg   float
    forward_unit_m  float
    observation     {uuid: {shape:[...], dtype:str, data_b64:str}, "instruction": {...}}
    occupancy_points_m  [[fwd,left], ...]  (optional; enables the external predictor)
    agent_radius_m  float                  (optional)
    decision_index  int                    (optional; which decision of the episode)
    history_observations  [observation, ...]  (optional; streaming VLA workers only --
                    the frames executed since the last decision, oldest first, which the
                    worker pushes into its video/KV history without generating)
Request (cmd="shutdown"): no other fields.

Response:
    ok                  bool
    primitive_actions   [int, ...]         -- STOP/MOVE_FORWARD/TURN_LEFT/TURN_RIGHT ids
    stop                bool
    risk_signals        {stop_prob, candidate_entropy, candidate_entropy_normalized,
                         top2_margin_prob, num_valid_candidates,
                         headroom_alt_exists_eps10pct, min_eps_for_safe_alt}
    predicted_clearance_by_step  [float|null, ...] | null
    hops                [[x,y,z], ...]     -- ETPNav-family only: the hop targets the
                                              primitives were decomposed from (Tryout)
    meta                {act, cur_vp, target_vp, stepk}
On failure: {ok: false, error: str}.
"""
PROTOCOL_VERSION = 1

REQUEST_KEYS = (
    "cmd", "episode_reset", "agent_position", "agent_rotation",
    "turn_unit_deg", "forward_unit_m", "observation",
)
RESPONSE_KEYS = ("ok", "primitive_actions", "risk_signals")


def validate_request(msg):
    if msg.get("cmd") == "shutdown":
        return
    if msg.get("cmd") != "decide":
        raise ValueError("unknown cmd {!r}".format(msg.get("cmd")))
    for k in ("agent_position", "agent_rotation", "observation"):
        if k not in msg:
            raise ValueError("request missing {!r}".format(k))
    if len(msg["agent_position"]) != 3 or len(msg["agent_rotation"]) != 4:
        raise ValueError("agent_position must be xyz, agent_rotation must be xyzw")


def validate_response(msg):
    if not msg.get("ok", False):
        raise ValueError("worker error: {}".format(msg.get("error")))
    if not isinstance(msg.get("primitive_actions"), list):
        raise ValueError("response missing primitive_actions list")
