"""Agent pose reads and the action <-> env.step conversion.

Who reads it: vlnsafe.rollout (every step), vlnsafe.agents.remote (pose for the worker
request).

Imports numpy; touches only `env.sim.get_agent_state()`. Must not import torch.

Ported from the pre-release harness: simulator_position (155-157), agent_pose_for_etpnav
(384-388), action_to_step (174-175), pose_json (791-792).
"""
import json

import numpy as np


def agent_position(env):
    """Agent xyz as a float64 numpy array."""
    return np.array(env.sim.get_agent_state().position, dtype=np.float64)


def agent_pose(env):
    """(position [x, y, z], rotation [qx, qy, qz, qw]) for a worker `decide` request."""
    state = env.sim.get_agent_state()
    position = state.position.tolist()
    rotation = list(state.rotation.imag) + [float(state.rotation.real)]
    return position, rotation


def action_to_step(action):
    return {"action": int(action)}


def pose_json(position):
    """Compact JSON of a position array, rounded to 6 dp — the `planned_chunk`-adjacent
    pose columns in steps.csv use this."""
    return json.dumps([round(float(v), 6) for v in np.asarray(position).tolist()])
