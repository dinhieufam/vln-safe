"""Deterministic input builders shared by the (pre-release) fixture generator and
test_geometry.py, so the fixture JSON only has to store a seed + params, never a 256x256
array. numpy's legacy RandomState stream is frozen across numpy versions, so the env that
generates the fixtures and the env that runs the test rebuild byte-identical inputs.
"""
import types

import numpy as np

ACTIONS = types.SimpleNamespace(STOP=0, MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3)

# The archived occupancy projection is written for the 256x256 clearance rig
# (the pre-release harness: _OCC_IMG_SIZE = 256, row_range = (16, 240)); inputs must be
# that size or the per-row floor geometry does not broadcast.
IMG_SIZE = 256


def depth_panorama(seed, n_keys, structure):
    """`n_keys` normalised-depth frames (uuid depth / depth_30 / ... ), values in [0, 1]."""
    rng = np.random.RandomState(seed)
    keys = ["depth"] + ["depth_{}".format(30 * k) for k in range(1, n_keys)]
    obs = {}
    for key in keys:
        if structure == "empty":
            frame = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        elif structure == "all_far":
            frame = np.full((IMG_SIZE, IMG_SIZE), 0.95, dtype=np.float32)
        elif structure == "all_close":
            frame = np.full((IMG_SIZE, IMG_SIZE), 0.02, dtype=np.float32)
        else:  # "mixed": textured depth + a near vertical obstacle + some invalid returns
            frame = rng.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)
            c0 = rng.randint(0, IMG_SIZE - 8)
            frame[:, c0:c0 + 6] = rng.uniform(0.01, 0.08)
            frame[rng.rand(IMG_SIZE, IMG_SIZE) < 0.05] = 0.0
        obs[key] = frame
    return obs, keys


def occupancy_cloud(seed, n_points):
    """A plausible ego-frame point cloud ([x_forward, y_left] metres) for the clearance
    walkers — they don't care how it was produced."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(-3.0, 3.0, n_points)
    ys = rng.uniform(-3.0, 3.0, n_points)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]


def hop_inputs(seed):
    """(quat coeffs [x,y,z,w], cur_pos [x,y,z], hops [[x,y,z], ...])."""
    rng = np.random.RandomState(seed)
    q = rng.randn(4)
    q = q / np.linalg.norm(q)
    cur_pos = rng.randn(3) * 3.0
    n_hops = int(rng.randint(1, 4))
    hops = [
        [float(cur_pos[0] + rng.randn()), float(cur_pos[1]), float(cur_pos[2] + rng.randn())]
        for _ in range(n_hops)
    ]
    return [float(v) for v in q], [float(v) for v in cur_pos], hops


def primitive_sequence(seed, n):
    rng = np.random.RandomState(seed)
    return [int(rng.choice([ACTIONS.MOVE_FORWARD, ACTIONS.TURN_LEFT, ACTIONS.TURN_RIGHT]))
            for _ in range(n)]
