"""The external clearance predictor for agents without one (T1), and the projection's
frame-size handling it depends on. Pure numpy; no habitat.

The golden 256x256 cases in test_geometry.py already pin the projection for the ETPNav rig;
these cover what is new: a non-square VLA camera (StreamVLN 640x480, HFOV 79) and the
single-primitive predictor built on it.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
from geometry_inputs import ACTIONS  # noqa: E402

from vlnsafe import geometry  # noqa: E402
from vlnsafe.clearance_predictor import PrimitiveClearancePredictor  # noqa: E402

MAX_DEPTH_M = 10.0
HFOV_DEG = 79.0
H, W = 480, 640


def _wall_frame(distance_m, height=H, width=W):
    """A flat wall `distance_m` ahead: every pixel reads that (normalised) depth."""
    return np.full((height, width, 1), distance_m / MAX_DEPTH_M, dtype=np.float32)


def _predictor():
    return PrimitiveClearancePredictor(
        turn_unit_deg=15, forward_unit_m=0.25, agent_radius_m=0.1, hfov_deg=HFOV_DEG,
        camera_height_m=1.25, sensor_keys=["depth"], min_depth_m=0.0,
        max_depth_m=MAX_DEPTH_M, actions=ACTIONS)


def test_projection_uses_the_frames_own_intrinsics():
    d = 2.0
    pts = geometry.project_depth_to_occupancy_points_m(
        {"depth": _wall_frame(d)}, sensor_keys=["depth"], hfov_deg=HFOV_DEG,
        camera_height_m=1.25)
    assert pts
    forward = np.array([p[0] for p in pts])
    left = np.array([p[1] for p in pts])
    # a flat wall projects to a line at x_forward == d, spanning the horizontal FOV
    assert np.allclose(forward, d, atol=1e-3)
    half_width = d * math.tan(math.radians(HFOV_DEG / 2.0)) * (W - 1) / W
    assert abs(left.max() - half_width) < 0.02 and abs(left.min() + half_width) < 0.02


def test_forward_primitive_sees_the_wall_and_turn_does_not_move():
    p = _predictor()
    obs = {"depth": _wall_frame(1.0)}
    # forward: 0.25 m closer to a wall 1.0 m away, minus the 0.1 m body radius
    assert abs(p(obs, ACTIONS.MOVE_FORWARD) - (0.75 - 0.1)) < 0.01
    # stop / turn keep the pose, so the clearance is the current one
    assert abs(p(obs, ACTIONS.STOP) - (1.0 - 0.1)) < 0.01
    assert abs(p(obs, ACTIONS.TURN_LEFT) - p(obs, ACTIONS.STOP)) < 1e-9


def test_a_closer_wall_gives_a_smaller_forward_clearance():
    p = _predictor()
    near = p({"depth": _wall_frame(0.5)}, ACTIONS.MOVE_FORWARD)
    far = p({"depth": _wall_frame(3.0)}, ACTIONS.MOVE_FORWARD)
    assert near < far


def test_no_obstacle_points_is_none_not_zero():
    p = _predictor()
    open_frame = {"depth": np.zeros((H, W, 1), dtype=np.float32)}  # no valid returns
    assert p(open_frame, ACTIONS.MOVE_FORWARD) is None
    assert p({}, ACTIONS.MOVE_FORWARD) is None
