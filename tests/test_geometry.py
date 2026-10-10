"""Pin vlnsafe.geometry to the reference implementation it was ported from.

tests/fixtures/geometry_cases.json holds a seed + params + the expected output for each
case; the 256x256 inputs are rebuilt here from the shared deterministic builders in
tests/fixtures/geometry_inputs.py. Expected outputs were produced once by the original
pre-release depth/occupancy and hop-geometry functions. This test asserts the current
implementation reproduces every one to 1e-9. It needs no habitat.
"""
import json
import os
import sys

import numpy as np
import pytest

from vlnsafe import geometry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
from geometry_inputs import (  # noqa: E402
    ACTIONS, depth_panorama, hop_inputs, occupancy_cloud, primitive_sequence,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "geometry_cases.json")
TOL = 1e-9

if not os.path.exists(FIXTURES):
    pytest.skip(
        "geometry_cases.json missing",
        allow_module_level=True,
    )

with open(FIXTURES) as _f:
    CASES = json.load(_f)


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOL + 1e-6 * abs(float(b))


@pytest.mark.parametrize("case", CASES["clearance"], ids=range(len(CASES["clearance"])))
def test_step_and_directional_clearance(case):
    obs, keys = depth_panorama(case["seed"], case["n_keys"], case["structure"])
    assert _close(
        geometry.compute_step_clearance_m(obs, sensor_keys=keys),
        case["step_clearance_m"],
    )
    got = geometry.compute_directional_clearance_m(obs, sensor_keys=keys)
    exp = case["directional_clearance_m"]
    assert len(got) == len(exp)
    assert all(_close(g, e) for g, e in zip(got, exp))


@pytest.mark.parametrize("case", CASES["occupancy"], ids=range(len(CASES["occupancy"])))
def test_occupancy_projection(case):
    obs, keys = depth_panorama(case["seed"], case["n_keys"], case["structure"])
    got = geometry.project_depth_to_occupancy_points_m(
        obs, sensor_keys=keys,
        hfov_deg=case["hfov_deg"], camera_height_m=case["camera_height_m"],
    )
    exp = case["points"]
    assert len(got) == len(exp)
    for (gx, gy), (ex, ey) in zip(got, exp):
        assert _close(gx, ex) and _close(gy, ey)


@pytest.mark.parametrize("case", CASES["primitive_walk"], ids=range(len(CASES["primitive_walk"])))
def test_predicted_clearance_by_primitive_step(case):
    cloud = occupancy_cloud(case["cloud_seed"], case["n_points"])
    prims = primitive_sequence(case["prim_seed"], case["n_prims"])
    got = geometry.predicted_clearance_by_primitive_step(
        prims, case["turn_unit_deg"], case["forward_unit_m"],
        cloud, ACTIONS, agent_radius_m=case["agent_radius_m"],
    )
    exp = case["predicted_clearance_m"]
    if exp is None:
        assert got is None
        return
    assert len(got) == len(exp)
    assert all(_close(g, e) for g, e in zip(got, exp))


@pytest.mark.parametrize("case", CASES["hop"], ids=range(len(CASES["hop"])))
def test_hop_decomposition_and_heading(case):
    quat, cur_pos, hops = hop_inputs(case["seed"])
    assert _close(geometry.heading_from_quaternion(np.asarray(quat)), case["heading"])

    got_prims = geometry.decompose_hops_to_primitives(
        hops, cur_pos, quat, case["turn_unit_deg"], case["forward_unit_m"], ACTIONS,
    )
    assert got_prims == case["primitives"]

    got_off = geometry.first_hop_relative_offset(hops, cur_pos, quat)
    ex, ey = case["first_hop_offset"]
    assert _close(got_off[0], ex) and _close(got_off[1], ey)


@pytest.mark.parametrize("case", CASES["hop"], ids=range(len(CASES["hop"])))
def test_hop_segments_flatten_to_the_decomposition(case):
    """vlnsafe.agents.tryout relies on this: its no-collision chunk is the landed chunk."""
    quat, cur_pos, hops = hop_inputs(case["seed"])
    segs = geometry.hop_segments(
        hops, cur_pos, quat, case["turn_unit_deg"], case["forward_unit_m"], ACTIONS)
    assert len(segs) == len(hops)
    flat = [a for turns, k in segs for a in turns + [int(ACTIONS.MOVE_FORWARD)] * k]
    assert flat == case["primitives"]


def test_care_rotation_zero_is_identity():
    """care_rotation_rad=0.0 must reproduce the unrotated decomposition exactly."""
    case = CASES["hop"][0]
    quat, cur_pos, hops = hop_inputs(case["seed"])
    common = (hops, cur_pos, quat, case["turn_unit_deg"], case["forward_unit_m"], ACTIONS)
    assert (
        geometry.decompose_hops_to_primitives(*common)
        == geometry.decompose_hops_to_primitives(*common, care_rotation_rad=0.0)
    )


def _projection_column_loop(obs, sensor_keys, hfov_deg, camera_height_m, max_depth_m=10.0,
                            min_depth_m=0.0, col_stride=2, floor_margin_m=0.15):
    """The pre-2026-09-23 column-at-a-time loop of project_depth_to_occupancy_points_m,
    verbatim, kept only as the reference the vectorised version must match exactly."""
    import math
    tan_half_hfov = math.tan(math.radians(hfov_deg / 2.0))
    points = []
    for key, yaw_deg in zip(sensor_keys, [30.0 * k for k in range(len(sensor_keys))]):
        arr = np.asarray(obs[key], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        height, width = arr.shape
        fx = (width / 2.0) / tan_half_hfov
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        lo, hi = int(round(16 * height / 256.0)), int(round(240 * height / 256.0))
        dv = np.arange(lo, hi) - cy
        with np.errstate(divide="ignore"):
            z_floor_per_row = np.where(dv > 0, camera_height_m * fx / np.maximum(dv, 1e-6), np.inf)
        band = arr[lo:hi, :]
        yaw = math.radians(yaw_deg)
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        for u in range(0, width, col_stride):
            raw_col = band[:, u].astype(np.float64)
            valid_mask = raw_col > 1e-6
            if not valid_mask.any():
                continue
            z_cam_col = raw_col * (max_depth_m - min_depth_m) + min_depth_m
            keep = (valid_mask & (z_cam_col < (z_floor_per_row - floor_margin_m))
                    & (z_cam_col < max_depth_m - 1e-6))
            if not keep.any():
                continue
            z_cam = float(z_cam_col[keep].min())
            x_cam = (u - cx) / fx * z_cam
            points.append([round(float(x_cam * syaw + z_cam * cyaw), 3),
                           round(float(z_cam * syaw - x_cam * cyaw), 3)])
    return points


@pytest.mark.parametrize("shape,hfov", [((256, 256), 90.0), ((480, 640), 79.0)])
def test_vectorised_projection_is_bit_identical_to_the_column_loop(shape, hfov):
    rng = np.random.RandomState(7)
    keys = ["depth"] + ["depth_{}".format(30 * k) for k in range(1, 12)]
    obs = {}
    for i, key in enumerate(keys):
        frame = rng.uniform(0.0, 1.0, shape).astype(np.float32)
        frame[rng.uniform(size=shape) < 0.1] = 0.0          # invalid returns
        if i % 3 == 0:
            frame[:, : shape[1] // 3] = 0.0                  # whole invalid columns
        if i % 4 == 1:
            frame[:] = 1.0                                   # beyond max range
        obs[key] = frame[..., None] if i % 2 else frame
    new = geometry.project_depth_to_occupancy_points_m(
        obs, sensor_keys=keys, hfov_deg=hfov, camera_height_m=1.25)
    assert new and new == _projection_column_loop(obs, keys, hfov, 1.25)
