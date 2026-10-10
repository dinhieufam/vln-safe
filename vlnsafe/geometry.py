"""Occupancy and clearance geometry.

What: turns a panoramic depth image into a 2D occupancy point cloud in the agent's frame
and answers "how close does this primitive take the agent to the nearest obstacle?", plus
the hop -> primitive decomposition ETPNav-family workers use to walk a graph decision out
into STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT.

Who reads it: vlnsafe.rollout (per step, for step_clearance_m) and every worker in
vlnsafe/workers/ (per candidate, for predicted_clearance_m). The workers run in four
different conda environments and all of them import this module.

Must not import: habitat, torch. numpy is the only dependency it is allowed. The habitat
quaternion helpers the ETPNav port originally leaned on (quaternion_from_coeff,
quaternion_rotate_vector, cartesian_to_polar) are reimplemented here in plain numpy so
this file stays habitat-free; they are pinned against habitat's own output by
tests/test_geometry.py.

Sign conventions: x forward along the agent's heading at snapshot time, y left, metres,
agent frame. Turn-left is the positive angle direction. See docs/schema.md.

Ported with zero behavioural change from:
  the pre-release harness  (compute_step_clearance_m,
      compute_directional_clearance_m, project_depth_to_occupancy_points_m)
  etpnav_policy_worker.py:182-316      (turn_actions_for_angle, occupancy_min_clearance,
      predicted_clearance_by_primitive_step, bin_clearance_toward_point,
      decompose_hops_to_primitives)
  hop_geometry.py:44-96                (heading_from_quaternion, calculate_vp_rel_pos,
      first_hop_relative_offset)
The three geometry functions in the first group are the load-bearing safety code and have
twice hidden bugs biased toward under-reporting risk; they are reproduced here verbatim
and locked to sampled (input, output) pairs.
"""

import math

import numpy as np

# --- rig / projection constants (read the real values from the
# merged benchmark config where one is in hand; these are the R2R-style defaults the
# archived runner used and are only fallbacks). ---

DEPTH_CLEARANCE_MIN_DEPTH_M = 0.0
DEPTH_CLEARANCE_MAX_DEPTH_M = 10.0
# 12 sensors, 30 degrees apart, spanning 360 degrees. habitat-lab default depth range,
# NORMALIZE_DEPTH=True, so raw obs values are in [0, 1] and
# metres = value * (max_depth - min_depth) + min_depth.
DEPTH_CLEARANCE_SENSOR_KEYS = ["depth"] + ["depth_{}".format(30 * k) for k in range(1, 12)]
CLEARANCE_DEPTH_SENSOR_KEYS = ["clearance_depth_0"] + [
    "clearance_depth_{}".format(30 * k) for k in range(1, 12)
]

DEFAULT_AGENT_RADIUS_M = 0.1  # habitat-lab SIMULATOR.AGENT_0.RADIUS default

_OCC_IMG_SIZE = 256
_OCC_ROW_RANGE = (16, 240)  # trims the most extreme ~6% of rows at each edge (steep,
# near-degenerate ray angles); NOT a floor/ceiling exclusion -- floor is dropped by a
# per-pixel physical check below.
_OCC_FLOOR_MARGIN_M = 0.15  # buffer below the geometric floor-intercept distance so
# grazing floor rays don't flicker between "obstacle" and "floor".
DEFAULT_OCC_HFOV_DEG = 90.0        # R2R; RxR is 63 -- pass the real value.
DEFAULT_OCC_CAMERA_HEIGHT_M = 1.25  # R2R; RxR is 0.88 -- pass the real value.


# --- pure-numpy replacements for the three habitat quaternion helpers -------------------
# habitat represents a rotation as coeffs [x, y, z, w]. These reproduce
# habitat.utils.geometry_utils.quaternion_from_coeff / quaternion_rotate_vector and
# habitat.tasks.utils.cartesian_to_polar for that coeff order, using the Hamilton product
# on (w, x, y, z) arrays. Locked to habitat's own output in tests/test_geometry.py.


def _quat_wxyz_from_coeff(coeffs):
    c = np.asarray(coeffs, dtype=np.float64)
    return np.array([c[3], c[0], c[1], c[2]], dtype=np.float64)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_inverse(q):
    # habitat uses np.quaternion.inverse() == conjugate / norm**2 (unit for agent poses).
    w, x, y, z = q
    n2 = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z], dtype=np.float64) / n2


def _quat_rotate_vector(q, v):
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    out = _quat_mul(_quat_mul(q, vq), _quat_inverse(q))
    return out[1:]


def cartesian_to_polar(x, y):
    return math.sqrt(x * x + y * y), math.atan2(y, x)


def heading_from_quaternion(quat_coeff):
    """Yaw of a habitat [x, y, z, w] rotation, in [0, 2*pi). Port of
    vlnce_baselines/models/graph_utils.py:54-59 (hop_geometry.py:44-49)."""
    q = _quat_wxyz_from_coeff(quat_coeff)
    heading_vector = _quat_rotate_vector(_quat_inverse(q), np.array([0.0, 0.0, -1.0]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)


def calculate_vp_rel_pos(p1, p2, base_heading=0.0, base_elevation=0.0):
    """Relative (heading, xz-distance) from p1 to p2. Verbatim port of
    vlnce_baselines/common/environments.py:26-42 (hop_geometry.py:52-67)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]  # noqa: F841 - kept verbatim; y is elevation, unused for xz heading
    dz = p2[2] - p1[2]
    xz_dist = max(np.sqrt(dx ** 2 + dz ** 2), 1e-8)

    heading = np.arcsin(-dx / xz_dist)  # (-pi/2, pi/2)
    if p2[2] > p1[2]:
        heading = np.pi - heading
    heading -= base_heading
    while heading < 0:
        heading += 2 * np.pi
    heading = heading % (2 * np.pi)
    return heading, xz_dist


# --- depth -> clearance ---------------------------------------------------------------


def compute_step_clearance_m(
    obs,
    min_depth_m=DEPTH_CLEARANCE_MIN_DEPTH_M,
    max_depth_m=DEPTH_CLEARANCE_MAX_DEPTH_M,
    sensor_keys=DEPTH_CLEARANCE_SENSOR_KEYS,
):
    """Nearest-obstacle distance around the agent this step, over the whole panoramic
    depth rig. Returns None if no depth sensor is present in obs (so callers can skip
    clearance logging rather than fabricate a value). A raw 0.0 reading means "no valid
    return", not "touching", and is excluded before the min. min_depth_m/max_depth_m MUST
    match the rig the caller built. Verbatim: the pre-release harness."""
    depth_min = None
    for key in sensor_keys:
        frame = obs.get(key)
        if frame is None:
            continue
        arr = np.asarray(frame, dtype=np.float32).reshape(-1)
        valid = arr[arr > 1e-6]
        if valid.size == 0:
            continue
        frame_min_m = float(valid.min()) * (max_depth_m - min_depth_m) + min_depth_m
        depth_min = frame_min_m if depth_min is None else min(depth_min, frame_min_m)
    return depth_min


def compute_directional_clearance_m(
    obs,
    min_depth_m=DEPTH_CLEARANCE_MIN_DEPTH_M,
    max_depth_m=DEPTH_CLEARANCE_MAX_DEPTH_M,
    sensor_keys=DEPTH_CLEARANCE_SENSOR_KEYS,
):
    """Per-direction analogue of compute_step_clearance_m, one reading per camera bin in
    sensor_keys' order. A bin with no valid return is reported at max_depth_m (open),
    never None, so the geometry always has a number. Verbatim:
    the pre-release harness."""
    values = []
    for key in sensor_keys:
        frame = obs.get(key) if obs is not None else None
        if frame is None:
            values.append(max_depth_m)
            continue
        arr = np.asarray(frame, dtype=np.float32).reshape(-1)
        valid = arr[arr > 1e-6]
        if valid.size == 0:
            values.append(max_depth_m)
            continue
        values.append(float(valid.min()) * (max_depth_m - min_depth_m) + min_depth_m)
    return values


def project_depth_to_occupancy_points_m(
    obs,
    min_depth_m=DEPTH_CLEARANCE_MIN_DEPTH_M,
    max_depth_m=DEPTH_CLEARANCE_MAX_DEPTH_M,
    sensor_keys=DEPTH_CLEARANCE_SENSOR_KEYS,
    camera_orientations_deg=None,
    row_range=None,
    col_stride=2,
    hfov_deg=DEFAULT_OCC_HFOV_DEG,
    camera_height_m=DEFAULT_OCC_CAMERA_HEIGHT_M,
    floor_margin_m=_OCC_FLOOR_MARGIN_M,
):
    """Project the depth panorama into a sparse 2D point cloud in the agent's current ego
    frame (x forward = sensor_keys[0]'s direction, y left). One point per subsampled
    column, at the closest valid non-floor surface in that column's vertical sliver.
    Floor is dropped by a per-pixel check against the pinhole floor-intercept depth
    z_floor(v) = camera_height_m * fy / (v - cy), not a blunt row cutoff, so close low
    obstacles survive. Returns [ego_forward_m, ego_left_m] pairs (each rounded to 3 dp),
    or [] if no depth sensor was present. Verbatim: the pre-release harness,
    except that the pinhole intrinsics come from each frame's own (height, width) instead
    of a fixed 256x256: a 640x480 VLA camera projects correctly, and at 256x256 every
    constant reduces to the original one (tests/test_geometry.py golden cases).
    `row_range` None = the (16, 240) band of a 256-row image, scaled to the frame height.
    """
    camera_orientations_deg = camera_orientations_deg or (
        [0.0] + [30.0 * k for k in range(1, len(sensor_keys))]
    )
    tan_half_hfov = math.tan(math.radians(hfov_deg / 2.0))
    floor_cache = {}  # (height, width) -> (fx, cx, cy, lo, hi, z_floor_per_row)

    def _intrinsics(height, width):
        if (height, width) not in floor_cache:
            fx = (width / 2.0) / tan_half_hfov
            fy = fx  # square pixels
            cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
            if row_range is None:
                lo = int(round(_OCC_ROW_RANGE[0] * height / float(_OCC_IMG_SIZE)))
                hi = int(round(_OCC_ROW_RANGE[1] * height / float(_OCC_IMG_SIZE)))
            else:
                lo, hi = row_range
            dv = np.arange(lo, hi) - cy  # > 0 => row looks below the horizon (floor-ward)
            with np.errstate(divide="ignore"):
                z_floor = np.where(
                    dv > 0, camera_height_m * fy / np.maximum(dv, 1e-6), np.inf)
            floor_cache[(height, width)] = (fx, cx, cy, lo, hi, z_floor)
        return floor_cache[(height, width)]

    points = []
    for key, yaw_deg in zip(sensor_keys, camera_orientations_deg):
        frame = obs.get(key) if obs is not None else None
        if frame is None:
            continue
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        fx, cx, _, lo, hi, z_floor_per_row = _intrinsics(arr.shape[0], arr.shape[1])
        band = arr[lo:hi, :]
        yaw = math.radians(yaw_deg)
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        # The per-column masks and minimum, for all subsampled columns at once (the
        # archived loop did one column per iteration; same elementwise float64 ops, same
        # min, same column order, so the points are bit-identical --
        # tests/test_geometry.py pins this against the loop). The per-point arithmetic
        # below stays scalar Python, as before.
        raw = band[:, ::col_stride].astype(np.float64)
        z_cam_cols = raw * (max_depth_m - min_depth_m) + min_depth_m
        keep = ((raw > 1e-6)
                & (z_cam_cols < (z_floor_per_row - floor_margin_m)[:, None])
                & (z_cam_cols < max_depth_m - 1e-6))
        z_min = np.where(keep, z_cam_cols, np.inf).min(axis=0)
        for j in np.flatnonzero(keep.any(axis=0)):
            u = int(j) * col_stride
            z_cam = float(z_min[j])
            x_cam = (u - cx) / fx * z_cam
            ego_forward = x_cam * syaw + z_cam * cyaw
            ego_left = z_cam * syaw - x_cam * cyaw
            points.append([round(float(ego_forward), 3), round(float(ego_left), 3)])
    return points


def occupancy_min_clearance(x_fwd, y_left, occupancy_points_m, agent_radius_m):
    """Nearest-surface distance from ego point (x_fwd, y_left) to the occupancy cloud,
    minus agent_radius_m (so the result is true clearance, not raw surface distance).
    None if no points. Verbatim: etpnav_policy_worker.py:200-213."""
    if not occupancy_points_m:
        return None
    pts = np.asarray(occupancy_points_m, dtype=np.float64)
    d = np.hypot(pts[:, 0] - x_fwd, pts[:, 1] - y_left)
    return float(d.min()) - agent_radius_m


def turn_actions_for_angle(ang_rad, turn_unit_deg, actions):
    """Angle (base-heading-relative radians) -> a list of TURN_LEFT / TURN_RIGHT ints and
    the actually-applied angle in radians (angle is snapped to turn_unit_deg first).
    `actions` needs int-able MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP attributes.
    Verbatim: etpnav_policy_worker.py:182-197 / hop_geometry.py:70-83."""
    left_action = int(actions.TURN_LEFT)
    right_action = int(actions.TURN_RIGHT)
    ang_degree = math.degrees(ang_rad)
    ang_degree = round(ang_degree / turn_unit_deg) * turn_unit_deg
    if 180 < ang_degree <= 360:
        ang_degree -= 360
    if ang_degree >= 0:
        turns = [left_action] * int(ang_degree // turn_unit_deg)
    else:
        turns = [right_action] * int(-ang_degree // turn_unit_deg)
    applied_rad = math.radians(ang_degree)
    return turns, applied_rad


def predicted_clearance_by_primitive_step(
    primitive_actions, turn_unit_deg, forward_unit_m, occupancy_points_m, actions,
    agent_radius_m=DEFAULT_AGENT_RADIUS_M,
):
    """Walk the exact decomposed primitive sequence about to run, accumulating true 2D
    ego position from the snapshot origin/heading, and at each step report the
    occupancy-cloud clearance there. Returns a list the length of primitive_actions, or
    None if no cloud. Verbatim: etpnav_policy_worker.py:216-260."""
    if not occupancy_points_m:
        return None
    left_action = int(actions.TURN_LEFT)
    right_action = int(actions.TURN_RIGHT)
    forward_action = int(actions.MOVE_FORWARD)
    heading_deg = 0.0
    x_fwd, y_left = 0.0, 0.0
    out = []
    for action in primitive_actions:
        if action == left_action:
            heading_deg += turn_unit_deg
        elif action == right_action:
            heading_deg -= turn_unit_deg
        elif action == forward_action:
            heading_rad = math.radians(heading_deg)
            x_fwd += forward_unit_m * math.cos(heading_rad)
            y_left += forward_unit_m * math.sin(heading_rad)
        out.append(occupancy_min_clearance(x_fwd, y_left, occupancy_points_m, agent_radius_m))
    return out


def bin_clearance_toward_point(
    point, cur_pos, cur_ori_arr, occupancy_points_m, agent_radius_m=DEFAULT_AGENT_RADIUS_M,
):
    """Occupancy clearance toward a target point (headroom sub-analysis). None if point
    is None, no cloud, or the point is coincident with cur_pos. Verbatim:
    etpnav_policy_worker.py:263-285 (habitat imports replaced by this module's own
    heading_from_quaternion / calculate_vp_rel_pos)."""
    if point is None or not occupancy_points_m:
        return None
    heading = float(heading_from_quaternion(np.array(cur_ori_arr)))
    ang, dist = calculate_vp_rel_pos(list(cur_pos), point, heading)
    if dist < 1e-6:
        return None
    x_fwd = dist * math.cos(ang)
    y_left = dist * math.sin(ang)
    return occupancy_min_clearance(x_fwd, y_left, occupancy_points_m, agent_radius_m)


def decompose_hops_to_primitives(
    hops, cur_pos, cur_ori_arr, turn_unit_deg, forward_unit_m, actions,
    care_rotation_rad=0.0,
):
    """A list of hop target positions (x, y, z) -> a flat list of primitive action ints,
    one hop's worth of turn-then-forward each (matching single_step_control /
    MoveHighToLowActionEval). care_rotation_rad, added to the FIRST hop's angle, defaults
    to 0.0 and reproduces the worker's own output exactly. Verbatim:
    etpnav_policy_worker.py:288-316 + hop_geometry.py:99-127."""
    forward_action = int(actions.MOVE_FORWARD)
    virtual_pos = list(cur_pos)
    virtual_heading = float(heading_from_quaternion(np.array(cur_ori_arr)))

    out = []
    for i, hop_pos in enumerate(hops):
        ang, dist = calculate_vp_rel_pos(virtual_pos, hop_pos, virtual_heading)
        if i == 0 and care_rotation_rad:
            ang = ang + care_rotation_rad
        turns, applied_rad = turn_actions_for_angle(ang, turn_unit_deg, actions)
        out.extend(turns)
        virtual_heading = (virtual_heading + applied_rad) % (2 * math.pi)
        ksteps = int(dist // forward_unit_m)
        out.extend([forward_action] * ksteps)
        virtual_pos = list(hop_pos)
    return out


def hop_segments(hops, cur_pos, cur_ori_arr, turn_unit_deg, forward_unit_m, actions):
    """decompose_hops_to_primitives (care_rotation_rad = 0) kept per hop: one
    (turn_actions, forward_steps) pair per hop. Flattening the pairs gives exactly its
    output (tests/test_geometry.py). vlnsafe.agents.tryout needs the hop boundaries,
    because Tryout acts on the forward steps of one hop."""
    virtual_pos = list(cur_pos)
    virtual_heading = float(heading_from_quaternion(np.array(cur_ori_arr)))
    out = []
    for hop_pos in hops:
        ang, dist = calculate_vp_rel_pos(virtual_pos, hop_pos, virtual_heading)
        turns, applied_rad = turn_actions_for_angle(ang, turn_unit_deg, actions)
        virtual_heading = (virtual_heading + applied_rad) % (2 * math.pi)
        out.append((turns, int(dist // forward_unit_m)))
        virtual_pos = list(hop_pos)
    return out


def first_hop_relative_offset(hops, cur_pos, cur_ori_arr):
    """Local (x_forward, y_left) metric offset of the first hop, positive angle = turn
    left. None if hops is empty. Verbatim: hop_geometry.py:86-96."""
    if not hops:
        return None
    virtual_heading = float(heading_from_quaternion(np.array(cur_ori_arr)))
    ang, dist = calculate_vp_rel_pos(list(cur_pos), hops[0], virtual_heading)
    return float(dist * math.cos(ang)), float(dist * math.sin(ang))
