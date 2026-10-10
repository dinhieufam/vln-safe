"""Camera rigs, each a named entry an agent config can request by `sensor_rig: <name>`.

What: builds the rigs an agent config can name by `sensor_rig:` —
  - `etpnav_12view`: 12 RGB + 12 depth sensors at 30-degree increments (uuids
    rgb/depth, rgb_30/depth_30, ...), the panoramic rig ETPNav's checkpoint expects
    (RGB size overridable by the agent's `rig:` block, for g3D-LF's 336);
  - `vla` (StreamVLN / NaVILA): one forward RGB + DEPTH at the model's resolution, plus
    GPS + COMPASS;
  - `omninav`: three RGB views (front / left / right) at OmniNav's camera, plus the
    benchmark's forward DEPTH;
  - `vlnce_recurrent` (CMA / Seq2Seq) uses VLN-CE's own base rig untouched, so it has no
    builder here;
  - the clearance-only panoramic depth ring (uuid clearance_depth_<deg>, MIN/MAX 0/10 m),
    independent of whatever range the policy's own depth input uses. Read locally in
    rollout for step_clearance_m; never added to the worker protocol payload.

Who reads it: vlnsafe.simulator.environment (adds the rig the run's agent asked for).

Imports the habitat task-config objects only. Must not import torch.

Ported verbatim from etpnav_policy_worker.py::build_etpnav_camera_sensors /
get_camera_orientations12_standalone and the pre-release harness::
build_clearance_depth_sensors.
"""
from copy import deepcopy

# 12 sensors, 30 degrees apart; raw depth obs in [0, 1], NORMALIZE_DEPTH=True, R2R uses
# habitat-lab's default 0/10 m range.
DEPTH_CLEARANCE_SENSOR_KEYS = ["clearance_depth_0"] + [
    "clearance_depth_{}".format(30 * k) for k in range(1, 12)
]


def get_camera_orientations12():
    """Pure-python copy of vlnce_baselines/common/utils.py::get_camera_orientations12."""
    base_angle_rad = 3.141592653589793 / 6
    return {str(30 * k): [0.0, base_angle_rad * k, 0.0] for k in range(1, 12)}


def build_etpnav_rig(task_config):
    """RGB_SENSOR/DEPTH_SENSOR + 11 copies each at 30-degree increments (uuids
    rgb_<deg>/depth_<deg>), mirroring ss_trainer_ETP.py::RLTrainer._set_config so a
    separate env can be built with the identical rig ETPNav's checkpoint was evaluated
    with. Returns the appended AGENT_0.SENSORS names."""
    camera_orientations = get_camera_orientations12()
    appended = []
    for sensor_type in ["RGB", "DEPTH"]:
        base_sensor = getattr(task_config.SIMULATOR, "{}_SENSOR".format(sensor_type))
        for action, orient in camera_orientations.items():
            template = "{}_{}".format(sensor_type, action)
            camera_config = deepcopy(base_sensor)
            camera_config.ORIENTATION = orient
            camera_config.UUID = template.lower()
            setattr(task_config.SIMULATOR, template, camera_config)
            task_config.SIMULATOR.AGENT_0.SENSORS.append(template)
            appended.append(template)
    return appended


def apply_rgb_size(task_config, rig=None):
    """Resize RGB_SENSOR from an agent's `rig: {rgb: {width, height}}` block before a rig
    is cloned from it (g3D-LF's 12 views are 336x336; docs/agent_notes/g3dlf.md). No
    block, or no rgb entry, leaves the sensor exactly as the benchmark config set it."""
    spec = (rig or {}).get("rgb") or {}
    sensor = task_config.SIMULATOR.RGB_SENSOR
    if spec.get("width") is not None:
        sensor.WIDTH = int(spec["width"])
    if spec.get("height") is not None:
        sensor.HEIGHT = int(spec["height"])


def build_vla_rig(task_config, rig=None):
    """The rig for a navigation VLA (StreamVLN / NaVILA): one forward RGB + one forward
    DEPTH, plus GPS + COMPASS in the task sensors (the VLAs condition on pose).

    `rig` is the agent config's own `rig:` block -- the VLA was trained at a specific
    camera and its SR degrades at any other, so the numbers live in configs/agents/<id>.yaml
    and never here. Shape:

        rig:
          rgb:   {width: 512, height: 512, hfov_deg: 90}
          depth: {width: 256, height: 256, min_m: 0.0, max_m: 10.0}
          camera_height_m: 1.25

    RGB and depth are sized separately because they genuinely differ (NaVILA is 512 RGB /
    256 depth). Anything absent is left exactly as the benchmark config set it. Forward
    step and turn angle are the benchmark's, not the agent's.

    Returns the AGENT_0.SENSORS / TASK.SENSORS names it set."""
    rig = rig or {}
    sim = task_config.SIMULATOR
    sim.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    for name, key in (("RGB_SENSOR", "rgb"), ("DEPTH_SENSOR", "depth")):
        spec = rig.get(key) or {}
        sensor = getattr(sim, name)
        if spec.get("width") is not None:
            sensor.WIDTH = int(spec["width"])
        if spec.get("height") is not None:
            sensor.HEIGHT = int(spec["height"])
        if spec.get("hfov_deg") is not None:
            sensor.HFOV = int(spec["hfov_deg"])
        if spec.get("min_m") is not None:
            sensor.MIN_DEPTH = float(spec["min_m"])
        if spec.get("max_m") is not None:
            sensor.MAX_DEPTH = float(spec["max_m"])
        if rig.get("camera_height_m") is not None:
            position = list(sensor.POSITION)
            position[1] = float(rig["camera_height_m"])
            sensor.POSITION = position

    task_sensors = [s for s in task_config.TASK.SENSORS if "INSTRUCTION_SENSOR" in s]
    for name in ("GPS_SENSOR", "COMPASS_SENSOR"):
        if name not in task_sensors:
            task_sensors.append(name)
    task_config.TASK.SENSORS = task_sensors
    return list(sim.AGENT_0.SENSORS) + list(task_sensors)


# OmniNav's tri-view: uuid -> yaw (rad), from its vlnce_task_r2r_pano.yaml (RGB_SENSOR_LEFT
# ORIENTATION [0, 1.5707963, 0], RIGHT [0, -1.5707963, 0]). Its BACK camera is rendered
# upstream but never read by the agent, so it is not built here.
OMNINAV_VIEWS = (("rgb_front", 0.0), ("rgb_left", 1.5707963), ("rgb_right", -1.5707963))


def build_omninav_rig(task_config, rig=None):
    """OmniNav (docs/agent_notes/omninav.md): three RGB cameras -- front, left, right, 90
    degrees apart -- at the size / HFOV / height of the agent's `rig: {rgb: ...}` block
    (720x640, HFOV 110, 1.25 m upstream), plus the benchmark's own forward DEPTH_SENSOR
    untouched (the worker never sees depth; rollout's step_clearance_m reads it, as for the
    other VLAs). No GPS/COMPASS: OmniNav's prompt carries no pose. Returns the
    AGENT_0.SENSORS names it set."""
    spec = (rig or {}).get("rgb") or {}
    sim = task_config.SIMULATOR
    base = sim.RGB_SENSOR
    names = []
    for uuid, yaw in OMNINAV_VIEWS:
        name = uuid.upper()
        camera = deepcopy(base)
        if spec.get("width") is not None:
            camera.WIDTH = int(spec["width"])
        if spec.get("height") is not None:
            camera.HEIGHT = int(spec["height"])
        if spec.get("hfov_deg") is not None:
            camera.HFOV = int(spec["hfov_deg"])
        if (rig or {}).get("camera_height_m") is not None:
            position = list(camera.POSITION)
            position[1] = float(rig["camera_height_m"])
            camera.POSITION = position
        camera.ORIENTATION = [0.0, yaw, 0.0]
        camera.UUID = uuid
        setattr(sim, name, camera)
        names.append(name)
    sim.AGENT_0.SENSORS = names + ["DEPTH_SENSOR"]
    task_config.TASK.SENSORS = [s for s in task_config.TASK.SENSORS
                                if "INSTRUCTION_SENSOR" in s]
    return list(sim.AGENT_0.SENSORS) + list(task_config.TASK.SENSORS)


def build_clearance_ring(task_config, min_depth_m=0.0, max_depth_m=10.0, base_sensor=None):
    """A second, clearance-only panoramic depth rig (uuid clearance_depth_<deg>)
    independent of the policy's own DEPTH_SENSOR range. Needed for RxR, whose policy
    depth input is floored at 0.5 m — reusing it for clearance collapses every reading
    below 0.5 m to the same value. Deep-copies whatever the base DEPTH_SENSOR is (so RxR's
    HFOV 63 is inherited), overriding only MIN/MAX/NORMALIZE/UUID. The worker never sees
    these sensors. `base_sensor` (default: the current DEPTH_SENSOR) lets a caller build the
    ring from the benchmark's own depth camera rather than an agent's rig override."""
    orientations = dict(get_camera_orientations12())
    orientations.setdefault("0", [0.0, 0.0, 0.0])
    if base_sensor is None:
        base_sensor = task_config.SIMULATOR.DEPTH_SENSOR
    appended = []
    for action, orient in orientations.items():
        template = "CLEARANCE_DEPTH_{}".format(action)
        camera_config = deepcopy(base_sensor)
        camera_config.ORIENTATION = orient
        camera_config.MIN_DEPTH = min_depth_m
        camera_config.MAX_DEPTH = max_depth_m
        camera_config.NORMALIZE_DEPTH = True
        camera_config.UUID = template.lower()
        setattr(task_config.SIMULATOR, template, camera_config)
        task_config.SIMULATOR.AGENT_0.SENSORS.append(template)
        appended.append(template)
    return appended
