"""External depth-based clearance predictor for agents that do not compute one themselves.

What: given the depth observation the agent is about to act on and the primitive it
proposed, project the depth into a 2D occupancy cloud and report the clearance at the pose
that primitive ends in -- the same quantity the ETPNav-lineage worker logs as
`predicted_clearance_m`, produced by the same functions in vlnsafe.geometry. It is a pure
function of (obs, primitive): the live rollout and the replay pass over a logged episode
(scripts/replay_geometry.py) call it identically, which is what lets the T1 geometry
column be recovered without re-running the agent.

Who reads it: vlnsafe.cli (built for every provider that does not send an occupancy cloud
to a worker), vlnsafe.rollout (one call per step), vlnsafe.guard (its `points`),
scripts/replay_geometry.py.

Every camera constant comes from the merged habitat config, never a default: R2R and RxR
differ in HFOV, camera height and depth range, and a VLA's camera differs again
(StreamVLN is 640x480, HFOV 79). Python 3.6 compatible; imports no habitat at module scope.
"""
from .geometry import (
    predicted_clearance_by_primitive_step, project_depth_to_occupancy_points_m,
)

# Default keys: the single forward camera. cli.py and the replay pass do NOT use this for
# T1 agents -- one forward camera cannot see a wall the agent hugs sideways, and the
# predictor then under-reports contact -- they pass the 12-camera clearance ring instead.
_FORWARD_DEPTH_KEYS = ("depth",)


class PrimitiveClearancePredictor(object):
    def __init__(self, turn_unit_deg, forward_unit_m, agent_radius_m, hfov_deg,
                 camera_height_m, sensor_keys, min_depth_m, max_depth_m, actions):
        self.turn_unit_deg = float(turn_unit_deg)
        self.forward_unit_m = float(forward_unit_m)
        self.agent_radius_m = float(agent_radius_m)
        self.hfov_deg = float(hfov_deg)
        self.camera_height_m = float(camera_height_m)
        self.sensor_keys = list(sensor_keys)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.actions = actions   # int-able MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP
        self._cached_obs = None
        self._cached_points = None

    @classmethod
    def from_habitat_config(cls, habitat_config, sensor_keys=None,
                            sensor_name="DEPTH_SENSOR"):
        """HFOV, camera height and depth range are read from the habitat sensor named
        `sensor_name` -- the one that produced `sensor_keys`' frames. `sensor_keys` None =
        the single forward "depth" camera; T1 agents pass the clearance ring's keys and
        "CLEARANCE_DEPTH_0" (geometry.CLEARANCE_DEPTH_SENSOR_KEYS)."""
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        sim = habitat_config.TASK_CONFIG.SIMULATOR
        cam = getattr(sim, sensor_name)
        return cls(
            turn_unit_deg=sim.TURN_ANGLE,
            forward_unit_m=sim.FORWARD_STEP_SIZE,
            agent_radius_m=sim.AGENT_0.RADIUS,
            hfov_deg=cam.HFOV,
            camera_height_m=cam.POSITION[1],
            sensor_keys=sensor_keys or _FORWARD_DEPTH_KEYS,
            min_depth_m=cam.MIN_DEPTH,
            max_depth_m=cam.MAX_DEPTH,
            actions=HabitatSimActions,
        )

    def points(self, obs):
        """The frame's ego-frame occupancy cloud. Cached on the identity of `obs`: the
        rollout's logged column and the guard (vlnsafe.guard) project the same pre-step
        frame, and env.step returns a fresh dict every step, so the cache never serves a
        stale frame."""
        if obs is not self._cached_obs:
            self._cached_points = project_depth_to_occupancy_points_m(
                obs,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
                sensor_keys=self.sensor_keys,
                hfov_deg=self.hfov_deg,
                camera_height_m=self.camera_height_m,
            )
            self._cached_obs = obs
        return self._cached_points

    def __call__(self, obs, primitive):
        """Clearance (m) at the pose `primitive` ends in, or None when the depth frame
        yields no obstacle point (same convention as the ETPNav worker: None ->
        UNAVAILABLE in the step log)."""
        out = predicted_clearance_by_primitive_step(
            [primitive], self.turn_unit_deg, self.forward_unit_m, self.points(obs),
            self.actions, agent_radius_m=self.agent_radius_m,
        )
        return None if out is None else out[0]
