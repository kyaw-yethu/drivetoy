"""NAVSIM inputs built from a DriveToy sample, and the NAVSIM code that tracks and scores plans on them.

DriveToy runs at 5 Hz and stores the centre of each body; NAVSIM runs at 10 Hz on the rear axle. A
DriveToy vehicle has its axles 0.35 L before and behind its centre (sim.Vehicle.pose), which fixes
the vehicle parameters of the ego. Road users replay their recorded motion, interpolated to 10 Hz.
"""
import warnings

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import LineString, Polygon, box

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, StateIndex, WeightedMetricIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

FPS = 5.0
T0 = 31                       # last observed frame
PLAN_FRAMES = 20              # 4 s
EGO = 0
AXLE = 0.35                   # axle offset from the centre, in body lengths
LANE_HALF = 1.75
SAMPLING = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
PLAN_SAMPLING = TrajectorySampling(time_horizon=4.0, interval_length=1 / FPS)
OBS_STEPS = SAMPLING.num_poses + 10   # NAVSIM looks 1 s further for TTC
VEH_TYPES, PED_TYPES = (1, 2, 3, 4), (5, 6)
TERMS = ('NC', 'DAC', 'TTC', 'HC')
_COLUMNS = dict(NC='no_at_fault_collisions', DAC='drivable_area_compliance',
                TTC='time_to_collision_within_bound', HC='history_comfort')


def vehicle_parameters(L, W, H):
    return VehicleParameters(width=float(W), front_length=(0.5 + AXLE) * L, rear_length=(0.5 - AXLE) * L,
                             cog_position_from_rear_axle=AXLE * L, wheel_base=2 * AXLE * L,
                             vehicle_name='drivetoy_ego', vehicle_type='car', height=float(H))


def body_length(vp):
    return vp.wheel_base / (2 * AXLE)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ego_states(ag, frames, vp):
    """nuPlan ego states of the recorded ego at the given frames (agents slot 0 rows: type, x, y,
    heading, speed, acceleration, L, W, H, valid). Yaw rate and steering come from the heading change
    since the previous frame, so a state uses nothing after its own frame."""
    L = body_length(vp)
    h = np.unwrap(ag[:, EGO, 3].astype(np.float64))
    out = []
    for f in frames:
        x, y, hd, v, a = (float(ag[f, EGO, i]) for i in (1, 2, 3, 4, 5))
        rate = (h[f] - h[f - 1]) * FPS if f > 0 else 0.0
        steer = float(np.arctan(vp.wheel_base * rate / v)) if v > 0.5 else 0.0
        rear = StateSE2(x - AXLE * L * np.cos(hd), y - AXLE * L * np.sin(hd), hd)
        out.append(EgoState.build_from_rear_axle(
            rear_axle_pose=rear, rear_axle_velocity_2d=StateVector2D(v, 0.0),
            rear_axle_acceleration_2d=StateVector2D(a, 0.0), tire_steering_angle=steer,
            time_point=TimePoint(int(round(f / FPS * 1e6))), vehicle_parameters=vp, angular_vel=rate))
    return out


def plans_to_rear_axle(paths, L):
    """(P, 20, 3) centre poses in the ego frame at the origin of the plan, as rear-axle poses in the
    frame of the rear axle at the origin, which is the planning frame of NAVSIM."""
    x, y, h = paths[..., 0], paths[..., 1], paths[..., 2]
    rx, ry = x - AXLE * L * np.cos(h), y - AXLE * L * np.sin(h)
    return np.stack([rx + AXLE * L, ry, h], -1)


def track(paths, initial, simulator):
    """The NAVSIM conversion of 4 s plans: interpolated to 10 Hz, then LQR tracking with the kinematic
    bicycle model from the initial state. Returns (P, 41, 11) state arrays."""
    rear = plans_to_rear_axle(np.asarray(paths, np.float64), body_length(initial.car_footprint.vehicle_parameters))
    states = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(p, PLAN_SAMPLING), initial),
                                               SAMPLING, initial.time_point) for p in rear])
    return simulator.simulate_proposals(states, initial)


def centres(states, vp):
    """Centre x, y, heading and speed of tracked states at the DriveToy frames after the origin."""
    s = states[:, 2::2]
    L = body_length(vp)
    h = s[..., StateIndex.HEADING]
    x = s[..., StateIndex.X] + AXLE * L * np.cos(h)
    y = s[..., StateIndex.Y] + AXLE * L * np.sin(h)
    v = np.hypot(s[..., StateIndex.VELOCITY_X], s[..., StateIndex.VELOCITY_Y])
    return np.stack([x, y, h, v], -1)


def drivable_map(m):
    """Lanes (LANE; connectors LANE_CONNECTOR) as their centrelines widened to 3.5 m, the road and
    school-zone pavement as DRIVABLE_AREA, and the junction square as INTERSECTION."""
    tokens, types, polys = [], [], []
    for ln in m.lanes:
        tokens.append(f'lane_{ln.id}')
        types.append(SemanticMapLayer.LANE_CONNECTOR if ln.kind == 'conn' else SemanticMapLayer.LANE)
        polys.append(LineString(np.asarray(ln.pts, np.float64)).buffer(LANE_HALF, cap_style=2))
    for i, s in enumerate(m.surfaces):
        if s['kind'] in ('road', 'school'):
            tokens.append(f'surface_{i}')
            types.append(SemanticMapLayer.DRIVABLE_AREA)
            polys.append(Polygon(np.asarray(s['poly'], np.float64)).buffer(0))
    if getattr(m, 'intersection', None):
        c, half = np.asarray(m.intersection['center'], np.float64), float(m.intersection['half'])
        tokens.append('intersection')
        types.append(SemanticMapLayer.INTERSECTION)
        polys.append(box(c[0] - half, c[1] - half, c[0] + half, c[1] + half))
    return PDMDrivableMap(tokens, types, np.array(polys, dtype=object))


def centreline(m, lane_ids):
    pts = np.concatenate([np.asarray(m.lanes[i].pts, np.float64) for i in lane_ids])
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3]
    pts = pts[keep]
    d = np.diff(pts, axis=0)
    head = np.arctan2(d[:, 1], d[:, 0])
    head = np.r_[head, head[-1]]
    return PDMPath([StateSE2(x, y, h) for (x, y), h in zip(pts, head)])


def _box(x, y, h, L, W, H):
    return OrientedBox(StateSE2(float(x), float(y), float(h)), float(L), float(W), float(H))


def detections(ag, statics, frame0=T0, steps=OBS_STEPS):
    """Road users of one future at 10 Hz from frame0. Odd steps lie halfway between two frames and
    keep only bodies present in both. The ego is left out; map statics stand still (parked cars as
    vehicles, cones as cones)."""
    t0_us = int(round(frame0 / FPS * 1e6))
    fixed = []
    for i, s in enumerate(statics):
        L, W, H = (float(v) for v in s['size'])
        meta = SceneObjectMetadata(timestamp_us=t0_us, token=f'static_{i}', track_id=None, track_token=f'static_{i}')
        b = _box(s['pos'][0], s['pos'][1], s['heading'], L, W, H)
        if s['kind'] == 'cone':
            fixed.append(StaticObject(TrackedObjectType.TRAFFIC_CONE, b, meta))
        else:
            fixed.append(Agent(TrackedObjectType.VEHICLE, b, StateVector2D(0.0, 0.0), meta))
    out = []
    for k in range(steps + 1):
        f, half = frame0 + k // 2, k % 2
        objs = list(fixed)
        for a in range(1, ag.shape[1]):
            r0 = ag[f, a]
            if r0[9] <= 0 or r0[0] not in VEH_TYPES + PED_TYPES:
                continue
            if half:
                r1 = ag[f + 1, a]
                if r1[9] <= 0 or r1[0] != r0[0]:
                    continue
                x, y = (r0[1:3] + r1[1:3]) / 2
                h, v = r0[3] + _wrap(r1[3] - r0[3]) / 2, (r0[4] + r1[4]) / 2
            else:
                x, y, h, v = r0[1], r0[2], r0[3], r0[4]
            tok = f'{a}_{int(r0[0])}_{r0[6]:.2f}'
            meta = SceneObjectMetadata(timestamp_us=t0_us + int(k * 1e5), token=f'{tok}_{k}', track_id=a, track_token=tok)
            kind = TrackedObjectType.VEHICLE if r0[0] in VEH_TYPES else TrackedObjectType.PEDESTRIAN
            objs.append(Agent(kind, _box(x, y, h, r0[6], r0[7], r0[8]),
                              StateVector2D(float(v * np.cos(h)), float(v * np.sin(h))), meta))
        out.append(DetectionsTracks(TrackedObjects(objs)))
    return out


def observation(tracks):
    obs = PDMObservation(SAMPLING, SAMPLING, map_radius=100.0, observation_sample_res=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        obs.update_detections_tracks(tracks)
    return obs


class _Scorer(PDMScorer):
    """The NAVSIM scorer without the terms this evaluation leaves out (DDC, TLC, EP and LK)."""
    def _calculate_driving_direction_compliance(self):
        self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION] = 1.0

    def _calculate_traffic_light_compliance(self):
        self._multi_metrics[MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE] = 1.0

    def _calculate_progress(self):
        self._progress_raw = np.zeros(self._num_proposals)

    def _calculate_lane_keeping(self):
        self._weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = 1.0


def score(states, obs, line, route_ids, dmap, past, vp):
    """NC, DAC, TTC and history comfort by the NAVSIM scorer for tracked states (P, 41, 11): (P, 4)."""
    scorer = _Scorer(SAMPLING, PDMScorerConfig(human_penalty_filter=False), vehicle_parameters=vp)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows = scorer.score_proposals(states, obs, line, route_ids, dmap, human_past_trajectory=past)
    return np.array([[float(r[_COLUMNS[t]].iloc[0]) for t in TERMS] for r in rows])


def past_trajectory(ag, vp, first=24, last=T0):
    return InterpolatedTrajectory(ego_states(ag, range(first, last + 1), vp))


def simulator():
    return PDMSimulator(SAMPLING)
