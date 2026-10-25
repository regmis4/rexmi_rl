"""Small, offline-fitted leg predictor and episode-level mission evidence.

No policy inference, simulator imports, or training in the control loop. A model
without held-out evidence cannot authorize a probability-qualified approach.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import beta


FEATURES = ('length_m', 'max_uphill_deg', 'max_downhill_deg', 'max_side_deg',
            'steep_length_m', 'roughness_m', 'clearance_m', 'turn_radians',
            'observed_fraction', 'start_speed', 'start_up_z', 'net_height_m',
            'shelf_endpoint', 'command_mps')
VERSION = 2


@dataclass(frozen=True)
class CapabilityProfile:
    slope_speed: float = .4
    max_ground_deg: float = 35.
    flat_deg: float = 5.
    normal_probability: float = .8
    fallback_probability: float = .6
    # Stage 1 deliberately does not authorize high-speed flat travel.
    flat_speed: float = .4
    preferred_uphill_deg: float = 20.
    uphill_limit_deg: float = 25.
    side_limit_deg: float = 25.

    def __post_init__(self):
        if not (0 < self.slope_speed <= .4 and self.flat_speed == self.slope_speed
                and 0 < self.max_ground_deg <= 35
                and 0 < self.preferred_uphill_deg <= self.uphill_limit_deg <= self.max_ground_deg
                and 0 < self.side_limit_deg <= self.max_ground_deg):
            raise ValueError('Ascent qualification requires a <=0.40 m/s ceiling everywhere and <=35 degrees')


@dataclass
class ReachabilityEstimate:
    probability: float | None
    interval: tuple | None
    validated: bool
    reason: str
    successes: int = 0
    failures: int = 0
    prior: float | None = None


@dataclass
class AttemptOutcome:
    attempt_id: str
    run_id: str
    features: list
    start_time: float
    end_time: float
    start: list
    endpoint: list
    position: list
    heading: float
    outcome: str
    reason: str
    prediction: dict
    updated_prediction: dict
    approach_key: str
    event: str = 'approach_outcome'
    feature_version: int = VERSION
    physical: bool = False
    terminal_maneuver: str | None = None
    executed_traversal: dict | None = None
    exclusion_applied: bool = False
    exclusion_reason: str = ''


@dataclass
class NavigationStatus:
    phase: str
    reason: str
    command_limit: float
    reachability: dict | None
    mode: str


def route_features(grid, path, yaw=0., speed=0., up_z=1., command=.4):
    """Features of the entire leg, with densely sampled crossed terrain."""
    length = uphill = downhill = side = steep = turns = 0.
    cells, along, prior_heading = [], [], yaw+(math.pi if command < 0 else 0.)
    for a, b in zip(path, path[1:]):
        delta = np.subtract(b, a); distance = float(np.linalg.norm(delta))
        if distance < 1e-8:
            continue
        direction = delta / distance
        heading = math.atan2(delta[1], delta[0])
        turns += abs(math.atan2(math.sin(heading-prior_heading), math.cos(heading-prior_heading)))
        prior_heading = heading
        samples = max(1, math.ceil(distance / (grid.cell_size*.5)))
        for t in np.linspace(0., 1., samples+1):
            point = np.asarray(a)+t*delta
            cell = grid.world_to_cell(*point)
            cells.append(cell)
            if cell is None or not grid.known[cell]:
                continue
            gx, gy = grid.gx[cell], grid.gy[cell]
            grade = math.degrees(math.atan(gx*direction[0]+gy*direction[1]))
            cross = abs(math.degrees(math.atan(-gx*direction[1]+gy*direction[0])))
            uphill, downhill, side = max(uphill, grade), max(downhill, -grade), max(side, cross)
            along.append(grade)
            if grade > 20:
                steep += distance/(samples+1)
        length += distance
    known = [c for c in cells if c is not None and grid.known[c]]
    rough = [float(grid.roughness[c]) for c in known if np.isfinite(grid.roughness[c])]
    clearance = min((float(grid.obstacle_distance[c])-grid.footprint_radius for c in known), default=0.)
    start, end = grid.world_to_cell(*path[0]), grid.world_to_cell(*path[-1])
    dz = float(grid.surface[end]-grid.surface[start]) if start is not None and end is not None else 0.
    shelf = bool(getattr(grid, 'shelf_mask', np.zeros_like(grid.known))[end]) if end is not None else False
    return [length, uphill, downhill, side, steep, max(rough, default=.1),
            min(3., max(0., clearance)), turns, len(known)/max(1, len(cells)),
            speed, up_z, dz if math.isfinite(dz) else 0., float(shelf), command]


class ReachabilityModel:
    def __init__(self, artifact=None):
        self.artifact = artifact
        self.model_id = None
        if artifact is not None:
            if artifact.get('version') != VERSION or artifact.get('features') != list(FEATURES):
                raise ValueError('Incompatible reachability feature version')
            for field in ('mean', 'scale', 'coefficients', 'minimum', 'maximum'):
                value = np.asarray(artifact[field], float)
                if value.shape != (len(FEATURES),) or not np.isfinite(value).all():
                    raise ValueError('Invalid model '+field)
            if np.any(np.asarray(artifact['scale']) <= 0):
                raise ValueError('Invalid feature scales')
            scalars = [artifact['intercept'], *artifact['calibration']]
            if len(artifact['calibration']) != 2 or not np.isfinite(scalars).all():
                raise ValueError('Invalid model calibration')
            groups = artifact.get('run_groups', {})
            splits = [set(groups.get(k, [])) for k in ('train', 'calibrate', 'evaluate')]
            if not all(splits) or any(splits[i] & splits[j] for i in range(3) for j in range(i)):
                raise ValueError('Training, calibration and evaluation runs must be nonempty and disjoint')
            self.model_id = hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, features):
        a = self.artifact
        if a is None:
            return None, 'no calibrated model; characterization only'
        x = np.asarray(features, float)
        if x.shape != (len(FEATURES),) or not np.isfinite(x).all():
            return None, 'invalid route features'
        low, high = np.asarray(a['minimum']), np.asarray(a['maximum'])
        allowance = np.maximum((high-low)*.15, 1e-5)
        if np.any(x < low-allowance) or np.any(x > high+allowance):
            return None, 'approach outside characterized feature range'
        if not a.get('evaluation', {}).get('qualified', False):
            return None, 'held-out probability validation has not passed'
        z = float(np.dot((x-np.asarray(a['mean']))/np.asarray(a['scale']), a['coefficients'])+a['intercept'])
        p = float(expit(a['calibration'][0]*z+a['calibration'][1]))
        return float(np.clip(p, .001, .999)), 'held-out calibrated leg estimate'


class ApproachEvidence:
    """Count completed attempts once, never telemetry frames or plan rejections."""
    def __init__(self):
        self.counts = {}
        self.completed = set()

    @staticmethod
    def key(start, endpoint, heading, command=.4, path=None):
        key = ':'.join(map(str, [*(round(float(x)) for x in (*start, *endpoint)),
                                 round(heading/(math.pi/4)) % 8, round(command, 1)]))
        if path is not None:
            # Same endpoints and final direction can hide different routes
            # around a rock. Keep their independent approach evidence apart.
            trace=[]
            for point in path:
                cell=tuple(round(float(value)) for value in point)
                if not trace or cell!=trace[-1]:trace.append(cell)
            key+=':'+hashlib.sha256(json.dumps(trace).encode()).hexdigest()[:16]
        return key

    def estimate(self, key, p0, reason='calibrated estimate'):
        s, f = self.counts.get(key, (0, 0))
        if p0 is None:
            return ReachabilityEstimate(None, None, False, reason, s, f)
        aa, bb = 4*p0+s, 4*(1-p0)+f
        # Conditional uncertainty given the terrain prior; not a claim about
        # model uncertainty outside its empirical validation range.
        interval = tuple(float(v) for v in beta.ppf([.05, .95], aa, bb))
        return ReachabilityEstimate(aa/(aa+bb), interval, True, reason, s, f, p0)

    def finish(self, attempt_id, key, outcome):
        if attempt_id in self.completed:
            return False
        self.completed.add(attempt_id)
        if outcome not in ('success', 'failure'):
            return False
        s, f = self.counts.get(key, (0, 0))
        self.counts[key] = (s+(outcome == 'success'), f+(outcome == 'failure'))
        return True


def eligibility(estimate, retreat, profile=CapabilityProfile()):
    if not estimate.validated:
        return 'unvalidated'
    if estimate.probability >= profile.normal_probability:
        return 'normal'
    if estimate.probability >= profile.fallback_probability and retreat:
        return 'fallback'
    return 'rejected'
