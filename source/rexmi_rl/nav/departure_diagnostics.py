"""Opt-in, observational capture before an unchanged corridor rejection.

Capture owns its arrays. File compression runs after the command decision on
the runner's export worker; neither observation nor writer errors authorize or
deny motion. This module never reads simulator terrain or computes a route.
"""
from dataclasses import asdict, is_dataclass
import copy
import json
import math
import time
from pathlib import Path

import numpy as np


MAP_FIELDS=('height','low','surface','last_seen','surface_seen','samples','known',
    'fresh','blocked','hazard','slope','gx','gy','roughness','roughness_support',
    'residual','obstacle_distance','steep_distance','local_slope','local_fit_support',
    'shelf_mask','failures','successes','cost','approach_blocked','no_go','no_go_depth',
    'failure_directions')
EXECUTOR_FIELDS=('recovery_turn','recovery_turn_finished','recovery_turn_samples','state','reason','entered','started','stable_since','samples',
    'stall_since','stall_origin','posture','pulse_direction','alignment_best',
    'alignment_progress_at','drive_exposure','last_output_time','forward_output',
    'release_origin','release_bad_since','uphill_patience','executed_traversal',
    'pending_traversal','suspended_release','release_memory','uphill_samples','uphill_stall_since',
    'uphill_goal_only_budget', 'ascent_intent_bound',
    'uphill_stall_origin','committed_heading','travel_axis','metrics','entry_samples','brake_episode_started',
    'posture_pause','pending_drive_continuation','uphill_exposure','posture_signal',
    'recovery_stage_started','recovery_stage_anchor','recovery_stage_samples',
    'uphill_speed_response_enabled','speed_response_used','speed_response_started',
    'drive_response_samples','drive_slow_since','recovery_continuation',
    'recovery_execution_heading','recovery_core_started','recovery_core_anchor',
    'recovery_motion_samples','recovery_region_samples','recovery_region_stall_since')
RECOVERY_FIELDS=('recovery_turn_used','recovery_turn_failure_id','recovery_request_id','recovering','recovery_since','recovery_trials','recovery_direction',
    'recovery_actions','recovery_progress','recovery_continuation_enabled','recovery_continuation_used',
    'recovery_permission','recovery_route','recovery_origin','supported_trail')


def plain(value):
    if is_dataclass(value):return plain(asdict(value))
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(set,frozenset)):return [plain(v) for v in sorted(value,key=repr)]
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


class DepartureCapture:
    """Only the first lateral rejection in this bounded diagnostic run."""
    def __init__(self,control,path,provenance,purpose='any'):
        if purpose not in ('any','mission','recovery'):
            raise ValueError('Departure capture purpose must be any, mission or recovery')
        self.control=control
        self.purpose=purpose
        self.path=Path(path).with_suffix('.departure.npz')
        self.provenance=copy.deepcopy(provenance)
        self.attempted=False
        self.metadata=None
        self.arrays=None
        self.error=None
        self.job=None

    def observe(self,obs,request,geometry):
        if self.attempted:return
        # A skipped event must reach the original failure/evidence handler with
        # no diagnostic work or state change; capture only the selected purpose.
        if self.purpose != 'any' and request.purpose != self.purpose:return
        self.attempted=True
        try:
            c=self.control;g=c.grid;e=c.executor
            # Copy the exact arrays, including freshness and exclusions, before
            # the rejection result can be consumed by the behavior supervisor.
            arrays={name:getattr(g,name).copy() for name in MAP_FIELDS}
            config={name:copy.deepcopy(getattr(g,name)) for name in
                ('world_size','cell_size','origin','footprint_radius','clearance',
                 'half_length','half_width','uphill_limit_deg','side_limit_deg')}
            config.update(STEP_THRESH=g.STEP_THRESH,MAX_SLOPE_DEG=g.MAX_SLOPE_DEG,
                planning_grade_reserve=getattr(g,'planning_grade_reserve',1.),
                motion_turn_limit=getattr(g,'motion_turn_limit',12.),
                diagonal_ascent_preference=getattr(g,'diagonal_ascent_preference',False))
            metadata=plain(dict(kind='pre_rejection_observed_map',version=1,
                time=obs.time,map_time=g.now,map_revision=g.revision,observation=obs,
                request=request,geometry=geometry,grid_config=config,capture_purpose=self.purpose,
                executor={name:copy.deepcopy(getattr(e,name)) for name in EXECUTOR_FIELDS},
                recovery={name:copy.deepcopy(getattr(c,name)) for name in RECOVERY_FIELDS},
                passive_heading_samples=copy.deepcopy(c.passive_heading.samples),
                previous_command=c.previous_command,attempt=copy.deepcopy(c.attempt),
                waypoint_index=c.wp_idx,mission_goals=[(w.x,w.y) for w in c.waypoints],
                checkpoint=c.checkpoint,path=copy.deepcopy(c.path),
                mission_route=copy.deepcopy(c.mission_route),
                terrain_prior=copy.deepcopy(getattr(g,'terrain_prior',None)),
                failure_records=copy.deepcopy(g.failure_records),
                provenance=self.provenance,counts_as_mission=False,counts_as_qualification=False,
                limitation='Exact observed evidence before original decision. Geometry is not success probability.'))
            self.arrays,self.metadata=arrays,metadata
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'

    def complete_tick(self,command,pose=None,velocity=None):
        if self.metadata is not None:
            e=self.control.executor
            self.metadata['returned_command']=plain(command)
            self.metadata['returned_result']=plain(e.result)
            self.metadata['returned_state']=e.state.value
            self.metadata['output_applied_to_physics']=False
            self.metadata['pose']=plain(pose)
            self.metadata['velocity_world_3d']=plain(velocity)

    def submit(self,worker):
        if not self.attempted or self.job is not None:return
        # The worker receives only detached data. No mutable controller or map.
        try:
            self.job=worker.submit(write_departure,self.path,self.arrays,
                                   copy.deepcopy(self.metadata),self.error)
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'


class EventSnapshots:
    """Bounded passive journal; snapshots never interrupt or authorize motion.

    Four captures and 64 MiB of cumulative admitted detached array/JSON data
    bound the writer queue even when the export worker is slow. Compression
    scratch space belongs to that single worker. No file I/O occurs in observe.
    """
    def __init__(self, control, path, provenance, *, max_captures=4, max_bytes=64*1024*1024):
        self.control, self.path, self.provenance = control, Path(path), provenance
        self.max_captures, self.max_bytes = min(4,max_captures), max_bytes
        self.captures, self.pending, self.seen = [], [], set()
        self.counts = dict(duplicates=0, cap_dropped=0, bytes_dropped=0, errors=0, ignored=0,
                           stage_skipped=0, reserved_skipped=0)
        # The round-trip runner already configures this outbound stage. Keep
        # other missions' first-four policy unchanged; do not infer a stage
        # from destination coordinates or alter the controller to capture it.
        self.outbound_waypoint = getattr(control, 'survey_ascent_waypoint', None)
        self.continuation_action = None
        self.general_captures = 0
        self.admitted_bytes = 0
        self.copy_count, self.copy_total_ms, self.copy_max_ms = 0, 0., 0.

    def observe(self, kind, obs, request, details):
        ordinary = kind in ('termination','brake_rejection','release_overspeed_brake')
        admission = kind == 'ascent_continuation_admitted'
        if not ordinary and not (admission and self.outbound_waypoint is not None):
            self.counts['ignored']+=1
            return
        priority = None
        if self.outbound_waypoint is not None:
            if self.control.wp_idx != self.outbound_waypoint:
                self.counts['stage_skipped'] += 1
                return
            if admission and self.continuation_action is None:
                priority = 'first_outbound_continuation'
            elif kind == 'termination' and request.action_id == self.continuation_action:
                priority = 'first_outbound_continuation_termination'
            elif admission or self.general_captures >= max(0, self.max_captures-2):
                self.counts['reserved_skipped'] += 1
                return
            else:
                priority = 'other_outbound_event'
        started = time.perf_counter()
        key = (request.action_id,kind)
        if key in self.seen:
            self.counts['duplicates'] += 1
            return
        if len(self.captures) >= self.max_captures or len(self.seen)>=128:
            self.counts['cap_dropped'] += 1
            return
        try:
            # Reject oversized worlds before allocating their detached copy.
            expected = sum(getattr(self.control.grid,name).nbytes for name in MAP_FIELDS)
            if self.admitted_bytes+expected > self.max_bytes:
                self.counts['bytes_dropped'] += 1
                return
            self.seen.add(key)
            capture = DepartureCapture(self.control,self.path,self.provenance)
            capture.path = self.path.with_suffix(f'.event-{len(self.captures)+1:02d}.npz')
            capture.observe(obs,request,details)
            if capture.metadata is None:
                raise ValueError(capture.error or 'snapshot unavailable')
            capture.metadata.update(kind='navigation_decision_snapshot',event_kind=kind,
                event_details=plain(details),output_returned=False,output_submitted_to_physics=False,
                output_step_completed=False,output_applied_to_physics=False,
                snapshot_phase='after held-interval observation; before terminal result or exclusion update',
                counts_as_mission=False,counts_as_qualification=False)
            if priority is not None:
                capture.metadata['capture_priority'] = priority
            size = expected+len(json.dumps(capture.metadata,allow_nan=False).encode())
            if self.admitted_bytes+size > self.max_bytes:
                self.counts['bytes_dropped'] += 1
                return
            self.admitted_bytes += size
            self.captures.append(capture)
            self.pending.append(capture)
            if priority == 'first_outbound_continuation':
                self.continuation_action = request.action_id
            elif priority == 'other_outbound_event':
                self.general_captures += 1
        except Exception as exc:
            self.counts['errors'] += 1
            # Only a small diagnostic message is appended, never a control result.
            self.control.escape_events.append(dict(event='event_snapshot_error',time=obs.time,
                action_id=request.action_id,error=f'{type(exc).__name__}: {exc}'))
        finally:
            elapsed=1000*(time.perf_counter()-started)
            self.copy_count+=1;self.copy_total_ms+=elapsed;self.copy_max_ms=max(self.copy_max_ms,elapsed)

    def complete_tick(self, command, pose=None, velocity=None):
        for capture in self.pending:
            try:
                capture.complete_tick(command,pose,velocity)
                capture.metadata['output_returned'] = True
                detail=self.control.executor.last_brake_evaluation
                if detail is not None and detail['time']==capture.metadata['time']:
                    capture.metadata['returned_tick_brake_evaluation']=plain(detail)
            except Exception as exc:
                self.counts['errors']+=1
                capture.metadata['output_record_error']=f'{type(exc).__name__}: {exc}'

    def submitted(self):
        for capture in self.pending:
            capture.metadata['output_submitted_to_physics'] = True

    def flush(self, worker, *, completed=False, reason='tick ended before physics step'):
        for capture in self.pending:
            capture.metadata.update(output_step_completed=completed,
                output_applied_to_physics=(True if completed else
                    None if capture.metadata['output_submitted_to_physics'] else False),
                output_dispatch_reason=reason)
            capture.submit(worker)
            # Worker owns the detached arrays; do not retain a second owner
            # after queueing. Cumulative admission bounds all outstanding jobs.
            capture.arrays = None
        self.pending.clear()

    def report(self):
        records=[]
        for capture in self.captures:
            try:
                result=capture.job.result() if capture.job is not None else dict(ok=False,error=capture.error or 'not submitted')
            except Exception as exc:
                result=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
            records.append(dict(path=str(capture.path),event_kind=capture.metadata['event_kind'],
                time=capture.metadata['time'],action_id=capture.metadata['request']['action_id'],
                output_returned=capture.metadata['output_returned'],
                output_submitted_to_physics=capture.metadata['output_submitted_to_physics'],
                output_step_completed=capture.metadata['output_step_completed'],result=result))
        return dict(captures=records,counts=dict(self.counts,
            write_errors=sum(not r['result'].get('ok',False) for r in records)),admitted_bytes=self.admitted_bytes,
            max_captures=self.max_captures,max_bytes=self.max_bytes,
            allocation_policy=('outbound: two general events plus first continuation and its termination'
                if self.outbound_waypoint is not None else 'first four eligible events'),
            copy_ms=dict(count=self.copy_count,mean=self.copy_total_ms/self.copy_count if self.copy_count else None,
                maximum=self.copy_max_ms if self.copy_count else None),
            limitation='Passive observed-map and controller snapshots; not a simulator state reset or success evidence.')


def write_departure(path,arrays,metadata,error=None):
    """Diagnostic writer failures are recorded, never propagated into control."""
    path=Path(path)
    try:
        if error:raise RuntimeError(error)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,metadata_json=np.asarray(json.dumps(metadata,allow_nan=False)),**arrays)
        path.with_suffix('.tmp').replace(path)
        return {'ok':True,'path':str(path)}
    except Exception as exc:
        detail={'ok':False,'path':str(path),'error':f'{type(exc).__name__}: {exc}'}
        try:path.with_suffix('.error.json').write_text(json.dumps(detail,indent=2)+'\n')
        except Exception:pass
        return detail


def load_departure(path):
    """Offline reconstruction of copied geometry; deliberately no map rebuild."""
    from .ascent_map import AscentTerrainMap
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata_json'].item()))
        cfg=metadata['grid_config']
        g=AscentTerrainMap(**{k:cfg[k] for k in ('world_size','cell_size','origin',
            'footprint_radius','clearance','half_length','half_width')})
        for name in MAP_FIELDS:
            getattr(g,name)[:]=data[name]
        for name,value in cfg.items():setattr(g,name,value)
        g.now=metadata['map_time'];g.revision=metadata['map_revision']
        g.terrain_prior=metadata['terrain_prior']
        g.failure_records=metadata['failure_records']
    return g,metadata
