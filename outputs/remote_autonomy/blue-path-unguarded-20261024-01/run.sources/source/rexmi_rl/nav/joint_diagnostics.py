"""Optional physics-rate ascent diagnostics; no motion or policy decisions."""
import copy
import time

from joint_telemetry import JointTelemetry, vector


class NavigationJointTelemetry(JointTelemetry):
    """Reuse teleop samples, with explicit pre-step navigation alignment.

    Raw policy output precedes wrapper clipping. Processed actuator targets and
    measured joint response remain in the inherited physics-rate CSV. Neither
    wheel speed nor net body contact force alone measures tire slip.
    """
    def __init__(self, *args, **kwargs):
        self.timing = {}
        self.completed_steps = 0
        self.partial_step = None
        self.final_error = None
        super().__init__(*args, **kwargs)

    def _elapsed(self, name, started):
        elapsed = 1000.*(time.perf_counter()-started)
        item = self.timing.setdefault(name, dict(count=0, total_ms=0., maximum_ms=0.))
        item['count'] += 1
        item['total_ms'] += elapsed
        item['maximum_ms'] = max(item['maximum_ms'], elapsed)

    def _capture(self, dt):
        started = time.perf_counter()
        try:
            super()._capture(dt)
        finally:
            self._elapsed('physics_sample_logging', started)

    def begin_navigation_step(self, step, nav_time, policy, state, action_id, vx, omega, actions):
        started = time.perf_counter()
        try:
            self.begin_step(step, policy, state, vx, omega)
            self.partial_step = dict(control_step=step, nav_time_s=nav_time,
                submitted=False, completed=False)
            self.event('policy_action', control_step=step, nav_time_s=nav_time,
                action_id=action_id, state=state,
                raw_policy_output_before_wrapper_clipping=copy.deepcopy(vector(actions)))
        finally:
            self._elapsed('control_pre_step_logging', started)

    def submitted(self):
        self.partial_step['submitted'] = True

    def physics_completed(self):
        self.partial_step['completed'] = True

    def end_navigation_step(self, **kwargs):
        started = time.perf_counter()
        try:
            # env.step returned even if subsequent logger validation fails.
            self.physics_completed()
            self.end_step(**kwargs)
            self.completed_steps += 1
            self.partial_step = None
        finally:
            self._elapsed('control_post_step_logging', started)

    def report(self):
        return dict(directory=str(self.directory), physics_samples=self.sample,
            completed_control_steps=self.completed_steps, partial_step=self.partial_step,
            final_error=self.final_error,
            logging_timing_ms={name:dict(count=item['count'],
                mean_ms=item['total_ms']/item['count'], maximum_ms=item['maximum_ms'])
                for name,item in self.timing.items()},
            alignment='nav CSV is pre-step; physics rows follow the same control_step. '
                'Navigation startup settling is included. Raw actions are before wrapper clipping; '
                'physics joint targets show processed actuator commands. Missing contact counts '
                'remain unavailable; raw per-body forces do not establish slip.')

    def close(self):
        if self.closed:return
        error=None
        try:
            if self.partial_step is not None:
                # A submitted step that raised can have partially advanced physics.
                self.event('partial_navigation_step', **self.partial_step,
                    captured_substeps=self.substep, expected_substeps=self.env.cfg.decimation,
                    physics_application='completed' if self.partial_step['completed'] else
                        'unknown_partial' if self.partial_step['submitted'] else 'not_submitted')
        except Exception as exc:error=exc
        try:
            super().close()
        except Exception as exc:
            if error is None:error=exc
        finally:
            for stream in (self.stream,self.events):
                try:stream.close()
                except Exception as exc:
                    if error is None:error=exc
            self.closed=True
        if error is not None:
            self.final_error=f'{type(error).__name__}: {error}'
            raise error
