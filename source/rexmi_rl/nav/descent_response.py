"""Descent-only patience for the existing progress-based reverse recovery."""
import numpy as np
from .reverse_recovery import ReverseRecovery


class DescentReverseRecovery(ReverseRecovery):
    """Give escalating reverse time to work; ascent keeps its existing behavior."""
    def tick(self, now, xy):
        distance=-float(np.dot(np.subtract(xy,self.origin),self.axis))
        if distance>=.5 or now-self.started>=18.:
            success=distance>=.5
            self.failed_attempts=0 if success else self.failed_attempts+1
            self.active=False
            return 0.,dict(success=success,reverse_displacement_m=distance,
                duration_s=now-self.started,peak_request_mps=self.SPEEDS[self.level],
                escalations=list(self.escalations))
        if distance>=self.credit+.1:
            self.credit=distance;self.progress_at=now
        if now-self.progress_at>=2.:
            self.level=min(2,self.level+1);self.progress_at=now
            self.escalations.append(dict(time=now,request_mps=self.SPEEDS[self.level],
                reverse_displacement_m=distance))
        return -self.SPEEDS[self.level],None
