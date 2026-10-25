"""Measured backout with escalating reverse commands, never faster forward drive."""
import math
import numpy as np


class ReverseRecovery:
    SPEEDS=(.4,.7,1.)

    def __init__(self):
        self.anchor=None;self.failed_attempts=0;self.active=False

    def start(self, now, xy, yaw):
        if self.anchor is None or math.dist(xy,self.anchor)>1.5:
            self.anchor=tuple(xy);self.failed_attempts=0
        self.origin=tuple(xy);self.axis=np.array([math.cos(yaw),math.sin(yaw)])
        self.started=self.progress_at=now;self.credit=0.
        self.level=min(self.failed_attempts,2);self.active=True
        self.escalations=[]

    def tick(self, now, xy):
        distance=-float(np.dot(np.subtract(xy,self.origin),self.axis))
        if distance>=.5 or now-self.started>=6.:
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
