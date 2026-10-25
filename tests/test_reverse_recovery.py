import test_remote_control
from rexmi_bt_tests.nav.reverse_recovery import ReverseRecovery


def test_stuck_reverse_escalates_and_repeated_failures_keep_memory():
    r=ReverseRecovery();r.start(0,(0,0),0)
    assert r.tick(0,(0,0))[0]==-.4
    assert r.tick(2,(0,0))[0]==-.7
    assert r.tick(4,(0,0))[0]==-1.
    command,result=r.tick(6,(0,0))
    assert command==0 and not result['success']
    r.start(7,(0,0),0)
    assert r.tick(7,(0,0))[0]==-.7


def test_productive_reverse_stays_slow_and_stops_on_measured_distance():
    r=ReverseRecovery();r.start(0,(0,0),0)
    for time,distance in ((1,.12),(2,.24),(3,.36)):
        assert r.tick(time,(-distance,0))[0]==-.4
    command,result=r.tick(4,(-.51,0))
    assert command==0 and result['success'] and not r.active
    r.start(5,(-.51,0),0)
    assert r.tick(5,(-.51,0))[0]==-.4


def test_sideways_motion_is_not_reverse_progress():
    r=ReverseRecovery();r.start(0,(0,0),0)
    assert r.tick(2,(0,2))[0]==-.7
