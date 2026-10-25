"""Display/recording isolation checks; no simulator launch."""
import csv
import importlib.util
import io
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_climb_burden import terrain

spec=importlib.util.spec_from_file_location('teleop_perception',Path(__file__).resolve().parents[1]/'scripts/teleop_perception.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_task_seed_frame_must_match_reference():
    meta={'config':{'terrain_prior':dict(task='crater',seed=42,coordinate_frame='world')}}
    m.validate_reference(meta,'crater',42)
    for task,seed in [('other',42),('crater',41)]:
        with pytest.raises(ValueError):m.validate_reference(meta,task,seed)
    meta['config']['terrain_prior']['coordinate_frame']='local'
    with pytest.raises(ValueError):m.validate_reference(meta,'crater',42)


def test_reading_is_directional_and_does_not_modify_map():
    g,_,_=terrain();g.gx[:]=.2;g.gy[:]=0.;g.slope[:]=.2
    g.cost.setflags(write=False);before=g.cost.copy()
    a=m.terrain_reading(g,(0.,0.),0.)
    b=m.terrain_reading(g,(0.,0.),math.pi/2)
    assert a['along_deg']==pytest.approx(math.degrees(math.atan(.2)))
    assert b['cross_deg']==pytest.approx(a['along_deg'])
    assert a['cost']==b['cost'] and np.array_equal(g.cost,before)
    assert not m.terrain_reading(g,(1000.,0.),0.)['known']


def test_live_trace_breaks_at_reset_and_records_unfiltered_commands():
    g,_,_=terrain();capture=m.TeleopPerception.__new__(m.TeleopPerception)
    capture.grid=g;capture.dt=.02;capture.episode=0;capture.trail=[]
    pose=SimpleNamespace(x=0.,y=0.,z=1.,yaw=0.)
    capture.localizer=SimpleNamespace(get_pose=lambda:pose)
    capture.state=SimpleNamespace(scenario_label=lambda:'manual')
    capture.writer=Mock();capture.nav=SimpleNamespace(shared={});capture.view=Mock()
    capture.update(0,'rocky_slope',-.4,.35)
    row=capture.writer.writerow.call_args.args[0]
    assert row['vx']==-.4 and row['omega']==.35 and row['sim_time_s']==.02
    pose.x=1.;capture.update(1,'rocky_slope',0.,0.)
    assert len(capture.trail)==2
    pose.x=0.;capture.update(2,'rocky_slope',0.,0.,done=True)
    assert not capture.trail and capture.episode==1
    capture.update(3,'rocky_slope',.4,0.)
    assert len(capture.trail)==1


def test_manual_palette_matches_plot_and_keeps_mask_separate():
    from matplotlib import colormaps
    capture=m.TeleopPerception.__new__(m.TeleopPerception)
    capture.palette=colormaps['plasma'];capture.grid=SimpleNamespace(blocked=np.array([[False,False,True]]))
    cost=np.array([[1.,20.,1.]]);known=np.ones_like(cost,bool)
    colors,legend=capture.colors(cost,known)
    np.testing.assert_allclose(colors[0,:2],colormaps['plasma'](np.array([0.,1.]))[:,:3],rtol=1e-6)
    np.testing.assert_allclose(colors[0,2],[.22,.24,.27])
    assert 'FROZEN' in legend
