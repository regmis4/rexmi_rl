"""Goal-region geometry, multi-goal search and controller mission semantics."""
import copy
import math
import time
from concurrent.futures import Future
from unittest.mock import patch

import numpy as np
import pytest
import test_remote_control as fixture
from rexmi_bt_tests.nav import exit_region as er, stage_graph as sg


def control():
    c = fixture.controller(fixture.bt.ground(), (4., 0.))
    # Synthetic small crater for behavioral tests. Constructor survey checks
    # are tested separately; no synthetic map is passed as a real survey.
    c.goal_mode = 'any_exit'
    c.exit_rim_radius = 1.
    return c


def test_unselected_and_selected_exit_status_keep_two_csv_coordinates():
    c=control()
    for selected in (None,(2.,3.)):
        c.selected_exit=selected
        status=er.mission_status(c)
        columns=[*(status['goal'] if status and status['goal'] else (None,None))]
        assert len(columns)==2
        assert all(v is None or isinstance(v,float) for v in columns)
        assert status['goal']==selected


def test_full_footprint_clearance_and_supported_gentle_exterior():
    c=control();g=c.grid
    assert er.reason(g,(1.2,0.),1.)
    assert not er.reason(g,(3.,0.),1.)
    cell=g.world_to_cell(3.,0.)
    for field,value in [('known',False),('hazard',True),('no_go',True),
                        ('slope',math.tan(math.radians(16.))),('surface',math.nan)]:
        old=getattr(g,field)[cell];getattr(g,field)[cell]=value
        assert er.reason(g,(3.,0.),1.),field
        getattr(g,field)[cell]=old


def test_multi_goal_prefers_complete_additive_route_with_temporary_height_loss():
    from test_stage_graph import graph
    data=graph();data['goal_indices']=[1,3]
    with patch.object(sg,'turn_reason',return_value=''),patch.object(sg,'turn_cost',return_value=0.):
        path,stats=sg.search(fixture.bt.ground(),data,0.)
    assert path==[(0.,0.),(1.,0.),(2.,0.)]
    assert stats['rank']==2. and stats['selected_goal_index']==3
    assert stats['optimal_in_sampled_graph']


def test_budget_keeps_checked_complete_incumbent():
    data=dict(points=[(0.,0.),(3.,0.),(1.,0.),(0.,3.)],heights=[0.]*4,
        edges={(0,1):dict(heading=0.,total=10.,uphill=0.),
               (0,2):dict(heading=0.,total=1.,uphill=0.),
               (2,3):dict(heading=1.,total=1.,uphill=0.)},
        turns={},goal_indices=[1,3],deadline=time.perf_counter()+30.)
    with patch.object(sg,'MAX_TRANSITIONS',2),patch.object(sg,'turn_reason',return_value=''),patch.object(sg,'turn_cost',return_value=0.):
        path,stats=sg.search(fixture.bt.ground(),data,0.)
    assert path==[(0.,0.),(3.,0.)]
    assert stats['complete'] and stats['budget_exhausted']
    assert not stats['optimal_in_sampled_graph']


def test_arrival_is_any_safe_exterior_not_old_fixed_goal_and_requires_settle():
    c=control()
    for i in range(36):
        t=3.+i*.02
        c.tick((0.,3.),0.,0.,float(t))
        if i<31:assert c.wp_idx==0
    assert c.wp_idx==1


def test_unsafe_radial_crossing_and_geometry_stop_cannot_complete():
    c=control();c.grid.slope[:]=math.tan(math.radians(20.))
    with patch.object(c,'_plan',return_value=False):
        c.path=[]
        for t in np.arange(3.,4.,.02):c.tick((0.,3.),0.,0.,float(t))
    assert c.wp_idx==0
    c=control()
    with patch.object(c,'_geometry',return_value='physical hazard'):
        for t in np.arange(3.,4.,.02):c.tick((0.,3.),0.,0.,float(t))
    assert c.wp_idx==0 and c.operator_mode=='stop'


def test_any_exit_owns_planning_above_floor_without_fixed_goal_fallback():
    c=control();stats=dict(complete=False,reason='budget exhausted')
    with patch.object(er,'plan',return_value=([],stats)) as plan:
        assert not c._propose_routes((6.,0.),0.,None,33.)
    assert plan.call_args.args[1]==(6.,0.)
    assert c.graph_plan_stats is stats
    from rexmi_bt_tests.nav.checkpoint_runner import route_comparison_key
    assert route_comparison_key(c)[1][0][:3]==(er.NAME,False,False)


def test_complete_route_keeps_scalar_rank_and_mission_endpoint():
    c=control();path=[(0.,0.),(0.,3.)]
    stats=dict(complete=True,rank=42.,path=path,reason='')
    with patch.object(er,'plan',return_value=(path,stats)):
        assert c._compute_plan((0.,0.),math.pi/2,3.)
    assert c.selected_exit==(0.,3.) and c.is_mission_checkpoint
    assert c.route_score(path,math.pi/2)==42.
    assert c.waypoints[0].x==4.  # No mutation of shared mission objects.
    from rexmi_bt_tests.nav.checkpoint_runner import route_comparison_key
    assert route_comparison_key(c)[1][0][:3]==(er.NAME,True,True)


def test_async_adoption_copies_exit_only_after_current_geometry_and_generation():
    c=control();path=[(0.,0.),(0.,3.)]
    shadow=copy.copy(c)
    with patch.object(er,'plan',return_value=(path,dict(complete=True,rank=42.,path=path,reason=''))):
        assert shadow._compute_plan((0.,0.),math.pi/2,3.)
    c.planner=object();c.plan_job=Future();c.last_yaw=shadow.last_yaw=math.pi/2
    c.plan_job.set_result((c.plan_generation,True,shadow))
    assert c._plan((0.,0.),math.pi/2,3.02)
    assert c.selected_exit==(0.,3.) and c.is_mission_checkpoint
    c.selected_exit=None;c.plan_job=Future()
    c.plan_job.set_result((c.plan_generation-1,True,shadow))
    assert not c._plan((0.,0.),math.pi/2,3.04) and c.selected_exit is None
    c.plan_job=Future();c.plan_job.set_result((c.plan_generation,True,shadow))
    c.grid.hazard[c.grid.world_to_cell(0.,3.)]=True
    assert not c._plan((0.,0.),math.pi/2,3.06) and c.selected_exit is None


def test_constructor_rejects_unvalidated_survey_and_mixed_objectives():
    c=control()
    with pytest.raises(ValueError,match='survey'):
        fixture.r.RemoteAscentController(c.grid,c.waypoints,mode='characterize',goal_mode='any_exit')
    with pytest.raises(ValueError,match='additive'):
        fixture.r.RemoteAscentController(c.grid,c.waypoints,mode='characterize',goal_mode='any_exit',route_objective='sustained_climb')
