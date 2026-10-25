"""Whole-leg probability ordering AFTER route eligibility; no learned model here.

Unknown confidence stays unknown. Historical manual motion is not an autonomous
success label. Terrain costs remain the explicit characterization fallback.
"""
import math

import numpy as np


def validated_domain(model, policy_sha256, source_id, capability_profile):
    artifact = getattr(model, 'artifact', None) or {}
    return bool(getattr(model, 'model_id', None)
        and artifact.get('evaluation', {}).get('qualified')
        and policy_sha256 and source_id and capability_profile
        and artifact.get('policy_sha256') == policy_sha256
        and artifact.get('source_id') == source_id
        and artifact.get('capability_profile') == capability_profile)


def select(control, candidates, xy):
    """Candidates have already passed completeness/tier/physical filtering.

    Only predictions of the entire same-start/same-destination route are
    comparable here. A checkpoint prediction cannot stand in for reaching exit.
    This ranks the existing model's approach-success outcome, not a product of
    cell estimates. That training label does not guarantee settling: historical
    frontier arrivals also count as approach successes. Mission completion and
    settling need a separately evaluated target contract before making that claim.
    """
    domain = bool(getattr(control, 'probability_domain_validated', False))
    ascent_first=bool(getattr(control,'ascent_first_active',False))
    climbing_burden=ascent_first and control.graph_plan_stats.get('objective')=='climb-burden-v1'
    wheelbase=ascent_first and control.graph_plan_stats.get('objective')=='wheelbase-demand-v2-fourth'
    sustained=ascent_first and control.graph_plan_stats.get('objective')=='sustained-climb-v1'
    prioritize_climb=not ascent_first and bool(getattr(control.grid,'uphill_first',False) or getattr(control,'graph_objective_active',False))
    def terrain_rank(row):
        if ascent_first:
            rank=control.graph_plan_stats['rank']
            row[9]['terrain_rank']=list(rank)
            return tuple(rank)
        if prioritize_climb:
            from .traversal_objective import route_cost
            burden=route_cost(control.grid,row[2],breakdown=True).get('uphill',math.inf)
            row[9]['terrain_rank']=[burden,row[1]]
            return burden,row[1]
        return (row[1],)
    waypoint = control.waypoints[control.wp_idx]
    goal = (waypoint.x, waypoint.y)
    known = []
    for candidate in candidates:
        tier, _, route, complete, leg, _, estimate, *_ = candidate
        info = candidate[9]
        probability = getattr(estimate, 'probability', None)
        comparable = (complete and leg is not None and len(leg) == len(route)
            and len(route) >= 2 and np.allclose(leg, route, atol=1e-8, rtol=0.)
            and math.dist(route[0], xy) <= .1 and math.dist(route[-1], goal) <= .1)
        numeric = isinstance(probability, (float, int)) and not isinstance(probability, bool)
        usable = bool(domain and comparable and tier in ('normal', 'fallback')
            and getattr(estimate, 'validated', False) and numeric
            and math.isfinite(probability) and 0. <= probability <= 1.)
        info['success_selection'] = dict(probability_used=probability if usable else None,
            comparable_whole_leg=bool(comparable), domain_validated=domain,
            prediction_scope='planning snapshot; whole-leg approach success, not a settling guarantee',
            model_id=getattr(control.model, 'model_id', None) if domain else None,
            probability_status='validated' if usable else 'unavailable for route ranking')
        if usable:
            known.append(candidate)
    if known:
        selected = min(known, key=lambda row: (-row[6].probability, terrain_rank(row)))
        reason = 'highest validated whole-leg approach-success estimate; terrain cost breaks exact ties'
        objective = 'success_probability'
    else:
        selected = min(candidates, key=terrain_rank)
        reason = ('no comparable validated whole-leg probability; unchanged terrain-cost characterization fallback')
        if prioritize_climb:
            reason='no calibrated probability; integrated uphill burden first, shared terrain cost second'
        if ascent_first:
            reason='no calibrated probability; ascent effort before checked gentle continuation, exterior effort then distance'
        if climbing_burden:
            reason='no calibrated probability; terrain difficulty times positive height gain, distance breaks ties'
        if wheelbase:
            reason='no calibrated probability; wheelbase uphill demand fourth power, distance breaks ties'
        if sustained:
            reason='no calibrated probability; sustained longitudinal climb bouts, distance breaks ties'
        objective = ('sustained_climb_characterization' if sustained else
            'wheelbase_demand_characterization' if wheelbase else
            'climb_burden_characterization' if climbing_burden else
            'ascent_first_characterization' if ascent_first else 'characterization_cost_fallback')
    for candidate in candidates:
        candidate[9]['success_selection'].update(objective=objective, reason=reason,
            comparable_validated_count=len(known), candidate_count=len(candidates))
        probability = candidate[9]['success_selection']['probability_used']
        candidate[9]['success_selection']['display'] = (
            ('Success probability unavailable — sustained longitudinal demand (characterization)' if sustained else
             'Success probability unavailable — wheelbase uphill demand fourth power (characterization)' if wheelbase else
             'Success probability unavailable — climbing burden (characterization)' if climbing_burden else
             'Success probability unavailable — ascent-first effort (characterization)' if ascent_first else
             'Success probability unavailable — uphill burden then terrain cost (characterization)'
             if prioritize_climb or ascent_first else 'Success probability unavailable — terrain-cost fallback (characterization)')
            if not known else 'Success probability unavailable — unranked by probability'
            if probability is None else
            f'Plan-time whole-leg approach success {probability:.0%} — validated model; probability ranked before terrain cost')
        candidate[9]['selection_rule'] = reason
    return selected


def selection_text(options):
    selected = next((row for row in options if row.get('selected')), {})
    status = selected.get('success_selection')
    if not status:
        return ''
    return status.get('display', '')
