#!/usr/bin/env python3
"""Fit/calibrate/evaluate complete-leg outcomes from three disjoint run sets.

Example: python scripts/fit_reachability.py --train a.events.jsonl b.events.jsonl
  --calibrate c.events.jsonl d.events.jsonl --evaluate e.events.jsonl f.events.jsonl
  --output outputs/ascent_v2/model.json

A file is written even when evidence is insufficient. Such a file is explicitly
unqualified and cannot authorize an approach in the autonomous controller.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def point_target(event):
    return (event.get('target_kind') in (None,'point','nominal_recovery_endpoint')
            and not event.get('approach_key','').startswith('recovery-region-'))


def definitions():
    path = Path(__file__).resolve().parents[1]/'source/rexmi_rl/nav/reachability.py'
    spec = importlib.util.spec_from_file_location('rexmi_reachability_fit', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_runs(paths):
    result, ids = [], set()
    version = definitions().VERSION
    for path in paths:
        if (Path(path).parent/'infrastructure_failure.json').exists():
            raise ValueError(f'{path}: simulator infrastructure failure; exclude this run')
        provenance_path = Path(str(path).removesuffix('.events.jsonl')+'.provenance.json')
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        summary_path=Path(str(path).removesuffix('.events.jsonl')+'.summary.json')
        if not summary_path.exists():
            raise ValueError(f'{path}: run has no final summary; unfinished runs cannot be split into calibration data')
        summary=json.loads(summary_path.read_text())
        if summary.get('result') not in ('complete','hold','step_limit') or summary.get('manual_intervention',True):
            raise ValueError(f'{path}: interrupted or invalid run; preserve it outside model qualification data')
        for line in Path(path).read_text().splitlines():
            event = json.loads(line)
            if event.get('event') != 'approach_outcome' or event.get('outcome') not in ('success', 'failure'):
                continue
            # Functional region outcomes remain in the source logs, but cannot
            # label the point-leg model without a separately validated target model.
            if not point_target(event):continue
            if event.get('feature_version') != version:
                raise ValueError(f'{path}: attempt feature version differs; do not mix incompatible characterization data')
            identity = (event['run_id'], event['attempt_id'])
            if identity in ids:
                raise ValueError('Duplicate completed attempt: '+str(identity))
            ids.add(identity)
            event['policy_sha256'] = provenance.get('policy_sha256')
            event['source_id'] = summary.get('source_id')
            event['capability_profile']=provenance.get('capability_profile')
            result.append(event)
    return sorted(result,key=lambda e:(e['run_id'],e['end_time'],e['attempt_id']))


def fit_logistic(x, y, penalty):
    x = np.c_[x, np.ones(len(x))]
    def objective(w):
        logits = x@w
        loss = np.logaddexp(0., logits).sum()-y@logits+penalty*np.dot(w[:-1], w[:-1])/2
        gradient = x.T@(expit(logits)-y)
        gradient[:-1] += penalty*w[:-1]
        return loss, gradient
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B')
    if not result.success:
        raise ValueError('Fit failed: '+result.message)
    return result.x


def reliability(predictions, outcomes, run_ids):
    bands = []
    run_ids=np.asarray(run_ids)
    groups=np.unique(run_ids)
    for lo, hi in ((0., .6), (.6, .8), (.8, 1.000001)):
        mask = (predictions >= lo) & (predictions < hi)
        n = int(mask.sum())
        mean = float(predictions[mask].mean()) if n else None
        rate = float(outcomes[mask].mean()) if n else None
        if n:
            z = 1.645
            centre = (rate+z*z/(2*n))/(1+z*z/n)
            half = z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            interval = [max(0., centre-half), min(1., centre+half)]
        else:
            interval = None
        supported_runs=int(len(np.unique(run_ids[mask])))
        cluster_interval=None
        if supported_runs>1:
            counts=np.array([int(np.sum(mask & (run_ids==run))) for run in groups])
            successes=np.array([float(outcomes[mask & (run_ids==run)].sum()) for run in groups])
            choices=np.random.default_rng(42).integers(0,len(groups),size=(2000,len(groups)))
            totals=counts[choices].sum(axis=1)
            rates=successes[choices].sum(axis=1)[totals>0]/totals[totals>0]
            cluster_interval=np.quantile(rates,[.05,.95]).tolist()
        bands.append(dict(lower=lo, upper=min(1., hi), count=n, run_count=supported_runs, predicted=mean,
                          observed=rate, descriptive_wilson_90_interval=interval,
                          run_bootstrap_90_interval=cluster_interval,
                          absolute_error=abs(mean-rate) if n else None))
    return bands


def fit(train, calibration, evaluation):
    if any(not point_target(row) for rows in (train,calibration,evaluation) for row in rows):
        raise ValueError('Point-leg calibration cannot use functional region outcomes')
    mod = definitions()
    sets = [sorted(rows,key=lambda e:(e['run_id'],e.get('end_time',0.))) for rows in (train,calibration,evaluation)]
    train,calibration,evaluation=sets
    groups = [set(e['run_id'] for e in events) for events in sets]
    if not all(groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Require nonempty, disjoint complete run sets for train/calibrate/evaluate')
    arrays = [np.asarray([e['features'] for e in events], float) for events in sets]
    ys = [np.asarray([e['outcome'] == 'success' for e in events], float) for events in sets]
    if any(x.ndim != 2 or x.shape[1] != len(mod.FEATURES) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Invalid or non-finite feature vectors')
    if any(len(np.unique(y)) < 2 for y in ys[:2]):
        raise ValueError('Training and calibration each require successful and failed completed legs')
    mean, scale = arrays[0].mean(axis=0), np.maximum(arrays[0].std(axis=0), .01)
    standardized = [(x-mean)/scale for x in arrays]
    w = fit_logistic(standardized[0], ys[0], penalty=1.)
    calibration_logits = standardized[1]@w[:-1]+w[-1]
    cal = fit_logistic(calibration_logits[:, None], ys[1], penalty=.1)
    p = expit(cal[0]*(standardized[2]@w[:-1]+w[-1])+cal[1])
    low,high=arrays[0].min(axis=0),arrays[0].max(axis=0)
    allowance=np.maximum((high-low)*.15,1e-5)
    supported=np.all((arrays[2]>=low-allowance)&(arrays[2]<=high+allowance),axis=1)
    run_ids=np.asarray([e['run_id'] for e in evaluation])
    bands = reliability(p[supported], ys[2][supported], run_ids[supported])
    # Replay online updates in chronological order without allowing later
    # outcomes to influence earlier predictions. Each run starts cold.
    evidence_by_run = {}
    updated = []
    repeated=0
    for row, p0, supported_row in zip(evaluation, p, supported):
        evidence = evidence_by_run.setdefault(row['run_id'], mod.ApproachEvidence())
        key = row['approach_key']
        repeated += int(bool(supported_row) and key in evidence.counts)
        updated.append(evidence.estimate(key, float(p0)).probability)
        evidence.finish(row['attempt_id'], key, row['outcome'])
    updated = np.asarray(updated)
    online_bands = reliability(updated[supported], ys[2][supported], run_ids[supported])
    reasons = []
    policies = {e.get('policy_sha256') for rows in sets for e in rows}
    if len(policies) != 1 or None in policies:
        reasons.append('all outcomes need provenance for the same frozen policy')
    sources={e.get('source_id') for rows in sets for e in rows}
    if len(sources)!=1 or None in sources:
        reasons.append('all qualification outcomes need the same controller source version')
    profiles={json.dumps(e.get('capability_profile'),sort_keys=True) for rows in sets for e in rows}
    if len(profiles)!=1 or 'null' in profiles:
        reasons.append('all qualification outcomes need the same recorded environment and capability profile')
    if any(len(g) < 3 for g in groups):
        reasons.append('need at least three independent runs per split')
    if len(train) < 50:
        reasons.append('need at least 50 completed training attempts')
    if repeated<10:
        reasons.append('need at least 10 held-out attempts with prior within-run evidence to validate the online update')
    for label, report in [('initial', bands), ('online', online_bands)]:
        for band in report[1:]:
            if band['count'] < 20:
                reasons.append(f'{label}: need 20 held-out attempts in {band["lower"]:.0%}–{band["upper"]:.0%} band')
            elif band['absolute_error'] > .1:
                reasons.append(f'{label}: calibration error exceeds 10 percentage points')
            if band['run_count']<3:
                reasons.append(f'{label}: each operational band needs support from three held-out runs')
    brier = float(np.mean((p[supported]-ys[2][supported])**2)) if supported.any() else None
    baseline = float(np.mean((ys[0].mean()-ys[2][supported])**2)) if supported.any() else None
    if brier is None:
        reasons.append('no held-out attempts within the characterized feature range')
    elif brier >= baseline:
        reasons.append('held-out Brier score does not improve on training base-rate prediction')
    return dict(version=mod.VERSION, features=list(mod.FEATURES), mean=mean.tolist(), scale=scale.tolist(),
                policy_sha256=next(iter(policies)) if len(policies)==1 else None,
                source_id=next(iter(sources)) if len(sources)==1 else None,
                capability_profile=json.loads(next(iter(profiles))) if len(profiles)==1 else None,
                coefficients=w[:-1].tolist(), intercept=float(w[-1]), calibration=cal.tolist(),
                minimum=low.tolist(), maximum=high.tolist(),
                run_groups=dict(zip(('train', 'calibrate', 'evaluate'), [sorted(g) for g in groups])),
                evaluation=dict(qualified=not reasons, reasons=reasons, brier=brier,
                                base_rate_brier=baseline,
                                online_brier=float(np.mean((updated[supported]-ys[2][supported])**2)) if supported.any() else None,
                                diagnostic_all_attempts_brier=float(np.mean((p-ys[2])**2)),
                                brier_scope='feature-supported held-out attempts only',
                                supported_evaluation_attempts=int(supported.sum()),
                                unsupported_evaluation_attempts=int((~supported).sum()),
                                reliability=bands, online_reliability=online_bands,
                                repeated_evidence_attempts=repeated,
                                sample_counts=[len(rows) for rows in sets]),
                uncertainty='90% local Beta interval conditional on fitted prior; held-out band intervals reported separately')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ('train', 'calibrate', 'evaluate'):
        parser.add_argument('--'+split, nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = [args.train, args.calibrate, args.evaluate]
    artifact = fit(*(read_runs(p) for p in paths))
    artifact['sources'] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for group in paths for p in group}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, allow_nan=False)+'\n')
    print(json.dumps(artifact['evaluation'], indent=2))


if __name__ == '__main__':
    main()
