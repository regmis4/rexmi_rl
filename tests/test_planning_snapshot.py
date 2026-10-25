from unittest.mock import patch
import numpy as np
import test_behavior_navigation as bt
from rexmi_bt_tests.nav import planning_snapshot as snap


def test_snapshot_roundtrip_does_not_rebuild_or_mutate_and_is_bounded(tmp_path):
    from types import SimpleNamespace
    g=bt.ground();g.cost[3,4]=12.345;g.last_seen[3,4]=1.23
    c=SimpleNamespace(grid=g,planning_snapshot_dir=tmp_path,wp_idx=0,
        waypoints=[SimpleNamespace(x=2.,y=1.)],plan_generation=4,protocol_version='test')
    record=snap.save(c,(0.,0.),.3,3.)
    restored,meta=snap.load(record['path'])
    assert meta['position']==[0.,0.] and meta['yaw']==.3
    for name in snap.ARRAYS:assert np.array_equal(getattr(g,name),getattr(restored,name),equal_nan=True)
    with patch.object(snap,'LIMIT',1):assert snap.save(c,(0.,0.),.3,4.)['status']=='snapshot limit reached'
