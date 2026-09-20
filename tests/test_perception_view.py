"""CPU geometry tests; no Isaac/torch import required."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

root = Path(__file__).resolve().parents[1]
module = root / 'source/rexmi_rl/nav/perception_view.py'
if not module.exists():
    module = Path(__file__).with_name('perception_view.py')
spec = importlib.util.spec_from_file_location('perception_view', module)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class PerceptionGeometryTests(unittest.TestCase):
    def test_high_cost_and_steep_colours_are_distinct(self):
        cost=np.array([[6.,20.,20.]])
        colors,legend=v.survey_colors(cost,np.ones_like(cost,bool),slope=np.array([[.1,.8,.1]]))
        np.testing.assert_allclose(colors[0],[[1,0,1],[1,.29,.12],[1,0,0]])
        self.assertIn('orange: slope >35',legend)

    def test_live_frames_reuse_tiles_until_map_or_layer_changes(self):
        from unittest.mock import MagicMock,patch
        from types import SimpleNamespace
        from threading import Lock
        g=SimpleNamespace(revision=1,traversal_revision=0,origin=(0,0),cell_size=1.,
                          get_visual_grid=lambda:(np.ones((2,2)),np.zeros((2,2)),np.ones((2,2),bool)))
        view=v.PerceptionView.__new__(v.PerceptionView)
        view.nav=SimpleNamespace(_omap=g,_lidar=None,_lock=Lock(),shared={'planned_path':[]},
                    _sim_localizer=SimpleNamespace(get_pose=lambda:SimpleNamespace(x=0.,y=0.,z=0.)))
        view.point_limit=2000;view._tile_key=None;view._tile_count=0;view.layer='terrain';view.status=None
        view._Usd=MagicMock();view._Sdf=MagicMock();view.stage=MagicMock()
        view.show=dict(lidar=True,costs=True,path=True,rays=False)
        for name in ('points','tiles','route','rays','checkpoint_marker','target_marker','_set_curve','_update_tiles'):
            setattr(view,name,MagicMock())
        view._update();view._update()
        self.assertEqual(view._update_tiles.call_count,1)
        self.assertEqual(view.points.GetPointsAttr().Set.call_count,2)
        g.revision+=1;view._update()
        self.assertEqual(view._update_tiles.call_count,2)
        view.layer='slope';view._update()
        self.assertEqual(view._update_tiles.call_count,3)

    def test_clearance_is_amber_without_changing_planning_costs(self):
        costs=np.full((2,2),20.)
        mask=np.array([[False,True],[True,False]])
        vertices,colors=v.cost_tiles(costs,np.ones((2,2)),np.ones((2,2),bool),
                                    (0,0),1,(0,0),clearance_mask=mask)
        self.assertEqual(int(np.sum(colors[:,1]==0)),2)
        self.assertEqual(int(np.sum(colors[:,1]>.4)),2)
        np.testing.assert_array_equal(costs,20.)
    def test_tile_identity_survives_reordering_and_budget_changes(self):
        g=np.ones((4,4))
        a,colors=v.cost_tiles(g,g,g.astype(bool),(0,0),1,(0,0),limit=3)
        pool=v.StableTilePool(3)
        first,_,_=pool.update(a,colors)
        slots=pool.slots.copy()
        second,_,_=pool.update(a.reshape(-1,4,3)[::-1].reshape(-1,3),colors[::-1])
        self.assertEqual(pool.slots,slots)
        np.testing.assert_array_equal(first,second)
        pool.update(a[:8],colors[:2])
        self.assertEqual(len(pool.slots),2)
        for key,slot in pool.slots.items(): self.assertEqual(slot,slots[key])
    def test_scan_filters_and_budget(self):
        pts = [[float('nan'),0,0],[float('inf'),0,0],[0,0,0],[1e6,0,0],[1,2,3],[2,3,4]]
        out = v.filter_hits(pts, [0,0,0], limit=1)
        self.assertEqual(out.shape, (1,3))
        np.testing.assert_allclose(out[0], [1,2,3])
        self.assertEqual(v.filter_hits([], [0,0,0]).shape, (0,3))

    def test_grid_axes_origin_and_unknown_mask(self):
        costs = np.array([[1,2],[6,20]],np.float32)
        heights = np.array([[1,2],[3,1e6]],np.float32)
        seen = np.array([[True,False],[True,True]])
        vertices, colors = v.cost_tiles(costs,heights,seen,(10,20),2,(10,20))
        centres = vertices.reshape(-1,4,3).mean(axis=1)
        np.testing.assert_allclose(centres, [[9,19,1.045],[11,19,3.045]],atol=1e-5)
        self.assertEqual(len(colors),2)
        self.assertGreater(colors[1,0],colors[0,0])

    def test_radius_and_budget(self):
        grid = np.ones((10,10))
        vertices, _ = v.cost_tiles(grid,grid,grid.astype(bool),(0,0),1,(0,0),radius=2,limit=3)
        self.assertEqual(len(vertices),12)
        self.assertTrue((np.linalg.norm(vertices.reshape(-1,4,3).mean(axis=1)[:,:2],axis=1)<=2).all())

    def test_route_does_not_bridge_unobserved_or_outside_cells(self):
        height = np.arange(9).reshape(3,3).astype(float)
        seen = np.ones((3,3),bool);seen[1,1]=False
        path = [(-1,-1),(-1,0),(0,0),(1,0),(1,1),(100,100)]
        segments = v.drape_path(path,height,seen,(0,0),1)
        self.assertEqual(segments.shape,(4,3))
        np.testing.assert_allclose(segments[:,2],[.12,1.12,7.12,8.12])
        self.assertEqual(v.drape_path([],height,seen,(0,0),1).shape,(0,3))

if __name__ == '__main__':
    unittest.main()
