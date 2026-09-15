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
