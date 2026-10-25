"""Frozen, episode-scoped permissions for incidental motion in failed patches.

This never grants intentional travel through a banned heading or any physical
geometry exception. Queries are pure; only actual observations retire patches.
"""
from dataclasses import dataclass, replace
import math

import numpy as np


def record_cells(grid, record):
    centre = grid.world_to_cell(*record['position'])
    if centre is None:
        return frozenset()
    radius = record['radius_m']
    count = math.ceil(radius/grid.cell_size)
    return frozenset((centre[0]+dr, centre[1]+dc)
        for dr in range(-count, count+1) for dc in range(-count, count+1)
        if math.hypot(dr, dc)*grid.cell_size <= radius
        and 0 <= centre[0]+dr < grid.n_cells and 0 <= centre[1]+dc < grid.n_cells)


def record_bins(grid, record):
    heading = grid.heading_bin(record['heading'])
    return frozenset(((heading-1) % 8, heading, (heading+1) % 8))


@dataclass(frozen=True)
class OccupiedPatch:
    failure_id: str
    cells: frozenset
    headings: frozenset


@dataclass(frozen=True)
class RecoveryPermission:
    episode_time: float
    patches: tuple

    @classmethod
    def capture(cls, grid, xy, now):
        cell = grid.world_to_cell(*xy)
        patches = []
        for record in grid.failure_records:
            cells = record_cells(grid, record)
            if cell in cells:
                # Freeze only exclusions actually present now, not hypothetical
                # coverage of a subsequently enlarged/changed failure record.
                for heading in sorted(record_bins(grid, record)):
                    blocked = frozenset(c for c in cells if grid.approach_blocked[*c, heading]
                                        or grid.failures[*c, heading] >= 2)
                    if cell in blocked:
                        patches.append(OccupiedPatch(record['attempt_id'], blocked, frozenset((heading,))))
        return cls(now, tuple(patches))

    def observe(self, grid, xy):
        cell = grid.world_to_cell(*xy)
        return replace(self, patches=tuple(p for p in self.patches if cell in p.cells))

    def reason(self, grid, start, end):
        length = math.dist(start, end)
        if length < 1e-9:
            return ''
        heading = grid.heading_bin(math.atan2(end[1]-start[1], end[0]-start[0]))
        first = grid.world_to_cell(*start)
        patches = tuple(p for p in self.patches if heading in p.headings and first in p.cells)
        allowed = frozenset(c for p in patches for c in p.cells)
        ids = {p.failure_id for p in patches}
        new_patches = [record_cells(grid, r) for r in grid.failure_records
                       if r['attempt_id'] not in ids and heading in record_bins(grid, r)]
        previous = None
        exited = not bool(patches)
        for t in np.linspace(0., 1., max(2, math.ceil(length/(grid.cell_size*.25))+1)):
            cell = grid.world_to_cell(*(np.asarray(start)+t*np.subtract(end, start)))
            if cell is None:
                return 'directional escape leaves observed map'
            touched = [cell]
            if previous is not None and cell[0] != previous[0] and cell[1] != previous[1]:
                touched += [(cell[0], previous[1]), (previous[0], cell[1])]
            for item in touched:
                if not grid.direction_allowed(item, heading):
                    if exited or item not in allowed or any(item in cells for cells in new_patches):
                        return 'directional escape enters an unowned or retired failure patch'
            if cell not in allowed:
                exited = True
            previous = cell
        return ''
