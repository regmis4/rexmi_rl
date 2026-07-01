# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Procedural heightfield generators for lunar south pole crater wall terrains.

Each generator produces a **cross-section tile** of a crater wall — a sloped
terrain tile that represents the traversal corridor from crater floor to rim.

All slope values are derived from NASA LOLA ``LDEM_80S_40MPP_ADJ.TIF`` (40 m/px
south polar DEM, NASA PGDA).  Processing pipeline: ``scripts/process_lola_dem.py``.
Full radial profiles and locomotion analysis are in ``data/craters/<name>/``.
See ``docs/lunar_crater_terrain_research.md`` for the complete methodology.

**Slope definition:** All values are radial slope at a **200 m baseline**,
computed from the P50 (median) axisymmetric radial profile.

Three crater morphology classes are implemented (DEM-derived):

  Class A — Broad, shallow (Haworth archetype)
  -----------------------------------------------------------------------
  d/D ≈ 0.09, P50 wall slope 1–3°, max 9°.  Very gentle; current policy trivial.
  Best ingress azimuth 160°, max wall slope 9.2° (Haworth DEM).

  Class B — Moderate complex (Faustini archetype)  ← PRIMARY DEMO TARGET
  -----------------------------------------------------------------------
  d/D ≈ 0.07, main wall 14–16° (200 m baseline).  All zones within current policy.
  Best ingress azimuth 330°, max wall slope 13.6° (Faustini DEM).

  Class C — Deep, steep (Shackleton archetype)
  -----------------------------------------------------------------------
  d/D ≈ 0.19, main wall uniformly 28–32° (200 m baseline).
  Best ingress azimuth 130°, max wall slope 31.0°.  Requires Phase 8.

Isaac Lab terrain function API
------------------------------
Sub-terrain functions must follow the signature:

    @height_field_to_mesh
    def my_terrain(difficulty: float, cfg: HfTerrainBaseCfg) -> np.ndarray:
        ...
        return height_field_int16  # shape (x_pixels, y_pixels)

The ``@height_field_to_mesh`` decorator (from isaaclab.terrains.height_field.utils):
  • allocates the full pixel buffer (including 1-pixel zero border)
  • calls the function with cfg.size shrunk to exclude the border
  • inserts the returned array into the buffer
  • converts the complete buffer to a trimesh.Trimesh
  • returns ([mesh], origin)

Inside the function, pixel counts are derived from cfg.size and cfg.horizontal_scale:
    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)  # slope direction
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)  # lateral (uniform)
Heights are in raw integer units: raw = int(height_metres / cfg.vertical_scale)

Terrain tile geometry
---------------------
  Tile size     : 32 m × 32 m (set in TerrainGeneratorCfg.size)
  Resolution    : 10 cm/pixel (horizontal_scale=0.1)
  Height units  : 5 mm/raw unit (vertical_scale=0.005)
  Slope direction : x-axis (axis 0 of the array), x=0 → floor, x=max → rim
"""

import math
from collections.abc import Callable

import numpy as np

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


# ===========================================================================
# Internal height-field builder — shared by all three crater types
# ===========================================================================

def _build_crater_hf(
    cfg,
    zone_slopes_deg: list[float],
    zone_fractions: list[float],
    roughness_m: float,
    seed: int,
) -> np.ndarray:
    """
    Build a piecewise-sloped crater wall heightfield (internal helper).

    The terrain slopes along the x-axis (axis 0 of the returned array).
    x = 0 is the crater floor (lowest); x = x_max is the rim (highest).
    Heights are computed by integrating tan(slope) × dx from x=0.
    Gaussian roughness is added to simulate lunar regolith texture.

    Parameters
    ----------
    cfg
        Isaac Lab sub-terrain config. Used fields:
          cfg.size               — (width_x, length_y) in metres
          cfg.horizontal_scale   — metres per pixel
          cfg.vertical_scale     — metres per raw height unit
    zone_slopes_deg : list[float]
        Mean slope angle (°) per zone, ordered floor → rim.
    zone_fractions : list[float]
        Fractional x-width of each zone (must sum to 1.0).
    roughness_m : float
        Regolith surface roughness: uniform noise ±roughness_m/2 per pixel.
    seed : int
        Random seed for reproducible noise pattern.

    Returns
    -------
    np.ndarray
        Integer height field, shape (x_pixels, y_pixels), dtype int16.
        Values in raw units: raw × cfg.vertical_scale = height in metres.
    """
    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)   # slope direction
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)   # lateral direction
    total_x_m = x_pixels * cfg.horizontal_scale          # total traverse in metres

    # -----------------------------------------------------------------------
    # Zone boundary positions along x (metres)
    # -----------------------------------------------------------------------
    boundaries_m = [0.0]
    for frac in zone_fractions:
        boundaries_m.append(boundaries_m[-1] + frac * total_x_m)

    # -----------------------------------------------------------------------
    # 1D height profile: integrate tan(slope) × dx from x=0
    # -----------------------------------------------------------------------
    heights_1d = np.zeros(x_pixels, dtype=np.float64)
    for i in range(1, x_pixels):
        x = i * cfg.horizontal_scale

        zone_idx = len(zone_fractions) - 1   # default: last zone
        for z in range(len(zone_fractions)):
            if x < boundaries_m[z + 1] + 1e-9:
                zone_idx = z
                break

        slope_rad = math.radians(zone_slopes_deg[zone_idx])
        heights_1d[i] = heights_1d[i - 1] + cfg.horizontal_scale * math.tan(slope_rad)

    # -----------------------------------------------------------------------
    # Expand to 2D: slope varies along axis-0 (x), uniform along axis-1 (y)
    # heights_1d shape (x_pixels,) → (x_pixels, y_pixels)
    # -----------------------------------------------------------------------
    heights_2d = np.tile(heights_1d[:, np.newaxis], (1, y_pixels)).astype(np.float64)

    # -----------------------------------------------------------------------
    # Add regolith roughness (uniform noise ±roughness_m/2)
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(seed=seed)
    noise = rng.uniform(-roughness_m / 2.0, roughness_m / 2.0, heights_2d.shape)
    heights_2d += noise

    # Ensure global minimum is at z=0 (the @height_field_to_mesh decorator
    # places the terrain origin at the centre-tile height, not at z=0, so this
    # shift is purely cosmetic but keeps values non-negative)
    heights_2d -= heights_2d.min()

    # -----------------------------------------------------------------------
    # Convert metres → raw integer units
    # -----------------------------------------------------------------------
    return (heights_2d / cfg.vertical_scale).astype(np.int16)


# ===========================================================================
# Public terrain generator functions — one per crater morphology class
# ===========================================================================

@height_field_to_mesh
def type1_ancient_crater_wall(difficulty: float, cfg) -> np.ndarray:
    """
    Class A crater wall — ancient, heavily degraded (Haworth archetype).

    DEM-derived morphometry (LOLA LDEM_80S_40MPP_ADJ.TIF):
      Rim radius: 27.06 km  |  Depth: 5 016 m  |  d/D: 0.093
      Best ingress azimuth: 160°
      Max wall slope (200 m baseline, best azimuth): 9.2°
      P50 wall slope: 1.3°

    Profile (for 32 m tile, floor → rim), slope at 200 m baseline:
      Approach floor (15%, 4.8 m):   1.0° — flat PSR floor region
      Lower wall    (35%, 11.2 m):   5.0° — very gentle degraded lower slope
      Mid wall      (25%, 8.0 m):    7.5° — moderate middle section
      Upper wall    (20%, 6.4 m):    9.0° — steepest zone (matches DEM max)
      Rim crest     ( 5%, 1.6 m):    4.0° — degraded, rounded rim

    Max tile height ≈ 2.1 m over 32 m.
    All zones ✅ within current Go2W policy capability (Phase 6-Optimized).
    """
    return _build_crater_hf(
        cfg,
        zone_slopes_deg=[1.0, 5.0, 7.5, 9.0, 4.0],
        zone_fractions=[0.15, 0.35, 0.25, 0.20, 0.05],
        roughness_m=cfg.roughness_m,
        seed=cfg.seed,
    )


@height_field_to_mesh
def type2_faustini_crater_wall(difficulty: float, cfg) -> np.ndarray:
    """
    Class B crater wall — Faustini archetype (PRIMARY DEMO TARGET).

    DEM-derived morphometry (LOLA LDEM_80S_40MPP_ADJ.TIF):
      Rim radius: 22.86 km  |  Depth: 3 336 m  |  d/D: 0.073
      Best ingress azimuth: 330°
      Max wall slope (200 m baseline, best azimuth): 13.6°
      P50 wall slope: 7.8°

    Radial profile zones (r_norm → slope at 200 m baseline):
      r=0.40 → 4.4°  (floor transition)
      r=0.50 → 16.0°  (lower wall — steepest)
      r=0.60 → 14.3°  (mid wall)
      r=0.70 → 14.3°  (mid wall)
      r=0.80 → 13.9°  (upper wall)
      r=0.90 →  9.2°  (rim approach)

    Profile (for 32 m tile, floor → rim), slope at 200 m baseline:
      Approach floor (10%, 3.2 m):   4.0° — floor transition (DEM: 4.4°)
      Lower wall     (25%, 8.0 m):  15.5° — steepest section (DEM: 14–16°)
      Mid wall       (35%, 11.2 m): 14.0° — main wall (DEM: 14.3°)
      Upper wall     (20%, 6.4 m):  12.0° — DEM: 13.9° → 9.2°
      Rim crest      (10%, 3.2 m):   6.5° — rim approach, decreasing

    Max tile height ≈ 7.0 m over 32 m.
    All zones ✅ within current Go2W policy capability (Phase 6-Optimized).
    """
    return _build_crater_hf(
        cfg,
        zone_slopes_deg=[4.0, 15.5, 14.0, 12.0, 6.5],
        zone_fractions=[0.10, 0.25, 0.35, 0.20, 0.10],
        roughness_m=cfg.roughness_m,
        seed=cfg.seed,
    )


@height_field_to_mesh
def type3_shackleton_crater_wall(difficulty: float, cfg) -> np.ndarray:
    """
    Class C crater wall — Shackleton archetype (young, steep, best-preserved).

    DEM-derived morphometry (LOLA LDEM_80S_40MPP_ADJ.TIF):
      Rim radius: 10.92 km  |  Depth: 4 077 m  |  d/D: 0.187
      Best ingress azimuth: 130°
      Max wall slope (200 m baseline, best azimuth): 31.0°
      P50 wall slope: 28.6°
      Validation: DEM depth 4 077 m vs. Zuber et al. 2012 (Science) 4 100 ± 50 m ✓

    Radial profile zones (r_norm → slope at 200 m baseline):
      r=0.20 →  5.3°  (lower floor transition)
      r=0.30 → 13.3°  (floor-to-wall transition)
      r=0.40 → 28.4°  (lower wall, steep onset)
      r=0.50 → 30.4°  (main wall)
      r=0.60 → 31.4°  (main wall, peak)
      r=0.70 → 31.9°  (main wall)
      r=0.80 → 30.6°  (main wall)
      r=0.90 → 21.8°  (upper wall / rim transition)

    Profile (for 32 m tile, floor → rim), slope at 200 m baseline:
      Floor           (10%, 3.2 m):   2.0° — near-flat floor (DEM: 0–5°)
      Transition      (15%, 4.8 m):  20.0° — DEM: 13–28° transition zone
      Main wall       (55%, 17.6 m): 31.0° — DEM: uniformly 28–32°
      Upper wall      (15%, 4.8 m):  22.0° — DEM: 21.8° at r=0.9
      Rim crest       ( 5%, 1.6 m):  12.0° — rim approach

    Max tile height ≈ 15.0 m over 32 m.
    ⚠️  Main wall (31°) and transition (20°) EXCEED current policy capability (~20°).
    Requires Phase 8 training (slope_range up to 30°+) before full demo.
    Floor and lower zone traversable with current policy.
    """
    return _build_crater_hf(
        cfg,
        zone_slopes_deg=[2.0, 20.0, 31.0, 22.0, 12.0],
        zone_fractions=[0.10, 0.15, 0.55, 0.15, 0.05],
        roughness_m=cfg.roughness_m,
        seed=cfg.seed,
    )


# ===========================================================================
# Sub-terrain config classes
# ===========================================================================

@configclass
class CraterType1WallCfg(HfTerrainBaseCfg):
    """
    Class A crater wall terrain configuration.

    Archetype: Haworth (51.4 km, Pre-Nectarian, d/D=0.093)
    DEM max wall slope: 9.2°  |  Demo readiness: ✅ ready now

    Inherits from HfTerrainBaseCfg:
      horizontal_scale = 0.1 m/pixel
      vertical_scale   = 0.005 m/unit
      border_width     = 0.0 m
      size             = overridden by TerrainGeneratorCfg.size (32 m × 32 m)
    """
    function: Callable = type1_ancient_crater_wall
    proportion: float = 1.0

    # Regolith roughness amplitude (±roughness_m/2 per pixel)
    roughness_m: float = 0.04

    # Fixed seed for reproducible tile geometry
    seed: int = 1


@configclass
class CraterType2WallCfg(HfTerrainBaseCfg):
    """
    Class B crater wall terrain configuration.

    Archetype: Faustini (42.5 km, Nectarian, d/D=0.073)
    DEM max wall slope: 13.6°, P50: 7.8°  |  Demo readiness: ✅ PRIMARY DEMO TARGET

    All zones traversable with the Phase 6-Optimized rough-terrain policy.
    Peak slope (15.5°) is well within the trained range (~20° max).
    """
    function: Callable = type2_faustini_crater_wall
    proportion: float = 1.0

    # 3 cm roughness matches training terrain amplitude
    roughness_m: float = 0.03

    seed: int = 2


@configclass
class CraterType3WallCfg(HfTerrainBaseCfg):
    """
    Class C crater wall terrain configuration.

    Archetype: Shackleton (20.9 km, Imbrian, d/D=0.187)
    DEM max wall slope: 31.0°, P50: 28.6°  |  Demo readiness: ✅ Phase 8 policy (~33°)

    Floor zone (2°) and main wall (31°) both within Phase 8 policy capability.
    Shackleton depth validated: DEM 4 077 m vs. Zuber et al. 2012 (4 100 ± 50 m).
    """
    function: Callable = type3_shackleton_crater_wall
    proportion: float = 1.0

    # Shackleton walls have less accumulated regolith — smoother surface
    roughness_m: float = 0.02

    seed: int = 3


# ===========================================================================
# Rocky Pyramid Slope — sloped terrain with boulders for robustness training
# ===========================================================================

@height_field_to_mesh
def rocky_pyramid_slope(difficulty: float, cfg) -> np.ndarray:
    """
    Pyramid slope with difficulty-scaled boulders and surface roughness.

    Used to train policy robustness on textured steep terrain — the scenario
    encountered in the crater bowl (sloped surface + scattered rocks simultaneously).

    The ``difficulty`` parameter (0.0 at curriculum row 0, 1.0 at row 9)
    simultaneously scales slope angle, boulder count, boulder size, and roughness:

      difficulty = 0.0  →  slope = 15°,  3 boulders, h_max =  5 cm, roughness = 1 cm
      difficulty = 0.5  →  slope = 25°, 14 boulders, h_max = 12 cm, roughness = 3.5 cm
      difficulty = 1.0  →  slope = 35°, 25 boulders, h_max = 20 cm, roughness = 6 cm

    The robot warms up on familiar slopes (15°, 3 tiny rocks) and progressively
    encounters the crater-wall reality (35°, 25 rocks, 6 cm texture) — no cold start.

    Tile layout:
      • 8 m × 8 m pyramid slope (same tile size as steep-slope training)
      • 2 m flat platform in the tile centre — robot spawns here
      • Slope rises outward from the platform edge to the tile border
      • Boulders scattered across the full tile (slope AND platform)
        so the robot learns to handle rocks in all phases of traversal

    Parameters
    ----------
    difficulty : float
        Curriculum difficulty in [0.0, 1.0].  Row 0 = 0.0, row 9 = 1.0.
    cfg : RockyPyramidSlopeCfg
        Configuration dataclass with all terrain parameters.

    Returns
    -------
    np.ndarray  shape (x_pixels, y_pixels), dtype int16
        Height field in raw units (raw × vertical_scale = metres).
    """
    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hx, hy = x_pixels // 2, y_pixels // 2

    # ------------------------------------------------------------------
    # Difficulty-scaled parameters
    # ------------------------------------------------------------------
    slope_deg    = cfg.slope_min_deg + difficulty * (cfg.slope_max_deg - cfg.slope_min_deg)
    n_boulders   = int(cfg.boulder_count_min
                       + difficulty * (cfg.boulder_count_max - cfg.boulder_count_min))
    b_height_max = (cfg.boulder_height_min
                    + difficulty * (cfg.boulder_height_max - cfg.boulder_height_min))
    roughness    = cfg.roughness_min_m + difficulty * (cfg.roughness_max_m - cfg.roughness_min_m)

    # ------------------------------------------------------------------
    # Build pyramid slope heightfield
    # ------------------------------------------------------------------
    # Tile-centred physical coordinates (metres)
    x_m = (np.arange(x_pixels) - hx) * cfg.horizontal_scale
    y_m = (np.arange(y_pixels) - hy) * cfg.horizontal_scale
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')

    # Distance from the flat platform edge (Chebyshev / L∞ metric creates a
    # square platform matching Isaac Lab's HfPyramidSlopedTerrainCfg geometry)
    platform_half = cfg.platform_width / 2.0
    dist_from_platform = np.maximum(
        0.0,
        np.maximum(np.abs(X) - platform_half, np.abs(Y) - platform_half),
    )
    h = math.tan(math.radians(slope_deg)) * dist_from_platform

    # ------------------------------------------------------------------
    # Surface roughness — uniform noise, difficulty-scaled amplitude
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=cfg.seed)
    h += rng.uniform(-roughness / 2.0, roughness / 2.0, h.shape)

    # ------------------------------------------------------------------
    # Scatter boulders across the full tile (slope + platform)
    # ------------------------------------------------------------------
    # h_min is fixed so there are always some small rocks even at row 0.
    # h_max scales with difficulty so hard rows have larger boulders.
    half_x = cfg.size[0] / 2.0
    half_y = cfg.size[1] / 2.0
    _add_boulders(
        h, X, Y, rng,
        n_boulders,
        cfg.boulder_height_min, b_height_max,
        cfg.boulder_radius_min, cfg.boulder_radius_max,
        -half_x, half_x, -half_y, half_y,
    )

    # Ensure global minimum = 0 (consistent with other terrain generators)
    h -= h.min()
    return (h / cfg.vertical_scale).astype(np.int16)


@configclass
class RockyPyramidSlopeCfg(HfTerrainBaseCfg):
    """
    Rocky pyramid slope terrain configuration.

    Combines a pyramid slope (identical geometry to HfPyramidSlopedTerrainCfg)
    with randomly scattered Gaussian boulder bumps and surface roughness.
    Used by ``Go2wRockySlopeEnvCfg`` to train robustness on textured steep terrain.

    Parameters tagged with "(difficulty-scaled)" are linearly interpolated between
    their _min and _max values as curriculum difficulty increases from 0.0 to 1.0.

    Inherits from HfTerrainBaseCfg:
      horizontal_scale = 0.1 m/pixel
      vertical_scale   = 0.005 m/unit
      border_width     = 0.0 m
      size             = set by TerrainGeneratorCfg.size (8 m × 8 m)
    """
    function: Callable = rocky_pyramid_slope
    proportion: float = 1.0

    # Slope angle range (difficulty-scaled)
    slope_min_deg: float = 15.0   # row 0: 15° — warm-up, already known from steep-slope training
    slope_max_deg: float = 35.0   # row 9: 35° — current physical capability ceiling

    # Flat platform in tile centre (robot spawn zone)
    platform_width: float = 2.0   # metres, matches HfPyramidSlopedTerrainCfg default

    # Surface roughness amplitude (difficulty-scaled, uniform noise ±roughness/2)
    roughness_min_m: float = 0.010   # 1 cm at easiest row — nearly clean
    roughness_max_m: float = 0.060   # 6 cm at hardest (matches crater floor roughness)

    # Boulder count (difficulty-scaled)
    boulder_count_min: int = 3    # 3 rocks at row 0 — almost a clean slope
    boulder_count_max: int = 25   # 25 rocks at row 9 — heavily cluttered crater wall

    # Boulder height: h_min fixed; h_max difficulty-scaled
    boulder_height_min: float = 0.05   # 5 cm always present (even easiest row)
    boulder_height_max: float = 0.20   # 20 cm at hardest (matches rough-terrain box heights)

    # Boulder width/sigma = radius / 2 — wide relative to height for flat slab appearance
    boulder_radius_min: float = 0.15   # narrow, sharp-edged rocks
    boulder_radius_max: float = 0.50   # wide, flat slabs (matches exterior bowl boulders)

    # Random seed for reproducible tile geometry
    seed: int = 99


# ===========================================================================
# Rocky Inverted Pyramid Slope — downhill variant for descent training
# ===========================================================================

@height_field_to_mesh
def rocky_pyramid_slope_down(difficulty: float, cfg) -> np.ndarray:
    """
    Inverted pyramid slope with difficulty-scaled boulders — DOWNHILL variant.

    Mirrors ``rocky_pyramid_slope()`` exactly but with the slope direction
    REVERSED: the flat platform at the tile centre is the **highest** point,
    and terrain falls away toward the tile edges.  The robot spawns at the
    centre (high) and is commanded FORWARD — it descends the slope.

    Used for controlled-descent training (Phase 8c): the policy must learn to
    actively brake wheels as gravity accelerates it down the crater wall.

    Difficulty scaling — identical to the uphill variant:
      difficulty = 0.0  →  slope = 15°,  3 boulders, h_max =  5 cm, roughness = 1 cm
      difficulty = 0.5  →  slope = 25°, 14 boulders, h_max = 12 cm, roughness = 3.5 cm
      difficulty = 1.0  →  slope = 35°, 25 boulders, h_max = 20 cm, roughness = 6 cm

    Tile layout: same 8 m × 8 m tile, 2 m flat platform at the peak (centre).
    Boulders scattered across the full tile including the flat platform — the
    robot must handle rocks on both the descent AND the launch zone.
    """
    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hx, hy = x_pixels // 2, y_pixels // 2

    # ------------------------------------------------------------------
    # Difficulty-scaled parameters (same as uphill variant)
    # ------------------------------------------------------------------
    slope_deg    = cfg.slope_min_deg + difficulty * (cfg.slope_max_deg - cfg.slope_min_deg)
    n_boulders   = int(cfg.boulder_count_min
                       + difficulty * (cfg.boulder_count_max - cfg.boulder_count_min))
    b_height_max = (cfg.boulder_height_min
                    + difficulty * (cfg.boulder_height_max - cfg.boulder_height_min))
    roughness    = cfg.roughness_min_m + difficulty * (cfg.roughness_max_m - cfg.roughness_min_m)

    # ------------------------------------------------------------------
    # Build INVERTED pyramid slope heightfield (peak at centre)
    # ------------------------------------------------------------------
    x_m = (np.arange(x_pixels) - hx) * cfg.horizontal_scale
    y_m = (np.arange(y_pixels) - hy) * cfg.horizontal_scale
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')

    platform_half = cfg.platform_width / 2.0
    dist_from_platform = np.maximum(
        0.0,
        np.maximum(np.abs(X) - platform_half, np.abs(Y) - platform_half),
    )

    # Maximum distance from platform to tile edge
    max_dist = max(cfg.size[0] / 2.0 - platform_half, cfg.size[1] / 2.0 - platform_half)
    max_h = math.tan(math.radians(slope_deg)) * max_dist

    # INVERTED: platform = max_h (highest), tile edges = 0 (lowest)
    h = max_h - math.tan(math.radians(slope_deg)) * dist_from_platform

    # ------------------------------------------------------------------
    # Surface roughness — use a different seed offset to produce a
    # different boulder/roughness pattern from the uphill companion tile
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=cfg.seed + 1000)
    h += rng.uniform(-roughness / 2.0, roughness / 2.0, h.shape)

    # ------------------------------------------------------------------
    # Scatter boulders across the full tile
    # ------------------------------------------------------------------
    half_x = cfg.size[0] / 2.0
    half_y = cfg.size[1] / 2.0
    _add_boulders(
        h, X, Y, rng,
        n_boulders,
        cfg.boulder_height_min, b_height_max,
        cfg.boulder_radius_min, cfg.boulder_radius_max,
        -half_x, half_x, -half_y, half_y,
    )

    # Ensure global minimum = 0
    h -= h.min()
    return (h / cfg.vertical_scale).astype(np.int16)


@configclass
class RockyPyramidSlopeDownCfg(RockyPyramidSlopeCfg):
    """
    Downhill rocky pyramid slope — robot descends from the centre platform.

    Inherits all parameters from ``RockyPyramidSlopeCfg`` (same slope range,
    boulder counts, heights, roughness, platform width, seed).  The only
    difference is the terrain generator function: ``rocky_pyramid_slope_down``
    inverts the height profile so the platform is the peak and the robot
    descends when commanded forward.

    Used alongside ``RockyPyramidSlopeCfg`` in a 50 / 50 terrain split so
    the policy trains equally on uphill and downhill scenarios.
    """
    function: Callable = rocky_pyramid_slope_down


# ===========================================================================
# Demo Bowl — full radial crater at robot scale (NEW HEADLINE DEMO TERRAIN)
# ===========================================================================

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_boulders(h, X, Y, rng, n, h_min, h_max, r_min, r_max,
                  x_lo, x_hi, y_lo, y_hi):
    """Scatter n Gaussian boulder bumps in the given rectangle, in-place."""
    if n <= 0:
        return
    bx = rng.uniform(x_lo, x_hi, n)
    by = rng.uniform(y_lo, y_hi, n)
    bh = rng.uniform(h_min, h_max, n)
    br = rng.uniform(r_min, r_max, n)
    for k in range(n):
        s = (br[k] / 2.0) ** 2
        h += bh[k] * np.exp(-((X - bx[k])**2 + (Y - by[k])**2) / (2.0 * s))


def _add_boulders_radial(h, X, Y, rng, n, h_min, h_max, r_min, r_max,
                          cx, cy, ring_r_lo, ring_r_hi):
    """Scatter n boulders in a radial ring around (cx, cy), in-place."""
    if n <= 0:
        return
    rr   = rng.uniform(ring_r_lo, ring_r_hi, n)
    phi  = rng.uniform(-math.pi, math.pi, n)
    bx   = cx + rr * np.cos(phi)
    by   = cy + rr * np.sin(phi)
    bh   = rng.uniform(h_min, h_max, n)
    br   = rng.uniform(r_min, r_max, n)
    for k in range(n):
        s = (br[k] / 2.0) ** 2
        h += bh[k] * np.exp(-((X - bx[k])**2 + (Y - by[k])**2) / (2.0 * s))


@height_field_to_mesh
def lunar_crater_demo_bowl(difficulty: float, cfg) -> np.ndarray:
    """
    Full radial crater bowl — geologically realistic lunar terrain.

    The 64 m × 64 m tile is filled with a complete Shackleton-class crater (22 m
    diameter) plus a rich SW mountain massif, secondary crater, scattered hills,
    linear fractures, and boulder fields across the entire tile.

    Geological components
    ---------------------
    1. Base undulating terrain       — multi-scale sinusoidal regolith topography
    2. Main crater bowl              — Shackleton-class, 25–35° walls
    3. Azimuthal variation + ejecta  — realism layers on the main crater
    4. Crater geological features    — concentric fractures, scarp, debris apron
    5. SW mountain massif            — large non-symmetric ridge at (−15, −13) m;
                                       8 overlapping asymmetric Gaussian peaks,
                                       max height 7.8 m, slope ~26–28°, ridgeline
                                       undulations and 50 boulders
    6. Secondary NW dome             — smaller hill at (−20, +16) m, 3.2 m tall
    7. Secondary SE dome             — hill at (+16, −20) m, 2.4 m tall
    8. Secondary small crater (SE)   — 8 m dia, 22° max slope at (+20, −18) m
    9. Linear ground fractures       — 5 thermal-contraction cracks across tile
    10. Boulder fields               — crater wall (10), exterior rim (80 default),
                                       tile-wide scatter (100), mountain (50),
                                       secondary crater (15)
    11. Spatially-varying roughness  — 2–8 cm, keyed to feature type
    """
    x_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    y_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # Tile-centred physical coordinates (metres); origin = crater centre
    hx, hy = x_pixels // 2, y_pixels // 2
    x_m = (np.arange(x_pixels) - hx) * cfg.horizontal_scale
    y_m = (np.arange(y_pixels) - hy) * cfg.horizontal_scale
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')
    R   = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    r_floor = cfg.r_floor    # 3.0 m
    r_rim   = cfg.r_rim      # 11.0 m

    rng = np.random.default_rng(seed=cfg.seed)

    # ==================================================================
    # 1. BASE TERRAIN — multi-scale undulating regolith plain
    # ==================================================================
    # Overlapping sine waves simulate the long-wavelength topographic
    # variation visible in LOLA DEMs (ancient ejecta redistribution,
    # isostatic adjustment, regolith creep over billions of years).
    h = (
        0.35 * np.sin(2*math.pi*X/28.0 + 0.73) * np.cos(2*math.pi*Y/23.0 - 0.31)
      + 0.22 * np.cos(2*math.pi*X/17.0 - 1.21) * np.sin(2*math.pi*Y/19.0 + 0.87)
      + 0.12 * np.sin(2*math.pi*(X+Y)/11.0 + 1.54)
      + 0.07 * np.cos(2*math.pi*X/7.0 + 0.42) * np.cos(2*math.pi*Y/6.0 - 1.10)
      + 0.04 * np.sin(2*math.pi*(X-Y)/5.0 - 0.88)
    )
    # Slight background tilt (~0.7° across tile, mimics far-field terrain slope)
    h += 0.012 * X - 0.008 * Y

    # ==================================================================
    # 2. MAIN CRATER BOWL — piecewise slope profile
    # ==================================================================
    wall_zones = [
        (r_floor,          r_floor + 1.0, 20.0),
        (r_floor + 1.0,    r_floor + 5.0, 30.0),
        (r_floor + 5.0,    r_rim   - 1.0, 33.0),
        (r_rim   - 1.0,    r_rim,         23.0),
    ]
    h_bowl = np.zeros((x_pixels, y_pixels), dtype=np.float64)
    h_at_rim = 0.0
    for (r0, r1, slope_deg) in wall_zones:
        tan_s = math.tan(math.radians(slope_deg))
        mask  = (R >= r0) & (R < r1)
        h_bowl[mask] = h_at_rim + tan_s * (R[mask] - r0)
        h_at_rim += tan_s * (r1 - r0)

    # Exterior ramp
    tan_ext = math.tan(math.radians(cfg.exterior_slope_deg))
    h_bowl[R >= r_rim] = h_at_rim - tan_ext * (R[R >= r_rim] - r_rim)
    h_bowl = np.maximum(h_bowl, 0.0)

    # Smooth out undulation inside the flat floor (floor should be nearly flat)
    floor_smooth = np.clip(1.0 - R / r_floor, 0.0, 1.0) ** 2
    h = h * (1.0 - floor_smooth)   # suppress undulation at crater centre
    h += h_bowl

    # Azimuthal slope variation (±az_variation_m at mid-wall)
    wall_frac   = np.clip((R - r_floor) / (r_rim - r_floor), 0.0, 1.0)
    az_envelope = 4.0 * wall_frac * (1.0 - wall_frac)
    az_signal   = (np.sin(2.0*PHI) + 0.40*np.sin(3.0*PHI+1.1) + 0.20*np.cos(5.0*PHI-0.3))
    h += cfg.az_variation_m * az_signal * az_envelope

    # Ejecta blanket — raised ring just outside rim
    h += 0.20 * np.exp(-((R - (r_rim + 2.5))**2) / (2*2.0**2)) * (R > r_rim - 1.0)

    # ==================================================================
    # 3. CRATER GEOLOGICAL FEATURES
    # ==================================================================
    frac_mask = (R > r_floor + 0.5) & (R < r_rim - 0.5)

    # Concentric circumferential fractures
    for r_frac, d_m, w_m in zip(cfg.fracture_radii, cfg.fracture_depths, cfg.fracture_widths):
        h -= d_m * np.exp(-((R - r_frac)**2) / (2*(w_m/2.5)**2)) * frac_mask

    # Mid-wall mass-wasting scarp
    h += (cfg.scarp_height_m
          * np.exp(-((R - cfg.scarp_radius_m)**2) / (2*(cfg.scarp_width_m/2.5)**2))
          * frac_mask)

    # Floor debris apron
    h += 0.06 * np.exp(-((R - (r_floor + 0.8))**2) / (2*0.7**2))

    # ==================================================================
    # 4. SW MOUNTAIN HIGHLAND — broad continuous elevated terrain
    # ==================================================================
    # NOT isolated peaks — a wide highland that forms the background landscape.
    # Three overlapping Gaussians of increasing width create a natural
    # continuous rise toward the SW corner.
    #
    # Summit placed at (−22, −20) so that interference at crater rim (r=11 m)
    # is ≤ 5 % of peak height:
    #   dist(summit, rim) ≈ 22.8 m → exp(−22.8²/(2×9.5²)) ≈ 0.056  (5.6 %)
    #
    # Near summit : tight Gaussian, clearly visible peak above horizon
    # Mid base    : wide base creates the broad "highland plateau" feel
    # Far fill    : terrain keeps rising toward SW tile edge (depth cue)
    h += 4.5 * np.exp(-((X + 22.0)**2 + (Y + 20.0)**2) / (2*9.5**2))   # near summit
    h += 3.0 * np.exp(-((X + 28.0)**2 + (Y + 24.0)**2) / (2*16.0**2))  # broad base
    h += 2.2 * np.exp(-((X + 31.0)**2 + (Y + 28.0)**2) / (2*11.0**2))  # SW depth fill

    # Mountain surface spurs — two elongated non-symmetric bumps simulate
    # geological outcrops / lateral ridges on the highland side.
    # These are simple asymmetric Gaussians, NOT regular sine waves.
    # Spur A (NE flank of highland): elongated NW–SE  (~0.9 m)
    h += 0.90 * np.exp(-((X + 18.0)**2 / (2*4.0**2) + (Y + 15.0)**2 / (2*8.5**2)))
    # Spur B (S flank): elongated E–W, lower (~0.65 m)
    h += 0.65 * np.exp(-((X + 25.0)**2 / (2*7.5**2) + (Y + 22.5)**2 / (2*3.5**2)))

    # m_env used downstream for roughness weighting
    m_env = np.exp(-((X + 22.0)**2 + (Y + 20.0)**2) / (2*14.0**2))

    # Mountain boulders — flat angular rocks on highland slopes.
    # Wide sigma (0.30–0.55 m) relative to height (0.10–0.35 m) → flat slab look.
    _add_boulders_radial(h, X, Y, rng,
                         cfg.mountain_boulder_count,
                         cfg.mountain_boulder_height_min, cfg.mountain_boulder_height_max,
                         cfg.mountain_boulder_radius_min, cfg.mountain_boulder_radius_max,
                         -22.0, -20.0,
                         0.5, 16.0)

    # ==================================================================
    # 4b. NW SECONDARY MOUNTAIN — half height, ~10 % rim interference
    # ==================================================================
    # Summit at (−20, +20): NW quadrant, visible as backdrop on opposite
    # side from SW massif.  Peak = 4.5 / 2 = 2.25 m.
    # Interference at rim: nearest rim pt ≈ (−7.8, +7.8); dist to summit
    #   = sqrt(12.2²+12.2²) = 17.3 m → exp(−17.3²/(2×8²)) ≈ 9.7 % ✓
    h += 2.25 * np.exp(-((X + 20.0)**2 + (Y - 20.0)**2) / (2*8.0**2))   # near summit
    h += 1.50 * np.exp(-((X + 26.0)**2 + (Y - 25.0)**2) / (2*14.0**2))  # broad base
    # Single asymmetric spur on NW mountain (no repeating waves)
    h += 0.55 * np.exp(-((X + 17.0)**2 / (2*3.0**2) + (Y - 17.0)**2 / (2*6.5**2)))
    _add_boulders_radial(h, X, Y, rng, 20,
                         0.08, 0.28, 0.25, 0.48,
                         -20.0, 20.0, 0.5, 12.0)


    # ==================================================================
    # 5. SECONDARY LANDSCAPE FEATURES
    # ==================================================================

    # NW rolling hill — gently elevated terrain NW of crater
    h += 1.8 * np.exp(-((X + 18.0)**2 + (Y - 15.0)**2) / (2*9.0**2))
    # Small asymmetric rock outcrop on NW hill (no repeating waves)
    h += 0.40 * np.exp(-((X + 16.0)**2 / (2*3.5**2) + (Y - 12.0)**2 / (2*7.0**2)))
    _add_boulders(h, X, Y, rng, 12,
                  0.08, 0.28, 0.28, 0.50,
                  -27.0, -10.0, 8.0, 22.0)

    # SE rolling terrain — broad gentle elevation SE
    h += 1.5 * np.exp(-((X - 18.0)**2 / (2*10.0**2) + (Y + 20.0)**2 / (2*7.0**2)))
    _add_boulders(h, X, Y, rng, 8,
                  0.06, 0.22, 0.25, 0.45,
                  10.0, 26.0, -28.0, -12.0)

    # NE ridge — behind robot spawn, adds depth to backdrop
    h += 1.2 * np.exp(-((X - 22.0)**2 / (2*6.0**2) + (Y - 13.0)**2 / (2*4.0**2)))

    # ==================================================================
    # 6. SECONDARY SMALL CRATER (SE quadrant)
    # ==================================================================
    # Diameter 8 m (r_rim = 4 m), depth 1.0 m → max slope 22° — within 25°.
    # Adds visual interest in the SE portion of the tile.
    sc_cx, sc_cy = 20.0, -18.0
    sc_r_floor   = 1.5
    sc_r_rim     = 4.0
    sc_depth     = 1.0   # atan(1.0 / 2.5) = 21.8° ✓
    sc_R = np.sqrt((X - sc_cx)**2 + (Y - sc_cy)**2)
    sc_floor_mask = sc_R < sc_r_floor
    sc_wall_mask  = (sc_R >= sc_r_floor) & (sc_R < sc_r_rim)
    sc_tan = sc_depth / (sc_r_rim - sc_r_floor)
    h[sc_floor_mask] -= sc_depth
    h[sc_wall_mask]  -= (sc_depth - sc_tan * (sc_R[sc_wall_mask] - sc_r_floor))
    # Ejecta rim
    h += 0.10 * np.exp(-((sc_R - (sc_r_rim + 1.0))**2) / (2*1.0**2)) * (sc_R > sc_r_rim)
    # Secondary crater boulders
    _add_boulders_radial(h, X, Y, rng, 15,
                         0.08, 0.35, 0.15, 0.35,
                         sc_cx, sc_cy, sc_r_floor, sc_r_rim + 3.0)

    # ==================================================================
    # 6b. TINY CRATER 2 — N/NE quadrant at (+14, +22)
    # ==================================================================
    # r_rim=3.5 m, depth=0.85 m → max slope atan(0.85/2.3) ≈ 20° ✓
    # Well clear of main crater (dist=26 m) and spawn zone (dist=22 m).
    sc2_cx, sc2_cy   = 14.0, 22.0
    sc2_r_floor, sc2_r_rim, sc2_depth = 1.2, 3.5, 0.85
    sc2_R   = np.sqrt((X - sc2_cx)**2 + (Y - sc2_cy)**2)
    sc2_tan = sc2_depth / (sc2_r_rim - sc2_r_floor)
    h[sc2_R < sc2_r_floor] -= sc2_depth
    sc2_wall = (sc2_R >= sc2_r_floor) & (sc2_R < sc2_r_rim)
    h[sc2_wall] -= (sc2_depth - sc2_tan * (sc2_R[sc2_wall] - sc2_r_floor))
    h += 0.09 * np.exp(-((sc2_R - (sc2_r_rim + 0.9))**2) / (2*0.8**2)) * (sc2_R > sc2_r_rim)
    _add_boulders_radial(h, X, Y, rng, 12,
                         0.07, 0.28, 0.15, 0.32,
                         sc2_cx, sc2_cy, sc2_r_floor, sc2_r_rim + 2.5)

    # ==================================================================
    # 6c. TINY CRATER 3 — NW quadrant at (−12, +24)
    # ==================================================================
    # r_rim=3.0 m, depth=0.75 m → max slope atan(0.75/2.0) ≈ 21° ✓
    # dist to main crater = 27 m; well outside NW mountain shoulder.
    sc3_cx, sc3_cy   = -12.0, 24.0
    sc3_r_floor, sc3_r_rim, sc3_depth = 1.0, 3.0, 0.75
    sc3_R   = np.sqrt((X - sc3_cx)**2 + (Y - sc3_cy)**2)
    sc3_tan = sc3_depth / (sc3_r_rim - sc3_r_floor)
    h[sc3_R < sc3_r_floor] -= sc3_depth
    sc3_wall = (sc3_R >= sc3_r_floor) & (sc3_R < sc3_r_rim)
    h[sc3_wall] -= (sc3_depth - sc3_tan * (sc3_R[sc3_wall] - sc3_r_floor))
    h += 0.08 * np.exp(-((sc3_R - (sc3_r_rim + 0.8))**2) / (2*0.7**2)) * (sc3_R > sc3_r_rim)
    _add_boulders_radial(h, X, Y, rng, 10,
                         0.06, 0.25, 0.14, 0.30,
                         sc3_cx, sc3_cy, sc3_r_floor, sc3_r_rim + 2.0)

    # ==================================================================
    # 7. LINEAR GROUND FRACTURES (thermal contraction cracks)
    # ==================================================================
    # Each fracture is a narrow Gaussian trough along a line segment.
    # Parameters: (line_x0, line_y0, direction_angle_rad, half_length_m, depth_m, width_m)
    fractures = [
        ( -5.0,   9.0,  0.35, 11.0, 0.07, 0.50),
        (  9.0,  -4.0,  1.20,  9.0, 0.06, 0.45),
        ( -8.0,  -4.0, -0.25,  8.0, 0.05, 0.40),
        (  5.0,  16.0,  0.60, 10.0, 0.06, 0.55),
        ( -3.0,   3.5,  2.10,  6.0, 0.08, 0.38),
    ]
    for (px, py, ang, half_len, depth, width) in fractures:
        dx, dy = math.cos(ang), math.sin(ang)
        dist_perp  = np.abs((X - px)*(-dy) + (Y - py)*dx)
        dist_along = (X - px)*dx + (Y - py)*dy
        h -= depth * np.exp(-(dist_perp**2) / (2*(width/2.5)**2)) * (np.abs(dist_along) < half_len)

    # ==================================================================
    # 8. COMPREHENSIVE BOULDER FIELDS
    # ==================================================================

    # 8a. Crater wall boulders (inner wall, r = r_floor+0.5 to r_rim-0.8)
    _add_boulders_radial(h, X, Y, rng,
                         cfg.boulder_count,
                         cfg.boulder_height_min, cfg.boulder_height_max,
                         cfg.boulder_radius_min, cfg.boulder_radius_max,
                         0.0, 0.0, r_floor + 0.5, r_rim - 0.8)

    # 8b. Exterior rim boulder field (r = r_rim+0.3 to r_rim+9 — wide dense ring)
    #     Wide, flat-ish boulders: sigma=0.28-0.55m, height=0.08-0.40m
    _add_boulders_radial(h, X, Y, rng,
                         cfg.exterior_boulder_count,
                         0.08, 0.40, 0.28, 0.55,    # flat slab profile
                         0.0, 0.0, r_rim + 0.3, r_rim + 9.0)

    # 8c. Tile-wide scattered boulders — flat angular rocks across the plain.
    #     Use wider sigma than height for a "sitting rock" appearance.
    _add_boulders(h, X, Y, rng,
                  cfg.scatter_boulder_count,
                  0.06, 0.28, 0.25, 0.50,   # height 6-28cm, sigma 25-50cm
                  -31.0, 31.0, -31.0, 31.0)

    # ==================================================================
    # 9. SPATIALLY-VARYING ROUGHNESS
    # ==================================================================
    roughness_map = np.full_like(h, 0.040)           # 4 cm background
    roughness_map += 0.040 * (R < r_floor + 1.0)     # 8 cm on crater floor
    roughness_map -= 0.020 * ((R > r_floor) & (R < r_rim))  # 2 cm on bare rock wall
    roughness_map += 0.020 * m_env                   # extra near mountain (loose talus)
    roughness_map = np.clip(roughness_map, 0.010, 0.090)
    noise = rng.uniform(-0.5, 0.5, h.shape) * roughness_map
    h += noise

    # Shift so global minimum = 0
    h -= h.min()
    return (h / cfg.vertical_scale).astype(np.int16)


@configclass
class LunarCraterDemoBowlCfg(HfTerrainBaseCfg):
    """
    Full lunar terrain bowl — geologically rich 64 m × 64 m tile.

    Features:
      • Multi-scale undulating regolith plain (4-component sine waves)
      • Main Shackleton-class crater (22 m dia, 25–35° walls)
      • SW mountain massif (8-peak non-symmetric ridge, max 7.8 m, ~27° slopes)
      • NW dome (3.2 m), SE dome (2.5 m), NE ridge (1.8 m)
      • Secondary small crater SE (8 m dia, 22° max)
      • 5 linear thermal-contraction fractures across tile
      • Boulder fields: crater wall (10), exterior ring (80), tile scatter (100),
        mountain (50), secondary features (25)
      • Spatially-varying roughness (2–9 cm)
    """
    function: Callable = lunar_crater_demo_bowl
    proportion: float = 1.0

    # --- Crater geometry ---
    r_floor:            float = 3.0
    r_rim:              float = 11.0
    exterior_slope_deg: float = 8.0

    # --- Azimuthal variation (±5° swing at mid-wall) ---
    az_variation_m: float = 0.8

    # --- Crater wall boulders (2× density per user request) ---
    boulder_count:      int   = 20
    boulder_height_min: float = 0.15
    boulder_height_max: float = 0.50
    boulder_radius_min: float = 0.20
    boulder_radius_max: float = 0.45

    # --- Crater concentric fractures (radii / depths / widths in m) ---
    fracture_radii:  tuple = (5.5, 8.0)
    fracture_depths: tuple = (0.10, 0.08)
    fracture_widths: tuple = (0.40, 0.35)

    # --- Mid-wall mass-wasting scarp ---
    scarp_radius_m: float = 7.0
    scarp_height_m: float = 0.18
    scarp_width_m:  float = 0.60

    # --- Exterior rim boulder field (10× density increase) ---
    exterior_boulder_count: int = 80

    # --- Mountain slope boulders (wide/flat = embedded-rock appearance) ---
    mountain_boulder_count:       int   = 50
    mountain_boulder_height_min:  float = 0.10   # flat, angular slabs
    mountain_boulder_height_max:  float = 0.35
    mountain_boulder_radius_min:  float = 0.30   # wide relative to height
    mountain_boulder_radius_max:  float = 0.55

    # --- Tile-wide scattered boulders (fills the exterior plain) ---
    scatter_boulder_count: int = 100

    # --- Roughness (legacy fields kept for compatibility) ---
    roughness_m:       float = 0.025
    floor_roughness_m: float = 0.08

    seed: int = 42
