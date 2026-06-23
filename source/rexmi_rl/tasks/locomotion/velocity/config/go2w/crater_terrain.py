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
    DEM max wall slope: 31.0°, P50: 28.6°  |  Demo readiness: ⚠️ Phase 8 required

    Floor zone (2°) traversable now.
    Main wall (31°) requires Phase 8 slope training.
    Shackleton depth validated: DEM 4 077 m vs. Zuber et al. 2012 (4 100 ± 50 m).
    """
    function: Callable = type3_shackleton_crater_wall
    proportion: float = 1.0

    # Shackleton walls have less accumulated regolith — smoother surface
    roughness_m: float = 0.02

    seed: int = 3
