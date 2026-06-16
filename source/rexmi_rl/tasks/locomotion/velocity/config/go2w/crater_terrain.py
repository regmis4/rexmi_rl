# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Procedural heightfield generators for lunar south pole crater wall terrains.

Each generator produces a **cross-section tile** of a crater wall — a sloped
terrain tile that represents the traversal corridor from crater floor to rim.
The macro slope profile is derived from NASA LOLA 5 m/pix DEM data and published
morphometry literature (see docs/lunar_crater_terrain_research.md).

Three crater morphology types are implemented:

  Type 1 — Ancient complex crater (Haworth / Shoemaker / Nobile archetype)
  -----------------------------------------------------------------------
  Large (≥52 km), Pre-Nectarian (~4.2 Ga), heavily degraded.
  Wall slopes are gentle (7.5–14°) — all zones within current policy capability.

  Type 2 — Intermediate crater (Faustini archetype)  ← PRIMARY DEMO TARGET
  -----------------------------------------------------------------------
  Intermediate (42 km), Nectarian, moderate degradation.
  Inner wall mean slope 15.4°, max ~25°.  All zones within current policy.

  Type 3 — Young steep crater (Shackleton archetype)
  -----------------------------------------------------------------------
  Young (21 km, Imbrian ~3.6 Ga), best-preserved geometry.
  Average wall slope 30.5°, max 35°.  Requires Phase 8 slope training.

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
# Public terrain generator functions — one per crater morphology type
# ===========================================================================

@height_field_to_mesh
def type1_ancient_crater_wall(difficulty: float, cfg) -> np.ndarray:
    """
    Type 1 crater wall — ancient, heavily degraded (Haworth / Shoemaker archetype).

    Source morphometry (LOLA, Smith et al. 2010; MDPI RS 2024):
      Diameter: 52 km  |  Depth: ~2.0 km  |  d/D: ~0.038
      Wall slope mean: 8–15°, max ~20°, heavily mass-wasted

    Profile (for 32 m tile, floor → rim):
      Floor (20%, 6.4m):  1.5° — flat PSR floor + debris aprons
      Lower wall (35%, 11.2m):  7.5° — mass-wasted lower terrace
      Mid wall (25%, 8.0m): 11.5° — degraded middle slope
      Upper wall (15%, 4.8m): 14.0° — steeper upper section
      Rim crest (5%, 1.6m):  6.5° — degraded, rounded rim

    Max tile height ≈ 3.9 m over 32 m.
    All zones ✅ within current Go2W policy capability (Phase 6-Optimized).
    """
    return _build_crater_hf(
        cfg,
        zone_slopes_deg=[1.5, 7.5, 11.5, 14.0, 6.5],
        zone_fractions=[0.20, 0.35, 0.25, 0.15, 0.05],
        roughness_m=cfg.roughness_m,
        seed=cfg.seed,
    )


@height_field_to_mesh
def type2_faustini_crater_wall(difficulty: float, cfg) -> np.ndarray:
    """
    Type 2 crater wall — Faustini archetype (PRIMARY DEMO TARGET).

    Source morphometry (LOLA; Hayne et al. PSJ 2024; Grokipedia):
      Diameter: 42 km  |  Depth: ~2.5 km  |  d/D: 0.059
      Rim mean slope: 7.2°  |  Inner wall mean: 15.4°  |  Max: ~25°

    Profile (for 32 m tile, floor → rim):
      Floor (15%, 4.8m):   1.5° — PSR floor (-2600 m level)
      Lower wall (35%, 11.2m): 11.5° — moderate lower wall
      Mid wall (30%, 9.6m):   15.0° — steeper mid section
      Upper wall (15%, 4.8m): 17.5° — approach to rim
      Rim crest (5%, 1.6m):   6.5° — degraded 7.2° mean rim

    Max tile height ≈ 6.7 m over 32 m.
    All zones ✅ within current Go2W policy capability (Phase 6-Optimized).
    Peak slope (17.5°) matches training distribution upper range.
    """
    return _build_crater_hf(
        cfg,
        zone_slopes_deg=[1.5, 11.5, 15.0, 17.5, 6.5],
        zone_fractions=[0.15, 0.35, 0.30, 0.15, 0.05],
        roughness_m=cfg.roughness_m,
        seed=cfg.seed,
    )


@height_field_to_mesh
def type3_shackleton_crater_wall(difficulty: float, cfg) -> np.ndarray:
    """
    Type 3 crater wall — Shackleton archetype (young, steep, best-preserved).

    Source morphometry (Zuber et al. 2012, Science 338; LPSC 2013 poster 2924):
      Diameter: 21 km  |  Depth: 4.1 km  |  d/D: 0.195
      Average wall slope: 30.5°  |  Max: 35°

    Profile (for 32 m tile, floor → rim):
      Floor (10%, 3.2m):    3.0° — near-flat floor
      Lower wall (20%, 6.4m): 22.5° — transition from floor
      Mid wall (55%, 17.6m): 31.5° — main 30.5° average wall
      Rim crest (15%, 4.8m): 35.0° — sharp rim

    Max tile height ≈ 16.9 m over 32 m.
    ⚠️  Mid wall (31.5°) and rim (35°) EXCEED current policy capability (~20°).
    Requires Phase 8 training (slope_range up to 35°) before full demo.
    """
    return _build_crater_hf(
        cfg,
        zone_slopes_deg=[3.0, 22.5, 31.5, 35.0],
        zone_fractions=[0.10, 0.20, 0.55, 0.15],
        roughness_m=cfg.roughness_m,
        seed=cfg.seed,
    )


# ===========================================================================
# Sub-terrain config classes
# ===========================================================================

@configclass
class CraterType1WallCfg(HfTerrainBaseCfg):
    """
    Type 1 ancient crater wall terrain configuration.

    Archetype: Haworth / Shoemaker / Nobile (52–79 km, Pre-Nectarian)
    Slope range: 7.5–14°  |  Demo readiness: ✅ ready now

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
    Type 2 Faustini-archetype crater wall terrain configuration.

    Archetype: Faustini (42 km, Nectarian)
    Slope range: 11.5–17.5°  |  Demo readiness: ✅ PRIMARY DEMO TARGET

    All zones traversable with the Phase 6-Optimized rough-terrain policy.
    Peak slope (17.5°) is within the trained range (~20° max).
    """
    function: Callable = type2_faustini_crater_wall
    proportion: float = 1.0

    # 3 cm roughness matches training terrain amplitude
    roughness_m: float = 0.03

    seed: int = 2


@configclass
class CraterType3WallCfg(HfTerrainBaseCfg):
    """
    Type 3 Shackleton-archetype crater wall terrain configuration.

    Archetype: Shackleton (21 km, Imbrian)
    Slope range: 22.5–35°  |  Demo readiness: ⚠️ Phase 8 required

    Floor zone (3°) traversable now.
    Mid wall (31.5°) and rim (35°) require Phase 8 slope training.
    """
    function: Callable = type3_shackleton_crater_wall
    proportion: float = 1.0

    # Shackleton walls are smooth (less accumulated regolith)
    roughness_m: float = 0.02

    seed: int = 3
