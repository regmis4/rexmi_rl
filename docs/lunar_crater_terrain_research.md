# Lunar South Pole Crater Terrain Research

> **Purpose:** Morphometric data from NASA LOLA and published literature to parameterise
> procedural heightfield terrains for Go2W traversability demos.  
> **Status:** Type 2 craters (Faustini / de Gerlache) targeted for first terrain build.
> Types 1 and 3 to follow as RL policy is refined for steeper terrain.

---

## 1. Crater Classification

The lunar south pole craters of interest span three morphological types defined by age,
diameter, and interior topographic relief.

| Type | Archetype Examples | Character |
|------|--------------------|-----------|
| **Type 1** | Haworth, Shoemaker, Nobile | Large (≥52 km), ancient (Pre-Nectarian ~4.2 Ga), heavily degraded, shallow relative to size, subdued rim, hummocky floor |
| **Type 2** | Faustini, de Gerlache | Intermediate (31–42 km), moderate degradation (Nectarian–Early Imbrian), two distinct sub-types: Faustini (shallow, gentle walls) and de Gerlache (deep, steep composite) |
| **Type 3** | Shackleton | Young (20 km, Imbrian ~3.6 Ga), best-preserved, steepest inner walls, sharp rim, near-flat floor, deepest d/D ratio |

---

## 2. Measured Morphometry — All Craters

Data sources: NASA LOLA 5 m/pix DEMs (PGDA, Goddard), Zuber et al. 2012 (Science),
Smith et al. 2010 (Science), Gläser et al. (JANSS 2023), Kokhanov et al. 2022 (Icarus),
Pike 1977 (empirical scaling).

### 2a. Global Dimensions

| Crater | Type | Diam. (km) | Depth (km) | d/D ratio | Rim height (km) | Age (Ga) | PSR area (km²) |
|--------|------|-----------|-----------|-----------|-----------------|----------|----------------|
| **Haworth** | 1 | 52 | ~2.0 (est.) | ~0.038 | ~0.6 | 4.18 (Pre-Nect.) | ~1700 |
| **Shoemaker** | 1 | 52 | ~2.2 (est.) | ~0.042 | ~0.7 | 4.18 (Pre-Nect.) | ~1760 |
| **Nobile** | 1 | 79 | ~3.0 (est.) | ~0.038 | ~0.8 | Pre-Nect. | ~1950 |
| **Faustini** | 2 | 42 | ~2.5 | 0.059 | ~0.5 | Nectarian | ~664 |
| **de Gerlache** | 2 | 31 | ~7.5 | 0.242 | ~0.9 | Early Imbrian | ~550 |
| **Shackleton** | 3 | 21 | 4.1 ± 0.05 | 0.195 | 1.3 | Imbrian (~3.6) | ~126 |

> **Notes:**
> - Haworth/Shoemaker/Nobile depths are estimates derived from diameter using Type 1
>   empirical scaling (d = 0.04 × D); direct LOLA depth measurements for these were not
>   available in the sources accessed.
> - de Gerlache depth of 7.5 km is unusually large for its diameter (d/D=0.24, well above
>   fresh crater scaling); this is likely due to a superposed inner crater that deepens the
>   topography — making it effectively a composite structure.
> - Shackleton data are the most precise: from Zuber et al. 2012 (Science), derived from
>   LOLA 10 m resolution topographic model.

---

### 2b. Wall Slope Data by Crater Zone

Slopes measured at two spatial baselines:
- **40 m baseline** (detailed local roughness, relevant for rover-scale navigation)
- **200 m baseline** (macroscopic slope, relevant for robot locomotion planning)

| Crater | Type | Rim (°) | Upper Wall (°) | Lower Wall (°) | Floor (°) | Max Slope (°) | Data source |
|--------|------|---------|---------------|---------------|----------|--------------|-------------|
| **Haworth** | 1 | ~5–8 | ~8–15 | ~5–12 | ~2–5 | ~20 | LOLA slope maps, estimated |
| **Shoemaker** | 1 | ~26.6 (rim face) | ~10–18 | ~8–14 | ~3–6 | ~30 | MDPI RS 2024 (Fig. 11d) |
| **Nobile** | 1 | ~5–10 | ~8–15 | ~5–12 | ~2–5 | ~22 | LOLA slope maps, estimated |
| **Faustini** | 2 | 7.2 (mean) | 15.4 (mean inner wall) | 10–15 | ~3–6 | ~25 | Grokipedia / Hayne et al. |
| **de Gerlache** | 2 | 10–15 | 25–40 | 15–25 | <10 (floor), <4 (P units) | 50 (at 40 m) | Kokhanov et al. 2022 (Icarus) |
| **Shackleton** | 3 | ~30–45 | 30–35 (avg 30.5°) | 20–25 | <5 (floor); up to 25 (mounds) | 35 | Zuber et al. 2012; LPSC 2013 |

---

## 3. Detailed Radial Slope Profile — Type 2 Craters

> These normalised profiles are the basis for the procedural heightfield generator.
> All distances are given as **r_norm = r / R** where R = crater rim radius.
> Height is given as **h_norm = h / depth** (0 = floor level, 1 = rim crest).

### 3a. Type 2a — Faustini (gentle walls, large shallow basin)

Diameter: 42 km | Depth: 2.5 km | d/D: 0.059

| Zone | r_norm range | Slope (°) | h_norm | Physical description |
|------|-------------|-----------|--------|----------------------|
| Central floor | 0.00 – 0.20 | 0–3 | 0.00 – 0.02 | Nearly flat, some small craterlets |
| Inner floor | 0.20 – 0.40 | 3–6 | 0.02 – 0.08 | Gentle rise, lobate scarps present |
| Lower wall | 0.40 – 0.65 | 8–15 | 0.08 – 0.35 | Moderate slope, mass-wasted debris |
| Mid wall | 0.65 – 0.82 | 12–18 | 0.35 – 0.70 | Steeper, some outcrops |
| Upper wall | 0.82 – 0.93 | 15–20 | 0.70 – 0.92 | Approaching rim, moderately steep |
| Rim crest | 0.93 – 1.00 | 5–8 | 0.92 – 1.00 | Degraded, hummocky |
| Ejecta blanket | 1.00 – 1.30 | 2–5 | 1.00 – 0.95 | Gentle outer slope, boulders |
| Far exterior | > 1.30 | 0–2 | ~0.95 – 1.00 | Returns to regional plateau level |

**Key traversal zones for robot demo (Faustini-type):**
- Rim approach (ejecta blanket): 2–5° — easy, analogous to trained rough terrain
- Rim crest: 5–8° — easy
- Upper wall descent: 15–20° — within current policy capability (~20° slopes trained)
- Mid wall: 12–18° — manageable
- Lower wall: 8–15° — easy
- Floor: 0–6° — easy

✅ **Faustini-type is within current policy capability at all zones.**

---

### 3b. Type 2b — de Gerlache (deep composite structure, steep interior)

Diameter: 31 km | Depth: ~7.5 km (composite) | d/D: ~0.24

> Note: de Gerlache has an inner superposed crater. The main crater morphology
> is used here; the inner crater adds localised steep zones (~30–50°).

| Zone | r_norm range | Slope (°) | h_norm | Physical description |
|------|-------------|-----------|--------|----------------------|
| Inner crater floor | 0.00 – 0.10 | < 5 | 0.00 – 0.01 | Small flat base of inner crater |
| Inner crater wall | 0.10 – 0.22 | 30–50 | 0.01 – 0.15 | Steep inner scarp (superposed crater) |
| Main floor | 0.22 – 0.38 | < 10 | 0.15 – 0.20 | Low-slope plain (P units), some girlands |
| Lower wall | 0.38 – 0.55 | 15–25 | 0.20 – 0.45 | Hummocky, lobate debris aprons |
| Mid wall (SW) | 0.55 – 0.72 | 25–40 | 0.45 – 0.72 | Steepest traversable zone, icy patches |
| Upper wall | 0.72 – 0.88 | 20–35 | 0.72 – 0.92 | Blocky outcrops, mass wasting |
| Rim crest | 0.88 – 1.00 | 10–15 | 0.92 – 1.00 | Degraded rim crest |
| Ejecta blanket | 1.00 – 1.25 | 3–8 | 1.00 – 0.97 | Outer slope, boulder scatter |
| Far exterior | > 1.25 | 0–3 | ~0.97 | Regional terrain |

**Key traversal challenges for robot demo (de Gerlache-type):**
- Mid wall (25–40°): **exceeds current policy capability** (trained to ~20°)
- Inner crater wall (30–50°): **well beyond current capability**
- Lower wall (15–25°): marginal, policy can handle ~20° sections
- Main floor and rim: ✅ within capability

⚠️ **De Gerlache inner zones require additional slope training before demo.**

---

## 4. Terrain Scale for Simulation

The actual craters are 31–42 km diameter. The Go2W robot is ~0.6 m wide.
We simulate a **representative terrain patch** (not the full crater) at accurate local slopes.

### Recommended simulation tile sizes

| Demo scenario | Tile size (m) | Height range (m) | Notes |
|---------------|--------------|-----------------|-------|
| Rim approach + descent | 200 × 200 | 15–30 m | Ejecta + rim crest + upper wall |
| Wall traverse (Faustini-type) | 100 × 100 | 20–40 m | Mid-wall slope at ~15° |
| Full wall to floor (Faustini) | 300 × 100 | 60–80 m | Long diagonal traverse |
| De Gerlache mid-wall | 100 × 100 | 30–60 m | Requires steeper policy |

---

## 5. Procedural Heightfield Parameters — Type 2a (Faustini) Target

These are the parameters to plug into Isaac Lab's `HfPyramidSlopedTerrainCfg` /
custom `crater_wall_terrain` generator. All values are for a **200 m × 200 m tile**
representing the mid-wall to rim zone.

```python
# Conceptual parameter set — for crater_env_cfg.py
CRATER_TYPE2A_PARAMS = {
    # Tile dimensions
    "tile_size_m": (200.0, 200.0),       # metres
    "height_range_m": (0.0, 30.0),       # floor at 0, rim crest at 30 m

    # Slope profile (across the 200m traverse from floor-side to rim-side)
    "floor_fraction": 0.15,              # 15% of tile = flat floor approach
    "lower_wall_fraction": 0.35,         # next 35%: 8–15° slope
    "mid_wall_fraction": 0.30,           # next 30%: 12–18° slope
    "upper_wall_fraction": 0.15,         # next 15%: 15–20° slope
    "rim_fraction": 0.05,                # last 5%: 5–8° (rim crest)

    # Slope values per zone (degrees)
    "slope_floor":       (0.0,  3.0),
    "slope_lower_wall":  (8.0, 15.0),
    "slope_mid_wall":   (12.0, 18.0),
    "slope_upper_wall": (15.0, 20.0),
    "slope_rim":         (5.0,  8.0),

    # Surface texture (superimposed on macro slope)
    "roughness_rms_m": 0.05,             # 5 cm RMS roughness (regolith-like)
    "boulder_density": 0.02,             # ~2% of surface covered by boulders
    "boulder_height_range_m": (0.05, 0.20),  # 5–20 cm height boulders

    # Horizontal scale
    "horizontal_scale_m": 0.1,           # 10 cm resolution heightfield
}
```

---

## 6. Training Policy Gap Analysis

| Terrain feature | Current policy capability | Faustini-type | de Gerlache mid-wall |
|----------------|--------------------------|---------------|----------------------|
| Slope 0–10° | ✅ Excellent (0.90+ tracking) | ✅ Floor, rim | ✅ Floor, outer rim |
| Slope 10–15° | ✅ Good (0.85+) | ✅ Lower wall | ✅ Lower wall |
| Slope 15–20° | ✅ Adequate (0.82–0.88) | ✅ Upper wall | ⚠️ Lower mid-wall |
| Slope 20–25° | ⚠️ Marginal (0.78–0.84) | 🔜 Edge of upper wall | ⚠️ Mid-wall |
| Slope 25–35° | ❌ Not trained | ❌ — | ❌ SW wall |
| Slope 35–50° | ❌ Not trained | ❌ — | ❌ Inner crater |
| Rough + slope | ✅ Good | ✅ With regolith | ⚠️ |
| Boulder obstacles 5–20 cm | ✅ Good (0.80–0.85) | ✅ Ejecta blanket | ✅ Ejecta |

**Conclusion:** The current Phase 6-Optimized policy can demonstrate **Faustini-type
(Type 2a) traversal at all zones**, making it the ideal first demo terrain. De Gerlache
(Type 2b) requires a Phase 8 training run targeting 25–35° slopes before a full wall
descent demo.

---

## 7. Next Steps

- [ ] Build `crater_env_cfg.py` — Faustini-type terrain (Type 2a), single-robot demo config
- [ ] Write `crater_terrain.py` — procedural heightfield generator using above parameters
- [ ] Register `RexmiRl-Go2w-Velocity-Crater-Play-v0` gym environment
- [ ] Test with `play.py --task RexmiRl-Go2w-Velocity-Crater-Play-v0`
- [ ] (Future) Phase 8 training: extend slope range to 25–35° for de Gerlache demo
- [ ] (Future) Add Type 1 (Haworth/Shoemaker) wide basin traversal
- [ ] (Future) Add Type 3 (Shackleton) steep wall — requires lunar gravity (g=1.62 m/s²)

---

## 8. Data Sources & References

| Source | What it provides |
|--------|-----------------|
| Zuber et al. 2012, *Science* 338 (LOLA Team) | Shackleton LOLA 10 m DEM; depth, d/D, avg/max wall slopes, roughness |
| PGDA/Goddard LOLA 5 m/pix products | 5 m GeoTIFF DEMs for Shackleton, Shoemaker, Haworth, Faustini, de Gerlache rims |
| Kokhanov et al. 2022, *Icarus* 388 | de Gerlache morphology, slope maps, girland features, floor units |
| Gläser et al. 2023, JANSS 40(4) | Shackleton mass wasting, slope deposits |
| Spudis et al. 2008, *GRL* | Early Shackleton radar profile; 20 km diam, massif height |
| Hayne et al. (IOPscience PSJ 2024) | Faustini PSR floor units, elevation range −2600 to −2900 m |
| Smith et al. LPSC 2013 (poster 2924) | 20 km crater wall slope profiles; upper 50% ~36°, lower wall 20–25° |
| MDPI Remote Sensing 16(2) 2024 | Slope stability analysis; Shackleton rim 45°, Shoemaker rim 26.6° |
| Pike 1977 (empirical scaling laws) | d/D ratios for simple/complex craters by diameter |
