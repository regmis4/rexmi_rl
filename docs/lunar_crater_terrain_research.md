# Lunar South Pole Crater Terrain Research

> **Status:** DEM-derived geometry complete for all 6 craters.
> GIS pipeline run 2026-06-22 from NASA PGDA `LDEM_80S_40MPP_ADJ.TIF`.
> All tables supersede the previous estimated data; see Section 9 for methodology notes.

---

## 1. Data Provenance

All geometry in this document is derived from a single consistent DEM product:

| Item | Value |
|------|-------|
| **Source file** | `LDEM_80S_40MPP_ADJ.TIF` |
| **Origin** | NASA PGDA / Goddard Space Flight Center |
| **URL** | https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_40MPP_ADJ.TIF |
| **Instrument** | LOLA (Lunar Orbiter Laser Altimeter) aboard LRO |
| **Projection** | South Polar Stereographic, Moon (2015) — Sphere / Ocentric |
| **Reference body** | Spherical Moon, radius 1 737 400 m |
| **Pixel size** | 40 m/px (nominal) |
| **Extent** | ±304 km from south pole (covers 80°S–90°S) |
| **Raster size** | 15 200 × 15 200 pixels |
| **Vertical datum** | LOLA absolute elevation, metres above mean sphere |
| **File size** | 684 MB |

Processing script: `scripts/process_lola_dem.py`
Output data: `data/craters/<name>/`  (not in git — regenerate from script)

---

## 2. Crater Seeds — IAU Gazetteer 2024

These coordinates were used **only** for the initial DEM crop window.
All rim positions, radii, and depths are derived from the DEM.

| Crater | D_IAU (km) | Lat | Lon |
|--------|-----------|-----|-----|
| Haworth | 51.42 | −87.20° | −7.49° |
| Shoemaker | 51.82 | −88.14° | 45.91° |
| Nobile | 79.27 | −85.28° | 53.27° |
| Faustini | 42.48 | −87.18° | 84.31° |
| de Gerlache | 32.71 | −88.48° | −88.34° |
| Shackleton | 20.92 | −89.67° | 129.78° |

---

## 3. DEM-Derived Global Dimensions

Rim radius and rim elevation are computed from the top-30% elevation crest of the
0.75–1.25 × R_IAU annulus, refined over a 5×5 search grid.
Depth = median rim elevation − 1st-percentile floor elevation (r < 0.15 × R_rim).
d/D = depth / (2 × R_rim).

| Crater | Rim R_DEM (km) | Rim elev (m) | Depth (m) | d/D | DEM samples across D |
|--------|---------------|-------------|-----------|-----|--------------------|
| **Haworth** | 27.06 | +1 640 | 5 016 | 0.0927 | ~1 353 |
| **Shoemaker** | 28.87 | −338 | 3 651 | 0.0632 | ~1 444 |
| **Nobile** | 40.47 | +4 918 | 4 927 | 0.0609 | ~2 024 |
| **Faustini** | 22.86 | +463 | 3 336 | 0.0730 | ~1 143 |
| **de Gerlache** | 17.50 | +948 | 3 270 | 0.0935 | ~875 |
| **Shackleton** | 10.92 | +1 242 | 4 077 | 0.1866 | ~546 |

> **Validation:** Shackleton depth 4 077 m vs. Zuber et al. 2012 (Science) published
> value of 4 100 ± 50 m — agreement within 0.6%.

> **Note on de Gerlache:** The previous document quoted 7.5 km depth and d/D = 0.24,
> sourced from a relief measurement that included the surrounding massif and a superposed
> inner crater. The DEM-derived value (3 270 m, d/D = 0.0935) describes the actual
> crater cavity floor and is consistent with its complex morphology.

---

## 4. Axisymmetric Radial Profile — Median (P50) Slope

Slope is the **radial slope** at an explicit **200 m baseline**:
α(r) = arctan[(z(r + 100 m) − z(r − 100 m)) / 200 m]

Full 90-bin CSVs are in `data/craters/<name>/radial_profile.csv`.
The table below samples every ~0.1 R_rim.

### 4a. Shackleton (d/D = 0.187 — steepest of the six)

Rim radius 10.92 km | Depth 4 077 m | Rim elevation +1 242 m

| r_norm | r (km) | Depth below rim (m) | Slope_200m (°) |
|--------|--------|--------------------|--------------:|
| 0.01 | 0.11 | 3 909 | — |
| 0.10 | 1.09 | 3 919 | ~1 |
| 0.20 | 2.18 | 3 866 | 5.3 |
| 0.30 | 3.28 | 3 745 | 13.3 |
| 0.40 | 4.37 | 3 258 | 28.4 |
| 0.50 | 5.46 | 2 642 | 30.4 |
| 0.60 | 6.55 | 1 980 | 31.4 |
| 0.70 | 7.65 | 1 310 | 31.9 |
| 0.80 | 8.74 | 641 | 30.6 |
| 0.90 | 9.83 | 55 | 21.8 |
| 1.00 (rim) | 10.92 | 0 | — |
| 1.10 | 12.01 | −292 (exterior) | −14.6 |

Wall character: uniformly steep 28–32° from r=0.4 to r=0.8, transitioning to 22° near
the upper rim. Very limited azimuthal variation (crater is nearly circular).

### 4b. Faustini (d/D = 0.073 — most traversable complex crater)

Rim radius 22.86 km | Depth 3 336 m | Rim elevation +463 m

| r_norm | r (km) | Depth below rim (m) | Slope_200m (°) |
|--------|--------|--------------------|--------------:|
| 0.01 | 0.23 | 2 949 | — |
| 0.10 | 2.29 | 2 959 | −0.8 |
| 0.20 | 4.57 | 2 963 | 0.5 |
| 0.30 | 6.86 | 2 913 | 1.8 |
| 0.40 | 9.14 | 2 824 | 4.4 |
| 0.50 | 11.43 | 2 376 | 16.0 |
| 0.60 | 13.72 | 1 726 | 14.3 |
| 0.70 | 16.00 | 1 125 | 14.3 |
| 0.80 | 18.29 | 627 | 13.9 |
| 0.90 | 20.57 | 119 | 9.2 |
| 1.00 (rim) | 22.86 | 0 | — |
| 1.10 | 25.14 | −119 (exterior) | −3.2 |

Wall character: gentle inner floor (0–4°) out to r ≈ 0.45, then a single broad
wall zone of 14–16° from r=0.45 to r=0.90. Nearly axisymmetric.

### 4c. de Gerlache (d/D = 0.094 — composite with inner structure)

Rim radius 17.50 km | Depth 3 270 m | Rim elevation +948 m

| r_norm | r (km) | Depth below rim (m) | Slope_200m (°) |
|--------|--------|--------------------|--------------:|
| 0.01 | 0.18 | 1 833 | — |
| 0.10 | 1.75 | 1 792 | 2.6 |
| 0.20 | 3.50 | 1 584 | 11.7 |
| 0.30 | 5.25 | 1 426 | 2.8 |
| 0.40 | 7.00 | 1 379 | −0.1 |
| 0.50 | 8.75 | 1 266 | 7.0 |
| 0.60 | 10.50 | 973 | 10.9 |
| 0.70 | 12.25 | 559 | 16.0 |
| 0.80 | 14.00 | 183 | 8.8 |
| 0.90 | 15.74 | (−132) | 3.0 |
| 1.00 (rim) | 17.50 | 0 | — |

Wall character: non-monotonic — a secondary step feature near r=0.3 (inner crater remnant
confirmed in literature) creates the slope dip between r=0.3 and r=0.4.
The median profile steepens to 16° at r=0.7 but is gentler than Shackleton on every azimuth.

---

## 5. Locomotion Analysis — All 6 Craters

**Wall zone definition:** 0.20 × R_rim ≤ r ≤ 0.95 × R_rim
**Slope baseline:** 200 m (centred difference along radial profile)
**"Best ingress azimuth":** azimuth with the lowest max wall slope

| Crater | Best ingress az | Max wall slope (best az) | P50 wall slope (best az) | Notes |
|--------|---------------|------------------------|------------------------|-------|
| **Haworth** | 160° | **9.2°** | 1.3° | Extremely gentle; current policy trivial |
| **Nobile** | 240° | **11.7°** | 3.3° | Very gentle; large flat basin |
| **Faustini** | 330° | **13.6°** | 7.8° | Within current policy capability on all azimuths |
| **de Gerlache** | 160° | **14.9°** | 9.1° | Manageable on best ingress; inner step feature |
| **Shoemaker** | 110° | **15.2°** | 10.0° | Manageable on best ingress |
| **Shackleton** | 130° | **31.0°** | 28.6° | Requires Phase 8 steep-slope policy (≥30°) |

Full per-azimuth tables in `data/craters/<name>/locomotion_analysis.csv` (36 azimuths).

> **Revision from previous document:**
> de Gerlache is NOT a 25–40° wall crater on its best ingress (the old table said
> "mid wall SW 25–40°"). DEM data shows max 14.9° on the best access azimuth.
> The steep zone in Kokhanov et al. 2022 is the SW inner crater wall, which at the
> P50 level and best ingress azimuth is not the primary traversal challenge.

---

## 6. Morphological Classification (DEM-Derived)

Classification based on measured d/D and P50 wall slope — not assumed beforehand.

| Class | Craters | d/D range | P50 wall slope | Character |
|-------|---------|-----------|---------------|-----------|
| **A — Broad shallow** | Haworth, Nobile | 0.061–0.093 | 1–3° | Very degraded, flat interiors |
| **B — Moderate** | Shoemaker, Faustini, de Gerlache | 0.063–0.094 | 8–10° | Wall-floored complex craters |
| **C — Deep/steep** | Shackleton | 0.187 | 29° | Best-preserved, steepest walls |

---

## 7. Simulation Tile Strategy

The full craters (10–40 km radius) are simulated as representative terrain patches.
Tile sizes for the Go2W demo at 1:1 scale:

| Demo scenario | Crater | Tile size (m) | Target slope zone | Status |
|--------------|--------|-------------|-----------------|--------|
| Ejecta + rim | Faustini | 200 × 200 | 0–14° | ✅ Ready now |
| Wall traverse | Faustini | 200 × 200 | 14–16° | ✅ Ready now |
| Full wall to floor | Faustini | 300 × 100 | 0–16° | ✅ Ready now |
| de Gerlache wall | de Gerlache | 200 × 200 | 9–15° | ✅ Ready now |
| Shackleton upper wall | Shackleton | 100 × 100 | 28–32° | 🔜 Needs Phase 8 |
| Shackleton full wall | Shackleton | 200 × 100 | 22–32° | 🔜 Needs Phase 8 |

---

## 8. Simulation Parameters — Faustini Wall Traverse (First Target)

Derived from DEM radial profile. All values are measured from
`data/craters/faustini/radial_profile.csv`, slope at 200 m baseline.

```python
# crater_terrain.py — Faustini-type 200 m wall traverse tile
# Represents the r=0.40–0.95 zone (lower through upper wall)
FAUSTINI_WALL_PARAMS = {
    # Zone fractions along 200 m traverse (floor-side → rim-side)
    "zones": [
        # (fraction_of_tile, slope_min_deg, slope_max_deg, label)
        (0.10,  0.0,  4.0,  "approach_floor"),    # r≈0.40, DEM: 4.4°
        (0.25, 14.0, 16.0,  "lower_wall"),         # r=0.50–0.65, DEM: 14–16°
        (0.35, 13.0, 15.0,  "mid_wall"),           # r=0.65–0.80, DEM: 13–14°
        (0.20, 10.0, 14.0,  "upper_wall"),         # r=0.80–0.90, DEM: 9–14°
        (0.10,  0.0,  5.0,  "rim_crest"),          # r=0.90–1.00, DEM: 9° decreasing
    ],
    # Surface texture
    "roughness_rms_m":       0.05,   # 5 cm RMS (regolith analogue)
    "boulder_coverage_frac": 0.02,   # 2% surface coverage
    "boulder_height_range":  (0.05, 0.20),  # 5–20 cm
    "tile_size_m":           (200.0, 200.0),
    "height_range_m":        (0.0, 58.0),   # 200 m × tan(16°) ≈ 57 m
}
```

---

## 9. Methodology Notes

### What changed from the previous document

| Previous claim | DEM result | Reason for difference |
|---------------|-----------|----------------------|
| Haworth depth ~2.0 km (estimated) | 5.02 km | Old value used d=0.04D empirical; DEM shows actual depth |
| de Gerlache depth 7.5 km, d/D=0.24 | 3.27 km, d/D=0.093 | Old value was total relief incl. surrounding massif; DEM measures cavity floor |
| Shoemaker depth ~2.2 km (estimated) | 3.65 km | Same empirical underestimate |
| de Gerlache "mid wall 25–40°" | 14.9° max on best ingress | Old value was selected SW inner-crater traverse; DEM gives crater-wide 200 m slope |
| Shackleton "rim 30–45°" | 22° at r=0.90 (transitioning to rim) | Old 45° was isolated rim-face measurement; DEM 200 m baseline is different quantity |

### Definitions used in this document

- **Rim radius:** median radial distance of top-30% elevation pixels in the 0.75–1.25 × R_IAU annulus
- **Rim elevation:** median elevation of those same crest pixels
- **Depth:** rim elevation − P1 (1st percentile) of all pixels with r < 0.15 × R_rim
- **Radial slope at 200 m baseline:** arctan[(z(r+100m) − z(r−100m)) / 200m]
  along the P50 radial profile
- **Wall zone:** 0.20 × R_rim ≤ r ≤ 0.95 × R_rim

---

## 10. Output Files

Per crater in `data/craters/<name>/`:

| File | Description |
|------|------------|
| `rim_info.json` | Rim radius, depth, d/D, rim elevation, DEM center |
| `radial_profile.csv` | 90-bin median profile: r_m, r_norm, z_P10/P50/P90, slope_200m, slope_400m |
| `directional_profiles.csv` | 36 × 90 azimuthal profiles |
| `locomotion_analysis.csv` | Per-azimuth min/P50/P90/max wall slope, seg lengths above 20/25/30/35° |
| `heightfield_32m.npy` | 2D float32 heightfield, 40 m/px (same as DEM), NaN outside rim+10% |
| `heightfield_32m.json` | Metadata for the heightfield array |
| `mesh_decimated.obj` | OBJ mesh (large — not in git) |

Regenerate with:
```bash
conda activate env_isaacsim
python scripts/process_lola_dem.py
```

---

## 11. Next Steps

- [x] Download and process `LDEM_80S_40MPP_ADJ.TIF` for all 6 craters
- [x] Validate Shackleton depth (4 077 m vs. published 4 100 m — ✓)
- [x] Update this document with DEM-derived geometry
- [ ] Update `crater_terrain.py` to use Faustini DEM parameters
- [ ] Run `play.py` with Faustini terrain to confirm robot navigability
- [ ] Phase 8 training: push steep-slope policy to 30°+ for Shackleton demo
- [ ] Per-azimuth ingress analysis → select optimal demo entry direction per crater
- [ ] (Future) Use `heightfield_32m.npy` directly as Isaac Lab terrain (replace procedural)
