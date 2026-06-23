#!/usr/bin/env python3
"""
process_lola_dem.py — Extract lunar crater geometry from NASA LOLA south polar DEM.

Usage
-----
    conda activate env_isaacsim
    python scripts/process_lola_dem.py [--crater NAME] [--dem PATH] [--out-dir DIR]

Inputs
------
    data/lola/LDEM_80S_40MPP_ADJ.TIF    NASA PGDA 40 m/px south polar DEM (default)
                                         URL: https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_40MPP_ADJ.TIF

Outputs per crater  →  data/craters/<name>/
----------------------------------------------
    rim_info.json              Rim radius (m), rim elevation (m, LOLA datum), depth,
                               d/D ratio, refined center (m, proj CRS), DEM pixel spacing
    radial_profile.csv         r_m, r_norm, z_P10, z_P50, z_P90, depth_P50,
                               slope_200m_deg, slope_400m_deg, n_pixels
    directional_profiles.csv   phi_deg, r_norm, z_m  (36 azimuths × N radial bins)
    locomotion_analysis.csv    phi_deg, wall_slope_min_deg, wall_slope_P50_deg,
                               wall_slope_P90_deg, max_slope_deg, elev_loss_m,
                               trav_dist_m, seg_len_above_20deg_m, seg_len_above_25deg_m,
                               seg_len_above_30deg_m, seg_len_above_35deg_m
    heightfield_32m.npy        2-D float32 array, 32 m/px, elevation in m (LOLA datum),
                               NaN outside crop circle.  Origin = crop SW corner.
    heightfield_32m.json       Metadata: origin_x_m, origin_y_m, pixel_size_m, shape,
                               crater_center_x_m, crater_center_y_m
    mesh_decimated.obj         Simulation-ready OBJ mesh (≤ 50 k faces)

Design decisions
----------------
* All slope values carry a baseline label — "slope_200m_deg" means Δz over 200 m radially.
  A slope without a baseline is meaningless (per the GIS review feedback).
* Rim radius is refined from the DEM, not taken from the IAU catalogue.
  Rim = locus of maximum elevation in the 0.75–1.25 × R_IAU annulus, smoothed
  over 10° azimuth windows.
* Rim reference elevation = median of accepted rim-crest elevations.
* Depth = z_rim_median − P1(floor elevations inside 0.15 × R_rim).
* Crater center is refined by minimising the standard deviation of rim heights
  over 36 azimuthal directions from the trial center.
* Radial profile bins: 0.00 to 1.80 × R_rim, step 0.02 (90 bins).
  Stops at 1.80 to avoid far-field noise while covering the full ejecta blanket.
* Directional profiles: 36 azimuths every 10°, sampled at 0.02 × R_rim intervals
  along straight radial lines using bilinear interpolation.
* Locomotion analysis defines "wall" as 0.20 R ≤ r ≤ 0.95 R to exclude
  the floor and the very top of the rim crest.

Dependencies
------------
    rasterio >= 1.4   (GeoTIFF I/O)
    pyproj   >= 3.0   (coordinate transforms)
    numpy    >= 1.24
    scipy    >= 1.10  (interpolation, stats)
    trimesh  >= 4.0   (OBJ/STL export)
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import xy as rio_xy
from pyproj import Transformer, CRS

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Crater seeds — IAU Gazetteer 2024 values.
# Used ONLY for initial DEM crop and R_IAU seed.  All geometry comes from DEM.
# ─────────────────────────────────────────────────────────────────────────────
CRATERS = {
    "haworth":     {"lat": -87.20, "lon":  -7.49, "D_iau_km": 51.42},
    "shoemaker":   {"lat": -88.14, "lon":  45.91, "D_iau_km": 51.82},
    "nobile":      {"lat": -85.28, "lon":  53.27, "D_iau_km": 79.27},
    "faustini":    {"lat": -87.18, "lon":  84.31, "D_iau_km": 42.48},
    "de_gerlache": {"lat": -88.48, "lon": -88.34, "D_iau_km": 32.71},
    "shackleton":  {"lat": -89.67, "lon": 129.78, "D_iau_km": 20.92},
}

MOON_RADIUS_M = 1_737_400.0  # IAU 2015 mean radius
N_AZ_BINS = 36               # directional profiles every 10°
N_RAD_BINS = 90              # radial bins: 0.00 to 1.80 × R_rim, step 0.02
RAD_MAX_NORM = 1.80          # max normalised radius to include in profile
WALL_R_INNER = 0.20          # inner wall boundary (normalised)
WALL_R_OUTER = 0.95          # outer wall boundary (normalised)
FLOOR_R_MAX  = 0.15          # floor region for depth calculation
TARGET_PX    = 32.0          # target pixel size (m) for output heightfield


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lon_lat_to_proj(transformer: Transformer, lon: float, lat: float):
    """Convert Moon geographic lon/lat (deg) → projected X, Y (m)."""
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def _make_lon_lat_to_proj_transformer(dem_crs: CRS) -> Transformer:
    """Build a Transformer from Moon geographic (lon/lat) to DEM projected CRS."""
    # The LOLA PGDA products use a spherical Moon (a = b = 1737400 m).
    # We define the geographic CRS explicitly with a lunar sphere.
    moon_geo_proj = (
        f"+proj=longlat +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +no_defs +type=crs"
    )
    try:
        t = Transformer.from_crs(moon_geo_proj, dem_crs.to_proj4(), always_xy=True)
    except Exception:
        # Fallback: if the DEM CRS doesn't parse cleanly, extract its PROJ string
        # manually and use a plain polar-stereographic definition.
        proj4 = dem_crs.to_proj4()
        t = Transformer.from_proj(moon_geo_proj, proj4, always_xy=True)
    return t


def _crop_dem(src: rasterio.DatasetReader, cx: float, cy: float, half_side: float):
    """
    Crop a square region [cx±half_side, cy±half_side] from a rasterio dataset.

    Returns
    -------
    data   : 2-D float32 array, nodata → NaN
    extent : (xmin, xmax, ymin, ymax) in projected CRS metres
    transform : rasterio.Affine for the cropped window
    """
    xmin = cx - half_side
    xmax = cx + half_side
    ymin = cy - half_side
    ymax = cy + half_side

    window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
    data = src.read(1, window=window).astype(np.float32)

    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    # Replace extreme sentinel values (LOLA uses -20000 in some products)
    data[data < -10_000] = np.nan
    data[data > 10_000] = np.nan

    win_transform = src.window_transform(window)
    return data, (xmin, xmax, ymin, ymax), win_transform


def _pixel_coords(data_shape, extent, transform):
    """
    Build X, Y coordinate arrays for each pixel centre.
    Returns arrays of shape == data_shape.
    """
    nrows, ncols = data_shape
    # pixel centres via rasterio transform
    col_idx = np.arange(ncols, dtype=np.float32)
    row_idx = np.arange(nrows, dtype=np.float32)
    cols, rows = np.meshgrid(col_idx, row_idx)
    # rasterio Affine: x = a*col + b*row + c, y = d*col + e*row + f
    a, b, c, d, e, f = (
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f,
    )
    px_x = a * (cols + 0.5) + b * (rows + 0.5) + c
    px_y = d * (cols + 0.5) + e * (rows + 0.5) + f
    return px_x, px_y


def _refine_center(z, px_x, px_y, cx_init, cy_init, r_iau_m):
    """
    Refine crater center by minimising std(z_rim) over 36 azimuthal directions.

    Searches on a 5×5 grid of ±0.05 × R_IAU offsets around the IAU center,
    each step = 0.01 × R_IAU.  Uses the ring 0.75R–1.25R to find the rim.

    Returns (cx_refined, cy_refined, r_rim_m, z_rim_m).
    """
    step = 0.01 * r_iau_m
    best_std = np.inf
    best_cx, best_cy = cx_init, cy_init

    for dcx in np.arange(-5, 6) * step:
        for dcy in np.arange(-5, 6) * step:
            cx_trial = cx_init + dcx
            cy_trial = cy_init + dcy
            r = np.sqrt((px_x - cx_trial)**2 + (py_y := px_y - cy_trial)**2)
            mask_rim = (r >= 0.75 * r_iau_m) & (r <= 1.25 * r_iau_m)
            if np.sum(mask_rim & np.isfinite(z)) < 36:
                continue
            z_rim_all = z[mask_rim & np.isfinite(z)]
            s = np.std(z_rim_all)
            if s < best_std:
                best_std = s
                best_cx, best_cy = cx_trial, cy_trial

    # Now compute rim properties from best center
    r_best = np.sqrt((px_x - best_cx)**2 + (px_y - best_cy)**2)
    mask_rim = (r_best >= 0.75 * r_iau_m) & (r_best <= 1.25 * r_iau_m)
    valid_rim = mask_rim & np.isfinite(z)
    if np.sum(valid_rim) < 10:
        # Fallback: use initial center
        best_cx, best_cy = cx_init, cy_init
        r_best = np.sqrt((px_x - best_cx)**2 + (px_y - best_cy)**2)
        mask_rim = (r_best >= 0.75 * r_iau_m) & (r_best <= 1.25 * r_iau_m)
        valid_rim = mask_rim & np.isfinite(z)

    # Rim radius: median radial distance of rim pixels weighted by elevation rank
    z_rim_all = z[valid_rim]
    r_rim_all = r_best[valid_rim]
    # Take top 30% of elevations in the rim annulus as the actual rim crest
    thresh = np.percentile(z_rim_all, 70)
    crest_mask = z_rim_all >= thresh
    r_rim_m = float(np.median(r_rim_all[crest_mask]))
    z_rim_m = float(np.median(z_rim_all[crest_mask]))

    return best_cx, best_cy, r_rim_m, z_rim_m


def _radial_profile(z, px_x, px_y, cx, cy, r_rim_m):
    """
    Compute median / P10 / P90 radial elevation profile and 200 m / 400 m slopes.

    Returns dict with keys:
        r_m, r_norm, z_P10, z_P50, z_P90, depth_P50,
        slope_200m_deg, slope_400m_deg, n_pixels
    """
    r = np.sqrt((px_x - cx)**2 + (px_y - cy)**2)
    r_norm = r / r_rim_m

    edges = np.linspace(0, RAD_MAX_NORM, N_RAD_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_r_m = centres * r_rim_m

    n  = np.zeros(N_RAD_BINS, dtype=int)
    z10 = np.full(N_RAD_BINS, np.nan)
    z50 = np.full(N_RAD_BINS, np.nan)
    z90 = np.full(N_RAD_BINS, np.nan)

    for i in range(N_RAD_BINS):
        mask = (r_norm >= edges[i]) & (r_norm < edges[i + 1]) & np.isfinite(z)
        n[i] = np.sum(mask)
        if n[i] >= 5:
            zv = z[mask]
            z10[i] = np.percentile(zv, 10)
            z50[i] = np.percentile(zv, 50)
            z90[i] = np.percentile(zv, 90)

    # Depth below rim
    z_rim = np.nanmedian(z50[np.abs(centres - 1.0) < 0.04])  # rim bin
    depth_P50 = z_rim - z50

    # Slope at 200 m and 400 m baseline (centred differences on radial profile)
    dr = (edges[1] - edges[0]) * r_rim_m  # bin width in metres
    step200 = max(1, round(100.0 / dr))   # bins for 100 m half-window (200 m total)
    step400 = max(1, round(200.0 / dr))

    def _slope(z_arr, half_step):
        sl = np.full_like(z_arr, np.nan)
        for i in range(half_step, len(z_arr) - half_step):
            if np.isfinite(z_arr[i - half_step]) and np.isfinite(z_arr[i + half_step]):
                dz = z_arr[i + half_step] - z_arr[i - half_step]
                dx = (2 * half_step * dr)
                sl[i] = np.degrees(np.arctan2(dz, dx))
        return sl

    sl200 = _slope(z50, step200)
    sl400 = _slope(z50, step400)

    return {
        "r_m":           bin_r_m,
        "r_norm":        centres,
        "z_P10":         z10,
        "z_P50":         z50,
        "z_P90":         z90,
        "depth_P50":     depth_P50,
        "slope_200m_deg": sl200,
        "slope_400m_deg": sl400,
        "n_pixels":      n,
    }


def _directional_profiles(z, px_x, px_y, cx, cy, r_rim_m, dem_pixel_m):
    """
    36 radial profiles sampled along azimuthal directions 0°, 10°, ..., 350°.
    Azimuth 0° = north (+Y direction), increases clockwise.

    Uses bilinear interpolation on the DEM pixel grid.

    Returns list of dicts: phi_deg, r_norm, z_m
    """
    # Build interpolator on the pixel grid
    # px_x and px_y are 2D arrays; we need to construct a regular grid
    nrows, ncols = z.shape
    # Compute grid axes from the pixel coordinate arrays
    # px_x varies along cols (axis 1), px_y varies along rows (axis 0)
    x_axis = px_x[0, :]        # x values for row 0 (monotone along cols)
    y_axis = px_y[:, 0]        # y values for col 0 (monotone along rows)

    # scipy RegularGridInterpolator expects axes to be strictly increasing
    # px_y may decrease with row index (north-up rasters have negative y step)
    flip_y = y_axis[-1] > y_axis[0]  # True if rows go south→north
    if not flip_y:
        # rows go north→south, y decreases: flip for interpolator
        y_axis_sorted = y_axis[::-1]
        z_sorted = z[::-1, :]
    else:
        y_axis_sorted = y_axis
        z_sorted = z

    interp = RegularGridInterpolator(
        (y_axis_sorted, x_axis), z_sorted,
        method="linear", bounds_error=False, fill_value=np.nan
    )

    r_norms = np.linspace(0, RAD_MAX_NORM, N_RAD_BINS)
    r_vals  = r_norms * r_rim_m

    results = []
    for i_az in range(N_AZ_BINS):
        phi_deg = i_az * (360.0 / N_AZ_BINS)
        phi_rad = np.radians(phi_deg)
        # Azimuth from north, clockwise: dx = sin(phi), dy = cos(phi)
        dx = np.sin(phi_rad)
        dy = np.cos(phi_rad)

        sample_x = cx + r_vals * dx
        sample_y = cy + r_vals * dy
        z_sampled = interp(np.column_stack([sample_y, sample_x]))

        results.append({
            "phi_deg": phi_deg,
            "r_norm": r_norms.tolist(),
            "z_m": z_sampled.tolist(),
        })
    return results


def _locomotion_analysis(dir_profiles, r_rim_m, z_rim_m, dem_pixel_m):
    """
    Per-azimuth traversability statistics.

    Parameters
    ----------
    dir_profiles : output of _directional_profiles
    r_rim_m      : rim radius in metres
    z_rim_m      : median rim crest elevation

    Returns list of dicts: one entry per azimuth.
    """
    r_norms = np.array(dir_profiles[0]["r_norm"])
    dr_m = r_norms[1] * r_rim_m if len(r_norms) > 1 else dem_pixel_m

    def _seg_len_above(slope_arr, r_arr, threshold_deg):
        """Total metres of continuous slope > threshold."""
        above = slope_arr > threshold_deg
        if not np.any(above):
            return 0.0
        # find longest contiguous run
        changes = np.diff(above.astype(int), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends   = np.where(changes == -1)[0]
        max_len = max((r_arr[e-1] - r_arr[s]) for s, e in zip(starts, ends)) if len(starts) > 0 else 0.0
        return float(max_len)

    results = []
    for prof in dir_profiles:
        phi = prof["phi_deg"]
        r_n = np.array(prof["r_norm"])
        z_s = np.array(prof["z_m"])
        r_m_arr = r_n * r_rim_m

        # Wall mask
        wall_mask = (r_n >= WALL_R_INNER) & (r_n <= WALL_R_OUTER) & np.isfinite(z_s)
        floor_mask = (r_n < FLOOR_R_MAX) & np.isfinite(z_s)

        if np.sum(wall_mask) < 5:
            results.append({"phi_deg": phi, "wall_slope_min_deg": np.nan,
                             "wall_slope_P50_deg": np.nan, "wall_slope_P90_deg": np.nan,
                             "max_slope_deg": np.nan, "elev_loss_m": np.nan,
                             "trav_dist_m": np.nan,
                             "seg_len_above_20deg_m": np.nan,
                             "seg_len_above_25deg_m": np.nan,
                             "seg_len_above_30deg_m": np.nan,
                             "seg_len_above_35deg_m": np.nan})
            continue

        # Slope at 200 m baseline along this profile
        step = max(1, round(100.0 / (r_m_arr[1] - r_m_arr[0]) if len(r_m_arr) > 1 else 1))
        z_wall = z_s.copy()
        z_wall[~wall_mask] = np.nan
        slopes = np.full_like(z_wall, np.nan)
        for i in range(step, len(z_wall) - step):
            if np.isfinite(z_wall[i - step]) and np.isfinite(z_wall[i + step]):
                dz = z_wall[i + step] - z_wall[i - step]
                dx = r_m_arr[i + step] - r_m_arr[i - step]
                slopes[i] = np.degrees(np.arctan2(abs(dz), dx))

        wall_slopes = slopes[wall_mask & np.isfinite(slopes)]
        r_wall = r_m_arr[wall_mask & np.isfinite(slopes)]

        # Elevation loss: rim height minus floor median
        z_floor = z_s[floor_mask] if np.sum(floor_mask) > 3 else np.array([np.nan])
        elev_loss = z_rim_m - float(np.nanmedian(z_floor)) if not np.all(np.isnan(z_floor)) else np.nan

        # Traversable distance (rim to floor)
        rim_r_idx = np.argmin(np.abs(r_n - 1.0))
        floor_r_idx = np.argmin(r_n)
        trav_dist = float(r_m_arr[rim_r_idx]) if rim_r_idx > floor_r_idx else np.nan

        seg_slopes = slopes[np.isfinite(slopes)]
        seg_r = r_m_arr[np.isfinite(slopes)]

        results.append({
            "phi_deg":               float(phi),
            "wall_slope_min_deg":    float(np.nanmin(wall_slopes))    if len(wall_slopes) > 0 else np.nan,
            "wall_slope_P50_deg":    float(np.nanmedian(wall_slopes)) if len(wall_slopes) > 0 else np.nan,
            "wall_slope_P90_deg":    float(np.nanpercentile(wall_slopes, 90)) if len(wall_slopes) > 0 else np.nan,
            "max_slope_deg":         float(np.nanmax(wall_slopes))    if len(wall_slopes) > 0 else np.nan,
            "elev_loss_m":           float(elev_loss)                 if not np.isnan(elev_loss) else np.nan,
            "trav_dist_m":           float(trav_dist),
            "seg_len_above_20deg_m": _seg_len_above(seg_slopes, seg_r, 20.0),
            "seg_len_above_25deg_m": _seg_len_above(seg_slopes, seg_r, 25.0),
            "seg_len_above_30deg_m": _seg_len_above(seg_slopes, seg_r, 30.0),
            "seg_len_above_35deg_m": _seg_len_above(seg_slopes, seg_r, 35.0),
        })
    return results


def _make_mesh(z_hf, pixel_m, out_path):
    """
    Build and export a decimated OBJ mesh from a 2D elevation array.

    z_hf    : 2D array (NaN outside crater)
    pixel_m : pixel size in metres
    out_path: output .obj path
    """
    try:
        import trimesh
        from trimesh.creation import box
    except ImportError:
        print("  [skip mesh] trimesh not available")
        return

    nrows, ncols = z_hf.shape
    # Build vertex grid
    x_idx = np.arange(ncols) * pixel_m
    y_idx = np.arange(nrows) * pixel_m
    xx, yy = np.meshgrid(x_idx, y_idx)

    valid = np.isfinite(z_hf)
    # Replace NaN with 0 for mesh (outside terrain = flat zero)
    z_clean = np.where(valid, z_hf, np.nanmin(z_hf[valid]) - 100 if np.any(valid) else 0)

    # Build faces from 2D grid
    verts = np.column_stack([
        xx.ravel(), yy.ravel(), z_clean.ravel()
    ])

    # Generate face indices for a regular grid
    def _grid_faces(nr, nc):
        rows = np.arange(nr - 1)
        cols = np.arange(nc - 1)
        r, c = np.meshgrid(rows, cols, indexing="ij")
        r, c = r.ravel(), c.ravel()
        tl = r * nc + c
        tr = tl + 1
        bl = tl + nc
        br = bl + 1
        f1 = np.column_stack([tl, bl, tr])
        f2 = np.column_stack([tr, bl, br])
        return np.vstack([f1, f2])

    faces = _grid_faces(nrows, ncols)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Decimate to ≤ 50 k faces
    target = 50_000
    if len(mesh.faces) > target:
        ratio = target / len(mesh.faces)
        try:
            mesh = mesh.simplify_quadric_decimation(int(len(mesh.faces) * ratio))
        except Exception:
            pass  # keep full mesh if decimation fails

    mesh.export(out_path)
    print(f"    mesh: {len(mesh.faces)} faces → {out_path}")


def _save_csv_radial(profile, out_path):
    import csv
    keys = ["r_m", "r_norm", "z_P10", "z_P50", "z_P90", "depth_P50",
            "slope_200m_deg", "slope_400m_deg", "n_pixels"]
    n = len(profile["r_m"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([
                f"{profile['r_m'][i]:.1f}",
                f"{profile['r_norm'][i]:.4f}",
                f"{profile['z_P10'][i]:.2f}"  if np.isfinite(profile['z_P10'][i])  else "",
                f"{profile['z_P50'][i]:.2f}"  if np.isfinite(profile['z_P50'][i])  else "",
                f"{profile['z_P90'][i]:.2f}"  if np.isfinite(profile['z_P90'][i])  else "",
                f"{profile['depth_P50'][i]:.2f}" if np.isfinite(profile['depth_P50'][i]) else "",
                f"{profile['slope_200m_deg'][i]:.2f}" if np.isfinite(profile['slope_200m_deg'][i]) else "",
                f"{profile['slope_400m_deg'][i]:.2f}" if np.isfinite(profile['slope_400m_deg'][i]) else "",
                str(profile['n_pixels'][i]),
            ])


def _save_csv_directional(dir_profiles, out_path):
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_deg", "r_norm", "z_m"])
        for prof in dir_profiles:
            phi = prof["phi_deg"]
            for rn, zm in zip(prof["r_norm"], prof["z_m"]):
                w.writerow([f"{phi:.1f}", f"{rn:.4f}",
                             f"{zm:.2f}" if zm is not None and np.isfinite(zm) else ""])


def _save_csv_locomotion(loco, out_path):
    import csv
    if not loco:
        return
    keys = list(loco[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in loco:
            w.writerow([
                f"{row[k]:.2f}" if isinstance(row[k], float) and not np.isnan(row[k]) else
                ("" if (isinstance(row[k], float) and np.isnan(row[k])) else row[k])
                for k in keys
            ])


# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────

def process_crater(name: str, cfg: dict, src: rasterio.DatasetReader,
                   transformer: Transformer, out_root: Path):
    print(f"\n{'='*60}")
    print(f"  Processing: {name.upper()}")
    print(f"{'='*60}")

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    D_iau_m = cfg["D_iau_km"] * 1000.0
    R_iau_m = D_iau_m / 2.0

    # 1. Convert IAU lat/lon to projected CRS
    cx_iau, cy_iau = _lon_lat_to_proj(transformer, cfg["lon"], cfg["lat"])
    print(f"  IAU center (proj):  X={cx_iau/1000:.3f} km, Y={cy_iau/1000:.3f} km")

    # 2. Crop 2 × D around IAU center
    half_side = D_iau_m  # crop = 2 × D (gives 1 D of margin on each side)
    z_crop, extent, win_transform = _crop_dem(src, cx_iau, cy_iau, half_side)
    print(f"  Crop shape: {z_crop.shape}  ({np.sum(np.isfinite(z_crop))} valid pixels)")

    if np.sum(np.isfinite(z_crop)) < 100:
        print(f"  WARNING: insufficient valid pixels for {name}, skipping.")
        return

    px_size_m = abs(win_transform.a)
    print(f"  DEM pixel size: {px_size_m:.1f} m")

    # 3. Build pixel coordinate arrays
    px_x, px_y = _pixel_coords(z_crop.shape, extent, win_transform)

    # 4. Refine crater center and find rim
    cx, cy, r_rim_m, z_rim_m = _refine_center(z_crop, px_x, px_y, cx_iau, cy_iau, R_iau_m)
    print(f"  Refined center:  dX={cx-cx_iau:.0f} m, dY={cy-cy_iau:.0f} m from IAU")
    print(f"  Rim radius:      {r_rim_m/1000:.3f} km  (IAU R_iau={R_iau_m/1000:.3f} km)")
    print(f"  Rim elevation:   {z_rim_m:.1f} m (LOLA datum)")

    # 5. Depth from floor P1
    r_all = np.sqrt((px_x - cx)**2 + (px_y - cy)**2)
    floor_mask = (r_all <= FLOOR_R_MAX * r_rim_m) & np.isfinite(z_crop)
    if np.sum(floor_mask) >= 10:
        z_floor_p1 = float(np.percentile(z_crop[floor_mask], 1))
        depth_m = z_rim_m - z_floor_p1
    else:
        z_floor_p1 = np.nan
        depth_m = np.nan

    d_over_D = depth_m / (2 * r_rim_m) if not np.isnan(depth_m) else np.nan
    print(f"  Depth (z_rim - floor P1): {depth_m:.1f} m")
    print(f"  d/D ratio:       {d_over_D:.4f}")

    # 6. Save rim info
    rim_info = {
        "name": name,
        "D_iau_km": cfg["D_iau_km"],
        "lat_iau": cfg["lat"],
        "lon_iau": cfg["lon"],
        "center_x_m": float(cx),
        "center_y_m": float(cy),
        "rim_radius_m": float(r_rim_m),
        "rim_radius_km": float(r_rim_m / 1000),
        "rim_elevation_m": float(z_rim_m),
        "floor_P1_elevation_m": float(z_floor_p1) if not np.isnan(z_floor_p1) else None,
        "depth_m": float(depth_m) if not np.isnan(depth_m) else None,
        "d_over_D": float(d_over_D) if not np.isnan(d_over_D) else None,
        "dem_pixel_m": float(px_size_m),
        "dem_source": "LDEM_80S_40MPP_ADJ.TIF (NASA PGDA, adjusted LOLA south polar DEM)",
        "slope_baseline_m": [200, 400],
        "notes": "All geometry derived from DEM; IAU values used only for initial crop."
    }
    with open(out_dir / "rim_info.json", "w") as f:
        json.dump(rim_info, f, indent=2)
    print(f"  rim_info.json saved")

    # 7. Radial profile
    print(f"  Computing radial profile...")
    rad_prof = _radial_profile(z_crop, px_x, px_y, cx, cy, r_rim_m)
    _save_csv_radial(rad_prof, out_dir / "radial_profile.csv")
    print(f"  radial_profile.csv saved  ({N_RAD_BINS} bins)")

    # 8. Directional profiles
    print(f"  Computing {N_AZ_BINS} directional profiles...")
    dir_profs = _directional_profiles(z_crop, px_x, px_y, cx, cy, r_rim_m, px_size_m)
    _save_csv_directional(dir_profs, out_dir / "directional_profiles.csv")
    print(f"  directional_profiles.csv saved")

    # 9. Locomotion analysis
    print(f"  Computing locomotion analysis...")
    loco = _locomotion_analysis(dir_profs, r_rim_m, z_rim_m, px_size_m)
    _save_csv_locomotion(loco, out_dir / "locomotion_analysis.csv")

    # Print best ingress azimuth
    valid_loco = [l for l in loco if not np.isnan(l["wall_slope_P50_deg"])]
    if valid_loco:
        best = min(valid_loco, key=lambda l: l["max_slope_deg"])
        print(f"  Best ingress azimuth: {best['phi_deg']:.0f}°  "
              f"(max wall slope {best['max_slope_deg']:.1f}°, "
              f"P50={best['wall_slope_P50_deg']:.1f}°)")
    print(f"  locomotion_analysis.csv saved")

    # 10. Heightfield at TARGET_PX resolution
    print(f"  Building {TARGET_PX:.0f} m/px heightfield...")
    # Crop to 2×R_rim circle, downsample to TARGET_PX
    r_crop = 1.10 * r_rim_m   # 10% margin beyond rim
    mask_circle = r_all <= r_crop
    z_hf_full = z_crop.copy()
    z_hf_full[~mask_circle] = np.nan

    # Downsample if needed
    factor = int(round(TARGET_PX / px_size_m))
    factor = max(1, factor)
    if factor > 1:
        def _downsample(arr, f):
            nr, nc = arr.shape
            nr2 = nr // f
            nc2 = nc // f
            out = np.full((nr2, nc2), np.nan)
            for di in range(f):
                for dj in range(f):
                    sl = arr[di::f, dj::f][:nr2, :nc2]
                    valid = np.isfinite(sl)
                    out = np.where(valid & ~np.isfinite(out), sl,
                                   np.where(valid & np.isfinite(out),
                                            (out + sl) / 2, out))
            return out
        z_hf = _downsample(z_hf_full, factor)
        hf_px_m = px_size_m * factor
    else:
        z_hf = z_hf_full
        hf_px_m = px_size_m

    np.save(str(out_dir / "heightfield_32m.npy"), z_hf.astype(np.float32))

    # Save heightfield metadata
    # Origin = SW corner of the crop (xmin, ymin)
    hf_meta = {
        "origin_x_m": float(cx - r_crop),
        "origin_y_m": float(cy - r_crop),
        "pixel_size_m": float(hf_px_m),
        "shape": list(z_hf.shape),
        "crater_center_x_m": float(cx),
        "crater_center_y_m": float(cy),
        "rim_radius_m": float(r_rim_m),
        "coordinate_system": "south polar stereographic, Moon 2000 (matches DEM CRS)",
    }
    with open(out_dir / "heightfield_32m.json", "w") as f:
        json.dump(hf_meta, f, indent=2)
    print(f"  heightfield_32m.npy saved  shape={z_hf.shape}, px={hf_px_m:.1f} m")

    # 11. Mesh
    print(f"  Building mesh...")
    _make_mesh(z_hf, hf_px_m, str(out_dir / "mesh_decimated.obj"))

    print(f"\n  ✓ {name} complete — outputs in {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process LOLA DEM for crater geometry")
    parser.add_argument("--dem",     default="data/lola/LDEM_80S_40MPP_ADJ.TIF",
                        help="Path to LOLA DEM GeoTIFF")
    parser.add_argument("--out-dir", default="data/craters",
                        help="Output root directory")
    parser.add_argument("--crater",  default=None,
                        help="Process only this crater (name from list, default=all)")
    args = parser.parse_args()

    dem_path = Path(args.dem)
    out_root = Path(args.out_dir)

    if not dem_path.exists():
        print(f"ERROR: DEM file not found: {dem_path}")
        print(f"Download with:")
        print(f"  wget -P data/lola/ https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_40MPP_ADJ.TIF")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    craters_to_process = (
        {args.crater: CRATERS[args.crater]}
        if args.crater and args.crater in CRATERS
        else CRATERS
    )

    print(f"Opening DEM: {dem_path}")
    with rasterio.open(str(dem_path)) as src:
        print(f"  Size:       {src.width} × {src.height} px")
        print(f"  Pixel size: {abs(src.transform.a):.2f} m")
        print(f"  CRS:        {src.crs.to_string()[:80]}...")
        print(f"  Bands:      {src.count}")
        print(f"  Nodata:     {src.nodata}")

        transformer = _make_lon_lat_to_proj_transformer(src.crs)

        for name, cfg in craters_to_process.items():
            try:
                process_crater(name, cfg, src, transformer, out_root)
            except Exception as e:
                import traceback
                print(f"\nERROR processing {name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  All done.  Results in {out_root}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
