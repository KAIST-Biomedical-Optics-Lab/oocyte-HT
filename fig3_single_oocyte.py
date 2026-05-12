import math
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_erosion, map_coordinates
from scipy.spatial import ConvexHull, distance_matrix
from scipy.special import sph_harm
from skimage import measure
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import convex_hull_image
from datetime import datetime



## data paths
PATHS_D0 = {
    "ooplasm": r"",
    "PB":      r"",
    "PVS":     r"",
    "ZP":      r"",
}

PATHS_D1 = {
    "ooplasm": r"",
    "PB":      r"",
    "PVS":     r"",
    "ZP":      r"",
}

PATHS_D2 = {
    "ooplasm": r"",
    "PB":      r"",
    "PVS":     r"",
    "ZP":      r"",
}

#output Excel directory for extracted featuares
OUT_DIR = Path(r"")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DX_UM = 0.1126
DY_UM = 0.1126
DZ_UM = 0.1126
VOX_UM3 = DX_UM * DY_UM * DZ_UM

THRESH_BACKGROUND = 0
NBINS_FLOAT = 256

ZP_THICKNESS_N_THETA = 90
ZP_THICKNESS_N_PHI = 180
ZP_THICKNESS_FRAC_CENTER_Z = 0.8
INCLUDE_ZP_THICKNESS_QC = True
PLOT_ZP_THICKNESS_QC = False

OFFSETS_13 = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
    (0, 1, 1), (0, 1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
]

# Structures for pairwise reporting
STRUCTURES_ORDER = ["ooplasm", "ZP", "PVS", "PB"]


# Basic I/O
def load_tiff_as_volume(path: str) -> np.ndarray:
    arr = tiff.imread(path)
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF ndim={arr.ndim} for {path}")
    return arr.astype(np.float32, copy=False)


def load_mask_bool(path: str) -> np.ndarray:
    vol = tiff.imread(path)
    vol = np.asarray(vol)
    vol = np.squeeze(vol)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    elif vol.ndim != 3:
        raise ValueError(f"Unsupported mask ndim={vol.ndim} for {path}")
    return vol > 0


def verify_paths(paths_dict: Dict[str, str], label: str) -> None:
    for name, p in paths_dict.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"[{label}] Missing path for {name}: {p}")


# Preserved from integrated pipeline: pixel histogram stats

def compute_hist_stats(vals: np.ndarray, nbins_float: int = 256):
    vals = np.asarray(vals)
    if vals.size == 0:
        return None

    is_int_like = np.allclose(vals, np.round(vals))
    if is_int_like:
        vals_int = vals.astype(np.int64)
        vmin = int(vals_int.min())
        if vmin < 0:
            vals_shift = vals_int - vmin
            hist = np.bincount(vals_shift)
            x = np.arange(hist.size) + vmin
        else:
            hist = np.bincount(vals_int)
            x = np.arange(hist.size)
    else:
        hist, edges = np.histogram(vals, bins=nbins_float)
        x = 0.5 * (edges[:-1] + edges[1:])

    if hist.sum() == 0:
        return None

    peak_idx = int(np.argmax(hist))
    peak_x = float(x[peak_idx])
    peak_val = float(hist[peak_idx])
    half_val = peak_val / 2.0

    left_x = float(x[0])
    for i in range(peak_idx, 0, -1):
        if hist[i] >= half_val and hist[i - 1] < half_val:
            x1, x2 = x[i - 1], x[i]
            y1, y2 = hist[i - 1], hist[i]
            frac = (half_val - y1) / (y2 - y1 + 1e-12)
            left_x = float(x1 + frac * (x2 - x1))
            break

    right_x = float(x[-1])
    for i in range(peak_idx, len(hist) - 1):
        if hist[i] >= half_val and hist[i + 1] < half_val:
            x1, x2 = x[i], x[i + 1]
            y1, y2 = hist[i], hist[i + 1]
            frac = (half_val - y2) / (y1 - y2 + 1e-12)
            right_x = float(x2 - frac * (x2 - x1))
            break

    mean_val = float(np.average(x, weights=hist))
    std_val = float(np.sqrt(np.average((x - mean_val) ** 2, weights=hist)))

    return {
        "mean": mean_val,
        "std": std_val,
        "peak_x": peak_x,
        "fwhm": float(right_x - left_x),
        "left_x": left_x,
        "right_x": right_x,
    }


def compute_drymass_metrics_stats(vol_ri: np.ndarray,
                                  mask: np.ndarray,
                                  dx_um: float = DX_UM,
                                  dy_um: float = DY_UM,
                                  dz_um: float = DZ_UM,
                                  n_m_scaled: float = 13370.0,
                                  alpha_dn: float = 0.18):
    vox_um3 = dx_um * dy_um * dz_um

    m = mask.astype(bool)
    vals = vol_ri[m]

    if vals.size == 0:
        warnings.warn("Mask is empty. Returning zero-valued dry mass metrics.", stacklevel=2)
        return {
            "voxel_count": 0,
            "volume_um3": 0.0,
            "drymass_pg_sum": 0.0,
            "conc_mg_per_ml": 0.0,
            "drymass_density_pg_per_um3": 0.0,
            "drymass_density_mg_per_ml": 0.0,
        }

    vals = vals.astype(np.float64, copy=False)
    vals = vals.copy()
    vals[vals < n_m_scaled] = n_m_scaled
    dn_scaled = vals - n_m_scaled

    voxel_count = int(m.sum())
    volume_um3 = voxel_count * vox_um3

    to_pg_factor = (1000.0 / (alpha_dn * 1e4)) * vox_um3 * 1e-3
    drymass_pg_sum = float(dn_scaled.sum() * to_pg_factor)
    drymass_density_pg_per_um3 = float(drymass_pg_sum / volume_um3) if volume_um3 > 0 else 0.0
    drymass_density_mg_per_ml = float(drymass_density_pg_per_um3 * 1e3)

    return {
        "voxel_count": voxel_count,
        "volume_um3": float(volume_um3),
        "drymass_pg_sum": drymass_pg_sum,
        "conc_mg_per_ml": drymass_density_mg_per_ml,
        "drymass_density_pg_per_um3": drymass_density_pg_per_um3,
        "drymass_density_mg_per_ml": drymass_density_mg_per_ml,
    }


def compute_drymass_pg_meanVol_integrated(vol_ri: np.ndarray,
                                          mask: np.ndarray,
                                          dx_um: float = DX_UM,
                                          n_m: float = 13370.0,
                                          alpha_dn: float = 0.18,
                                          scaleRI: float = 1e4) -> float:
    vox_um3 = dx_um ** 3
    m = mask.astype(bool)
    ri_vals = vol_ri[m]
    if ri_vals.size == 0:
        return 0.0
    ri_clamped = ri_vals * scaleRI
    mean_ri_clamped = ri_clamped.mean()
    conc_mg_per_ml = (mean_ri_clamped - n_m) * (1000.0 / alpha_dn / scaleRI)
    volume_um3 = int(m.sum()) * vox_um3
    return float(conc_mg_per_ml * volume_um3 * (1 / 1e12) * 1e9)


def get_sum_and_diff_probs(P: np.ndarray):
    L = P.shape[0]
    pxpy = np.zeros(2 * L - 1, dtype=np.float64)
    pxmy = np.zeros(L, dtype=np.float64)

    rows, cols = P.nonzero()
    vals = P[rows, cols]
    np.add.at(pxpy, rows + cols, vals)
    np.add.at(pxmy, np.abs(rows - cols), vals)
    return pxpy, pxmy


def calculate_features_stats(P: np.ndarray, vmin: float):
    L = P.shape[0]
    vals = np.arange(L, dtype=np.float64) + vmin
    vi, vj = np.meshgrid(vals, vals, indexing="ij")

    asm = np.sum(P ** 2)
    contrast = np.sum(((vi - vj) ** 2) * P)

    px = P.sum(axis=1)
    py = P.sum(axis=0)

    mux = np.sum(vals * px)
    muy = np.sum(vals * py)
    sigx = np.sqrt(np.sum((vals - mux) ** 2 * px))
    sigy = np.sqrt(np.sum((vals - muy) ** 2 * py))

    if sigx * sigy == 0:
        correlation = 0.0
    else:
        correlation = float(np.sum((vi - mux) * (vj - muy) * P) / (sigx * sigy))

    variance = np.sum(((vi - mux) ** 2) * P)

    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    homogeneity = np.sum(P / (1.0 + (ii - jj) ** 2))

    eps = 1e-12
    p_safe = np.clip(P, eps, None)
    entropy = -np.sum(p_safe * np.log(p_safe))

    pxpy, pxmy = get_sum_and_diff_probs(P)

    k_sum = np.arange(len(pxpy))
    sum_vals = k_sum + 2 * vmin
    sum_avg = np.sum(sum_vals * pxpy)
    sum_var = np.sum(((sum_vals - sum_avg) ** 2) * pxpy)

    pxpy_safe = np.clip(pxpy, eps, None)
    sum_ent = -np.sum(pxpy_safe * np.log(pxpy_safe))

    k_diff = np.arange(len(pxmy))
    diff_avg = np.sum(k_diff * pxmy)
    diff_var = np.sum(((k_diff - diff_avg) ** 2) * pxmy)

    pxmy_safe = np.clip(pxmy, eps, None)
    diff_ent = -np.sum(pxmy_safe * np.log(pxmy_safe))

    cluster_shade = np.sum(((vi + vj - mux - muy) ** 3) * P)
    cluster_tend = np.sum(((vi + vj - mux - muy) ** 4) * P)
    dissimilarity = np.sum(np.abs(vi - vj) * P)

    px_safe = np.clip(px, eps, None)
    py_safe = np.clip(py, eps, None)
    log_px = np.log(px_safe)
    log_py = np.log(py_safe)

    hxy1 = -np.sum(P * (log_px[:, None] + log_py[None, :]))
    hx = -np.sum(px_safe * log_px)
    hy = -np.sum(py_safe * log_py)
    hxy2 = hx + hy

    max_h = max(hx, hy)
    imc1 = (entropy - hxy1) / (max_h + eps)
    imc2_term = 1.0 - np.exp(-2.0 * (hxy2 - entropy))
    imc2 = np.sqrt(max(0.0, imc2_term))


    cluster_prominence = np.sum(((vi + vj - mux - muy) ** 4) * P)

    return {
        "glcm_asm": float(asm),
        "glcm_contrast": float(contrast),
        "glcm_correlation": float(correlation),
        "glcm_variance": float(variance),
        "glcm_homogeneity": float(homogeneity),
        "glcm_sum_average": float(sum_avg),
        "glcm_sum_variance": float(sum_var),
        "glcm_sum_entropy": float(sum_ent),
        "glcm_entropy": float(entropy),
        "glcm_diff_variance": float(diff_var),
        "glcm_diff_entropy": float(diff_ent),
        "glcm_imc1": float(imc1),
        "glcm_imc2": float(imc2),
        "glcm_dissimilarity": float(dissimilarity),
        "glcm_cluster_shade": float(cluster_shade),
        "glcm_cluster_tendency": float(cluster_tend),
        "glcm_cluster_prominence_extra": float(cluster_prominence),
        "glcm_entropy_nat_extra": float(entropy),  # same natural-log entropy, preserved naming convenience
    }


def quantize_ht_round_int(vol3d: np.ndarray, mask: np.ndarray):
    if np.issubdtype(vol3d.dtype, np.floating):
        vol_i = np.rint(vol3d).astype(np.int32)
    else:
        vol_i = vol3d.astype(np.int32, copy=False)

    valid = vol_i[mask]
    if valid.size == 0:
        return None, 0.0, 0

    vmin = int(valid.min())
    vmax = int(valid.max())
    levels = vmax - vmin + 1

    if levels <= 1:
        return None, float(vmin), int(levels)

    q = vol_i - vmin
    return q, float(vmin), int(levels)


def glcm_3d_stats(vol_q: np.ndarray, mask: np.ndarray, vmin: float, levels: int, offsets=OFFSETS_13):
    D, H, W = vol_q.shape
    C = np.zeros((levels, levels), dtype=np.float64)

    for dz, dy, dx in offsets:
        z1 = slice(max(0, dz), D + min(0, dz))
        z2 = slice(max(0, -dz), D + min(0, -dz))
        y1 = slice(max(0, dy), H + min(0, dy))
        y2 = slice(max(0, -dy), H + min(0, -dy))
        x1 = slice(max(0, dx), W + min(0, dx))
        x2 = slice(max(0, -dx), W + min(0, -dx))

        a = vol_q[z1, y1, x1].ravel()
        b = vol_q[z2, y2, x2].ravel()
        m1 = mask[z1, y1, x1].ravel()
        m2 = mask[z2, y2, x2].ravel()

        ok = m1 & m2
        if not np.any(ok):
            continue

        va = a[ok]
        vb = b[ok]

        idx = va * levels + vb
        counts = np.bincount(idx, minlength=levels * levels)
        C += counts.reshape(levels, levels)

    C = C + C.T
    s = C.sum()
    if s == 0:
        return {}
    P = C / s
    return calculate_features_stats(P, vmin)


def ellipse_axes_regionprops_2d(mask2d: np.ndarray, dx: float, dy: float):
    lbl = measure.label(mask2d.astype(np.uint8))
    props = measure.regionprops(lbl)
    if not props:
        return np.nan, np.nan

    prop = max(props, key=lambda p: p.area)
    major = float(prop.major_axis_length * dx)
    minor = float(prop.minor_axis_length * dy)
    return major, minor


def choose_max_area_slice(mask3d: np.ndarray) -> int:
    areas = mask3d.astype(bool).sum(axis=(1, 2))
    return int(np.argmax(areas))


def choose_max_width_slice_regionprops(mask3d: np.ndarray, dx: float, dy: float) -> int:
    best_z = None
    best_major = -np.inf
    best_area = -1

    for z in range(mask3d.shape[0]):
        mask2d = mask3d[z]
        area = int(mask2d.sum())
        if area == 0:
            continue

        major_um, minor_um = ellipse_axes_regionprops_2d(mask2d, dx, dy)
        if np.isnan(major_um):
            continue

        if (major_um > best_major) or (np.isclose(major_um, best_major) and area > best_area):
            best_major = major_um
            best_area = area
            best_z = z

    if best_z is None:
        return choose_max_area_slice(mask3d)

    return int(best_z)


def sample_sphere_directions(n: int = 1024) -> np.ndarray:
    i = np.arange(n, dtype=float)
    phi_g = (1 + np.sqrt(5)) / 2
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    theta = 2 * np.pi * i / phi_g
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def surface_voxel_points(mask3d: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    m = mask3d.astype(bool)
    if not np.any(m):
        return np.empty((0, 3), dtype=np.float64)

    eroded = binary_erosion(m)
    surf = m & (~eroded)

    z, y, x = np.nonzero(surf)
    x_um = x.astype(np.float64) * dx
    y_um = y.astype(np.float64) * dy
    z_um = z.astype(np.float64) * dz
    return np.column_stack([x_um, y_um, z_um])


def ellipsoid_axes_from_mask_3d(mask3d: np.ndarray, dx: float, dy: float, dz: float):
    z, y, x = np.nonzero(mask3d)
    if len(z) < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    pts = np.column_stack([x * dx, y * dy, z * dz]).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)

    cov = np.cov(pts, rowvar=False)
    evals, _ = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    evals = np.sort(evals)[::-1]
    full_axes = 2.0 * np.sqrt(5.0 * evals)
    return full_axes.astype(np.float64)  # [major, middle, minor]


def cartesian_to_spherical_xyz(pts: np.ndarray):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.clip(z / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_bandpower_l0_l2_pointcloud(mask3d: np.ndarray, dx: float, dy: float, dz: float):
    pts = surface_voxel_points(mask3d, dx, dy, dz)
    if pts.shape[0] < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    cx_um = float(np.mean(pts[:, 0]))
    cy_um = float(np.mean(pts[:, 1]))
    cz_um = float(np.mean(pts[:, 2]))

    x = pts[:, 0] - cx_um
    y = pts[:, 1] - cy_um
    z = pts[:, 2] - cz_um

    r = np.sqrt(x * x + y * y + z * z)
    valid = r > 0
    if not np.any(valid):
        return np.nan, np.nan, np.nan, cx_um, cy_um, cz_um

    pts_centered = np.column_stack([x[valid], y[valid], z[valid]])
    r, theta, phi = cartesian_to_spherical_xyz(pts_centered)

    coeffs = {}
    for l in [0, 2]:
        for m in range(-l, l + 1):
            Ylm = sph_harm(m, l, phi, theta)
            alm = np.mean(r * np.conjugate(Ylm))
            coeffs[(l, m)] = alm

    power_l0 = float(np.abs(coeffs[(0, 0)]) ** 2)
    power_l2 = float(np.sum([np.abs(coeffs[(2, m)]) ** 2 for m in range(-2, 3)]))
    ratio = float(power_l2 / power_l0) if power_l0 > 0 else np.nan
    return power_l0, power_l2, ratio, cx_um, cy_um, cz_um


def surface_area_from_mask(mask3d: np.ndarray, dx: float, dy: float, dz: float) -> float:
    verts, faces, _, _ = marching_cubes(mask3d.astype(np.uint8), level=0.5, spacing=(dz, dy, dx))
    return float(mesh_surface_area(verts, faces))


N_ANGLES_FERET = 180


def projection_xy_from_3d(mask3d: np.ndarray) -> np.ndarray:
    return mask3d.any(axis=0)


def feret_diameters_projection(mask_2d: np.ndarray, n_angles: int = N_ANGLES_FERET):
    coords = np.column_stack(np.nonzero(mask_2d))
    if coords.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0, (np.nan, np.nan)

    ys = coords[:, 0].astype(float)
    xs = coords[:, 1].astype(float)

    cy, cx = ys.mean(), xs.mean()
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    max_d = 0.0
    min_d = np.inf
    theta_max = 0.0
    theta_min = 0.0

    for theta in angles:
        proj = xs * np.cos(theta) + ys * np.sin(theta)
        d = proj.max() - proj.min()
        if d > max_d:
            max_d = d
            theta_max = theta
        if d < min_d:
            min_d = d
            theta_min = theta

    if min_d == np.inf:
        min_d = 0.0

    return float(min_d), float(max_d), float(theta_min), float(theta_max), (cy, cx)


def compute_projection_elongation(mask3d: np.ndarray, px_um: float = DX_UM) -> Dict[str, float]:
    proj2d = projection_xy_from_3d(mask3d)
    proj2d_filled = binary_fill_holes(proj2d)

    dmin_px, dmax_px, _, _, _ = feret_diameters_projection(proj2d_filled)
    elongation = 1 - (dmin_px / dmax_px) if dmax_px > 0 else np.nan

    return {
        "shape_extra_proj_Dmin_um": float(dmin_px * px_um),
        "shape_extra_proj_Dmax_um": float(dmax_px * px_um),
        "shape_extra_proj_elongation": float(elongation),
    }


def compute_P2_integrated(mask: np.ndarray, L_max: int = 6, level: float = 0.5):
    com_z, com_y, com_x = ndimage.center_of_mass(mask)
    verts, faces, normals, values = measure.marching_cubes(mask.astype(float), level=level)

    verts_centered = verts.copy()
    verts_centered[:, 0] -= com_z
    verts_centered[:, 1] -= com_y
    verts_centered[:, 2] -= com_x

    zc, yc, xc = verts_centered[:, 0], verts_centered[:, 1], verts_centered[:, 2]
    r = np.sqrt(xc ** 2 + yc ** 2 + zc ** 2)

    theta = np.arccos(np.clip(zc / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(yc, xc)
    phi[phi < 0] += 2 * np.pi

    r_mean = r.mean()
    r_norm = r / (r_mean + 1e-12)

    lm_list = [(l, m) for l in range(0, L_max + 1) for m in range(-l, l + 1)]

    A = np.zeros((len(theta), len(lm_list)), dtype=np.complex128)
    for j, (l, m) in enumerate(lm_list):
        A[:, j] = sph_harm(m, l, phi, theta)

    c, *_ = np.linalg.lstsq(A, r_norm.astype(np.complex128), rcond=None)

    P_l = np.zeros(L_max + 1)
    idx = 0
    for l in range(0, L_max + 1):
        band_power = 0.0
        for _m in range(-l, l + 1):
            band_power += np.abs(c[idx]) ** 2
            idx += 1
        P_l[l] = band_power

    return float(P_l[2])


def compute_spharm_entropy_integrated(mask: np.ndarray, L_max: int = 10, level: float = 0.5):
    com_z, com_y, com_x = ndimage.center_of_mass(mask)
    verts, faces, normals, values = measure.marching_cubes(mask.astype(float), level=level)

    verts_centered = verts.copy()
    verts_centered[:, 0] -= com_z
    verts_centered[:, 1] -= com_y
    verts_centered[:, 2] -= com_x

    zc, yc, xc = verts_centered[:, 0], verts_centered[:, 1], verts_centered[:, 2]
    r = np.sqrt(xc ** 2 + yc ** 2 + zc ** 2)

    theta = np.arccos(np.clip(zc / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(yc, xc)
    phi[phi < 0] += 2 * np.pi

    r_mean = r.mean()
    r_norm = r / (r_mean + 1e-12)

    lm_list = [(l, m) for l in range(0, L_max + 1) for m in range(-l, l + 1)]

    A = np.zeros((len(theta), len(lm_list)), dtype=np.complex128)
    for j, (l, m) in enumerate(lm_list):
        A[:, j] = sph_harm(m, l, phi, theta)

    c, *_ = np.linalg.lstsq(A, r_norm.astype(np.complex128), rcond=None)

    P_l = np.zeros(L_max + 1, dtype=np.float64)
    idx = 0
    for l in range(0, L_max + 1):
        band_power = 0.0
        for _m in range(-l, l + 1):
            band_power += np.abs(c[idx]) ** 2
            idx += 1
        P_l[l] = band_power.real

    l_min_shape = 2
    P_shape = P_l[l_min_shape:]
    l_indices = np.arange(l_min_shape, L_max + 1)

    if P_shape.sum() > 0:
        max_mode_l = int(l_indices[np.argmax(P_shape)])
        p = P_shape / P_shape.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        max_mode_l = 0
        entropy = 0.0

    return {
        "shape_extra_P2": float(P_l[2]) if len(P_l) > 2 else np.nan,
        "shape_extra_sph_entropy_lge2": float(entropy),
        "shape_extra_sph_maxmode_l": int(max_mode_l),
    }



def fill_zp_and_get_center(mask_zp: np.ndarray):
    mask_filled = binary_fill_holes(mask_zp.astype(bool))
    if not np.any(mask_filled):
        raise ValueError("Filled ZP mask is empty. Cannot compute geometric center.")
    cz, cy, cx = ndimage.center_of_mass(mask_filled.astype(np.float32))
    center_vox = np.array([cz, cy, cx], dtype=np.float64)
    return mask_filled.astype(bool), center_vox


def compute_directional_radii_with_z(mask: np.ndarray,
                                     center_vox: np.ndarray,
                                     voxel_size_um: float = DX_UM,
                                     n_theta: int = ZP_THICKNESS_N_THETA,
                                     n_phi: int = ZP_THICKNESS_N_PHI):
    if not np.any(mask):
        radii_um = np.full((n_theta, n_phi), np.nan, dtype=np.float64)
        z_boundary = np.full((n_theta, n_phi), np.nan, dtype=np.float64)
        return radii_um, z_boundary

    inds = np.argwhere(mask)
    coords = inds.astype(np.float64) - center_vox.reshape(1, 3)

    dz = coords[:, 0]
    dy = coords[:, 1]
    dx = coords[:, 2]

    r_vox = np.sqrt(dx**2 + dy**2 + dz**2)

    valid = r_vox > 1e-6
    if not np.any(valid):
        radii_um = np.full((n_theta, n_phi), np.nan, dtype=np.float64)
        z_boundary = np.full((n_theta, n_phi), np.nan, dtype=np.float64)
        return radii_um, z_boundary

    r_vox = r_vox[valid]
    dx = dx[valid]
    dy = dy[valid]
    dz = dz[valid]

    theta = np.arccos(np.clip(dz / r_vox, -1.0, 1.0))
    phi = np.arctan2(dy, dx)
    phi[phi < 0] += 2 * np.pi

    theta_bin = (theta / np.pi * n_theta).astype(int)
    phi_bin = (phi / (2 * np.pi) * n_phi).astype(int)

    theta_bin = np.clip(theta_bin, 0, n_theta - 1)
    phi_bin = np.clip(phi_bin, 0, n_phi - 1)

    radii_vox_grid = np.full((n_theta, n_phi), np.nan, dtype=np.float64)
    z_boundary_grid = np.full((n_theta, n_phi), np.nan, dtype=np.float64)

    cz = center_vox[0]

    for r, dz_i, it, ip in zip(r_vox, dz, theta_bin, phi_bin):
        if np.isnan(radii_vox_grid[it, ip]) or (r > radii_vox_grid[it, ip]):
            radii_vox_grid[it, ip] = r
            z_boundary_grid[it, ip] = cz + dz_i

    radii_um_grid = radii_vox_grid * voxel_size_um
    return radii_um_grid, z_boundary_grid


def compute_zp_thickness_stats(mask_zp_raw: np.ndarray,
                               mask_pvs_raw: np.ndarray,
                               voxel_size_um: float = DX_UM,
                               n_theta: int = ZP_THICKNESS_N_THETA,
                               n_phi: int = ZP_THICKNESS_N_PHI,
                               frac_center_z: float = ZP_THICKNESS_FRAC_CENTER_Z):
    mask_zp_filled, center_vox = fill_zp_and_get_center(mask_zp_raw)

    radii_zp_grid, z_zp_grid = compute_directional_radii_with_z(
        mask_zp_filled,
        center_vox,
        voxel_size_um=voxel_size_um,
        n_theta=n_theta,
        n_phi=n_phi,
    )
    radii_pvs_grid, z_pvs_grid = compute_directional_radii_with_z(
        mask_pvs_raw,
        center_vox,
        voxel_size_um=voxel_size_um,
        n_theta=n_theta,
        n_phi=n_phi,
    )

    Z_total = mask_zp_raw.shape[0]
    half_span = (Z_total * frac_center_z) / 2.0
    z_min = center_vox[0] - half_span
    z_max = center_vox[0] + half_span

    valid = (
        (~np.isnan(radii_zp_grid)) &
        (~np.isnan(radii_pvs_grid)) &
        (z_zp_grid >= z_min) & (z_zp_grid <= z_max) &
        (z_pvs_grid >= z_min) & (z_pvs_grid <= z_max)
    )

    thickness_grid = np.full_like(radii_zp_grid, np.nan, dtype=np.float64)
    thickness_grid[valid] = radii_zp_grid[valid] - radii_pvs_grid[valid]
    thickness_vals = thickness_grid[valid]

    if thickness_vals.size == 0:
        return {
            "zp_thickness_mean_um": np.nan,
            "zp_thickness_std_um": np.nan,
            "zp_thickness_min_um": np.nan,
            "zp_thickness_max_um": np.nan,
            "zp_thickness_n_valid": 0,
            "zp_thickness_center_z_vox": float(center_vox[0]),
            "zp_thickness_center_y_vox": float(center_vox[1]),
            "zp_thickness_center_x_vox": float(center_vox[2]),
            "zp_thickness_z_min_vox": float(z_min),
            "zp_thickness_z_max_vox": float(z_max),
            "zp_thickness_frac_center_z": float(frac_center_z),
            "_zp_thickness_grid": thickness_grid,
            "_zp_radii_outer_grid_um": radii_zp_grid,
            "_zp_radii_inner_grid_um": radii_pvs_grid,
        }

    return {
        "zp_thickness_mean_um": float(np.mean(thickness_vals)),
        "zp_thickness_std_um": float(np.std(thickness_vals)),
        "zp_thickness_min_um": float(np.min(thickness_vals)),
        "zp_thickness_max_um": float(np.max(thickness_vals)),
        "zp_thickness_n_valid": int(thickness_vals.size),
        "zp_thickness_center_z_vox": float(center_vox[0]),
        "zp_thickness_center_y_vox": float(center_vox[1]),
        "zp_thickness_center_x_vox": float(center_vox[2]),
        "zp_thickness_z_min_vox": float(z_min),
        "zp_thickness_z_max_vox": float(z_max),
        "zp_thickness_frac_center_z": float(frac_center_z),
        "_zp_thickness_grid": thickness_grid,
        "_zp_radii_outer_grid_um": radii_zp_grid,
        "_zp_radii_inner_grid_um": radii_pvs_grid,
    }


def plot_center_slice(mask_zp: np.ndarray, mask_pvs: np.ndarray, center_vox: np.ndarray, title: str):
    union_mask = (mask_zp | mask_pvs).astype(np.uint8)
    if not np.any(union_mask):
        print("[WARN] Union mask empty. Skip plotting.")
        return

    cz, cy, cx = center_vox
    z_idx = int(round(cz))
    z_idx = max(0, min(z_idx, union_mask.shape[0] - 1))

    slice_2d = union_mask[z_idx, :, :]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(slice_2d, cmap="gray", interpolation="nearest")
    ax.set_title(f"{title}\nZP center at (z={cz:.2f}, y={cy:.2f}, x={cx:.2f}), shown at z={z_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
    plt.tight_layout()
    plt.show()


def plot_thickness_plane_with_bar(mask_zp: np.ndarray,
                                  mask_pvs: np.ndarray,
                                  center_vox: np.ndarray,
                                  theta_idx: int,
                                  phi_idx: int,
                                  n_theta: int,
                                  n_phi: int,
                                  voxel_size_um: float,
                                  r_inner_um: float,
                                  r_outer_um: float,
                                  thickness_um: float,
                                  s_range_um=None,
                                  t_range_um=None,
                                  n_s: int = 400,
                                  n_t: int = 200):
    theta = (theta_idx + 0.5) / n_theta * np.pi
    phi = (phi_idx + 0.5) / n_phi * 2.0 * np.pi

    u = np.array([
        np.cos(theta),
        np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
    ], dtype=float)
    u = u / np.linalg.norm(u)

    w = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u, w)) > 0.9:
        w = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, w)
    if np.linalg.norm(v) < 1e-6:
        v = np.array([0.0, 1.0, 0.0])
    v = v / np.linalg.norm(v)

    max_r = float(max(r_inner_um, r_outer_um))
    if s_range_um is None:
        s_range_um = max_r * 1.2
    if t_range_um is None:
        t_range_um = max_r * 1.2

    s_vals = np.linspace(-s_range_um, s_range_um, n_s)
    t_vals = np.linspace(-t_range_um, t_range_um, n_t)

    S, T = np.meshgrid(s_vals, t_vals)

    coords = (
        center_vox.reshape(1, 1, 3) +
        (S[..., None] / voxel_size_um) * u.reshape(1, 1, 3) +
        (T[..., None] / voxel_size_um) * v.reshape(1, 1, 3)
    )

    zc = coords[..., 0]
    yc = coords[..., 1]
    xc = coords[..., 2]

    coords_stack = np.stack([zc, yc, xc], axis=0)
    coords_flat = coords_stack.reshape(3, -1)

    zp_plane = map_coordinates(mask_zp.astype(float), coords_flat, order=0, mode="constant", cval=0.0).reshape(n_t, n_s)
    pvs_plane = map_coordinates(mask_pvs.astype(float), coords_flat, order=0, mode="constant", cval=0.0).reshape(n_t, n_s)

    plane_vis = pvs_plane + 2 * zp_plane

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(plane_vis, origin="lower", interpolation="nearest", extent=[s_vals[0], s_vals[-1], t_vals[0], t_vals[-1]])
    ax.set_title(f"Ray-plane (θ={theta_idx}, φ={phi_idx})\nZP thickness ≈ {thickness_um:.2f} µm")
    ax.set_xlabel("s (along ray) [µm]")
    ax.set_ylabel("t (perpendicular) [µm]")
    ax.axhline(0.0, color="white", linestyle="--", linewidth=1.0, label="ray line (t=0)")

    if np.isfinite(r_inner_um) and np.isfinite(r_outer_um):
        s1 = min(r_inner_um, r_outer_um)
        s2 = max(r_inner_um, r_outer_um)
        ax.plot([s1, s2], [0.0, 0.0], "-", linewidth=3.0, label="ZP thickness", alpha=0.9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mask (0=bg, 1=PVS, 2=ZP)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_zp_thickness_qc(mask_zp_raw: np.ndarray,
                         mask_pvs_raw: np.ndarray,
                         thickness_stats: Dict[str, object],
                         title_prefix: str = "ZP"):
    thickness_grid = thickness_stats.get("_zp_thickness_grid")
    radii_zp_grid = thickness_stats.get("_zp_radii_outer_grid_um")
    radii_pvs_grid = thickness_stats.get("_zp_radii_inner_grid_um")

    if thickness_grid is None or not np.any(np.isfinite(thickness_grid)):
        print("[WARN] No valid ZP thickness values to plot.")
        return

    valid_t, valid_p = np.where(np.isfinite(thickness_grid))
    vals_valid = thickness_grid[valid_t, valid_p]

    idx_min = int(np.argmin(vals_valid))
    idx_max = int(np.argmax(vals_valid))
    t_min, p_min = valid_t[idx_min], valid_p[idx_min]
    t_max, p_max = valid_t[idx_max], valid_p[idx_max]

    fig_map, ax_map = plt.subplots(figsize=(6, 5))
    im = ax_map.imshow(thickness_grid, cmap="coolwarm", interpolation="nearest")
    ax_map.set_title(f"{title_prefix} – ZP Thickness Map (theta × phi)")
    ax_map.set_xlabel("phi index")
    ax_map.set_ylabel("theta index")
    cbar = plt.colorbar(im, ax=ax_map)
    cbar.set_label("Thickness (µm)")
    ax_map.plot(p_min, t_min, "bo", markersize=8, label=f"min={vals_valid[idx_min]:.2f} µm")
    ax_map.plot(p_max, t_max, "ro", markersize=8, label=f"max={vals_valid[idx_max]:.2f} µm")
    ax_map.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    center_vox = np.array([
        thickness_stats["zp_thickness_center_z_vox"],
        thickness_stats["zp_thickness_center_y_vox"],
        thickness_stats["zp_thickness_center_x_vox"],
    ], dtype=np.float64)

    plot_thickness_plane_with_bar(
        mask_zp=mask_zp_raw,
        mask_pvs=mask_pvs_raw,
        center_vox=center_vox,
        theta_idx=t_min,
        phi_idx=p_min,
        n_theta=thickness_grid.shape[0],
        n_phi=thickness_grid.shape[1],
        voxel_size_um=DX_UM,
        r_inner_um=radii_pvs_grid[t_min, p_min],
        r_outer_um=radii_zp_grid[t_min, p_min],
        thickness_um=thickness_grid[t_min, p_min],
    )

    plot_thickness_plane_with_bar(
        mask_zp=mask_zp_raw,
        mask_pvs=mask_pvs_raw,
        center_vox=center_vox,
        theta_idx=t_max,
        phi_idx=p_max,
        n_theta=thickness_grid.shape[0],
        n_phi=thickness_grid.shape[1],
        voxel_size_um=DX_UM,
        r_inner_um=radii_pvs_grid[t_max, p_max],
        r_outer_um=radii_zp_grid[t_max, p_max],
        thickness_um=thickness_grid[t_max, p_max],
    )

    plot_center_slice(mask_zp_raw, mask_pvs_raw, center_vox, title=f"{title_prefix} ZP/PVS")



def extract_ht_features_stats_aligned(vol3d: np.ndarray) -> Dict[str, float]:
    roi = vol3d != 0
    vals = vol3d[roi].astype(np.float32)
    if vals.size == 0:
        return {}

    voxel_count = int(roi.sum())
    volume_um3 = float(voxel_count * VOX_UM3)
    volume_pL = float(volume_um3 * 1e-3)

    iz = choose_max_width_slice_regionprops(roi, DX_UM, DY_UM)
    roi2d = roi[iz]
    major2d_um, minor2d_um = ellipse_axes_regionprops_2d(roi2d, DX_UM, DY_UM)

    axes3d = ellipsoid_axes_from_mask_3d(roi, DX_UM, DY_UM, DZ_UM)
    major3d_um = float(axes3d[0])
    middle3d_um = float(axes3d[1])
    minor3d_um = float(axes3d[2])
    aspect_ratio = float(major3d_um / minor3d_um) if minor3d_um > 0 else np.nan

    power_l0, power_l2, ellipticity, center_x_um, center_y_um, center_z_um = \
        spherical_bandpower_l0_l2_pointcloud(roi, DX_UM, DY_UM, DZ_UM)

    surface_um2 = surface_area_from_mask(roi, DX_UM, DY_UM, DZ_UM)
    surface_over_volume = float(surface_um2 / volume_um3) if volume_um3 > 0 else np.nan

    ri_mean = float(vals.mean())
    ri_std = float(vals.std(ddof=0))
    ri_cv = float(ri_std / ri_mean) if ri_mean != 0 else np.nan

    dry_stats = compute_drymass_metrics_stats(vol3d.astype(np.float32), roi.astype(bool))
    drymass_pg_meanVol_extra = compute_drymass_pg_meanVol_integrated(vol3d.astype(np.float32), roi.astype(bool))

    out = {
        "ht_max_width_slice_index": int(iz),
        "ht_voxel_count": voxel_count,
        "ht_volume_um3": volume_um3,
        "ht_volume_pL": volume_pL,
        "ht_major_axis_2d_um": float(major2d_um),
        "ht_minor_axis_2d_um": float(minor2d_um),
        "ht_major_axis_3d_um": major3d_um,
        "ht_middle_axis_3d_um": middle3d_um,
        "ht_minor_axis_3d_um": minor3d_um,
        "ht_aspect_ratio": aspect_ratio,
        "ht_sph_power_l0": float(power_l0),
        "ht_sph_power_l2": float(power_l2),
        "ht_ellipticity": float(ellipticity),
        "ht_surface_um2": float(surface_um2),
        "ht_surface_over_volume": float(surface_over_volume),
        "ht_RI_mean": ri_mean,
        "ht_RI_std": ri_std,
        "ht_RI_cv": ri_cv,
        "ht_center_x_um": float(center_x_um),
        "ht_center_y_um": float(center_y_um),
        "ht_center_z_um": float(center_z_um),
        "ht_dry_mass_pg": float(dry_stats["drymass_pg_sum"]),
        "ht_drymass_density_mg_per_ml": float(dry_stats["drymass_density_mg_per_ml"]),
        "ht_drymass_density_pg_per_um3": float(dry_stats["drymass_density_pg_per_um3"]),
        "ht_drymass_conc_mg_per_ml": float(dry_stats["conc_mg_per_ml"]),
        "ht_drymass_pg_meanVol_extra": float(drymass_pg_meanVol_extra),
    }

    q, vmin, levels = quantize_ht_round_int(vol3d, roi)
    if q is not None and levels > 1:
        out.update(glcm_3d_stats(q, roi, vmin, levels, offsets=OFFSETS_13))

    # preserved extra shape metrics from integrated pipeline
    out.update(compute_projection_elongation(roi))
    out["shape_extra_P2"] = float(compute_P2_integrated(roi.astype(np.uint8), L_max=6))
    out.update(compute_spharm_entropy_integrated(roi.astype(np.uint8), L_max=10))

    return out



def compute_size_metrics(vol: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    voxels = int(mask.sum())
    vol_um3 = float(voxels * VOX_UM3)

    proj2d = mask.any(axis=0)
    proj2d_filled = binary_fill_holes(proj2d)
    area_px = int(proj2d_filled.sum())
    area_um2 = float(area_px * DX_UM * DY_UM)

    return {
        "size_voxels": voxels,
        "size_vol_um3": vol_um3,
        "size_area_um2": area_um2,
    }


def analyze_structure(volume_path: str) -> Dict[str, float]:
    vol = load_tiff_as_volume(volume_path)
    mask = vol > THRESH_BACKGROUND

    vals = vol[mask]
    hist_stats = compute_hist_stats(vals, nbins_float=NBINS_FLOAT)
    size_stats = compute_size_metrics(vol, mask)
    ht_stats = extract_ht_features_stats_aligned(vol)

    rec = {}
    if hist_stats is not None:
        rec["pixel_mean"] = hist_stats["mean"]
        rec["pixel_std"] = hist_stats["std"]
        rec["pixel_peak"] = hist_stats["peak_x"]
        rec["pixel_fwhm"] = hist_stats["fwhm"]
        rec["pixel_left_x"] = hist_stats["left_x"]
        rec["pixel_right_x"] = hist_stats["right_x"]

    rec.update(size_stats)
    rec.update(ht_stats)
    return rec


def build_timepoint_summary(paths_dict: Dict[str, str], tp_label: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    verify_paths(paths_dict, tp_label)

    wide = {}
    long_records = []
    all_vals = []
    structure_metrics = {}

    for region in STRUCTURES_ORDER:
        if region not in paths_dict:
            continue

        vol = load_tiff_as_volume(paths_dict[region])
        mask = vol > THRESH_BACKGROUND
        vals = vol[mask]
        if vals.size > 0:
            all_vals.append(vals.astype(np.float64))

        rec = analyze_structure(paths_dict[region])
        structure_metrics[region] = rec

    if ("ZP" in paths_dict) and ("PVS" in paths_dict) and ("ZP" in structure_metrics):
        mask_zp_raw = load_mask_bool(paths_dict["ZP"])
        mask_pvs_raw = load_mask_bool(paths_dict["PVS"])
        thickness_stats = compute_zp_thickness_stats(
            mask_zp_raw,
            mask_pvs_raw,
            voxel_size_um=DX_UM,
            n_theta=ZP_THICKNESS_N_THETA,
            n_phi=ZP_THICKNESS_N_PHI,
            frac_center_z=ZP_THICKNESS_FRAC_CENTER_Z,
        )

        thickness_record = {
            "zp_thickness_mean_um": thickness_stats["zp_thickness_mean_um"],
        }
        if INCLUDE_ZP_THICKNESS_QC:
            thickness_record.update({
                "zp_thickness_std_um": thickness_stats["zp_thickness_std_um"],
                "zp_thickness_min_um": thickness_stats["zp_thickness_min_um"],
                "zp_thickness_max_um": thickness_stats["zp_thickness_max_um"],
                "zp_thickness_n_valid": thickness_stats["zp_thickness_n_valid"],
                "zp_thickness_center_z_vox": thickness_stats["zp_thickness_center_z_vox"],
                "zp_thickness_center_y_vox": thickness_stats["zp_thickness_center_y_vox"],
                "zp_thickness_center_x_vox": thickness_stats["zp_thickness_center_x_vox"],
                "zp_thickness_z_min_vox": thickness_stats["zp_thickness_z_min_vox"],
                "zp_thickness_z_max_vox": thickness_stats["zp_thickness_z_max_vox"],
                "zp_thickness_frac_center_z": thickness_stats["zp_thickness_frac_center_z"],
            })

        structure_metrics["ZP"].update(thickness_record)

        if PLOT_ZP_THICKNESS_QC:
            plot_zp_thickness_qc(mask_zp_raw, mask_pvs_raw, thickness_stats, title_prefix=tp_label)

    vol_ooplasm = structure_metrics.get("ooplasm", {}).get("size_vol_um3", np.nan)

    for region, rec in structure_metrics.items():
        ratio_to_ooplasm = float(rec["size_vol_um3"] / vol_ooplasm) if np.isfinite(vol_ooplasm) and vol_ooplasm > 0 else np.nan
        rec["size_ratio_to_ooplasm"] = ratio_to_ooplasm

        for k, v in rec.items():
            wide[f"{k}_{region}"] = v

        long_row = {
            "timepoint": tp_label,
            "region": region,
            "filepath": paths_dict[region],
            **rec,
        }
        long_records.append(long_row)

    # preserved integrated total pixel stats across structures
    vals_total = np.concatenate(all_vals) if len(all_vals) > 0 else np.array([], dtype=np.float64)
    total_hist = compute_hist_stats(vals_total, nbins_float=NBINS_FLOAT)
    if total_hist is not None:
        wide["pixel_mean_total"] = total_hist["mean"]
        wide["pixel_std_total"] = total_hist["std"]
        wide["pixel_peak_total"] = total_hist["peak_x"]
        wide["pixel_fwhm_total"] = total_hist["fwhm"]


    wide["timepoint"] = tp_label
    df_long = pd.DataFrame(long_records)
    return wide, df_long


def main():
    sample_dict = {
        "D0": PATHS_D0,
        "D1": PATHS_D1,
        "D2": PATHS_D2,
    }

    run_samples = ["D0", "D1", "D2"]

    wide_list = []
    long_list = []

    for tp in run_samples:
        wide_tp, long_tp = build_timepoint_summary(sample_dict[tp], tp)
        wide_list.append(wide_tp)
        long_list.append(long_tp)

    df_wide = pd.DataFrame(wide_list).set_index("timepoint")
    df_long = pd.concat(long_list, ignore_index=True)


    run_tag = "_".join(run_samples)   # 예: D1 또는 D0_D1_D2
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_xlsx = OUT_DIR / f"fig3_ht_{run_tag}_{time_tag}.xlsx"


    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_wide.to_excel(writer, sheet_name="summary_wide")
        df_long.to_excel(writer, index=False, sheet_name="summary_long")

    print(f"\nSaved: {out_xlsx}")


if __name__ == "__main__":
    main()
