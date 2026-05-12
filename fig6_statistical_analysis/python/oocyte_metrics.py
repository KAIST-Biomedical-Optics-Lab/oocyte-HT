from datetime import datetime
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff


from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull, distance_matrix
from scipy.special import sph_harm
from skimage import measure
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import convex_hull_image


# ============================================================
# Rust extension import
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
RUST_DIR = THIS_DIR.parent / "rust"

if str(RUST_DIR) not in sys.path:
    sys.path.insert(0, str(RUST_DIR))

USE_RUST_GLCM = True

try:
    import fast_glcm
    FAST_GLCM_AVAILABLE = True
except Exception as e:
    FAST_GLCM_AVAILABLE = False
    fast_glcm = None
    print(f"[WARN] fast_glcm import failed. Falling back to Python GLCM. ({e})")


# ============================================================
# Config
# ============================================================

ROOT_BASE = Path(r"G:\RealData\ooplasm")
DATE_DIRS = ["260129", "260204", "260209"]
RUN_DATE_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODALITY_MAP = {
    "2DBF_renewed": "2DBF",
    "2.5DBF": "2.5DBF",
    "HT": "HT",
}

TARGET_KEYWORD = ""
VALID_EXTS = {".tif", ".tiff"}
BF_BINS = 64
CALCULATE_HT_3DBF_INTENSITY = True
HT_3DBF_FOLDER_NAME = "3DBF"

DX_UM = 0.1126
DY_UM = 0.1126
DZ_UM = 0.1126
VOX_UM3 = DX_UM * DY_UM * DZ_UM

OFFSETS_13 = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
    (0, 1, 1), (0, 1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
]

OFFSETS_2D_4 = [
    (0, 1), (1, 0), (1, 1), (1, -1)
]

OUT_XLSX = Path(fr"G:\RealData\excels\all_features_{RUN_DATE_TIME}.xlsx")
EXCLUDE_EXCEL_COLUMNS = [
    "ellipticity_center_x_um",
    "ellipticity_center_y_um",
]


# ============================================================
# Basic utils
# ============================================================

def file_matches_target(path: Path, target_keyword: str) -> bool:
    if target_keyword is None:
        return True

    target_keyword = str(target_keyword).strip()
    if target_keyword == "":
        return True

    return target_keyword in path.name


def infer_group_from_name(name: str) -> str:
    up = name.upper()
    if up.startswith("D0"):
        return "D0"
    if up.startswith("D1"):
        return "D1"
    if up.startswith("D2"):
        return "D2"
    raise ValueError(f"Cannot infer group from filename: {name}")


def list_files_recursive(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]


def build_file_index_by_name(folder: Path):
    index = {}
    duplicates = set()

    if not folder.exists():
        return index, duplicates

    for path in list_files_recursive(folder):
        key = path.name.lower()
        if key in index:
            duplicates.add(key)
            continue
        index[key] = path

    return index, duplicates


def read_image_any(path: Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)

    return arr


def ensure_2d_or_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported ndim={arr.ndim}, shape={arr.shape}")
    return arr


# ============================================================
# Dry mass
# ============================================================

def compute_drymass_metrics_from_ri(vol_ri: np.ndarray,
                                    mask: np.ndarray,
                                    dx_um: float = 0.1126,
                                    dy_um: float = 0.1126,
                                    dz_um: float = 0.1126,
                                    n_m_scaled: float = 13370.0,
                                    alpha_dn: float = 0.18):

    vox_um3 = dx_um * dy_um * dz_um

    m = mask.astype(bool)
    vals = vol_ri[m]

    if vals.size == 0:
        warnings.warn("Mask is empty. Returning zero-valued dry mass metrics.", stacklevel=2)
        return {
            "volume_um3": 0.0,
            "drymass_pg_sum": 0.0,
            "drymass_density_mg_per_ml": 0.0,
        }

    vals = vals.astype(np.float64, copy=False)
    vals[vals < n_m_scaled] = n_m_scaled
    dn_scaled = vals - n_m_scaled

    voxel_count = int(m.sum())
    volume_um3 = voxel_count * vox_um3

    to_pg_factor = (1000.0 / (alpha_dn * 1e4)) * vox_um3 * 1e-3
    drymass_pg_sum = float(dn_scaled.sum() * to_pg_factor)
    drymass_density_pg_per_um3 = float(drymass_pg_sum / volume_um3) if volume_um3 > 0 else 0.0
    drymass_density_mg_per_ml = drymass_density_pg_per_um3 * 1e3

    return {
        "volume_um3": float(volume_um3),
        "drymass_pg_sum": drymass_pg_sum,
        "drymass_density_mg_per_ml": float(drymass_density_mg_per_ml),
    }


# ============================================================
# 2D geometry
# ============================================================

def mask_to_boundary_coords_2d(mask2d: np.ndarray) -> np.ndarray:
    contours = measure.find_contours(mask2d.astype(np.uint8), 0.5)
    if len(contours) == 0:
        raise ValueError("No contour found in mask.")
    contour = max(contours, key=len)
    xy = np.column_stack([contour[:, 1], contour[:, 0]])
    return xy.astype(np.float64)


def resample_closed_curve_by_arclength(xy: np.ndarray, n_samples: int = 1024) -> np.ndarray:
    pts = np.vstack([xy, xy[0]])
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])

    if s[-1] <= 0:
        raise ValueError("Boundary length is zero.")

    target = np.linspace(0.0, s[-1], n_samples, endpoint=False)
    x = np.interp(target, s, pts[:, 0])
    y = np.interp(target, s, pts[:, 1])

    return np.column_stack([x, y])


def polygon_perimeter_from_boundary(boundary_xy: np.ndarray, dx: float, dy: float) -> float:
    pts_um = np.column_stack([boundary_xy[:, 0] * dx, boundary_xy[:, 1] * dy])
    dif = np.diff(np.vstack([pts_um, pts_um[0]]), axis=0)
    return float(np.sqrt((dif**2).sum(axis=1)).sum())


def feret_diameters_2d(mask2d: np.ndarray, dx: float, dy: float):
    hull_mask = convex_hull_image(mask2d)
    boundary_xy = mask_to_boundary_coords_2d(hull_mask)
    pts_um = np.column_stack([boundary_xy[:, 0] * dx, boundary_xy[:, 1] * dy])

    dmat = distance_matrix(pts_um, pts_um)
    major = float(dmat.max())

    pts_closed = np.vstack([pts_um, pts_um[0]])
    edges = np.diff(pts_closed, axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    widths = []
    for theta in np.unique(np.round(angles, 6)):
        n = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])
        proj = pts_um @ n
        widths.append(proj.max() - proj.min())

    minor = float(np.min(widths))
    return major, minor


def ellipse_axes_regionprops_2d(mask2d: np.ndarray, dx: float, dy: float):
    lbl = measure.label(mask2d.astype(np.uint8))
    props = measure.regionprops(lbl)
    if not props:
        return np.nan, np.nan

    prop = max(props, key=lambda p: p.area)
    major = float(prop.major_axis_length * dx)
    minor = float(prop.minor_axis_length * dy)
    return major, minor


def fourier_ellipticity_2d(mask2d: np.ndarray,
                           dx_um: float,
                           dy_um: float,
                           n_samples: int = 2048):
    mask2d = mask2d.astype(bool)
    if not np.any(mask2d):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    boundary_xy = mask_to_boundary_coords_2d(mask2d)
    boundary_xy = resample_closed_curve_by_arclength(boundary_xy, n_samples=n_samples)

    x_um = boundary_xy[:, 0] * dx_um
    y_um = boundary_xy[:, 1] * dy_um

    cx_um = float(np.mean(x_um))
    cy_um = float(np.mean(y_um))

    x_um = x_um - cx_um
    y_um = y_um - cy_um

    z = x_um + 1j * y_um
    c = np.fft.fft(z) / len(z)

    power_c1 = float(np.abs(c[1]) ** 2 + np.abs(c[-1]) ** 2)
    power_c2 = float(np.abs(c[2]) ** 2 + np.abs(c[-2]) ** 2)

    ratio = np.nan
    if power_c1 > 0:
        ratio = float(power_c2 / power_c1)

    return power_c1, power_c2, ratio, cx_um, cy_um


# ============================================================
# 3D point cloud shape
# ============================================================

def sample_sphere_directions(n: int = 1024) -> np.ndarray:
    i = np.arange(n, dtype=float)
    phi_g = (1 + np.sqrt(5)) / 2
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    theta = 2 * np.pi * i / phi_g
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def surface_voxel_points(mask3d: np.ndarray,
                         dx: float,
                         dy: float,
                         dz: float) -> np.ndarray:
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


def feret_diameters_3d_pointcloud(mask3d: np.ndarray,
                                  dx: float,
                                  dy: float,
                                  dz: float,
                                  n_dir_minor: int = 1024):
    pts = surface_voxel_points(mask3d, dx, dy, dz)

    hull = ConvexHull(pts)
    hull_pts = pts[np.unique(hull.vertices)]

    dmat = distance_matrix(hull_pts, hull_pts)
    major = float(dmat.max())

    dirs = sample_sphere_directions(n_dir_minor)
    widths = []
    for v in dirs:
        proj = hull_pts @ v
        widths.append(proj.max() - proj.min())

    minor = float(np.min(widths))
    return major, minor


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
    return full_axes.astype(np.float64)


def cartesian_to_spherical_xyz(pts: np.ndarray):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_bandpower_l0_l2_pointcloud(mask3d: np.ndarray,
                                         dx: float,
                                         dy: float,
                                         dz: float):
    pts = surface_voxel_points(mask3d, dx, dy, dz)
    if pts.shape[0] < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    cx_um = float(np.mean(pts[:, 0]))
    cy_um = float(np.mean(pts[:, 1]))
    cz_um = float(np.mean(pts[:, 2]))

    x = pts[:, 0] - cx_um
    y = pts[:, 1] - cy_um
    z = pts[:, 2] - cz_um

    r = np.sqrt(x*x + y*y + z*z)
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
    power_l2 = float(np.sum([np.abs(coeffs[(2, m)])**2 for m in range(-2, 3)]))

    ratio = np.nan
    if power_l0 > 0:
        ratio = float(power_l2 / power_l0)

    return power_l0, power_l2, ratio, cx_um, cy_um, cz_um


def surface_area_from_mask(mask3d: np.ndarray, dx: float, dy: float, dz: float) -> float:
    verts, faces, _, _ = marching_cubes(mask3d.astype(np.uint8), level=0.5, spacing=(dz, dy, dx))
    return float(mesh_surface_area(verts, faces))


# ============================================================
# Haralick helpers
# ============================================================

def get_sum_and_diff_probs(P: np.ndarray):
    L = P.shape[0]
    pxpy = np.zeros(2 * L - 1, dtype=np.float64)
    pxmy = np.zeros(L, dtype=np.float64)

    rows, cols = P.nonzero()
    vals = P[rows, cols]
    np.add.at(pxpy, rows + cols, vals)
    np.add.at(pxmy, np.abs(rows - cols), vals)
    return pxpy, pxmy


def calculate_features(P: np.ndarray, vmin: float):
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
        correlation = np.sum((vi - mux) * (vj - muy) * P) / (sigx * sigy)

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

    return {
        "asm": float(asm),
        "contrast": float(contrast),
        "correlation": float(correlation),
        "variance": float(variance),
        "homogeneity": float(homogeneity),
        "sum_average": float(sum_avg),
        "sum_variance": float(sum_var),
        "sum_entropy": float(sum_ent),
        "entropy": float(entropy),
        "diff_variance": float(diff_var),
        "diff_entropy": float(diff_ent),
        "imc1": float(imc1),
        "imc2": float(imc2),
        "dissimilarity": float(dissimilarity),
        "cluster_shade": float(cluster_shade),
        "cluster_tendency": float(cluster_tend),
    }


# ============================================================
# Quantization
# ============================================================

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


def quantize_bf_bins(vol: np.ndarray, mask: np.ndarray, bins: int = 64):
    valid = vol[mask]
    if valid.size == 0:
        return None, 0.0, 0

    vmin = float(valid.min())
    vmax = float(valid.max())

    if vmax <= vmin:
        return None, vmin, 0

    scaled = (vol - vmin) / (vmax - vmin)
    q = np.floor(scaled * (bins - 1) + 1e-12).astype(np.int32)
    q = np.clip(q, 0, bins - 1)
    return q, vmin, bins


# ============================================================
# GLCM builders
# ============================================================

def _glcm2d_counts_for_slice_python(img_q2d: np.ndarray, mask2d: np.ndarray, levels: int, offsets=OFFSETS_2D_4):
    H, W = img_q2d.shape
    C = np.zeros((levels, levels), dtype=np.int64)

    for dy, dx in offsets:
        y1 = slice(max(0, dy), H + min(0, dy))
        y2 = slice(max(0, -dy), H + min(0, -dy))
        x1 = slice(max(0, dx), W + min(0, dx))
        x2 = slice(max(0, -dx), W + min(0, -dx))

        a = img_q2d[y1, x1].ravel()
        b = img_q2d[y2, x2].ravel()
        m1 = mask2d[y1, x1].ravel()
        m2 = mask2d[y2, x2].ravel()

        ok = m1 & m2
        if not np.any(ok):
            continue

        va = a[ok]
        vb = b[ok]

        idx = va * levels + vb
        counts = np.bincount(idx, minlength=levels * levels).reshape(levels, levels)
        C += counts

    C = C + C.T
    return C


def _glcm2d_counts_for_slice(img_q2d: np.ndarray, mask2d: np.ndarray, levels: int, offsets=OFFSETS_2D_4):
    if USE_RUST_GLCM and FAST_GLCM_AVAILABLE:
        return np.asarray(
            fast_glcm.glcm2d_counts(
                np.ascontiguousarray(img_q2d.astype(np.int32)),
                np.ascontiguousarray(mask2d.astype(bool)),
                int(levels),
                list(offsets),
            )
        )
    return _glcm2d_counts_for_slice_python(img_q2d, mask2d, levels, offsets=offsets)


def glcm_2d(img_q2d: np.ndarray, mask2d: np.ndarray, vmin: float, levels: int, offsets=OFFSETS_2D_4):
    C = _glcm2d_counts_for_slice(img_q2d, mask2d, levels, offsets=offsets)
    s = C.sum()
    if s == 0:
        return {}
    P = C.astype(np.float64) / float(s)
    return calculate_features(P, vmin)


def glcm_3d_python(vol_q: np.ndarray, mask: np.ndarray, vmin: float, levels: int, offsets=OFFSETS_13):
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
    return calculate_features(P, vmin)


def glcm_3d(vol_q: np.ndarray, mask: np.ndarray, vmin: float, levels: int, offsets=OFFSETS_13):
    if USE_RUST_GLCM and FAST_GLCM_AVAILABLE:
        C = np.asarray(
            fast_glcm.glcm3d_counts(
                np.ascontiguousarray(vol_q.astype(np.int32)),
                np.ascontiguousarray(mask.astype(bool)),
                int(levels),
                list(offsets),
            ),
            dtype=np.int64
        )
        s = C.sum()
        if s == 0:
            return {}
        P = C.astype(np.float64) / float(s)
        return calculate_features(P, vmin)

    return glcm_3d_python(vol_q, mask, vmin, levels, offsets=offsets)


def glcm_25d_mrg_from_stack(vol_q: np.ndarray, mask: np.ndarray, vmin: float, levels: int, offsets2d=OFFSETS_2D_4):
    D = vol_q.shape[0]
    combined = np.zeros((levels, levels), dtype=np.int64)

    for z in range(D):
        m2d = mask[z]
        if not np.any(m2d):
            continue

        img2d = vol_q[z]
        C = _glcm2d_counts_for_slice(img2d, m2d, levels, offsets=offsets2d)
        if C.sum() > 0:
            combined += C

    s = combined.sum()
    if s == 0:
        return {}

    P = combined.astype(np.float64) / float(s)
    return calculate_features(P, vmin)


# ============================================================
# Feature extractors
# ============================================================

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


def extract_bf_intensity_stats(img: np.ndarray):
    roi = img != 0
    vals = img[roi].astype(np.float32)
    if vals.size == 0:
        return {
            "intensity_mean": np.nan,
            "intensity_std": np.nan,
            "intensity_cv": np.nan,
        }

    mean_int = float(vals.mean())
    std_int = float(vals.std(ddof=0))
    cv_int = float(std_int / mean_int) if mean_int != 0 else np.nan

    return {
        "intensity_mean": mean_int,
        "intensity_std": std_int,
        "intensity_cv": cv_int,
    }


def extract_2dbf_features(img2d: np.ndarray):
    roi = img2d != 0
    vals = img2d[roi].astype(np.float32)
    if vals.size == 0:
        return {}

    area_px = int(roi.sum())
    area_um2 = area_px * DX_UM * DY_UM

    major_um, minor_um = ellipse_axes_regionprops_2d(roi, DX_UM, DY_UM)
    aspect_ratio_2d = major_um / minor_um if minor_um != 0 else np.nan

    power_c1, power_c2, ellipticity_2d, _, _ = fourier_ellipticity_2d(
        roi, DX_UM, DY_UM
    )

    boundary = mask_to_boundary_coords_2d(roi)
    perimeter_um = polygon_perimeter_from_boundary(boundary, DX_UM, DY_UM)
    perimeter_over_area = perimeter_um / area_um2 if area_um2 != 0 else np.nan

    mean_int = float(vals.mean())
    std_int = float(vals.std(ddof=0))
    cv_int = float(std_int / mean_int) if mean_int != 0 else np.nan

    feats = {
        "area_um2": area_um2,
        "major_axis_2d_um": major_um,
        "minor_axis_2d_um": minor_um,
        "aspect_ratio_2d": aspect_ratio_2d,
        "fourier_power_c1": power_c1,
        "fourier_power_c2": power_c2,
        "ellipticity_2d": ellipticity_2d,
        "perimeter_um": perimeter_um,
        "perimeter_over_area": perimeter_over_area,
        "intensity_mean": mean_int,
        "intensity_std": std_int,
        "intensity_cv": cv_int,
    }

    q, vmin, levels = quantize_bf_bins(img2d.astype(np.float32), roi, bins=BF_BINS)
    if q is not None and levels > 1:
        feats.update(glcm_2d(q, roi, vmin, levels, offsets=OFFSETS_2D_4))

    return feats


def extract_25dbf_features(img3d: np.ndarray):
    roi3d = img3d != 0
    vals = img3d[roi3d].astype(np.float32)
    if vals.size == 0:
        return {}

    iz = choose_max_area_slice(roi3d)
    roi2d = roi3d[iz]

    area_px = int(roi2d.sum())
    area_um2 = area_px * DX_UM * DY_UM

    major_um, minor_um = ellipse_axes_regionprops_2d(roi2d, DX_UM, DY_UM)
    aspect_ratio_2d = major_um / minor_um if minor_um != 0 else np.nan

    power_c1, power_c2, ellipticity_2d, _, _ = fourier_ellipticity_2d(
        roi2d, DX_UM, DY_UM
    )

    boundary = mask_to_boundary_coords_2d(roi2d)
    perimeter_um = polygon_perimeter_from_boundary(boundary, DX_UM, DY_UM)
    perimeter_over_area = perimeter_um / area_um2 if area_um2 != 0 else np.nan

    mean_int = float(vals.mean())
    std_int = float(vals.std(ddof=0))
    cv_int = float(std_int / mean_int) if mean_int != 0 else np.nan

    feats = {
        "area_um2": area_um2,
        "major_axis_2d_um": major_um,
        "minor_axis_2d_um": minor_um,
        "aspect_ratio_2d": aspect_ratio_2d,
        "fourier_power_c1": power_c1,
        "fourier_power_c2": power_c2,
        "ellipticity_2d": ellipticity_2d,
        "perimeter_um": perimeter_um,
        "perimeter_over_area": perimeter_over_area,
        "intensity_mean": mean_int,
        "intensity_std": std_int,
        "intensity_cv": cv_int,
    }

    q, vmin, levels = quantize_bf_bins(img3d.astype(np.float32), roi3d, bins=BF_BINS)
    if q is not None and levels > 1:
        feats.update(glcm_25d_mrg_from_stack(q, roi3d, vmin, levels, offsets2d=OFFSETS_2D_4))

    return feats


def extract_ht_features(vol3d: np.ndarray, bf_intensity_path: Path = None):
    roi = vol3d != 0
    vals = vol3d[roi].astype(np.float32)
    if vals.size == 0:
        return {}

    voxel_count = int(roi.sum())
    volume_um3 = voxel_count * VOX_UM3
    volume_pL = volume_um3 * 1e-3

    iz = choose_max_area_slice(roi)
    roi2d = roi[iz]

    area_px_2d = int(roi2d.sum())
    area_um2 = area_px_2d * DX_UM * DY_UM

    major2d_um, minor2d_um = ellipse_axes_regionprops_2d(roi2d, DX_UM, DY_UM)
    aspect_ratio_2d = major2d_um / minor2d_um if minor2d_um != 0 else np.nan
    
    power_c1_2d, power_c2_2d, ellipticity_2d, _, _ = fourier_ellipticity_2d(
        roi2d, DX_UM, DY_UM
    )

    boundary = mask_to_boundary_coords_2d(roi2d)
    perimeter_um = polygon_perimeter_from_boundary(boundary, DX_UM, DY_UM)
    perimeter_over_area = perimeter_um / area_um2 if area_um2 != 0 else np.nan

    axes3d = ellipsoid_axes_from_mask_3d(roi, DX_UM, DY_UM, DZ_UM)
    major3d_um = float(axes3d[0])
    middle3d_um = float(axes3d[1])
    minor3d_um = float(axes3d[2])

    aspect_ratio_3d = major3d_um / minor3d_um if minor3d_um != 0 else np.nan

    power_l0, power_l2, ellipticity_3d, center_x_um, center_y_um, center_z_um = spherical_bandpower_l0_l2_pointcloud(
        roi, DX_UM, DY_UM, DZ_UM
    )

    surface_um2 = surface_area_from_mask(roi, DX_UM, DY_UM, DZ_UM)
    surface_over_volume = surface_um2 / volume_um3 if volume_um3 != 0 else np.nan

    ri_mean = float(vals.mean())
    ri_std = float(vals.std(ddof=0))
    ri_cv = float(ri_std / ri_mean) if ri_mean != 0 else np.nan

    dry = compute_drymass_metrics_from_ri(
        vol_ri=vol3d.astype(np.float32),
        mask=roi.astype(bool),
        dx_um=DX_UM,
        dy_um=DY_UM,
        dz_um=DZ_UM,
    )

    feats = {
        "volume_pL": volume_pL,
        "area_um2": area_um2,
        "major_axis_2d_um": major2d_um,
        "minor_axis_2d_um": minor2d_um,
        "aspect_ratio_2d": aspect_ratio_2d,
        "fourier_power_c1": power_c1_2d,
        "fourier_power_c2": power_c2_2d,
        "ellipticity_2d": ellipticity_2d,
        "perimeter_um": perimeter_um,
        "perimeter_over_area": perimeter_over_area,
        "ellipticity_2d": ellipticity_2d,
        "major_axis_3d_um": major3d_um,
        "minor_axis_3d_um": minor3d_um,
        "aspect_ratio_3d": aspect_ratio_3d,
        "sph_power_l0": power_l0,
        "sph_power_l2": power_l2,
        "ellipticity_3d": ellipticity_3d,
        "surface_um2": surface_um2,
        "surface_over_volume": surface_over_volume,
        "RI_mean": ri_mean,
        "RI_std": ri_std,
        "RI_cv": ri_cv,
        "dry_mass_pg": dry["drymass_pg_sum"],
        "drymass_density_mg_per_ml": dry["drymass_density_mg_per_ml"],
    }

    q, vmin, levels = quantize_ht_round_int(vol3d, roi)
    if q is not None and levels > 1:
        feats.update(glcm_3d(q, roi, vmin, levels, offsets=OFFSETS_13))

    if CALCULATE_HT_3DBF_INTENSITY:
        if bf_intensity_path is None:
            warnings.warn("No matching 3DBF file found for HT intensity statistics.", stacklevel=2)
        else:
            bf_arr = ensure_2d_or_3d(read_image_any(bf_intensity_path))
            if bf_arr.ndim != 3:
                raise ValueError(f"Matched HT 3DBF intensity file must be 3D, got shape={bf_arr.shape}")
            feats.update(extract_bf_intensity_stats(bf_arr))

    return feats


# ============================================================
# Main
# ============================================================

def analyze_one_file(path: Path,
                     root_base: Path,
                     folder_name: str,
                     modality_name: str,
                     ht_3dbf_index=None):
    arr = read_image_any(path)
    arr = ensure_2d_or_3d(arr)

    rec = {
        "root_date": root_base.name,
        "modality": modality_name,
        "source_folder": folder_name,
        "group": infer_group_from_name(path.name),
        "filename": path.name,
        "filepath": str(path),
    }

    if modality_name == "2DBF":
        if arr.ndim != 2:
            raise ValueError(f"2DBF must be 2D, got shape={arr.shape}")
        feats = extract_2dbf_features(arr)

    elif modality_name == "2.5DBF":
        if arr.ndim != 3:
            raise ValueError(f"2.5DBF must be 3D, got shape={arr.shape}")
        feats = extract_25dbf_features(arr)

    elif modality_name == "HT":
        if arr.ndim != 3:
            raise ValueError(f"HT must be 3D, got shape={arr.shape}")
        bf_intensity_path = None
        if CALCULATE_HT_3DBF_INTENSITY and ht_3dbf_index is not None:
            bf_intensity_path = ht_3dbf_index.get(path.name.lower())
        feats = extract_ht_features(arr.astype(np.float32), bf_intensity_path=bf_intensity_path)

    else:
        raise ValueError(f"Unknown modality: {modality_name}")

    rec.update(feats)
    return rec


def run_all_or_target():
    records = []

    print(f"\n[ROOT_BASE] {ROOT_BASE}")
    print(f"[DATE_DIRS] {DATE_DIRS}")
    print(f"[TARGET_KEYWORD] {TARGET_KEYWORD if str(TARGET_KEYWORD).strip() != '' else '(ALL FILES)'}")
    print(f"[USE_RUST_GLCM] {USE_RUST_GLCM and FAST_GLCM_AVAILABLE}")
    print(f"[CALCULATE_HT_3DBF_INTENSITY] {CALCULATE_HT_3DBF_INTENSITY}")

    for date_dir in DATE_DIRS:
        root_dir = ROOT_BASE / date_dir

        if not root_dir.exists():
            print(f"\n[Skip date] {date_dir} (folder not found)")
            continue

        print(f"\n[DATE] {date_dir}")

        ht_3dbf_index = {}
        if CALCULATE_HT_3DBF_INTENSITY:
            ht_3dbf_folder = root_dir / HT_3DBF_FOLDER_NAME
            ht_3dbf_index, duplicates = build_file_index_by_name(ht_3dbf_folder)
            print(f"  - {HT_3DBF_FOLDER_NAME} (HT intensity source): {len(ht_3dbf_index)} indexed files")
            if duplicates:
                print(f"    [WARN] Duplicate 3DBF filenames ignored: {len(duplicates)}")

        for folder_name, modality_name in MODALITY_MAP.items():
            folder = root_dir / folder_name

            if not folder.exists():
                print(f"  - {folder_name} ({modality_name}): folder not found")
                continue

            files_all = list_files_recursive(folder)
            files = [f for f in files_all if file_matches_target(f, TARGET_KEYWORD)]

            print(f"  - {folder_name} ({modality_name}): {len(files)} matched files")

            for i, f in enumerate(files, 1):
                print(f"    [{i}/{len(files)}] {f.name}")
                rec = analyze_one_file(
                    f,
                    root_dir,
                    folder_name,
                    modality_name,
                    ht_3dbf_index=ht_3dbf_index,
                )
                records.append(rec)

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("\nNo files matched.")
        return df

    key_cols = [
        "root_date", "modality", "source_folder", "group",
        "filename", "filepath"
    ]
    other_cols = [c for c in df.columns if c not in key_cols]
    df = df[key_cols + other_cols]

    return df


if __name__ == "__main__":
    df = run_all_or_target()

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print("\nNothing to save.")
    else:
        excel_df = df.drop(columns=EXCLUDE_EXCEL_COLUMNS, errors="ignore")
        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
            excel_df.to_excel(writer, index=False, sheet_name="all_features")
            for mod in ["2DBF", "2.5DBF", "HT"]:
                sub = excel_df[excel_df["modality"] == mod].copy()
                sub.to_excel(writer, index=False, sheet_name=mod.replace(".", "p"))

        print("\nSaved:", OUT_XLSX)
        print(excel_df)
