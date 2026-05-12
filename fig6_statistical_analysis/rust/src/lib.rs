use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn check_levels(levels: usize) -> PyResult<()> {
    if levels < 2 {
        return Err(PyValueError::new_err("levels must be >= 2"));
    }
    Ok(())
}

#[pyfunction]
fn glcm2d_counts<'py>(
    py: Python<'py>,
    img_q2d: PyReadonlyArray2<'py, i32>,
    mask2d: PyReadonlyArray2<'py, bool>,
    levels: usize,
    offsets: Vec<(isize, isize)>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    check_levels(levels)?;

    let img = img_q2d.as_array();
    let mask = mask2d.as_array();

    let shape_img = img.shape();
    let shape_mask = mask.shape();

    if shape_img != shape_mask {
        return Err(PyValueError::new_err("img_q2d and mask2d must have the same shape"));
    }

    let h = shape_img[0] as isize;
    let w = shape_img[1] as isize;

    let mut counts = Array2::<i64>::zeros((levels, levels));

    for (dy, dx) in offsets {
        let y_start = 0.max(dy);
        let y_end = h.min(h + dy);
        let x_start = 0.max(dx);
        let x_end = w.min(w + dx);

        for y1 in y_start..y_end {
            let y2 = y1 - dy;
            for x1 in x_start..x_end {
                let x2 = x1 - dx;

                let y1u = y1 as usize;
                let x1u = x1 as usize;
                let y2u = y2 as usize;
                let x2u = x2 as usize;

                if !(mask[[y1u, x1u]] && mask[[y2u, x2u]]) {
                    continue;
                }

                let a = img[[y1u, x1u]];
                let b = img[[y2u, x2u]];

                if a < 0 || b < 0 {
                    return Err(PyValueError::new_err("quantized image values must be nonnegative"));
                }

                let ai = a as usize;
                let bi = b as usize;

                if ai >= levels || bi >= levels {
                    return Err(PyValueError::new_err("quantized image value out of range for levels"));
                }

                counts[[ai, bi]] += 1;
                counts[[bi, ai]] += 1;
            }
        }
    }

    Ok(counts.into_pyarray_bound(py))
}

#[pyfunction]
fn glcm3d_counts<'py>(
    py: Python<'py>,
    vol_q: PyReadonlyArray3<'py, i32>,
    mask: PyReadonlyArray3<'py, bool>,
    levels: usize,
    offsets: Vec<(isize, isize, isize)>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    check_levels(levels)?;

    let vol = vol_q.as_array();
    let msk = mask.as_array();

    let shape_vol = vol.shape();
    let shape_mask = msk.shape();

    if shape_vol != shape_mask {
        return Err(PyValueError::new_err("vol_q and mask must have the same shape"));
    }

    let d = shape_vol[0] as isize;
    let h = shape_vol[1] as isize;
    let w = shape_vol[2] as isize;

    let mut counts = Array2::<i64>::zeros((levels, levels));

    for (dz, dy, dx) in offsets {
        let z_start = 0.max(dz);
        let z_end = d.min(d + dz);
        let y_start = 0.max(dy);
        let y_end = h.min(h + dy);
        let x_start = 0.max(dx);
        let x_end = w.min(w + dx);

        for z1 in z_start..z_end {
            let z2 = z1 - dz;
            for y1 in y_start..y_end {
                let y2 = y1 - dy;
                for x1 in x_start..x_end {
                    let x2 = x1 - dx;

                    let z1u = z1 as usize;
                    let y1u = y1 as usize;
                    let x1u = x1 as usize;

                    let z2u = z2 as usize;
                    let y2u = y2 as usize;
                    let x2u = x2 as usize;

                    if !(msk[[z1u, y1u, x1u]] && msk[[z2u, y2u, x2u]]) {
                        continue;
                    }

                    let a = vol[[z1u, y1u, x1u]];
                    let b = vol[[z2u, y2u, x2u]];

                    if a < 0 || b < 0 {
                        return Err(PyValueError::new_err("quantized volume values must be nonnegative"));
                    }

                    let ai = a as usize;
                    let bi = b as usize;

                    if ai >= levels || bi >= levels {
                        return Err(PyValueError::new_err("quantized volume value out of range for levels"));
                    }

                    counts[[ai, bi]] += 1;
                    counts[[bi, ai]] += 1;
                }
            }
        }
    }

    Ok(counts.into_pyarray_bound(py))
}

#[pymodule]
fn fast_glcm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(glcm2d_counts, m)?)?;
    m.add_function(wrap_pyfunction!(glcm3d_counts, m)?)?;
    Ok(())
}