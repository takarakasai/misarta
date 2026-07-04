//! Conversion helpers between numpy.ndarray and nalgebra types.
//!
//! All conversions copy data — no aliasing between Python and Rust memory.
//! Shape mismatches raise `ValueError` on the Python side.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ─── numpy → nalgebra ───────────────────────────────────────────────────────

pub fn pyarray_to_vec(arr: PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    let slice = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be contiguous"))?;
    Ok(slice.to_vec())
}

pub fn pyarray_to_vector3(arr: PyReadonlyArray1<f64>) -> PyResult<Vector3<f64>> {
    let slice = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be contiguous"))?;
    if slice.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "expected length-3 vector, got length {}",
            slice.len()
        )));
    }
    Ok(Vector3::new(slice[0], slice[1], slice[2]))
}

pub fn pyarray_to_matrix3(arr: PyReadonlyArray2<f64>) -> PyResult<Matrix3<f64>> {
    let view = arr.as_array();
    let shape = view.shape();
    if shape != [3, 3] {
        return Err(PyValueError::new_err(format!(
            "expected (3, 3) matrix, got {:?}",
            shape
        )));
    }
    let mut out = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            out[(i, j)] = view[[i, j]];
        }
    }
    Ok(out)
}

// ─── nalgebra → numpy ───────────────────────────────────────────────────────

pub fn dvector_to_pyarray<'py>(
    py: Python<'py>,
    v: &DVector<f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice_bound(py, v.as_slice())
}

pub fn vector3_to_pyarray<'py>(
    py: Python<'py>,
    v: &Vector3<f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice_bound(py, &[v[0], v[1], v[2]])
}

pub fn matrix3_to_pyarray<'py>(
    py: Python<'py>,
    m: &Matrix3<f64>,
) -> Bound<'py, PyArray2<f64>> {
    // nalgebra is column-major; build a row-major Vec for numpy.
    let mut data = Vec::with_capacity(9);
    for i in 0..3 {
        for j in 0..3 {
            data.push(m[(i, j)]);
        }
    }
    PyArray2::from_vec2_bound(py, &[
        vec![data[0], data[1], data[2]],
        vec![data[3], data[4], data[5]],
        vec![data[6], data[7], data[8]],
    ])
    .expect("3x3 vec2 is well-shaped")
}

pub fn dmatrix_to_pyarray<'py>(
    py: Python<'py>,
    m: &DMatrix<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let (rows, cols) = m.shape();
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(m[(i, j)]);
        }
        data.push(row);
    }
    PyArray2::from_vec2_bound(py, &data).expect("rectangular shape")
}

// ─── Length validation helpers ──────────────────────────────────────────────

pub fn check_len(name: &str, got: usize, expected: usize) -> PyResult<()> {
    if got != expected {
        return Err(PyValueError::new_err(format!(
            "{name} has length {got}, expected {expected}"
        )));
    }
    Ok(())
}
