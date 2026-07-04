//! PyData — Python wrapper around `misarta::data::Data<f64>`.
//!
//! Mirrors misarta's pure-functional API: algorithms return fresh `Data`
//! values rather than mutating an existing one.

use crate::conv;
use crate::model::PyModel;
use crate::se3::PySE3;
use misarta::data::Data;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "Data", module = "misarta._misarta")]
#[derive(Clone)]
pub struct PyData {
    pub inner: Data<f64>,
}

impl PyData {
    pub fn from_data(d: Data<f64>) -> Self {
        PyData { inner: d }
    }
}

#[allow(non_snake_case)]
#[pymethods]
impl PyData {
    /// Allocate empty data sized for the given model.
    #[new]
    fn new(model: &PyModel) -> Self {
        PyData { inner: Data::new(&model.inner) }
    }

    /// Absolute placement of joint `i` in the world frame (oMi).
    fn oMi(&self, i: usize) -> PyResult<PySE3> {
        if i >= self.inner.oMi.len() {
            return Err(PyValueError::new_err(format!(
                "joint index {} out of range (0..{})",
                i,
                self.inner.oMi.len()
            )));
        }
        Ok(PySE3 { inner: self.inner.oMi[i].clone() })
    }

    /// Joint placement relative to its parent (parent_M_joint).
    fn joint_placement(&self, i: usize) -> PyResult<PySE3> {
        if i >= self.inner.joint_placements.len() {
            return Err(PyValueError::new_err(format!(
                "joint index {} out of range (0..{})",
                i,
                self.inner.joint_placements.len()
            )));
        }
        Ok(PySE3 { inner: self.inner.joint_placements[i].clone() })
    }

    /// Body-frame Jacobian matrix (6 x nv). Populated only when a Jacobian
    /// algorithm has run; otherwise zeros.
    #[getter]
    fn J<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        conv::dmatrix_to_pyarray(py, &self.inner.J)
    }

    /// Body-frame spatial velocity [omega; v_lin] of joint `i` (length 6).
    fn body_velocity<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if i >= self.inner.v.len() {
            return Err(PyValueError::new_err(format!(
                "joint index {} out of range (0..{})", i, self.inner.v.len()
            )));
        }
        let v = &self.inner.v[i];
        Ok(PyArray1::from_slice_bound(
            py,
            &[v[0], v[1], v[2], v[3], v[4], v[5]],
        ))
    }

    /// Body-frame spatial acceleration [alpha; a_lin] of joint `i` (length 6).
    fn body_acceleration<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if i >= self.inner.a.len() {
            return Err(PyValueError::new_err(format!(
                "joint index {} out of range (0..{})", i, self.inner.a.len()
            )));
        }
        let a = &self.inner.a[i];
        Ok(PyArray1::from_slice_bound(
            py,
            &[a[0], a[1], a[2], a[3], a[4], a[5]],
        ))
    }
}
