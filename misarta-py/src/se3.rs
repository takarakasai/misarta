//! PySE3 — Python wrapper around `misarta::se3::SE3<f64>` (= nalgebra Isometry3).

use crate::conv;
use misarta::se3::{self, SE3};
use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "SE3", module = "misarta._misarta")]
#[derive(Clone)]
pub struct PySE3 {
    pub inner: SE3<f64>,
}

#[pymethods]
impl PySE3 {
    /// Build an SE3 from a 3x3 rotation matrix and a length-3 translation vector.
    #[new]
    #[pyo3(signature = (rotation=None, translation=None))]
    fn new(
        rotation: Option<PyReadonlyArray2<f64>>,
        translation: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let rot = match rotation {
            Some(r) => {
                let m = conv::pyarray_to_matrix3(r)?;
                Rotation3::from_matrix_unchecked(m)
            }
            None => Rotation3::identity(),
        };
        let trans = match translation {
            Some(t) => conv::pyarray_to_vector3(t)?,
            None => nalgebra::Vector3::zeros(),
        };
        Ok(PySE3 {
            inner: SE3::from_parts(
                Translation3::from(trans),
                UnitQuaternion::from_rotation_matrix(&rot),
            ),
        })
    }

    #[staticmethod]
    fn identity() -> Self {
        PySE3 { inner: se3::identity() }
    }

    /// Build an SE3 from a 4x4 homogeneous matrix.
    #[staticmethod]
    fn from_homogeneous(matrix: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let view = matrix.as_array();
        let shape = view.shape();
        if shape != [4, 4] {
            return Err(PyValueError::new_err(format!(
                "expected (4, 4) matrix, got {:?}",
                shape
            )));
        }
        let mut m = nalgebra::Matrix4::zeros();
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] = view[[i, j]];
            }
        }
        Ok(PySE3 {
            inner: se3::from_homogeneous(&m),
        })
    }

    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let r = se3::rotation_matrix(&self.inner);
        conv::matrix3_to_pyarray(py, &r)
    }

    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let t = se3::translation(&self.inner);
        conv::vector3_to_pyarray(py, &t)
    }

    fn homogeneous<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let h = se3::to_homogeneous(&self.inner);
        // Build a row-major (4,4) numpy array from column-major nalgebra.
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(4);
        for i in 0..4 {
            let mut row = Vec::with_capacity(4);
            for j in 0..4 {
                row.push(h[(i, j)]);
            }
            rows.push(row);
        }
        PyArray2::from_vec2_bound(py, &rows).expect("4x4 shape ok")
    }

    fn inverse(&self) -> Self {
        PySE3 { inner: se3::inverse(&self.inner) }
    }

    fn __mul__(&self, other: &PySE3) -> PySE3 {
        PySE3 { inner: se3::compose(&self.inner, &other.inner) }
    }

    fn __repr__(&self) -> String {
        let t = se3::translation(&self.inner);
        format!("SE3(translation=[{:.6}, {:.6}, {:.6}], rotation=...)",
                t[0], t[1], t[2])
    }
}
