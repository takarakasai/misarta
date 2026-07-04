//! URDF / SDF model loaders.

use crate::model::PyModel;
use misarta_formats::urdf;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

/// Build a `Model` from a URDF XML string.
///
/// The optional `root` argument is accepted for Pinocchio API parity but
/// ignored — misarta's URDF loader infers the root from the topology.
#[pyfunction]
#[pyo3(signature = (urdf_str, root=None))]
pub fn build_model_from_urdf(urdf_str: &str, root: Option<&str>) -> PyResult<PyModel> {
    let _ = root; // currently unused; kept for API parity with Pinocchio.
    let model = urdf::load_urdf_string(urdf_str)
        .map_err(|e| PyValueError::new_err(format!("URDF parse error: {:?}", e)))?;
    Ok(PyModel::from_model(model))
}

/// Load a `Model` from a URDF file path.
#[pyfunction]
pub fn load_urdf(path: PathBuf) -> PyResult<PyModel> {
    let model = urdf::load_urdf(&path)
        .map_err(|e| PyIOError::new_err(format!("URDF load error: {:?}", e)))?;
    Ok(PyModel::from_model(model))
}
