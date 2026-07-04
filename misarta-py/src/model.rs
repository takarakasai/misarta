//! PyModel — Python wrapper around `misarta::model::Model<f64>`.
//!
//! Models are immutable; the inner `Model<f64>` is held in an `Arc` so that
//! `PyData` (and any algorithm result holding model references) can share it
//! cheaply across the FFI boundary.

use crate::conv;
use crate::se3::PySE3;
use misarta::joint::JointType;
use misarta::model::{JointModel, Model};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Python-facing joint type enum.
///
/// Use the static constructors `JointType.revolute(axis)`,
/// `JointType.prismatic(axis)`, `JointType.fixed()`, `JointType.free_flyer()`.
#[pyclass(name = "JointType", module = "misarta._misarta")]
#[derive(Clone)]
pub struct PyJointType {
    pub inner: JointType<f64>,
}

#[pymethods]
impl PyJointType {
    #[staticmethod]
    fn revolute(axis: numpy::PyReadonlyArray1<f64>) -> PyResult<Self> {
        let a = conv::pyarray_to_vector3(axis)?;
        Ok(PyJointType { inner: JointType::Revolute { axis: a } })
    }

    #[staticmethod]
    fn prismatic(axis: numpy::PyReadonlyArray1<f64>) -> PyResult<Self> {
        let a = conv::pyarray_to_vector3(axis)?;
        Ok(PyJointType { inner: JointType::Prismatic { axis: a } })
    }

    #[staticmethod]
    fn fixed() -> Self {
        PyJointType { inner: JointType::Fixed }
    }

    #[staticmethod]
    fn free_flyer() -> Self {
        PyJointType { inner: JointType::FreeFlyer }
    }

    #[getter]
    fn nq(&self) -> usize {
        self.inner.nq()
    }

    #[getter]
    fn nv(&self) -> usize {
        self.inner.nv()
    }

    /// Lowercase type name: "revolute", "prismatic", "fixed", "free_flyer".
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            JointType::Revolute { .. } => "revolute",
            JointType::Prismatic { .. } => "prismatic",
            JointType::Fixed => "fixed",
            JointType::FreeFlyer => "free_flyer",
        }
    }

    fn __repr__(&self) -> String {
        format!("JointType({})", self.kind())
    }
}

/// Read-only view of a single joint in the kinematic tree.
#[pyclass(name = "JointModel", module = "misarta._misarta")]
#[derive(Clone)]
pub struct PyJointModel {
    pub inner: JointModel<f64>,
}

#[pymethods]
impl PyJointModel {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn parent(&self) -> usize {
        self.inner.parent
    }

    #[getter]
    fn joint_type(&self) -> PyJointType {
        PyJointType { inner: self.inner.joint_type.clone() }
    }

    #[getter]
    fn placement(&self) -> PySE3 {
        PySE3 { inner: self.inner.placement.clone() }
    }

    fn __repr__(&self) -> String {
        let kind = match &self.inner.joint_type {
            JointType::Revolute { .. } => "revolute",
            JointType::Prismatic { .. } => "prismatic",
            JointType::Fixed => "fixed",
            JointType::FreeFlyer => "free_flyer",
        };
        format!(
            "JointModel(name={:?}, parent={}, type={})",
            self.inner.name, self.inner.parent, kind
        )
    }
}

/// Immutable robot model.
#[pyclass(name = "Model", module = "misarta._misarta")]
#[derive(Clone)]
pub struct PyModel {
    pub inner: Arc<Model<f64>>,
}

impl PyModel {
    pub fn from_model(m: Model<f64>) -> Self {
        PyModel { inner: Arc::new(m) }
    }
}

#[pymethods]
impl PyModel {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn nq(&self) -> usize {
        self.inner.nq
    }

    #[getter]
    fn nv(&self) -> usize {
        self.inner.nv
    }

    /// Number of joints (excluding the universe at index 0).
    #[getter]
    fn njoints(&self) -> usize {
        self.inner.num_joints()
    }

    /// Total number of frames including the universe (= len(joints)).
    #[getter]
    fn n_total(&self) -> usize {
        self.inner.joints.len()
    }

    #[getter]
    fn gravity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let g = &self.inner.gravity;
        PyArray1::from_slice_bound(py, &[g[0], g[1], g[2]])
    }

    /// Get the joint at the given 1-based index.
    fn joint(&self, idx: usize) -> PyResult<PyJointModel> {
        if idx == 0 || idx >= self.inner.joints.len() {
            return Err(PyValueError::new_err(format!(
                "joint index {} out of range (1..{})", idx, self.inner.joints.len()
            )));
        }
        Ok(PyJointModel { inner: self.inner.joints[idx].clone() })
    }

    /// Find the 1-based joint index for the given joint name, or `None`.
    fn joint_id(&self, name: &str) -> Option<usize> {
        self.inner
            .joints
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, j)| j.name == name)
            .map(|(i, _)| i)
    }

    /// Find the index for the given link name (0 = root), or `None`.
    fn link_id(&self, name: &str) -> Option<usize> {
        self.inner
            .link_names
            .iter()
            .position(|n| n == name)
    }

    /// All joint names (excluding universe).
    fn joint_names(&self) -> Vec<String> {
        self.inner
            .joints
            .iter()
            .skip(1)
            .map(|j| j.name.clone())
            .collect()
    }

    /// All link names.
    fn link_names(&self) -> Vec<String> {
        self.inner.link_names.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(name={:?}, nq={}, nv={}, njoints={})",
            self.inner.name, self.inner.nq, self.inner.nv, self.inner.num_joints()
        )
    }
}
