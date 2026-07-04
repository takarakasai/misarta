//! Algorithm wrappers — FK, Jacobian, RNEA, CRBA, ABA.

use crate::conv;
use crate::data::PyData;
use crate::model::PyModel;
use misarta::{aba as misa_aba, crba as misa_crba, fk as misa_fk, jacobian as misa_jac, rnea as misa_rnea};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Forward kinematics — returns a fresh `Data` with `oMi[i]` filled.
#[pyfunction]
pub fn forward_kinematics(model: &PyModel, q: PyReadonlyArray1<f64>) -> PyResult<PyData> {
    let q = conv::pyarray_to_vec(q)?;
    conv::check_len("q", q.len(), model.inner.nq)?;
    let data = misa_fk::forward_kinematics(&model.inner, &q);
    Ok(PyData::from_data(data))
}

/// World-frame geometric Jacobian (6 x nv) for the given joint.
///
/// `joint_id` is 1-based. `ref_frame` is one of `LOCAL`, `WORLD`,
/// `LOCAL_WORLD_ALIGNED` (matching Pinocchio convention). For Phase 1 we
/// expose only the world-frame and local-frame variants exposed by misarta.
#[pyfunction]
#[pyo3(signature = (model, q, joint_id, ref_frame=crate::WORLD))]
pub fn compute_joint_jacobian<'py>(
    py: Python<'py>,
    model: &PyModel,
    q: PyReadonlyArray1<f64>,
    joint_id: usize,
    ref_frame: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let q = conv::pyarray_to_vec(q)?;
    conv::check_len("q", q.len(), model.inner.nq)?;

    let m = &model.inner;
    let j = if ref_frame == crate::LOCAL {
        misa_jac::compute_joint_jacobian_local(m, &q, joint_id)
    } else {
        // WORLD or LOCAL_WORLD_ALIGNED — both use world-frame Jacobian here.
        // (LOCAL_WORLD_ALIGNED differs only in linear-part interpretation,
        // which Phase 1 does not distinguish.)
        misa_jac::compute_joint_jacobian(m, &q, joint_id)
    };
    Ok(conv::dmatrix_to_pyarray(py, &j))
}

/// Inverse dynamics via RNEA. Returns the generalized force vector tau (length nv).
#[pyfunction]
pub fn rnea<'py>(
    py: Python<'py>,
    model: &PyModel,
    q: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = conv::pyarray_to_vec(q)?;
    let v = conv::pyarray_to_vec(v)?;
    let a = conv::pyarray_to_vec(a)?;
    conv::check_len("q", q.len(), model.inner.nq)?;
    conv::check_len("v", v.len(), model.inner.nv)?;
    conv::check_len("a", a.len(), model.inner.nv)?;
    let tau = misa_rnea::rnea(&model.inner, &q, &v, &a);
    Ok(conv::dvector_to_pyarray(py, &tau))
}

/// Joint-space inertia matrix M(q) via CRBA. Returns an `nv x nv` matrix.
#[pyfunction]
pub fn crba<'py>(
    py: Python<'py>,
    model: &PyModel,
    q: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let q = conv::pyarray_to_vec(q)?;
    conv::check_len("q", q.len(), model.inner.nq)?;
    let m = misa_crba::crba(&model.inner, &q);
    Ok(conv::dmatrix_to_pyarray(py, &m))
}

/// Forward dynamics via ABA. Returns acceleration vector q_ddot (length nv).
#[pyfunction]
pub fn aba<'py>(
    py: Python<'py>,
    model: &PyModel,
    q: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    tau: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = conv::pyarray_to_vec(q)?;
    let v = conv::pyarray_to_vec(v)?;
    let tau = conv::pyarray_to_vec(tau)?;
    conv::check_len("q", q.len(), model.inner.nq)?;
    conv::check_len("v", v.len(), model.inner.nv)?;
    conv::check_len("tau", tau.len(), model.inner.nv)?;
    let ddq = misa_aba::aba(&model.inner, &q, &v, &tau);
    Ok(conv::dvector_to_pyarray(py, &ddq))
}
