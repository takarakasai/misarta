//! misarta Python bindings (PyO3).
//!
//! This crate exposes the f64 specialization of misarta to Python via PyO3.
//! See `doc/python-binding-plan.md` in the misarta crate for the design.

// PyO3 0.22 macros emit unsafe calls without explicit `unsafe { }` blocks;
// Rust 2024 lints these by default. The PyO3 expansion is sound — silence
// the warnings until we move to a newer PyO3 release.
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod algorithms;
mod conv;
mod data;
mod loaders;
mod model;
mod se3;

/// Reference-frame constants matching Pinocchio convention.
const LOCAL: i32 = 0;
const WORLD: i32 = 1;
const LOCAL_WORLD_ALIGNED: i32 = 2;

#[pymodule]
fn _misarta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add("LOCAL", LOCAL)?;
    m.add("WORLD", WORLD)?;
    m.add("LOCAL_WORLD_ALIGNED", LOCAL_WORLD_ALIGNED)?;

    m.add_class::<se3::PySE3>()?;
    m.add_class::<model::PyJointType>()?;
    m.add_class::<model::PyJointModel>()?;
    m.add_class::<model::PyModel>()?;
    m.add_class::<data::PyData>()?;

    m.add_function(wrap_pyfunction!(algorithms::forward_kinematics, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::compute_joint_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::rnea, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::crba, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::aba, m)?)?;

    m.add_function(wrap_pyfunction!(loaders::build_model_from_urdf, m)?)?;
    m.add_function(wrap_pyfunction!(loaders::load_urdf, m)?)?;

    Ok(())
}
