//! Rigid constraint model — Pinocchio-compatible constraint Jacobian framework.
//!
//! This module provides the building blocks for:
//!
//! - **Loop-closure constraints** (closed kinematic chains / parallel mechanisms)
//! - **Cross-branch IK** (e.g. both hands holding one object)
//! - **Relative pose constraints** between any two frames in the kinematic tree
//!
//! # Key Concepts
//!
//! A [`RigidConstraint`] specifies a desired relative placement between two
//! operational frames (*frame1* and *frame2*).  The frames can live on the
//! same chain, on different branches, or one of them can be the world frame
//! (joint index 0).
//!
//! The **constraint error** is the se(3) log of the discrepancy:
//!
//! $$e = \log\bigl(M_1^{-1}\, M_2\, M_{\text{des}}^{-1}\bigr)$$
//!
//! The **constraint Jacobian** is:
//!
//! $$J_c = J_2 - J_1 \quad\text{(world frame)}$$
//!
//! which maps joint velocities to the constraint-error rate.
//!
//! [`ConstraintModel`] aggregates multiple constraints.
//! [`compute_constraint_jacobian`] and [`compute_constraint_error`] evaluate
//! the stacked Jacobian and error for all constraints simultaneously.
//!
//! # Constraint types
//!
//! | Type | Rows | Description |
//! |------|------|-------------|
//! | `Contact6D` | 6 | Full pose (position + orientation) |
//! | `Contact3D` | 3 | Position only |
//!
//! # Example
//!
//! ```
//! use misarta::{model::*, joint, se3};
//! use misarta::constraint::{
//!     RigidConstraint, ConstraintType, ConstraintModel,
//!     compute_constraint_error, compute_constraint_jacobian,
//! };
//! use misarta::frames::Frame;
//!
//! // Build a Y-shaped tree: universe → j1 → j2 (left arm)
//! //                                    ↘ j3 (right arm)
//! let model = ModelBuilder::<f64>::new()
//!     .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
//!     .add_joint("j2", 1, joint::revolute_x(), se3::identity(), LinkInertia::zero())
//!     .add_joint("j3", 1, joint::revolute_y(), se3::identity(), LinkInertia::zero())
//!     .build();
//!
//! // Constrain j2 and j3 tips to be at the same position
//! let frame_left = Frame { name: "left".into(), parent_joint: 2, placement: se3::identity() };
//! let frame_right = Frame { name: "right".into(), parent_joint: 3, placement: se3::identity() };
//!
//! let c = RigidConstraint::position(frame_left, frame_right);
//! let cm = ConstraintModel::from_constraints(vec![c]);
//!
//! let q = vec![0.0; model.nq];
//! let err = compute_constraint_error(&model, &q, &cm);
//! let jc = compute_constraint_jacobian(&model, &q, &cm);
//! assert_eq!(jc.nrows(), 3);
//! assert_eq!(jc.ncols(), model.nv);
//! ```

// ─── Submodules ─────────────────────────────────────────────────────────────

pub mod types;
pub mod error;
pub mod jacobian;
pub mod ik;
pub mod qp_ik;

// ─── Re-exports (flat public API) ──────────────────────────────────────────

pub use types::{
    ConstraintModel, ConstraintType, ReferenceFrame, RigidConstraint,
};

pub use error::{
    compute_constraint_error, compute_constraint_error_from_data,
};

pub use jacobian::{
    compute_constraint_jacobian, compute_constraint_jacobian_from_data,
};

pub use ik::{
    ConstrainedIkConfig, ConstrainedIkResult,
    solve_constrained_ik, solve_frame_task_with_constraints,
    solve_task_with_constraints,
};

pub use qp_ik::{
    QpIkConfig,
    build_joint_limit_inequalities, build_max_step_inequalities,
    stack_inequalities,
    solve_constrained_ik_qp, solve_frame_task_with_constraints_qp,
    solve_task_with_constraints_qp,
};
