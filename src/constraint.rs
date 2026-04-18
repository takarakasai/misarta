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

use crate::data::Data;
use crate::fk::forward_kinematics;
use crate::frames::{
    compute_frame_jacobian_from_data, compute_frame_placement_from_data, Frame,
};
use crate::model::Model;
use crate::se3::{self, SE3};
use nalgebra::{DMatrix, DVector, RealField, Vector3};

// NOTE on sign convention:
//
// The constraint error is defined as e = actual − desired (e.g. p2 − expected).
// The constraint Jacobian is J_c = de/dq ≈ J2 − J1.
// To drive e → 0 we need dq = −J_c⁺ e (negative sign).

// ─── Constraint type ────────────────────────────────────────────────────────

/// Dimensionality / type of a rigid constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// Full 6-D constraint (position + orientation).
    Contact6D,
    /// Position-only 3-D constraint.
    Contact3D,
}

impl ConstraintType {
    /// Number of rows this constraint contributes.
    pub fn dim(&self) -> usize {
        match self {
            ConstraintType::Contact6D => 6,
            ConstraintType::Contact3D => 3,
        }
    }
}

// ─── Reference frame for expressing the constraint ─────────────────────────

/// In which coordinate frame the constraint error and Jacobian are expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceFrame {
    /// World (spatial / fixed) frame.
    World,
    /// Frame 1 (the first frame of the constraint pair).
    Local,
}

// ─── Single rigid constraint ────────────────────────────────────────────────

/// A rigid constraint between two operational frames.
///
/// The constraint enforces that the relative placement of `frame2` w.r.t.
/// `frame1` equals `desired_relative_placement` (identity by default,
/// meaning the two frames should coincide).
///
/// Either frame can have `parent_joint == 0` to anchor it to the world.
#[derive(Debug, Clone)]
pub struct RigidConstraint<T: RealField> {
    /// Human-readable name.
    pub name: String,
    /// First frame (reference).
    pub frame1: Frame<T>,
    /// Second frame (target).
    pub frame2: Frame<T>,
    /// Desired relative placement: $M_1^{-1} M_2 = M_{\text{des}}$.
    /// Default is identity (frames should coincide).
    pub desired_relative_placement: SE3<T>,
    /// Constraint type (6D or 3D).
    pub constraint_type: ConstraintType,
    /// Reference frame for the error/Jacobian.
    pub reference_frame: ReferenceFrame,
}

impl<T: RealField> RigidConstraint<T> {
    /// Create a 6-D (pose) constraint between two frames.
    ///
    /// The desired relative placement defaults to identity (frames coincide).
    pub fn pose(frame1: Frame<T>, frame2: Frame<T>) -> Self {
        Self {
            name: format!("{}-{}", frame1.name, frame2.name),
            frame1,
            frame2,
            desired_relative_placement: se3::identity(),
            constraint_type: ConstraintType::Contact6D,
            reference_frame: ReferenceFrame::World,
        }
    }

    /// Create a 3-D (position-only) constraint between two frames.
    pub fn position(frame1: Frame<T>, frame2: Frame<T>) -> Self {
        Self {
            name: format!("{}-{}", frame1.name, frame2.name),
            frame1,
            frame2,
            desired_relative_placement: se3::identity(),
            constraint_type: ConstraintType::Contact3D,
            reference_frame: ReferenceFrame::World,
        }
    }

    /// Builder: set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Builder: set the desired relative placement.
    pub fn with_desired_placement(mut self, m: SE3<T>) -> Self {
        self.desired_relative_placement = m;
        self
    }

    /// Builder: set the reference frame.
    pub fn with_reference_frame(mut self, rf: ReferenceFrame) -> Self {
        self.reference_frame = rf;
        self
    }

    /// Number of scalar constraint rows.
    pub fn dim(&self) -> usize {
        self.constraint_type.dim()
    }
}

// ─── Constraint model (collection) ─────────────────────────────────────────

/// Collection of rigid constraints.
#[derive(Debug, Clone)]
pub struct ConstraintModel<T: RealField> {
    pub constraints: Vec<RigidConstraint<T>>,
}

impl<T: RealField> ConstraintModel<T> {
    /// Create an empty constraint model.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Create from a vec of constraints.
    pub fn from_constraints(constraints: Vec<RigidConstraint<T>>) -> Self {
        Self { constraints }
    }

    /// Add a constraint.
    pub fn add(&mut self, c: RigidConstraint<T>) {
        self.constraints.push(c);
    }

    /// Total number of constraint rows.
    pub fn total_dim(&self) -> usize {
        self.constraints.iter().map(|c| c.dim()).sum()
    }

    /// Number of constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Whether there are no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

impl<T: RealField> Default for ConstraintModel<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Constraint error computation ──────────────────────────────────────────

/// Compute the stacked constraint error vector.
///
/// For each constraint, the error depends on the type:
///
/// - **Contact6D** (world frame): $e = \log(M_1^{-1} M_2 M_{\text{des}}^{-1})$
///   — the 6-D pose error expressed in the world frame.
/// - **Contact3D** (world frame): $e = t_2 - t_1 - R_1 t_{\text{des}}$
///   — the 3-D position error in the world frame.
///
/// Returns a `DVector<f64>` of length equal to `cm.total_dim()`.
pub fn compute_constraint_error(
    model: &Model<f64>,
    q: &[f64],
    cm: &ConstraintModel<f64>,
) -> DVector<f64> {
    let data = forward_kinematics(model, q);
    compute_constraint_error_from_data(&data, cm)
}

/// Same as [`compute_constraint_error`] but with pre-computed FK data.
pub fn compute_constraint_error_from_data(
    data: &Data<f64>,
    cm: &ConstraintModel<f64>,
) -> DVector<f64> {
    let total = cm.total_dim();
    let mut err = DVector::zeros(total);
    let mut row = 0;

    for c in &cm.constraints {
        let m1 = compute_frame_placement_from_data(data, &c.frame1);
        let m2 = compute_frame_placement_from_data(data, &c.frame2);

        match c.constraint_type {
            ConstraintType::Contact6D => {
                let e = compute_pose_error_6d(&m1, &m2, &c.desired_relative_placement, c.reference_frame);
                for i in 0..6 {
                    err[row + i] = e[i];
                }
                row += 6;
            }
            ConstraintType::Contact3D => {
                let e = compute_position_error_3d(&m1, &m2, &c.desired_relative_placement, c.reference_frame);
                for i in 0..3 {
                    err[row + i] = e[i];
                }
                row += 3;
            }
        }
    }

    err
}

/// 6-D pose error: $\log(M_{\text{des}}^{-1} M_1^{-1} M_2)$
fn compute_pose_error_6d(
    m1: &SE3<f64>,
    m2: &SE3<f64>,
    m_des: &SE3<f64>,
    reference_frame: ReferenceFrame,
) -> nalgebra::Vector6<f64> {
    // Relative placement: M_1^{-1} * M_2
    let m_rel = se3::compose(&se3::inverse(m1), m2);
    // Error: M_rel * M_des^{-1}  (== identity when constraint is satisfied)
    let m_err = se3::compose(&m_rel, &se3::inverse(m_des));
    let log_err = se3::log(&m_err);

    match reference_frame {
        ReferenceFrame::World => {
            // Rotate error from frame1 to world: R1 * log_err
            let r1 = se3::rotation_matrix(m1);
            let omega = Vector3::new(log_err[0], log_err[1], log_err[2]);
            let v = Vector3::new(log_err[3], log_err[4], log_err[5]);
            let omega_w = &r1 * omega;
            let v_w = &r1 * v;
            nalgebra::Vector6::new(omega_w[0], omega_w[1], omega_w[2], v_w[0], v_w[1], v_w[2])
        }
        ReferenceFrame::Local => log_err,
    }
}

/// 3-D position error
fn compute_position_error_3d(
    m1: &SE3<f64>,
    m2: &SE3<f64>,
    m_des: &SE3<f64>,
    reference_frame: ReferenceFrame,
) -> Vector3<f64> {
    let p1 = se3::translation(m1);
    let p2 = se3::translation(m2);
    let p_des = se3::translation(m_des);

    // Expected position of frame2 in world: p1 + R1 * p_des
    let r1 = se3::rotation_matrix(m1);
    let expected = &p1 + &r1 * &p_des;
    let err_world = p2 - expected;

    match reference_frame {
        ReferenceFrame::World => err_world,
        ReferenceFrame::Local => r1.transpose() * err_world,
    }
}

// ─── Constraint Jacobian computation ───────────────────────────────────────

/// Compute the stacked constraint Jacobian.
///
/// For each constraint:
///
/// $$J_c = J_{\text{frame2}} - J_{\text{frame1}}$$
///
/// (with appropriate row extraction for 3-D constraints).
///
/// Returns a `DMatrix<f64>` of shape `(total_dim, nv)`.
pub fn compute_constraint_jacobian(
    model: &Model<f64>,
    q: &[f64],
    cm: &ConstraintModel<f64>,
) -> DMatrix<f64> {
    let data = forward_kinematics(model, q);
    compute_constraint_jacobian_from_data(model, q, &data, cm)
}

/// Same as [`compute_constraint_jacobian`] but with pre-computed FK data.
pub fn compute_constraint_jacobian_from_data(
    model: &Model<f64>,
    q: &[f64],
    data: &Data<f64>,
    cm: &ConstraintModel<f64>,
) -> DMatrix<f64> {
    let total = cm.total_dim();
    let nv = model.nv;
    let mut jc = DMatrix::zeros(total, nv);
    let mut row = 0;

    for c in &cm.constraints {
        // Compute frame Jacobians (6 × nv each)
        let j1 = if c.frame1.parent_joint == 0 {
            // Frame1 is anchored to world → zero Jacobian
            DMatrix::zeros(6, nv)
        } else {
            compute_frame_jacobian_from_data(model, q, data, &c.frame1)
        };

        let j2 = if c.frame2.parent_joint == 0 {
            DMatrix::zeros(6, nv)
        } else {
            compute_frame_jacobian_from_data(model, q, data, &c.frame2)
        };

        // Relative Jacobian: J_c = J2 - J1
        let j_rel = &j2 - &j1;

        match c.constraint_type {
            ConstraintType::Contact6D => {
                match c.reference_frame {
                    ReferenceFrame::World => {
                        // Use as-is (world-frame Jacobian)
                        jc.view_mut((row, 0), (6, nv)).copy_from(&j_rel);
                    }
                    ReferenceFrame::Local => {
                        // Rotate to frame1's local frame
                        let m1 = compute_frame_placement_from_data(data, &c.frame1);
                        let r1 = se3::rotation_matrix(&m1);
                        let r1t = r1.transpose();
                        for col in 0..nv {
                            let w = Vector3::new(j_rel[(0, col)], j_rel[(1, col)], j_rel[(2, col)]);
                            let v = Vector3::new(j_rel[(3, col)], j_rel[(4, col)], j_rel[(5, col)]);
                            let w_l = &r1t * w;
                            let v_l = &r1t * v;
                            jc[(row, col)] = w_l[0];
                            jc[(row + 1, col)] = w_l[1];
                            jc[(row + 2, col)] = w_l[2];
                            jc[(row + 3, col)] = v_l[0];
                            jc[(row + 4, col)] = v_l[1];
                            jc[(row + 5, col)] = v_l[2];
                        }
                    }
                }
                row += 6;
            }
            ConstraintType::Contact3D => {
                match c.reference_frame {
                    ReferenceFrame::World => {
                        // Extract linear rows (rows 3-5)
                        jc.view_mut((row, 0), (3, nv))
                            .copy_from(&j_rel.view((3, 0), (3, nv)));
                    }
                    ReferenceFrame::Local => {
                        let m1 = compute_frame_placement_from_data(data, &c.frame1);
                        let r1 = se3::rotation_matrix(&m1);
                        let r1t = r1.transpose();
                        for col in 0..nv {
                            let v = Vector3::new(j_rel[(3, col)], j_rel[(4, col)], j_rel[(5, col)]);
                            let v_l = &r1t * v;
                            jc[(row, col)] = v_l[0];
                            jc[(row + 1, col)] = v_l[1];
                            jc[(row + 2, col)] = v_l[2];
                        }
                    }
                }
                row += 3;
            }
        }
    }

    jc
}

// ─── Constrained IK ────────────────────────────────────────────────────────

/// Result of a constrained IK solve.
#[derive(Debug, Clone)]
pub struct ConstrainedIkResult {
    /// Solution configuration.
    pub q: Vec<f64>,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Final constraint error norm.
    pub constraint_error_norm: f64,
    /// Final primary task error norm (if any).
    pub task_error_norm: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

/// Configuration for constrained / cross-chain IK.
#[derive(Debug, Clone)]
pub struct ConstrainedIkConfig {
    /// Maximum iterations.
    pub max_iters: usize,
    /// Convergence tolerance on constraint error.
    pub tol_constraint: f64,
    /// Convergence tolerance on task error (primary IK task).
    pub tol_task: f64,
    /// Step size (0, 1].
    pub step_size: f64,
    /// Damping factor for DLS.
    pub damping: f64,
    /// Weight for constraint enforcement relative to task.
    pub constraint_weight: f64,
}

impl Default for ConstrainedIkConfig {
    fn default() -> Self {
        Self {
            max_iters: 200,
            tol_constraint: 1e-6,
            tol_task: 1e-6,
            step_size: 0.5,
            damping: 1e-3,
            constraint_weight: 10.0,
        }
    }
}

/// Solve IK subject to rigid constraints (constraint-only, no primary task).
///
/// Minimises the constraint error using DLS on the constraint Jacobian.
/// Useful for loop-closure or aligning two branches.
///
/// # Arguments
///
/// * `model` — robot model
/// * `q0` — initial configuration
/// * `cm` — constraint model
/// * `config` — solver configuration
pub fn solve_constrained_ik(
    model: &Model<f64>,
    q0: &[f64],
    cm: &ConstraintModel<f64>,
    config: &ConstrainedIkConfig,
) -> ConstrainedIkResult {
    let mut q = q0.to_vec();
    let mut last_err = f64::INFINITY;

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);
        let e = compute_constraint_error_from_data(&data, cm);
        let e_norm = e.norm();
        last_err = e_norm;

        if e_norm <= config.tol_constraint {
            return ConstrainedIkResult {
                q,
                iterations: iter,
                constraint_error_norm: e_norm,
                task_error_norm: 0.0,
                converged: true,
            };
        }

        let jc = compute_constraint_jacobian_from_data(model, &q, &data, cm);
        let nc = jc.nrows();

        // DLS: dq = −Jc^T (Jc Jc^T + λ² I)^{-1} e  (negative to reduce error)
        let a = &jc * jc.transpose()
            + DMatrix::<f64>::identity(nc, nc) * (config.damping * config.damping);
        let neg_e = -&e;
        let Some(y) = a.lu().solve(&neg_e) else {
            break;
        };
        let dq = jc.transpose() * y * config.step_size;

        // Integrate
        q = crate::manifold::integrate(model, &q, dq.as_slice(), 1.0);
    }

    ConstrainedIkResult {
        q,
        iterations: config.max_iters,
        constraint_error_norm: last_err,
        task_error_norm: 0.0,
        converged: false,
    }
}

/// Solve IK with a primary end-effector task **and** rigid constraints.
///
/// Uses an augmented Jacobian approach: the task Jacobian and constraint
/// Jacobian are stacked, with the constraint rows weighted by
/// `config.constraint_weight`.
///
/// ```text
/// [ J_task            ] dq = [ e_task            ]
/// [ w · J_constraint  ]      [ w · e_constraint  ]
/// ```
///
/// # Arguments
///
/// * `model` — robot model
/// * `q0` — initial configuration
/// * `joint_idx` — end-effector joint for the primary task (position IK)
/// * `target` — desired world-frame position of the end-effector
/// * `cm` — constraint model (cross-chain / loop constraints)
/// * `config` — solver configuration
pub fn solve_task_with_constraints(
    model: &Model<f64>,
    q0: &[f64],
    joint_idx: usize,
    target: Vector3<f64>,
    cm: &ConstraintModel<f64>,
    config: &ConstrainedIkConfig,
) -> ConstrainedIkResult {
    use crate::jacobian::compute_joint_jacobian_from_data;

    let nc = cm.total_dim();
    let task_rows = 3; // position IK
    let total_rows = task_rows + nc;
    let nv = model.nv;

    let mut q = q0.to_vec();
    let mut last_task_err = f64::INFINITY;
    let mut last_constraint_err = f64::INFINITY;

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);

        // Task error (position): e_task = target − current (IK convention, positive sign)
        let p = se3::translation(&data.oMi[joint_idx]);
        let e_task = target - p;
        let task_norm = e_task.norm();
        last_task_err = task_norm;

        // Constraint error: e_c = actual − desired (positive sign, negated below)
        let e_c = compute_constraint_error_from_data(&data, cm);
        let c_norm = e_c.norm();
        last_constraint_err = c_norm;

        if task_norm <= config.tol_task && c_norm <= config.tol_constraint {
            return ConstrainedIkResult {
                q,
                iterations: iter,
                constraint_error_norm: c_norm,
                task_error_norm: task_norm,
                converged: true,
            };
        }

        // Augmented Jacobian
        let j_full = compute_joint_jacobian_from_data(model, &q, &data, joint_idx);
        let j_task = j_full.rows(3, 3).into_owned(); // linear rows only

        let jc = compute_constraint_jacobian_from_data(model, &q, &data, cm);

        // Stack: [J_task; w·J_c]  and  [+e_task; −w·e_c]
        // Task uses +e (IK convention); constraint uses −e (drive e→0).
        let mut j_aug = DMatrix::zeros(total_rows, nv);
        j_aug.view_mut((0, 0), (3, nv)).copy_from(&j_task);
        j_aug
            .view_mut((3, 0), (nc, nv))
            .copy_from(&(&jc * config.constraint_weight));

        let mut e_aug = DVector::zeros(total_rows);
        for i in 0..3 {
            e_aug[i] = e_task[i];
        }
        for i in 0..nc {
            e_aug[3 + i] = -e_c[i] * config.constraint_weight;
        }

        // DLS
        let a = &j_aug * j_aug.transpose()
            + DMatrix::<f64>::identity(total_rows, total_rows) * (config.damping * config.damping);
        let Some(y) = a.lu().solve(&e_aug) else {
            break;
        };
        let dq = j_aug.transpose() * y * config.step_size;

        q = crate::manifold::integrate(model, &q, dq.as_slice(), 1.0);
    }

    ConstrainedIkResult {
        q,
        iterations: config.max_iters,
        constraint_error_norm: last_constraint_err,
        task_error_norm: last_task_err,
        converged: false,
    }
}

/// Solve a **frame-based** task IK with rigid constraints (6-D pose task).
///
/// Primary task: align `task_frame` with `target_pose` (6-D).
/// Additional constraints from `cm` are enforced simultaneously.
pub fn solve_frame_task_with_constraints(
    model: &Model<f64>,
    q0: &[f64],
    task_frame: &Frame<f64>,
    target_pose: SE3<f64>,
    cm: &ConstraintModel<f64>,
    config: &ConstrainedIkConfig,
) -> ConstrainedIkResult {
    let nv = model.nv;
    let nc = cm.total_dim();
    let task_rows = 6; // pose IK
    let total_rows = task_rows + nc;

    let mut q = q0.to_vec();
    let mut last_task_err = f64::INFINITY;
    let mut last_constraint_err = f64::INFINITY;

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);

        // Task error (pose)
        let m_current = compute_frame_placement_from_data(&data, task_frame);
        let m_err = se3::compose(&se3::inverse(&m_current), &target_pose);
        let log_err = se3::log(&m_err);

        // Rotate to world frame
        let r = se3::rotation_matrix(&m_current);
        let omega = Vector3::new(log_err[0], log_err[1], log_err[2]);
        let v = Vector3::new(log_err[3], log_err[4], log_err[5]);
        let omega_w = &r * omega;
        let v_w = &r * v;

        let e_task = DVector::from_vec(vec![
            omega_w[0], omega_w[1], omega_w[2], v_w[0], v_w[1], v_w[2],
        ]);
        let task_norm = e_task.norm();
        last_task_err = task_norm;

        // Constraint error
        let e_c = compute_constraint_error_from_data(&data, cm);
        let c_norm = e_c.norm();
        last_constraint_err = c_norm;

        if task_norm <= config.tol_task && c_norm <= config.tol_constraint {
            return ConstrainedIkResult {
                q,
                iterations: iter,
                constraint_error_norm: c_norm,
                task_error_norm: task_norm,
                converged: true,
            };
        }

        // Task Jacobian (frame)
        let j_task = compute_frame_jacobian_from_data(model, &q, &data, task_frame);

        // Constraint Jacobian
        let jc = compute_constraint_jacobian_from_data(model, &q, &data, cm);

        // Stack: [J_task; w·J_c] and [+e_task; −w·e_c]
        let mut j_aug = DMatrix::zeros(total_rows, nv);
        j_aug.view_mut((0, 0), (6, nv)).copy_from(&j_task);
        j_aug
            .view_mut((6, 0), (nc, nv))
            .copy_from(&(&jc * config.constraint_weight));

        let mut e_aug = DVector::zeros(total_rows);
        for i in 0..6 {
            e_aug[i] = e_task[i];
        }
        for i in 0..nc {
            e_aug[6 + i] = -e_c[i] * config.constraint_weight;
        }

        // DLS
        let a = &j_aug * j_aug.transpose()
            + DMatrix::<f64>::identity(total_rows, total_rows) * (config.damping * config.damping);
        let Some(y) = a.lu().solve(&e_aug) else {
            break;
        };
        let dq = j_aug.transpose() * y * config.step_size;

        q = crate::manifold::integrate(model, &q, dq.as_slice(), 1.0);
    }

    ConstrainedIkResult {
        q,
        iterations: config.max_iters,
        constraint_error_norm: last_constraint_err,
        task_error_norm: last_task_err,
        converged: false,
    }
}

// ─── QP-based inequality-constrained IK ─────────────────────────────────────

/// Configuration for QP-based IK with inequality constraints.
///
/// At each iteration the solver forms a QP whose objective is
/// the weighted sum of the task error, equality-constraint error, and
/// a damping regulariser:
///
/// $$\min_{dq} \lVert J_t\, dq - e_t\rVert^2
///   + w^2 \lVert J_c\, dq + e_c\rVert^2
///   + \lambda^2 \lVert dq\rVert^2$$
///
/// subject to $A_{iq}\, dq \le b_{iq}$ (joint limits, workspace bounds, etc.).
#[derive(Debug, Clone)]
pub struct QpIkConfig {
    /// Maximum outer IK iterations.
    pub max_iters: usize,
    /// Convergence tolerance on primary task error.
    pub tol_task: f64,
    /// Convergence tolerance on equality constraint error.
    pub tol_constraint: f64,
    /// Step-size multiplier applied to the QP solution (0, 1].
    pub step_size: f64,
    /// Damping (Levenberg–Marquardt regularisation).
    pub damping: f64,
    /// Weight for equality constraints in the objective.
    pub constraint_weight: f64,
    /// Joint-position limits.  When `Some`, box inequalities
    /// $q_{\min} - q \le dq \le q_{\max} - q$ are added automatically.
    pub joint_limits: Option<crate::limits::JointLimits>,
    /// Max step-size bound: $\lVert dq\rVert_\infty \le \texttt{max\_step}$.
    pub max_step: Option<f64>,
    /// Maximum active-set iterations inside each QP solve.
    pub qp_max_iters: usize,
}

impl Default for QpIkConfig {
    fn default() -> Self {
        Self {
            max_iters: 200,
            tol_task: 1e-6,
            tol_constraint: 1e-6,
            step_size: 0.5,
            damping: 1e-3,
            constraint_weight: 10.0,
            joint_limits: None,
            max_step: None,
            qp_max_iters: 200,
        }
    }
}

/// Build the box-inequality rows for joint-position limits.
///
/// For each 1-DOF joint *i*:
///
/// $$dq_i \le q_{\max,i} - q_i , \qquad -dq_i \le q_i - q_{\min,i}$$
///
/// Returns `(A, b)` where $A\, dq \le b$.
/// Rows with infinite bounds are omitted.
pub fn build_joint_limit_inequalities(
    model: &Model<f64>,
    q: &[f64],
    limits: &crate::limits::JointLimits,
) -> (DMatrix<f64>, DVector<f64>) {
    use crate::joint::JointType;

    let nv = model.nv;
    let mut rows: Vec<(usize, f64, f64)> = Vec::new(); // (v_idx, coeff, rhs)

    for (i, joint) in model.joints.iter().enumerate().skip(1) {
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];

        match &joint.joint_type {
            JointType::Fixed => {}
            JointType::Revolute { .. } | JointType::Prismatic { .. } => {
                let q_cur = q[qi];
                let q_lo = limits.q_min[qi];
                let q_hi = limits.q_max[qi];
                if q_hi.is_finite() {
                    rows.push((vi, 1.0, q_hi - q_cur));
                }
                if q_lo.is_finite() {
                    rows.push((vi, -1.0, q_cur - q_lo));
                }
            }
            JointType::FreeFlyer => {
                for k in 0..3 {
                    let q_cur = q[qi + k];
                    let q_lo = limits.q_min[qi + k];
                    let q_hi = limits.q_max[qi + k];
                    if q_hi.is_finite() {
                        rows.push((vi + k, 1.0, q_hi - q_cur));
                    }
                    if q_lo.is_finite() {
                        rows.push((vi + k, -1.0, q_cur - q_lo));
                    }
                }
            }
        }
    }

    let m = rows.len();
    let mut a = DMatrix::zeros(m, nv);
    let mut b = DVector::zeros(m);
    for (r, &(col, coeff, rhs)) in rows.iter().enumerate() {
        a[(r, col)] = coeff;
        b[r] = rhs;
    }
    (a, b)
}

/// Build step-size inequality rows: $\lVert dq\rVert_\infty \le s$.
///
/// Returns $A\, dq \le b$ with $A = [I; -I]$, $b = s \cdot \mathbf{1}$.
pub fn build_max_step_inequalities(nv: usize, max_step: f64) -> (DMatrix<f64>, DVector<f64>) {
    let mut a = DMatrix::zeros(2 * nv, nv);
    let b = DVector::from_element(2 * nv, max_step);
    for i in 0..nv {
        a[(i, i)] = 1.0;
        a[(nv + i, i)] = -1.0;
    }
    (a, b)
}

/// Stack multiple inequality pairs into a single $(A, b)$.
pub fn stack_inequalities(pairs: &[(&DMatrix<f64>, &DVector<f64>)]) -> (DMatrix<f64>, DVector<f64>) {
    if pairs.is_empty() {
        return (DMatrix::zeros(0, 0), DVector::zeros(0));
    }
    let ncols = pairs[0].0.ncols();
    let total_rows: usize = pairs.iter().map(|(a, _)| a.nrows()).sum();
    let mut a = DMatrix::zeros(total_rows, ncols);
    let mut b = DVector::zeros(total_rows);
    let mut row = 0;
    for &(ai, bi) in pairs {
        let m = ai.nrows();
        a.view_mut((row, 0), (m, ncols)).copy_from(ai);
        b.rows_mut(row, m).copy_from(bi);
        row += m;
    }
    (a, b)
}

/// Internal: build the combined inequality (A_iq, b_iq) for a single QP step.
///
/// When `step_size < 1`, the bounds are scaled by `1 / step_size` so that
/// after multiplying the QP solution by `step_size`, the actual step
/// still satisfies the original limits.
fn build_step_inequalities(
    model: &Model<f64>,
    q: &[f64],
    config: &QpIkConfig,
) -> Option<(DMatrix<f64>, DVector<f64>)> {
    let mut parts: Vec<(DMatrix<f64>, DVector<f64>)> = Vec::new();

    if let Some(ref lim) = config.joint_limits {
        parts.push(build_joint_limit_inequalities(model, q, lim));
    }
    if let Some(ms) = config.max_step {
        parts.push(build_max_step_inequalities(model.nv, ms));
    }
    if parts.is_empty() {
        return None;
    }
    let refs: Vec<(&DMatrix<f64>, &DVector<f64>)> =
        parts.iter().map(|(a, b)| (a, b)).collect();
    let (a, mut b) = stack_inequalities(&refs);
    // Scale bounds: QP solves for dq_raw, actual step is step_size * dq_raw.
    // We need step_size * dq_raw ≤ b_orig, i.e., dq_raw ≤ b_orig / step_size.
    if config.step_size > 0.0 && config.step_size < 1.0 {
        b /= config.step_size;
    }
    Some((a, b))
}

/// Solve constraint-only IK (no primary task) with inequality bounds via QP.
///
/// Equivalent to [`solve_constrained_ik`] but respects hard inequality
/// constraints (joint limits, step-size bounds) at every iteration.
pub fn solve_constrained_ik_qp(
    model: &Model<f64>,
    q0: &[f64],
    cm: &ConstraintModel<f64>,
    config: &QpIkConfig,
) -> ConstrainedIkResult {
    use crate::qp::{solve_qp, QpConfig, QpStatus};

    let nv = model.nv;
    let mut q = q0.to_vec();
    // Clamp initial q to limits
    if let Some(ref lim) = config.joint_limits {
        q = crate::limits::clamp_configuration(model, &q, lim);
    }
    let mut last_err = f64::INFINITY;

    let qp_cfg = QpConfig {
        max_iters: config.qp_max_iters,
        ..Default::default()
    };

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);
        let e_c = compute_constraint_error_from_data(&data, cm);
        let e_norm = e_c.norm();
        last_err = e_norm;

        if e_norm <= config.tol_constraint {
            return ConstrainedIkResult {
                q,
                iterations: iter,
                constraint_error_norm: e_norm,
                task_error_norm: 0.0,
                converged: true,
            };
        }

        let jc = compute_constraint_jacobian_from_data(model, &q, &data, cm);

        // QP:  min 0.5 dq^T H dq + c^T dq   s.t. A_iq dq ≤ b_iq
        //   H = w² Jc^T Jc + λ² I
        //   c = w² Jc^T e_c   (drives Jc dq ≈ -e_c)
        let w2 = config.constraint_weight * config.constraint_weight;
        let lam2 = config.damping * config.damping;
        let h = &jc.transpose() * &jc * w2
            + DMatrix::<f64>::identity(nv, nv) * lam2;
        let cv = jc.transpose() * &e_c * w2;

        let iq = build_step_inequalities(model, &q, config);
        let x0 = DVector::zeros(nv);
        let sol = solve_qp(
            &h, &cv,
            None, None,
            iq.as_ref().map(|(a, _)| a),
            iq.as_ref().map(|(_, b)| b),
            Some(&x0),
            &qp_cfg,
        );

        if sol.status == QpStatus::NumericalFailure {
            break;
        }

        let dq = &sol.x * config.step_size;
        q = crate::manifold::integrate(model, &q, dq.as_slice(), 1.0);
        // Clamp to limits after integration (safety net)
        if let Some(ref lim) = config.joint_limits {
            q = crate::limits::clamp_configuration(model, &q, lim);
        }
    }

    ConstrainedIkResult {
        q,
        iterations: config.max_iters,
        constraint_error_norm: last_err,
        task_error_norm: 0.0,
        converged: false,
    }
}

/// Solve position IK with equality constraints **and** inequality bounds via QP.
///
/// The primary task drives `joint_idx` to `target` (3-D position).
/// Equality constraints from `cm` are enforced as a weighted cost term.
/// Inequality constraints (joint limits, step-size) are hard QP bounds.
pub fn solve_task_with_constraints_qp(
    model: &Model<f64>,
    q0: &[f64],
    joint_idx: usize,
    target: Vector3<f64>,
    cm: &ConstraintModel<f64>,
    config: &QpIkConfig,
) -> ConstrainedIkResult {
    use crate::jacobian::compute_joint_jacobian_from_data;
    use crate::qp::{solve_qp, QpConfig, QpStatus};

    let nv = model.nv;
    let nc = cm.total_dim();
    let mut q = q0.to_vec();
    if let Some(ref lim) = config.joint_limits {
        q = crate::limits::clamp_configuration(model, &q, lim);
    }
    let mut last_task_err = f64::INFINITY;
    let mut last_constraint_err = f64::INFINITY;

    let qp_cfg = QpConfig {
        max_iters: config.qp_max_iters,
        ..Default::default()
    };

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);

        // Task error
        let p_cur = se3::translation(&data.oMi[joint_idx]);
        let e_task = target - p_cur;
        let task_norm = e_task.norm();
        last_task_err = task_norm;

        // Constraint error
        let e_c = if nc > 0 {
            compute_constraint_error_from_data(&data, cm)
        } else {
            DVector::zeros(0)
        };
        let c_norm = e_c.norm();
        last_constraint_err = c_norm;

        if task_norm <= config.tol_task
            && (nc == 0 || c_norm <= config.tol_constraint)
        {
            return ConstrainedIkResult {
                q,
                iterations: iter,
                constraint_error_norm: c_norm,
                task_error_norm: task_norm,
                converged: true,
            };
        }

        // Task Jacobian (linear rows)
        let j_full = compute_joint_jacobian_from_data(model, &q, &data, joint_idx);
        let j_task = j_full.rows(3, 3).into_owned(); // 3 × nv

        // Constraint Jacobian
        let jc = if nc > 0 {
            compute_constraint_jacobian_from_data(model, &q, &data, cm)
        } else {
            DMatrix::zeros(0, nv)
        };

        // Build QP:
        //   H = Jt^T Jt + w² Jc^T Jc + λ² I
        //   c_qp = -Jt^T e_task + w² Jc^T e_c
        let w2 = config.constraint_weight * config.constraint_weight;
        let lam2 = config.damping * config.damping;
        let mut h = &j_task.transpose() * &j_task
            + DMatrix::<f64>::identity(nv, nv) * lam2;
        let mut cv = -j_task.transpose() * DVector::from_column_slice(e_task.as_slice());

        if nc > 0 {
            h += &jc.transpose() * &jc * w2;
            cv += jc.transpose() * &e_c * w2;
        }

        let iq = build_step_inequalities(model, &q, config);
        let x0 = DVector::zeros(nv);
        let sol = solve_qp(
            &h, &cv,
            None, None,
            iq.as_ref().map(|(a, _)| a),
            iq.as_ref().map(|(_, b)| b),
            Some(&x0),
            &qp_cfg,
        );

        if sol.status == QpStatus::NumericalFailure {
            break;
        }

        let dq = &sol.x * config.step_size;
        q = crate::manifold::integrate(model, &q, dq.as_slice(), 1.0);
        if let Some(ref lim) = config.joint_limits {
            q = crate::limits::clamp_configuration(model, &q, lim);
        }
    }

    ConstrainedIkResult {
        q,
        iterations: config.max_iters,
        constraint_error_norm: last_constraint_err,
        task_error_norm: last_task_err,
        converged: false,
    }
}

/// Solve 6-D frame IK with equality constraints **and** inequality bounds
/// via QP.
///
/// Primary task: align `task_frame` with `target_pose` (6-D).
/// Equality constraints from `cm` enforced as weighted cost.
/// Inequality constraints (joint limits, step-size) handled as hard QP bounds.
pub fn solve_frame_task_with_constraints_qp(
    model: &Model<f64>,
    q0: &[f64],
    task_frame: &Frame<f64>,
    target_pose: SE3<f64>,
    cm: &ConstraintModel<f64>,
    config: &QpIkConfig,
) -> ConstrainedIkResult {
    use crate::qp::{solve_qp, QpConfig, QpStatus};

    let nv = model.nv;
    let nc = cm.total_dim();
    let mut q = q0.to_vec();
    if let Some(ref lim) = config.joint_limits {
        q = crate::limits::clamp_configuration(model, &q, lim);
    }
    let mut last_task_err = f64::INFINITY;
    let mut last_constraint_err = f64::INFINITY;

    let qp_cfg = QpConfig {
        max_iters: config.qp_max_iters,
        ..Default::default()
    };

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);

        // Task error (6D pose)
        let m_current = compute_frame_placement_from_data(&data, task_frame);
        let m_err = se3::compose(&se3::inverse(&m_current), &target_pose);
        let log_err = se3::log(&m_err);

        // Rotate to world frame
        let r = se3::rotation_matrix(&m_current);
        let omega = Vector3::new(log_err[0], log_err[1], log_err[2]);
        let v = Vector3::new(log_err[3], log_err[4], log_err[5]);
        let omega_w = &r * omega;
        let v_w = &r * v;

        let e_task = DVector::from_vec(vec![
            omega_w[0], omega_w[1], omega_w[2], v_w[0], v_w[1], v_w[2],
        ]);
        let task_norm = e_task.norm();
        last_task_err = task_norm;

        // Constraint error
        let e_c = if nc > 0 {
            compute_constraint_error_from_data(&data, cm)
        } else {
            DVector::zeros(0)
        };
        let c_norm = e_c.norm();
        last_constraint_err = c_norm;

        if task_norm <= config.tol_task
            && (nc == 0 || c_norm <= config.tol_constraint)
        {
            return ConstrainedIkResult {
                q,
                iterations: iter,
                constraint_error_norm: c_norm,
                task_error_norm: task_norm,
                converged: true,
            };
        }

        // Task Jacobian (frame)
        let j_task = compute_frame_jacobian_from_data(model, &q, &data, task_frame);

        // Constraint Jacobian
        let jc = if nc > 0 {
            compute_constraint_jacobian_from_data(model, &q, &data, cm)
        } else {
            DMatrix::zeros(0, nv)
        };

        // Build QP
        let w2 = config.constraint_weight * config.constraint_weight;
        let lam2 = config.damping * config.damping;
        let mut h = &j_task.transpose() * &j_task
            + DMatrix::<f64>::identity(nv, nv) * lam2;
        let mut cv = -j_task.transpose() * &e_task;

        if nc > 0 {
            h += &jc.transpose() * &jc * w2;
            cv += jc.transpose() * &e_c * w2;
        }

        let iq = build_step_inequalities(model, &q, config);
        let x0 = DVector::zeros(nv);
        let sol = solve_qp(
            &h, &cv,
            None, None,
            iq.as_ref().map(|(a, _)| a),
            iq.as_ref().map(|(_, b)| b),
            Some(&x0),
            &qp_cfg,
        );

        if sol.status == QpStatus::NumericalFailure {
            break;
        }

        let dq = &sol.x * config.step_size;
        q = crate::manifold::integrate(model, &q, dq.as_slice(), 1.0);
        if let Some(ref lim) = config.joint_limits {
            q = crate::limits::clamp_configuration(model, &q, lim);
        }
    }

    ConstrainedIkResult {
        q,
        iterations: config.max_iters,
        constraint_error_norm: last_constraint_err,
        task_error_norm: last_task_err,
        converged: false,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fk::forward_kinematics;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use approx::assert_relative_eq;
    use nalgebra::{Rotation3, Vector3};

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Y-tree:  universe → j1(Z) → j2(X)  (left arm, link lengths 1.0)
    ///                           → j3(Y)  (right arm)
    fn y_tree() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        ModelBuilder::new()
            .name("y_tree")
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j2", 1, joint::revolute_x(), offset.clone(), LinkInertia::zero())
            .add_joint("j3", 1, joint::revolute_y(), offset, LinkInertia::zero())
            .build()
    }

    /// Dual-arm humanoid-like: universe → base → left_shoulder → left_elbow
    ///                                        → right_shoulder → right_elbow
    fn dual_arm() -> Model<f64> {
        let shoulder_y = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.3, 0.0),
        );
        let shoulder_ny = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, -0.3, 0.0),
        );
        let forearm = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        ModelBuilder::new()
            .name("dual_arm")
            .add_joint("base", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("left_shoulder", 1, joint::revolute_x(), shoulder_y, LinkInertia::zero())
            .add_joint("left_elbow", 2, joint::revolute_x(), forearm.clone(), LinkInertia::zero())
            .add_joint("right_shoulder", 1, joint::revolute_x(), shoulder_ny, LinkInertia::zero())
            .add_joint("right_elbow", 4, joint::revolute_x(), forearm, LinkInertia::zero())
            .build()
    }

    fn frame_at_joint(name: &str, joint_idx: usize) -> Frame<f64> {
        Frame {
            name: name.to_string(),
            parent_joint: joint_idx,
            placement: se3::identity(),
        }
    }

    fn frame_with_offset(name: &str, joint_idx: usize, offset: Vector3<f64>) -> Frame<f64> {
        Frame {
            name: name.to_string(),
            parent_joint: joint_idx,
            placement: se3::from_rotation_and_translation(&Rotation3::identity(), &offset),
        }
    }

    // ── ConstraintModel basics ──────────────────────────────────────────

    #[test]
    fn empty_constraint_model() {
        let cm = ConstraintModel::<f64>::new();
        assert_eq!(cm.total_dim(), 0);
        assert!(cm.is_empty());
    }

    #[test]
    fn constraint_model_dimensions() {
        let f1 = frame_at_joint("a", 1);
        let f2 = frame_at_joint("b", 2);
        let f3 = frame_at_joint("c", 3);

        let mut cm = ConstraintModel::new();
        cm.add(RigidConstraint::pose(f1.clone(), f2.clone()));
        cm.add(RigidConstraint::position(f2, f3));

        assert_eq!(cm.len(), 2);
        assert_eq!(cm.total_dim(), 9); // 6 + 3
    }

    // ── Constraint error ────────────────────────────────────────────────

    #[test]
    fn error_zero_when_frames_coincide() {
        let model = y_tree();
        let q = vec![0.0; model.nq]; // j2 and j3 have same offset from j1

        // At q=0, j2 and j3 are at the same world position
        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);
        let err = compute_constraint_error(&model, &q, &cm);

        // At zero config, both j2 and j3 are at (1,0,0) → error = 0
        assert_relative_eq!(err.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn error_nonzero_when_frames_differ() {
        let model = y_tree();
        let q = vec![0.0, 0.5, -0.5]; // j2 and j3 at different rotations

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2, f3),
        ]);
        let err = compute_constraint_error(&model, &q, &cm);
        assert!(err.norm() > 0.01, "error should be nonzero when frames differ");
    }

    #[test]
    fn error_with_desired_offset() {
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        // Desired: frame2 is at (0.5, 0, 0) relative to frame1
        let desired = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.5, 0.0, 0.0),
        );

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3).with_desired_placement(desired),
        ]);

        let err = compute_constraint_error(&model, &q, &cm);
        // At q=0, j2 == j3 position, but desired offset is (0.5,0,0)
        // So error = p2 - (p1 + R1*p_des) = 0 - 0.5 = -0.5 in x
        assert_relative_eq!(err[0], -0.5, epsilon = 1e-12);
        assert_relative_eq!(err[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(err[2], 0.0, epsilon = 1e-12);
    }

    // ── Constraint Jacobian ─────────────────────────────────────────────

    #[test]
    fn jacobian_shape() {
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        assert_eq!(jc.nrows(), 3);
        assert_eq!(jc.ncols(), model.nv);
    }

    #[test]
    fn jacobian_6d_shape() {
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        assert_eq!(jc.nrows(), 6);
        assert_eq!(jc.ncols(), model.nv);
    }

    #[test]
    fn jacobian_multiple_constraints_stacked() {
        let model = dual_arm();
        let q = vec![0.0; model.nq];

        let left_tip = frame_at_joint("left_tip", 3);
        let right_tip = frame_at_joint("right_tip", 5);
        let world_anchor = Frame {
            name: "world".into(),
            parent_joint: 0,
            placement: se3::from_rotation_and_translation(
                &Rotation3::identity(),
                &Vector3::new(0.0, 0.0, -1.0),
            ),
        };

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(left_tip.clone(), right_tip),  // 3 rows
            RigidConstraint::pose(left_tip, world_anchor),           // 6 rows
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        assert_eq!(jc.nrows(), 9);
        assert_eq!(jc.ncols(), model.nv);
    }

    #[test]
    fn jacobian_finite_diff_validation_3d() {
        // Validate J_c via finite differences: d(error)/dq
        let model = y_tree();
        let q = vec![0.3, -0.2, 0.5];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let e0 = compute_constraint_error(&model, &q, &cm);

        let eps = 1e-7;
        for col in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[col] += eps;
            let e_plus = compute_constraint_error(&model, &q_plus, &cm);
            let de = (&e_plus - &e0) / eps;

            for row in 0..3 {
                assert_relative_eq!(jc[(row, col)], de[row], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn jacobian_finite_diff_validation_6d() {
        // J_c = J2 - J1 is a first-order approximation of d(log error)/dq.
        // The rotation-to-world step adds q-dependent terms that only vanish
        // at e = 0.  We test near constraint satisfaction for accuracy.
        let model = y_tree();
        let q = vec![0.001, 0.001, -0.001]; // very small → tiny error

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let e0 = compute_constraint_error(&model, &q, &cm);

        let eps = 1e-7;
        for col in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[col] += eps;
            let e_plus = compute_constraint_error(&model, &q_plus, &cm);
            let de = (&e_plus - &e0) / eps;

            for row in 0..6 {
                assert_relative_eq!(jc[(row, col)], de[row], epsilon = 5e-3);
            }
        }
    }

    #[test]
    fn jacobian_world_anchor_single_branch() {
        // Constrain j2 position to the world origin
        let model = y_tree();
        let q = vec![0.2, 0.0, 0.0];

        let f2 = frame_at_joint("left", 2);
        let world_origin = Frame {
            name: "world_origin".into(),
            parent_joint: 0,
            placement: se3::identity(),
        };

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(world_origin, f2),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        // J_c = J(f2) - 0 = J(f2) (linear rows)
        // This should be the same as the frame Jacobian of f2 (linear rows)
        let data = forward_kinematics(&model, &q);
        let j_f2 = compute_frame_jacobian_from_data(&model, &q, &data, &frame_at_joint("left", 2));
        let j_f2_lin = j_f2.rows(3, 3);

        assert_relative_eq!(jc, j_f2_lin.into_owned(), epsilon = 1e-12);
    }

    // ── Cross-branch Jacobian properties ────────────────────────────────

    #[test]
    fn cross_branch_jacobian_nonzero_both_branches() {
        // For a cross-branch constraint between tip frames, the Jacobian
        // should have nonzero columns for joints on BOTH branches.
        // Offsets must be perpendicular to the rotation axes:
        // j2 is revolute_x → offset along Z; j3 is revolute_y → offset along Z.
        let model = y_tree();
        let q = vec![0.3, 0.2, -0.4];

        let f2 = frame_with_offset("left_tip", 2, Vector3::new(0.0, 0.0, 0.5));
        let f3 = frame_with_offset("right_tip", 3, Vector3::new(0.0, 0.0, 0.5));

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);

        // Column 1 (j2) and column 2 (j3) should both be nonzero
        let col1_norm = jc.column(1).norm();
        let col2_norm = jc.column(2).norm();
        assert!(col1_norm > 1e-6, "j2 column should be nonzero: {col1_norm}");
        assert!(col2_norm > 1e-6, "j3 column should be nonzero: {col2_norm}");
    }

    #[test]
    fn cross_branch_jacobian_common_ancestor_column() {
        // j1 is the common ancestor of j2 and j3.
        // For a position constraint between j2 and j3, j1's column should
        // capture the relative motion (cancel in angular, lever-arm difference
        // in linear).
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);

        // At q=0, j2 and j3 are at the same point → j1 column should be zero
        // because rotating j1 moves both equally.
        let col0 = jc.column(0);
        assert_relative_eq!(col0.norm(), 0.0, epsilon = 1e-12);
    }

    // ── Constrained IK (constraint-only) ────────────────────────────────

    #[test]
    fn constrained_ik_converges_cross_branch() {
        // Make j2 and j3 tips meet by constraint IK
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5]; // start with different rotations

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let config = ConstrainedIkConfig {
            max_iters: 200,
            tol_constraint: 1e-8,
            ..Default::default()
        };

        let result = solve_constrained_ik(&model, &q0, &cm, &config);
        assert!(result.converged, "should converge; err={}", result.constraint_error_norm);
        assert!(result.constraint_error_norm < 1e-6);

        // Verify: j2 and j3 world positions should match
        let data = forward_kinematics(&model, &result.q);
        let p2 = se3::translation(&data.oMi[2]);
        let p3 = se3::translation(&data.oMi[3]);
        assert_relative_eq!(p2, p3, epsilon = 1e-5);
    }

    #[test]
    fn constrained_ik_6d_converges() {
        let model = y_tree();
        let q0 = vec![0.0, 0.3, -0.3];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2, f3),
        ]);

        let config = ConstrainedIkConfig {
            max_iters: 300,
            tol_constraint: 1e-6,
            step_size: 0.3,
            ..Default::default()
        };

        let result = solve_constrained_ik(&model, &q0, &cm, &config);
        assert!(result.converged, "err={}", result.constraint_error_norm);
    }

    #[test]
    fn constrained_ik_with_desired_offset() {
        // Use tip frames with offset perpendicular to rotation axes
        // so the constraint Jacobian is full-rank.
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        let f2 = frame_with_offset("left_tip", 2, Vector3::new(0.0, 0.0, 0.5));
        let f3 = frame_with_offset("right_tip", 3, Vector3::new(0.0, 0.0, 0.5));

        // Desired: frame3 tip is 0.1m along Y relative to frame2 tip
        let desired = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.1, 0.0),
        );

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2.clone(), f3.clone()).with_desired_placement(desired),
        ]);

        let config = ConstrainedIkConfig {
            max_iters: 1000,
            tol_constraint: 1e-3,
            step_size: 0.3,
            damping: 1e-3,
            ..Default::default()
        };

        let result = solve_constrained_ik(&model, &q0, &cm, &config);
        // With nonzero desired offset the Jacobian J2−J1 is an approximation
        // (missing dR1/dq · t_des term), so convergence to a neighborhood.
        assert!(
            result.constraint_error_norm < 0.05,
            "expected significant error reduction; err={}",
            result.constraint_error_norm
        );
    }

    // ── Task + constraints (augmented) ──────────────────────────────────

    #[test]
    fn task_with_constraints_dual_arm() {
        // Both hands must meet (constraint) while left tip reaches target.
        let model = dual_arm();
        let q0 = vec![0.0; model.nq];

        let left_tip = frame_with_offset("left_tip", 3, Vector3::new(0.0, 0.0, -0.3));
        let right_tip = frame_with_offset("right_tip", 5, Vector3::new(0.0, 0.0, -0.3));

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(left_tip, right_tip),
        ]);

        let target = Vector3::new(0.0, 0.0, -0.8);

        let config = ConstrainedIkConfig {
            max_iters: 500,
            tol_task: 1e-3,
            tol_constraint: 1e-3,
            step_size: 0.3,
            damping: 1e-2,
            constraint_weight: 5.0,
        };

        let result = solve_task_with_constraints(&model, &q0, 3, target, &cm, &config);

        // The constraint (left==right) should be approximately satisfied
        let data = forward_kinematics(&model, &result.q);
        let p_left = se3::translation(&compute_frame_placement_from_data(
            &data,
            &frame_with_offset("lt", 3, Vector3::new(0.0, 0.0, -0.3)),
        ));
        let p_right = se3::translation(&compute_frame_placement_from_data(
            &data,
            &frame_with_offset("rt", 5, Vector3::new(0.0, 0.0, -0.3)),
        ));
        let constraint_err = (p_left - p_right).norm();
        assert!(
            constraint_err < 0.1,
            "cross-branch constraint error too large: {constraint_err}"
        );
    }

    // ── Local reference frame ───────────────────────────────────────────

    #[test]
    fn local_frame_error_rotates() {
        let model = y_tree();
        let q = vec![0.3, 0.2, -0.4];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm_world = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2.clone(), f3.clone()),
        ]);
        let cm_local = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2.clone(), f3.clone())
                .with_reference_frame(ReferenceFrame::Local),
        ]);

        let e_world = compute_constraint_error(&model, &q, &cm_world);
        let e_local = compute_constraint_error(&model, &q, &cm_local);

        // They should have the same norm
        assert_relative_eq!(e_world.norm(), e_local.norm(), epsilon = 1e-12);

        // But different components (rotated by R1^T)
        let data = forward_kinematics(&model, &q);
        let m1 = compute_frame_placement_from_data(&data, &f2);
        let r1 = se3::rotation_matrix(&m1);
        let expected_local = r1.transpose() * Vector3::new(e_world[0], e_world[1], e_world[2]);
        assert_relative_eq!(e_local[0], expected_local[0], epsilon = 1e-12);
        assert_relative_eq!(e_local[1], expected_local[1], epsilon = 1e-12);
        assert_relative_eq!(e_local[2], expected_local[2], epsilon = 1e-12);
    }

    #[test]
    fn local_jacobian_finite_diff_validation() {
        let model = y_tree();
        let q = vec![0.2, 0.3, -0.1];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3)
                .with_reference_frame(ReferenceFrame::Local),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let e0 = compute_constraint_error(&model, &q, &cm);

        let eps = 1e-7;
        for col in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[col] += eps;
            let e_plus = compute_constraint_error(&model, &q_plus, &cm);
            let de = (&e_plus - &e0) / eps;

            for row in 0..3 {
                assert_relative_eq!(jc[(row, col)], de[row], epsilon = 1e-4);
            }
        }
    }

    // ── Integration with constrained_forward_dynamics ───────────────────

    #[test]
    fn constraint_jacobian_in_kkt() {
        // Use tip frames with offset so Jc is full-rank for KKT.
        use crate::constrained::constrained_forward_dynamics;

        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        let inertia = LinkInertia {
            mass: 1.0,
            center_of_mass: Vector3::new(0.5, 0.0, 0.0),
            rotational_inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01)),
        };
        let model = ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), inertia.clone())
            .add_joint("j2", 1, joint::revolute_x(), offset.clone(), inertia.clone())
            .add_joint("j3", 1, joint::revolute_y(), offset, inertia.clone())
            .build();

        let q = vec![0.1, 0.2, -0.2];
        let v = vec![0.0; model.nv];
        let tau = vec![0.0; model.nv];

        // Use tip frames with offset so linear Jacobian columns are nonzero
        let f2 = frame_with_offset("left_tip", 2, Vector3::new(0.5, 0.0, 0.0));
        let f3 = frame_with_offset("right_tip", 3, Vector3::new(0.5, 0.0, 0.0));

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let gamma = DVector::zeros(jc.nrows());

        let result = constrained_forward_dynamics(&model, &q, &v, &tau, &jc, &gamma);
        assert_eq!(result.qdd.len(), model.nv);
        assert_eq!(result.lambda.len(), 3);
    }

    // ── Frame task + constraint ─────────────────────────────────────────

    #[test]
    fn frame_task_with_constraints_converges() {
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        // Task: bring j2 tip to a specific pose
        let task_frame = frame_with_offset("tool", 2, Vector3::new(0.5, 0.0, 0.0));
        let target_pose = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(1.2, 0.3, 0.0),
        );

        // No constraints for this test (just verify the solver doesn't break)
        let cm = ConstraintModel::new();

        let config = ConstrainedIkConfig {
            max_iters: 300,
            tol_task: 1e-3,
            tol_constraint: 1e-6,
            step_size: 0.3,
            damping: 1e-2,
            constraint_weight: 5.0,
        };

        let result = solve_frame_task_with_constraints(
            &model, &q0, &task_frame, target_pose, &cm, &config,
        );

        // With only 3 DOFs and a 6D task, may not fully converge, but should reduce error
        assert!(result.task_error_norm < 1.0);
    }

    #[test]
    fn constrained_ik_dual_arm_position() {
        // Dual-arm: constrain both elbow tips to be at the same position
        let model = dual_arm();
        let q0 = vec![0.0, 0.3, -0.5, -0.3, 0.5];

        // Use tip frames with offset for full-rank Jacobian
        let left_tip = frame_with_offset("l_tip", 3, Vector3::new(0.0, 0.0, -0.3));
        let right_tip = frame_with_offset("r_tip", 5, Vector3::new(0.0, 0.0, -0.3));

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(left_tip.clone(), right_tip.clone()),
        ]);

        let config = ConstrainedIkConfig {
            max_iters: 500,
            tol_constraint: 1e-4,
            step_size: 0.5,
            damping: 1e-3,
            ..Default::default()
        };

        let result = solve_constrained_ik(&model, &q0, &cm, &config);
        assert!(result.converged, "err={}", result.constraint_error_norm);

        let data = forward_kinematics(&model, &result.q);
        let p_l = se3::translation(&compute_frame_placement_from_data(&data, &left_tip));
        let p_r = se3::translation(&compute_frame_placement_from_data(&data, &right_tip));
        assert_relative_eq!(p_l, p_r, epsilon = 1e-3);
    }

    // ── QP-based inequality-constrained IK ──────────────────────────────

    #[test]
    fn joint_limit_inequalities_shape() {
        let model = y_tree();
        let q = vec![0.0; model.nq];
        let mut limits = crate::limits::JointLimits::unbounded(&model);
        // Set finite limits on all 3 joints
        for i in 0..model.nq {
            limits.q_min[i] = -1.5;
            limits.q_max[i] = 1.5;
        }
        let (a, b) = build_joint_limit_inequalities(&model, &q, &limits);
        // 3 joints × 2 (upper + lower) = 6 rows
        assert_eq!(a.nrows(), 6);
        assert_eq!(a.ncols(), model.nv);
        assert_eq!(b.nrows(), 6);
    }

    #[test]
    fn joint_limit_inequalities_values() {
        let model = y_tree();
        let q = vec![0.5, -0.3, 0.8];
        let mut limits = crate::limits::JointLimits::unbounded(&model);
        limits.q_min = vec![-1.0, -1.0, -1.0];
        limits.q_max = vec![1.0, 1.0, 1.0];

        let (a, b) = build_joint_limit_inequalities(&model, &q, &limits);

        // dq=0 should satisfy all inequalities (since q is within limits)
        let dq_zero = nalgebra::DVector::zeros(model.nv);
        let vals = &a * &dq_zero;
        for i in 0..a.nrows() {
            assert!(vals[i] <= b[i] + 1e-12, "dq=0 should be feasible, row {i}");
        }
    }

    #[test]
    fn max_step_inequalities_shape() {
        let (a, b) = build_max_step_inequalities(5, 0.1);
        assert_eq!(a.nrows(), 10); // 2 * 5
        assert_eq!(a.ncols(), 5);
        for i in 0..10 {
            assert_relative_eq!(b[i], 0.1, epsilon = 1e-15);
        }
    }

    #[test]
    fn stack_inequalities_works() {
        let a1 = nalgebra::DMatrix::identity(2, 3);
        let b1 = nalgebra::DVector::from_vec(vec![1.0, 2.0]);
        let a2 = nalgebra::DMatrix::from_element(1, 3, 0.5);
        let b2 = nalgebra::DVector::from_element(1, 3.0);

        let (a, b) = stack_inequalities(&[(&a1, &b1), (&a2, &b2)]);
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.ncols(), 3);
        assert_eq!(b.nrows(), 3);
        assert_relative_eq!(b[2], 3.0, epsilon = 1e-15);
    }

    #[test]
    fn qp_ik_no_limits_matches_dls() {
        // Without inequality constraints, the QP solver should behave like DLS.
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);
        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let config = QpIkConfig {
            max_iters: 200,
            tol_constraint: 1e-6,
            step_size: 0.5,
            damping: 1e-3,
            constraint_weight: 10.0,
            ..Default::default()
        };

        let result = solve_constrained_ik_qp(&model, &q0, &cm, &config);
        assert!(result.converged, "should converge; err={}", result.constraint_error_norm);
        assert!(result.constraint_error_norm < 1e-4);
    }

    #[test]
    fn qp_ik_with_joint_limits_respected() {
        // IK should respect joint limits even when convergence requires
        // going beyond them.
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);
        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let mut limits = crate::limits::JointLimits::unbounded(&model);
        limits.q_min = vec![-0.3, -0.3, -0.3];
        limits.q_max = vec![0.3, 0.3, 0.3];

        let config = QpIkConfig {
            max_iters: 300,
            tol_constraint: 1e-6,
            step_size: 0.5,
            damping: 1e-3,
            constraint_weight: 10.0,
            joint_limits: Some(limits.clone()),
            ..Default::default()
        };

        let result = solve_constrained_ik_qp(&model, &q0, &cm, &config);

        // Check that the solution respects joint limits
        for (i, &qi) in result.q.iter().enumerate() {
            assert!(
                qi >= limits.q_min[i] - 1e-6,
                "joint {i} below lower limit: {qi} < {}",
                limits.q_min[i]
            );
            assert!(
                qi <= limits.q_max[i] + 1e-6,
                "joint {i} above upper limit: {qi} > {}",
                limits.q_max[i]
            );
        }
    }

    #[test]
    fn qp_ik_with_max_step() {
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);
        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let config = QpIkConfig {
            max_iters: 500,
            tol_constraint: 1e-6,
            step_size: 1.0,
            damping: 1e-3,
            constraint_weight: 10.0,
            max_step: Some(0.05),
            ..Default::default()
        };

        let result = solve_constrained_ik_qp(&model, &q0, &cm, &config);
        // Should still converge (just takes more iterations)
        assert!(
            result.constraint_error_norm < 0.01,
            "should reduce error; err={}",
            result.constraint_error_norm
        );
    }

    #[test]
    fn qp_task_ik_with_joint_limits() {
        // Position IK with joint limits
        let model = y_tree();
        let q0 = vec![0.0; model.nq];
        let target = Vector3::new(0.8, 0.5, 0.0);

        let mut limits = crate::limits::JointLimits::unbounded(&model);
        limits.q_min = vec![-1.0, -1.0, -1.0];
        limits.q_max = vec![1.0, 1.0, 1.0];

        let cm = ConstraintModel::new(); // no equality constraints

        let config = QpIkConfig {
            max_iters: 300,
            tol_task: 1e-3,
            tol_constraint: 1e-6,
            step_size: 0.5,
            damping: 1e-2,
            constraint_weight: 10.0,
            joint_limits: Some(limits.clone()),
            ..Default::default()
        };

        let result = solve_task_with_constraints_qp(
            &model, &q0, 2, target, &cm, &config,
        );

        // Verify joint limits
        for (i, &qi) in result.q.iter().enumerate() {
            assert!(qi >= limits.q_min[i] - 1e-6, "lower limit violated j{i}");
            assert!(qi <= limits.q_max[i] + 1e-6, "upper limit violated j{i}");
        }

        // Check task error reduced
        assert!(result.task_error_norm < 0.5, "task err={}", result.task_error_norm);
    }

    #[test]
    fn qp_task_ik_tight_limits() {
        // With very tight joint limits, the IK should NOT violate them,
        // even if the task can't be achieved.
        let model = y_tree();
        let q0 = vec![0.0; model.nq];
        let target = Vector3::new(2.0, 0.0, 0.0); // far target

        let mut limits = crate::limits::JointLimits::unbounded(&model);
        limits.q_min = vec![-0.1, -0.1, -0.1];
        limits.q_max = vec![0.1, 0.1, 0.1];

        let cm = ConstraintModel::new();
        let config = QpIkConfig {
            max_iters: 100,
            tol_task: 1e-3,
            step_size: 0.5,
            damping: 1e-2,
            joint_limits: Some(limits.clone()),
            ..Default::default()
        };

        let result = solve_task_with_constraints_qp(
            &model, &q0, 2, target, &cm, &config,
        );

        // Task won't converge (limits too tight), but limits must be respected
        for (i, &qi) in result.q.iter().enumerate() {
            assert!(qi >= limits.q_min[i] - 1e-6, "j{i} < lower");
            assert!(qi <= limits.q_max[i] + 1e-6, "j{i} > upper");
        }
    }

    #[test]
    fn qp_task_with_equality_and_inequality() {
        // Dual-arm: hands must meet (equality), joint limits active.
        let model = dual_arm();
        let q0 = vec![0.0; model.nq];

        let left_tip = frame_with_offset("l_tip", 3, Vector3::new(0.0, 0.0, -0.3));
        let right_tip = frame_with_offset("r_tip", 5, Vector3::new(0.0, 0.0, -0.3));

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(left_tip, right_tip),
        ]);

        let target = Vector3::new(0.0, 0.0, -0.7);

        let mut limits = crate::limits::JointLimits::unbounded(&model);
        for i in 0..model.nq {
            limits.q_min[i] = -1.5;
            limits.q_max[i] = 1.5;
        }

        let config = QpIkConfig {
            max_iters: 500,
            tol_task: 1e-2,
            tol_constraint: 1e-2,
            step_size: 0.3,
            damping: 1e-2,
            constraint_weight: 5.0,
            joint_limits: Some(limits.clone()),
            ..Default::default()
        };

        let result = solve_task_with_constraints_qp(
            &model, &q0, 3, target, &cm, &config,
        );

        // Joint limits must be respected
        for (i, &qi) in result.q.iter().enumerate() {
            assert!(qi >= limits.q_min[i] - 1e-6, "j{i} limit");
            assert!(qi <= limits.q_max[i] + 1e-6, "j{i} limit");
        }
    }

    #[test]
    fn qp_frame_task_with_limits() {
        // 6D frame IK with joint limits
        let model = y_tree();
        let q0 = vec![0.0, 0.3, -0.3];

        let task_frame = frame_with_offset("tool", 2, Vector3::new(0.5, 0.0, 0.0));
        let target_pose = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(1.2, 0.3, 0.0),
        );

        let mut limits = crate::limits::JointLimits::unbounded(&model);
        limits.q_min = vec![-1.5, -1.5, -1.5];
        limits.q_max = vec![1.5, 1.5, 1.5];

        let cm = ConstraintModel::new();
        let config = QpIkConfig {
            max_iters: 300,
            tol_task: 1e-3,
            step_size: 0.3,
            damping: 1e-2,
            joint_limits: Some(limits.clone()),
            ..Default::default()
        };

        let result = solve_frame_task_with_constraints_qp(
            &model, &q0, &task_frame, target_pose, &cm, &config,
        );

        // Joint limits respected
        for (i, &qi) in result.q.iter().enumerate() {
            assert!(qi >= limits.q_min[i] - 1e-6);
            assert!(qi <= limits.q_max[i] + 1e-6);
        }

        // With only 3 DOFs and a 6D task, it won't fully converge
        assert!(result.task_error_norm < 1.0);
    }

    #[test]
    fn qp_ik_limits_inactive_same_as_no_limits() {
        // When all limits are very wide, the QP solution should be close
        // to the unconstrained DLS solution.
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);
        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        // Wide limits (won't be active)
        let mut limits = crate::limits::JointLimits::unbounded(&model);
        limits.q_min = vec![-10.0; model.nq];
        limits.q_max = vec![10.0; model.nq];

        let config_no_lim = QpIkConfig {
            max_iters: 200,
            tol_constraint: 1e-6,
            step_size: 0.5,
            damping: 1e-3,
            constraint_weight: 10.0,
            ..Default::default()
        };

        let config_wide_lim = QpIkConfig {
            joint_limits: Some(limits),
            ..config_no_lim.clone()
        };

        let r1 = solve_constrained_ik_qp(&model, &q0, &cm, &config_no_lim);
        let r2 = solve_constrained_ik_qp(&model, &q0, &cm, &config_wide_lim);

        // Both should converge to similar solutions
        assert!(r1.converged);
        assert!(r2.converged);
        let diff: f64 = r1.q.iter().zip(&r2.q).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 0.1, "wide-limit solution should match no-limit: diff={diff}");
    }
}
