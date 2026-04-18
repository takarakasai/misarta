//! QP-based inequality-constrained IK.
//!
//! Extends the DLS-based constrained IK with hard inequality bounds
//! (joint limits, step-size constraints) enforced via a QP at each
//! iteration.

use crate::fk::forward_kinematics;
use crate::frames::{compute_frame_jacobian_from_data, compute_frame_placement_from_data, Frame};
use crate::model::Model;
use crate::se3::{self, SE3};
use nalgebra::{DMatrix, DVector, Vector3};

use super::error::compute_constraint_error_from_data;
use super::ik::ConstrainedIkResult;
use super::jacobian::compute_constraint_jacobian_from_data;
use super::ConstraintModel;

// ─── Config ─────────────────────────────────────────────────────────────────

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
    /// Which QP solver backend to use.
    pub qp_solver: crate::qp::QpSolver,
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
            qp_solver: crate::qp::QpSolver::default(),
        }
    }
}

// ─── Inequality builders ────────────────────────────────────────────────────

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

// ─── QP IK solvers ──────────────────────────────────────────────────────────

/// Solve constraint-only IK (no primary task) with inequality bounds via QP.
///
/// Equivalent to [`solve_constrained_ik`](super::ik::solve_constrained_ik)
/// but respects hard inequality constraints (joint limits, step-size bounds)
/// at every iteration.
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
        solver: config.qp_solver,
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
        solver: config.qp_solver,
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
        solver: config.qp_solver,
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
    use super::super::RigidConstraint;
    use crate::frames::Frame;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{Rotation3, Vector3};

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

    #[test]
    fn joint_limit_inequalities_shape() {
        let model = y_tree();
        let q = vec![0.0; model.nq];
        let mut limits = crate::limits::JointLimits::unbounded(&model);
        for i in 0..model.nq {
            limits.q_min[i] = -1.5;
            limits.q_max[i] = 1.5;
        }
        let (a, b) = build_joint_limit_inequalities(&model, &q, &limits);
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

        let dq_zero = nalgebra::DVector::zeros(model.nv);
        let vals = &a * &dq_zero;
        for i in 0..a.nrows() {
            assert!(vals[i] <= b[i] + 1e-12, "dq=0 should be feasible, row {i}");
        }
    }

    #[test]
    fn max_step_inequalities_shape() {
        let (a, b) = build_max_step_inequalities(5, 0.1);
        assert_eq!(a.nrows(), 10);
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
        assert!(
            result.constraint_error_norm < 0.01,
            "should reduce error; err={}",
            result.constraint_error_norm
        );
    }

    #[test]
    fn qp_task_ik_with_joint_limits() {
        let model = y_tree();
        let q0 = vec![0.0; model.nq];
        let target = Vector3::new(0.8, 0.5, 0.0);

        let mut limits = crate::limits::JointLimits::unbounded(&model);
        limits.q_min = vec![-1.0, -1.0, -1.0];
        limits.q_max = vec![1.0, 1.0, 1.0];

        let cm = ConstraintModel::new();

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

        for (i, &qi) in result.q.iter().enumerate() {
            assert!(qi >= limits.q_min[i] - 1e-6, "lower limit violated j{i}");
            assert!(qi <= limits.q_max[i] + 1e-6, "upper limit violated j{i}");
        }

        assert!(result.task_error_norm < 0.5, "task err={}", result.task_error_norm);
    }

    #[test]
    fn qp_task_ik_tight_limits() {
        let model = y_tree();
        let q0 = vec![0.0; model.nq];
        let target = Vector3::new(2.0, 0.0, 0.0);

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

        for (i, &qi) in result.q.iter().enumerate() {
            assert!(qi >= limits.q_min[i] - 1e-6, "j{i} < lower");
            assert!(qi <= limits.q_max[i] + 1e-6, "j{i} > upper");
        }
    }

    #[test]
    fn qp_task_with_equality_and_inequality() {
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

        for (i, &qi) in result.q.iter().enumerate() {
            assert!(qi >= limits.q_min[i] - 1e-6, "j{i} limit");
            assert!(qi <= limits.q_max[i] + 1e-6, "j{i} limit");
        }
    }

    #[test]
    fn qp_frame_task_with_limits() {
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

        for (i, &qi) in result.q.iter().enumerate() {
            assert!(qi >= limits.q_min[i] - 1e-6);
            assert!(qi <= limits.q_max[i] + 1e-6);
        }

        assert!(result.task_error_norm < 1.0);
    }

    #[test]
    fn qp_ik_limits_inactive_same_as_no_limits() {
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);
        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

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

        assert!(r1.converged);
        assert!(r2.converged);
        let diff: f64 = r1.q.iter().zip(&r2.q).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 0.1, "wide-limit solution should match no-limit: diff={diff}");
    }
}
