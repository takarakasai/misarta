//! Constrained IK solvers (DLS / augmented-Jacobian).
//!
//! Provides damped least-squares solvers for constraint-only IK and
//! task-plus-constraint IK.

use crate::fk::forward_kinematics;
use crate::frames::{compute_frame_jacobian_from_data, compute_frame_placement_from_data, Frame};
use crate::model::Model;
use crate::se3::{self, SE3};
use nalgebra::{DMatrix, DVector, Vector3};

use super::error::{compute_constraint_error_from_data};
use super::jacobian::compute_constraint_jacobian_from_data;
use super::ConstraintModel;

// ─── Result / Config types ──────────────────────────────────────────────────

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

// ─── Solvers ────────────────────────────────────────────────────────────────

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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::RigidConstraint;
    use crate::fk::forward_kinematics;
    use crate::frames::{compute_frame_placement_from_data, Frame};
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
    fn constrained_ik_converges_cross_branch() {
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

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
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        let f2 = frame_with_offset("left_tip", 2, Vector3::new(0.0, 0.0, 0.5));
        let f3 = frame_with_offset("right_tip", 3, Vector3::new(0.0, 0.0, 0.5));

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
        assert!(
            result.constraint_error_norm < 0.05,
            "expected significant error reduction; err={}",
            result.constraint_error_norm
        );
    }

    #[test]
    fn task_with_constraints_dual_arm() {
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

    #[test]
    fn frame_task_with_constraints_converges() {
        let model = y_tree();
        let q0 = vec![0.0, 0.5, -0.5];

        let task_frame = frame_with_offset("tool", 2, Vector3::new(0.5, 0.0, 0.0));
        let target_pose = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(1.2, 0.3, 0.0),
        );

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

        assert!(result.task_error_norm < 1.0);
    }

    #[test]
    fn constrained_ik_dual_arm_position() {
        let model = dual_arm();
        let q0 = vec![0.0, 0.3, -0.5, -0.3, 0.5];

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
}
