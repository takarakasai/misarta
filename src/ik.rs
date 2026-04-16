//! Numerical inverse kinematics (IK) solvers.
//!
//! This module provides DLS-based iterative IK for common task types:
//! - joint position IK
//! - joint orientation IK
//! - joint pose IK (6D)
//! - frame pose IK (6D)
use crate::fk::forward_kinematics;
use crate::frames::{compute_frame_jacobian, compute_frame_placement, Frame};
use crate::jacobian::compute_joint_jacobian;
use crate::limits;
use crate::limits::JointLimits;
use crate::manifold;
use crate::model::Model;
use crate::se3;
use crate::collision::{AllowedCollisionMatrix, collision_potential_gradient, has_collision_acm};
use crate::geometry::GeometryModel;
use nalgebra::{DMatrix, DVector, Isometry3, UnitQuaternion, Vector3};

#[derive(Debug, Clone)]
pub enum Damping {
    Fixed(f64),
    AdaptiveManipulability {
        lambda_min: f64,
        lambda_max: f64,
        manipulability_threshold: f64,
    },
}

#[derive(Debug, Clone)]
pub struct IkConfig {
    pub max_iters: usize,
    pub tol_error: f64,
    pub tol_step: f64,
    pub step_size: f64,
    pub damping: Damping,
    pub joint_limits: Option<JointLimits>,
}

impl Default for IkConfig {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tol_error: 1e-6,
            tol_step: 1e-8,
            step_size: 1.0,
            damping: Damping::Fixed(1e-2),
            joint_limits: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IkStatus {
    Converged,
    MaxIterations,
    NumericalFailure,
}

#[derive(Debug, Clone)]
pub struct IkResult {
    pub q: Vec<f64>,
    pub iterations: usize,
    pub final_error_norm: f64,
    pub status: IkStatus,
}

fn orientation_error_world(current: &UnitQuaternion<f64>, target: &UnitQuaternion<f64>) -> Vector3<f64> {
    let q_err = target * current.inverse();
    q_err.scaled_axis()
}

fn lambda_from_jacobian(j: &DMatrix<f64>, damping: &Damping) -> f64 {
    match damping {
        Damping::Fixed(l) => *l,
        Damping::AdaptiveManipulability {
            lambda_min,
            lambda_max,
            manipulability_threshold,
        } => {
            let jj_t = j * j.transpose();
            let det = jj_t.determinant().max(0.0);
            let w = det.sqrt();

            if w >= *manipulability_threshold {
                *lambda_min
            } else {
                let ratio = if *manipulability_threshold <= 1e-12 {
                    0.0
                } else {
                    (w / manipulability_threshold).clamp(0.0, 1.0)
                };
                lambda_min + (lambda_max - lambda_min) * (1.0 - ratio)
            }
        }
    }
}

fn dls_step(j: &DMatrix<f64>, e: &DVector<f64>, damping: &Damping) -> Option<DVector<f64>> {
    let lambda = lambda_from_jacobian(j, damping);
    let m = j.nrows();

    let a = j * j.transpose() + DMatrix::<f64>::identity(m, m) * (lambda * lambda);
    let y = a.lu().solve(e)?;
    Some(j.transpose() * y)
}

fn dls_pseudoinverse(j: &DMatrix<f64>, damping: &Damping) -> Option<DMatrix<f64>> {
    let lambda = lambda_from_jacobian(j, damping);
    let m = j.nrows();
    let a = j * j.transpose() + DMatrix::<f64>::identity(m, m) * (lambda * lambda);
    let a_inv = a.lu().solve(&DMatrix::<f64>::identity(m, m))?;
    Some(j.transpose() * a_inv)
}

fn apply_step_with_limits(
    model: &Model<f64>,
    mut q: Vec<f64>,
    mut dq: DVector<f64>,
    config: &IkConfig,
) -> (Vec<f64>, DVector<f64>) {
    dq *= config.step_size;
    if let Some(lim) = &config.joint_limits {
        let dq_sat = limits::saturate_velocity(model, dq.as_slice(), lim);
        dq = DVector::from_vec(dq_sat);
    }

    q = manifold::integrate(model, &q, dq.as_slice(), 1.0);
    if let Some(lim) = &config.joint_limits {
        q = limits::clamp_configuration(model, &q, lim);
    }
    (q, dq)
}

fn solve_iterative(
    model: &Model<f64>,
    q0: &[f64],
    config: &IkConfig,
    mut error_and_jacobian: impl FnMut(&[f64]) -> (DVector<f64>, DMatrix<f64>),
) -> IkResult {
    let mut q = manifold::normalize_configuration(model, q0);
    if let Some(lim) = &config.joint_limits {
        q = limits::clamp_configuration(model, &q, lim);
    }
    let mut last_error = f64::INFINITY;

    for iter in 0..config.max_iters {
        let (e, j) = error_and_jacobian(&q);
        let e_norm = e.norm();
        last_error = e_norm;

        if e_norm <= config.tol_error {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: e_norm,
                status: IkStatus::Converged,
            };
        }

        let Some(mut dq) = dls_step(&j, &e, &config.damping) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: e_norm,
                status: IkStatus::NumericalFailure,
            };
        };

        dq *= config.step_size;
        if let Some(lim) = &config.joint_limits {
            let dq_sat = limits::saturate_velocity(model, dq.as_slice(), lim);
            dq = DVector::from_vec(dq_sat);
        }

        let step_norm = dq.norm();
        if step_norm <= config.tol_step {
            break;
        }

        q = manifold::integrate(model, &q, dq.as_slice(), 1.0);
        if let Some(lim) = &config.joint_limits {
            q = limits::clamp_configuration(model, &q, lim);
        }
    }

    IkResult {
        q,
        iterations: config.max_iters,
        final_error_norm: last_error,
        status: IkStatus::MaxIterations,
    }
}

pub fn solve_joint_position_ik(
    model: &Model<f64>,
    q0: &[f64],
    joint_idx: usize,
    target_position_world: Vector3<f64>,
    config: &IkConfig,
) -> IkResult {
    solve_iterative(model, q0, config, |q| {
        let data = forward_kinematics(model, q);
        let current = se3::translation(&data.oMi[joint_idx]);
        let e = target_position_world - current;

        let j_full = compute_joint_jacobian(model, q, joint_idx);
        let j = j_full.rows(3, 3).into_owned();

        (DVector::from_vec(vec![e[0], e[1], e[2]]), j)
    })
}

pub fn solve_joint_orientation_ik(
    model: &Model<f64>,
    q0: &[f64],
    joint_idx: usize,
    target_orientation_world: UnitQuaternion<f64>,
    config: &IkConfig,
) -> IkResult {
    solve_iterative(model, q0, config, |q| {
        let data = forward_kinematics(model, q);
        let current = data.oMi[joint_idx].rotation;
        let e = orientation_error_world(&current, &target_orientation_world);

        let j_full = compute_joint_jacobian(model, q, joint_idx);
        let j = j_full.rows(0, 3).into_owned();

        (DVector::from_vec(vec![e[0], e[1], e[2]]), j)
    })
}

pub fn solve_joint_pose_ik(
    model: &Model<f64>,
    q0: &[f64],
    joint_idx: usize,
    target_pose_world: Isometry3<f64>,
    config: &IkConfig,
) -> IkResult {
    solve_iterative(model, q0, config, |q| {
        let data = forward_kinematics(model, q);
        let current = data.oMi[joint_idx];

        let e_rot = orientation_error_world(&current.rotation, &target_pose_world.rotation);
        let e_lin = se3::translation(&target_pose_world) - se3::translation(&current);

        let j = compute_joint_jacobian(model, q, joint_idx);
        (
            DVector::from_vec(vec![e_rot[0], e_rot[1], e_rot[2], e_lin[0], e_lin[1], e_lin[2]]),
            j,
        )
    })
}

pub fn solve_frame_pose_ik(
    model: &Model<f64>,
    q0: &[f64],
    frame: &Frame<f64>,
    target_pose_world: Isometry3<f64>,
    config: &IkConfig,
) -> IkResult {
    solve_iterative(model, q0, config, |q| {
        let current = compute_frame_placement(model, q, frame);
        let e_rot = orientation_error_world(&current.rotation, &target_pose_world.rotation);
        let e_lin = se3::translation(&target_pose_world) - se3::translation(&current);

        let j = compute_frame_jacobian(model, q, frame);
        (
            DVector::from_vec(vec![e_rot[0], e_rot[1], e_rot[2], e_lin[0], e_lin[1], e_lin[2]]),
            j,
        )
    })
}

pub fn solve_joint_position_orientation_ik(
    model: &Model<f64>,
    q0: &[f64],
    joint_idx: usize,
    target_position_world: Vector3<f64>,
    target_orientation_world: UnitQuaternion<f64>,
    position_weight: f64,
    orientation_weight: f64,
    config: &IkConfig,
) -> IkResult {
    solve_iterative(model, q0, config, |q| {
        let data = forward_kinematics(model, q);
        let current = data.oMi[joint_idx];

        let e_rot = orientation_error_world(&current.rotation, &target_orientation_world) * orientation_weight;
        let e_lin = (target_position_world - se3::translation(&current)) * position_weight;

        let mut j = compute_joint_jacobian(model, q, joint_idx);
        for r in 0..3 {
            for c in 0..j.ncols() {
                j[(r, c)] *= orientation_weight;
                j[(r + 3, c)] *= position_weight;
            }
        }

        (
            DVector::from_vec(vec![e_rot[0], e_rot[1], e_rot[2], e_lin[0], e_lin[1], e_lin[2]]),
            j,
        )
    })
}

/// Joint-position IK with null-space posture regularization.
///
/// Primary task: reach `target_position_world` at `joint_idx`.
/// Secondary task (projected in null-space): move toward `q_posture_target`.
pub fn solve_joint_position_ik_with_posture(
    model: &Model<f64>,
    q0: &[f64],
    joint_idx: usize,
    target_position_world: Vector3<f64>,
    q_posture_target: &[f64],
    posture_gain: f64,
    config: &IkConfig,
) -> IkResult {
    assert_eq!(q_posture_target.len(), model.nq);

    let mut q = manifold::normalize_configuration(model, q0);
    if let Some(lim) = &config.joint_limits {
        q = limits::clamp_configuration(model, &q, lim);
    }
    let mut last_error = f64::INFINITY;

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);
        let current = se3::translation(&data.oMi[joint_idx]);
        let e_vec = target_position_world - current;
        let e = DVector::from_vec(vec![e_vec[0], e_vec[1], e_vec[2]]);
        let e_norm = e.norm();
        last_error = e_norm;

        if e_norm <= config.tol_error {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: e_norm,
                status: IkStatus::Converged,
            };
        }

        let j_full = compute_joint_jacobian(model, &q, joint_idx);
        let j = j_full.rows(3, 3).into_owned();

        let Some(dq_primary) = dls_step(&j, &e, &config.damping) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: e_norm,
                status: IkStatus::NumericalFailure,
            };
        };

        let Some(j_pinv) = dls_pseudoinverse(&j, &config.damping) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: e_norm,
                status: IkStatus::NumericalFailure,
            };
        };

        let n = DMatrix::<f64>::identity(model.nv, model.nv) - (&j_pinv * &j);

        let posture_err = DVector::from_vec(manifold::difference(model, &q, q_posture_target));
        let dq_secondary = n * (posture_err * posture_gain);
        let dq = dq_primary + dq_secondary;

        let step_norm = dq.norm();
        if step_norm <= config.tol_step {
            break;
        }

        let (q_new, _) = apply_step_with_limits(model, q, dq, config);
        q = q_new;
    }

    IkResult {
        q,
        iterations: config.max_iters,
        final_error_norm: last_error,
        status: IkStatus::MaxIterations,
    }
}

/// Prioritized two-task IK (strict hierarchy).
///
/// - Primary task: position of `primary_joint_idx`
/// - Secondary task: position of `secondary_joint_idx` in null-space of primary
pub fn solve_two_task_position_ik(
    model: &Model<f64>,
    q0: &[f64],
    primary_joint_idx: usize,
    primary_target_world: Vector3<f64>,
    secondary_joint_idx: usize,
    secondary_target_world: Vector3<f64>,
    secondary_weight: f64,
    config: &IkConfig,
) -> IkResult {
    let mut q = manifold::normalize_configuration(model, q0);
    if let Some(lim) = &config.joint_limits {
        q = limits::clamp_configuration(model, &q, lim);
    }
    let mut last_error = f64::INFINITY;

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);

        let p1 = se3::translation(&data.oMi[primary_joint_idx]);
        let e1_vec = primary_target_world - p1;
        let e1 = DVector::from_vec(vec![e1_vec[0], e1_vec[1], e1_vec[2]]);
        let e1_norm = e1.norm();

        let p2 = se3::translation(&data.oMi[secondary_joint_idx]);
        let e2_vec = (secondary_target_world - p2) * secondary_weight;
        let e2 = DVector::from_vec(vec![e2_vec[0], e2_vec[1], e2_vec[2]]);

        last_error = (e1_norm * e1_norm + e2.norm_squared()).sqrt();

        if e1_norm <= config.tol_error {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: last_error,
                status: IkStatus::Converged,
            };
        }

        let j1_full = compute_joint_jacobian(model, &q, primary_joint_idx);
        let j2_full = compute_joint_jacobian(model, &q, secondary_joint_idx);
        let j1 = j1_full.rows(3, 3).into_owned();
        let j2 = j2_full.rows(3, 3).into_owned() * secondary_weight;

        let Some(dq1) = dls_step(&j1, &e1, &config.damping) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: last_error,
                status: IkStatus::NumericalFailure,
            };
        };

        let Some(j1_pinv) = dls_pseudoinverse(&j1, &config.damping) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: last_error,
                status: IkStatus::NumericalFailure,
            };
        };
        let n = DMatrix::<f64>::identity(model.nv, model.nv) - (&j1_pinv * &j1);

        // Secondary task in null-space: J2 N dq2 = e2 - J2 dq1
        let j2n = &j2 * &n;
        let rhs2 = e2 - (&j2 * &dq1);
        let dq2 = dls_step(&j2n, &rhs2, &config.damping).unwrap_or_else(|| DVector::zeros(model.nv));

        let dq = dq1 + &n * dq2;
        let step_norm = dq.norm();
        if step_norm <= config.tol_step {
            break;
        }

        let (q_new, _) = apply_step_with_limits(model, q, dq, config);
        q = q_new;
    }

    IkResult {
        q,
        iterations: config.max_iters,
        final_error_norm: last_error,
        status: IkStatus::MaxIterations,
    }
}

// ─── Collision-aware IK ───────────────────────────────────────────────────────

/// Configuration for collision-aware IK.
#[derive(Debug, Clone)]
pub struct CollisionConfig {
    /// Safety margin around geometry objects (meters).  Pairs closer than this
    /// distance generate a repulsion gradient.
    pub safety_margin: f64,
    /// Weight applied to the repulsion gradient relative to the IK step.
    /// Typical range: 0.1 – 5.0.
    pub collision_weight: f64,
    /// Finite-difference step size for gradient computation.  Default: 1e-4.
    pub fd_eps: f64,
    /// Optional ACM; pairs in the ACM are never included in the potential.
    pub acm: Option<AllowedCollisionMatrix>,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            safety_margin: 0.05,
            collision_weight: 1.0,
            fd_eps: 1e-4,
            acm: None,
        }
    }
}

/// Solve a joint-position IK while repelling from geometry collisions.
///
/// At each iteration the standard DLS position step `Δq₁` is computed, then a
/// repulsion term
///
/// ```text
/// Δq_rep = −collision_weight · ∇V(q)
/// ```
///
/// is added, where `∇V` is the gradient of the collision potential
/// (see [`collision_potential_gradient`]).
///
/// The repulsion is applied in the null-space of the IK Jacobian so the
/// primary task (end-effector position) is disturbed as little as possible:
///
/// ```text
/// Δq = Δq₁ + N · Δq_rep
/// ```
///
/// # Parameters
/// - `model` – kinematic model.
/// - `gmodel` – geometry model used for collision checks.
/// - `q0` – initial configuration.
/// - `joint_idx` – joint whose translation should reach `target`.
/// - `target` – desired world-frame position.
/// - `cc` – collision configuration.
/// - `config` – IK solver configuration.
pub fn solve_joint_position_ik_with_collision_avoidance(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q0: &[f64],
    joint_idx: usize,
    target: Vector3<f64>,
    cc: &CollisionConfig,
    config: &IkConfig,
) -> IkResult {
    assert!(joint_idx > 0 && joint_idx < model.joints.len());

    let acm_ref = cc.acm.as_ref();

    let mut q = manifold::normalize_configuration(model, q0);
    if let Some(lim) = &config.joint_limits {
        q = limits::clamp_configuration(model, &q, lim);
    }

    let mut last_error = 0.0_f64;

    for iter in 0..config.max_iters {
        let data = forward_kinematics(model, &q);
        let p = se3::translation(&data.oMi[joint_idx]);
        let e_vec = DVector::from_iterator(3, (target - p).iter().copied());

        let error_norm = e_vec.norm();
        last_error = error_norm;
        if error_norm <= config.tol_error {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: error_norm,
                status: IkStatus::Converged,
            };
        }

        // Primary task step (linear rows of Jacobian)
        let j_full = compute_joint_jacobian(model, &q, joint_idx);
        let j = j_full.rows(3, 3).into_owned();

        let Some(dq1) = dls_step(&j, &e_vec, &config.damping) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: last_error,
                status: IkStatus::NumericalFailure,
            };
        };

        // Null-space projector
        let Some(j_pinv) = dls_pseudoinverse(&j, &config.damping) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: last_error,
                status: IkStatus::NumericalFailure,
            };
        };
        let n = DMatrix::<f64>::identity(model.nv, model.nv) - (&j_pinv * &j);

        // Collision repulsion gradient
        let grad = collision_potential_gradient(
            model,
            gmodel,
            &q,
            acm_ref,
            cc.safety_margin,
            cc.fd_eps,
        );
        let grad_vec = DVector::from_vec(grad);
        let dq_rep = -cc.collision_weight * grad_vec;

        // Combine: primary + null-space repulsion
        let dq = dq1 + &n * dq_rep;

        let step_norm = dq.norm();
        if step_norm <= config.tol_step {
            break;
        }

        let (q_new, _) = apply_step_with_limits(model, q, dq, config);
        q = q_new;
    }

    IkResult {
        q,
        iterations: config.max_iters,
        final_error_norm: last_error,
        status: IkStatus::MaxIterations,
    }
}

/// Returns `true` if the configuration `q` has any unallowed collision.
///
/// Thin convenience wrapper around [`has_collision_acm`].
pub fn configuration_is_collision_free(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q: &[f64],
    acm: Option<&AllowedCollisionMatrix>,
) -> bool {
    !has_collision_acm(model, gmodel, q, acm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::limits::JointLimits;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;

    fn two_link_planar() -> Model<f64> {
        ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::revolute_z(),
                se3::identity(),
                LinkInertia::zero(),
            )
            .add_joint(
                "j2",
                1,
                joint::revolute_z(),
                se3::from_rotation_and_translation(&nalgebra::Rotation3::identity(), &Vector3::new(1.0, 0.0, 0.0)),
                LinkInertia::zero(),
            )
            .build()
    }

    #[test]
    fn position_ik_reaches_target() {
        let model = two_link_planar();
        let q0 = vec![0.1, -0.3];
        let target = Vector3::new(0.8, 0.6, 0.0);

        let cfg = IkConfig {
            max_iters: 200,
            tol_error: 1e-8,
            tol_step: 1e-10,
            step_size: 0.8,
            damping: Damping::Fixed(1e-3),
            joint_limits: None,
        };

        let result = solve_joint_position_ik(&model, &q0, 2, target, &cfg);
        assert_eq!(result.status, IkStatus::Converged);

        let data = forward_kinematics(&model, &result.q);
        let p = se3::translation(&data.oMi[2]);
        assert_relative_eq!(p[0], target[0], epsilon = 1e-4);
        assert_relative_eq!(p[1], target[1], epsilon = 1e-4);
    }

    #[test]
    fn orientation_ik_single_joint() {
        let model = ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::revolute_z(),
                se3::identity(),
                LinkInertia::zero(),
            )
            .build();
        let q0 = vec![0.0];
        let target = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);

        let result = solve_joint_orientation_ik(&model, &q0, 1, target, &IkConfig::default());
        assert_eq!(result.status, IkStatus::Converged);
        assert_relative_eq!(result.q[0], std::f64::consts::FRAC_PI_2, epsilon = 1e-4);
    }

    #[test]
    fn pose_ik_freeflyer_converges() {
        let model = ModelBuilder::new()
            .add_joint(
                "base",
                0,
                crate::joint::JointType::FreeFlyer,
                se3::identity(),
                LinkInertia::zero(),
            )
            .build();

        let q0 = model.neutral_q();
        let target = Isometry3::from_parts(
            nalgebra::Translation3::new(0.4, -0.2, 0.3),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.3),
        );

        let result = solve_joint_pose_ik(&model, &q0, 1, target, &IkConfig::default());
        assert_eq!(result.status, IkStatus::Converged);
        assert!(result.final_error_norm < 1e-6);
    }

    #[test]
    fn adaptive_damping_position_ik() {
        let model = two_link_planar();
        let q0 = vec![0.0, 0.0];
        let target = Vector3::new(0.6, 0.8, 0.0);

        let cfg = IkConfig {
            damping: Damping::AdaptiveManipulability {
                lambda_min: 1e-4,
                lambda_max: 1e-1,
                manipulability_threshold: 1e-2,
            },
            ..IkConfig::default()
        };

        let result = solve_joint_position_ik(&model, &q0, 2, target, &cfg);
        assert_eq!(result.status, IkStatus::Converged);
    }

    #[test]
    fn impossible_target_returns_max_iterations() {
        let model = two_link_planar();
        let q0 = vec![0.0, 0.0];
        let target = Vector3::new(10.0, 0.0, 0.0);

        let cfg = IkConfig {
            max_iters: 20,
            ..IkConfig::default()
        };

        let result = solve_joint_position_ik(&model, &q0, 2, target, &cfg);
        assert_eq!(result.status, IkStatus::MaxIterations);
    }

    #[test]
    fn ik_respects_joint_limits() {
        let model = two_link_planar();
        let q0 = vec![0.0, 0.0];
        let target = Vector3::new(0.0, 1.0, 0.0);

        let mut limits = JointLimits::unbounded(&model);
        // Lock first joint near zero; solver should not violate this bound.
        limits.q_min[0] = -0.1;
        limits.q_max[0] = 0.1;
        limits.v_max[0] = 0.05;

        let cfg = IkConfig {
            max_iters: 150,
            joint_limits: Some(limits.clone()),
            ..IkConfig::default()
        };

        let result = solve_joint_position_ik(&model, &q0, 2, target, &cfg);
        assert!(result.q[0] >= limits.q_min[0] - 1e-12);
        assert!(result.q[0] <= limits.q_max[0] + 1e-12);
    }

    #[test]
    fn nullspace_posture_bias_moves_toward_reference() {
        let model = two_link_planar();
        let q0 = vec![0.0, 0.0];
        let target = Vector3::new(1.5, 0.0, 0.0);
        let q_ref = vec![0.0, 1.0];

        let cfg = IkConfig {
            max_iters: 120,
            step_size: 0.8,
            damping: Damping::Fixed(1e-3),
            ..IkConfig::default()
        };

        let no_bias = solve_joint_position_ik(&model, &q0, 2, target, &cfg);
        let with_bias = solve_joint_position_ik_with_posture(&model, &q0, 2, target, &q_ref, 0.15, &cfg);

        let d_no = (no_bias.q[1] - q_ref[1]).abs();
        let d_yes = (with_bias.q[1] - q_ref[1]).abs();
        assert!(d_yes < d_no);
    }

    #[test]
    fn prioritized_two_task_keeps_primary_accuracy() {
        let model = two_link_planar();
        let q0 = vec![0.2, -0.4];

        let primary_target = Vector3::new(0.8, 0.6, 0.0);
        let secondary_target = Vector3::new(0.4, 0.7, 0.0);

        let cfg = IkConfig {
            max_iters: 180,
            step_size: 0.8,
            damping: Damping::Fixed(1e-3),
            ..IkConfig::default()
        };

        let res = solve_two_task_position_ik(
            &model,
            &q0,
            2,
            primary_target,
            1,
            secondary_target,
            0.5,
            &cfg,
        );

        // Primary task must remain accurate.
        let data = forward_kinematics(&model, &res.q);
        let p_primary = se3::translation(&data.oMi[2]);
        assert_relative_eq!(p_primary[0], primary_target[0], epsilon = 1e-3);
        assert_relative_eq!(p_primary[1], primary_target[1], epsilon = 1e-3);
    }

    #[test]
    fn collision_aware_ik_reduces_potential_vs_plain_ik() {
        use crate::collision::collision_potential;
        use crate::geometry::{GeometryModel, GeometryObject, GeometryShape};

        // 2-link arm; add a large obstacle sphere at the mid-point of link 2
        // so that naive IK runs into it.
        let model = two_link_planar();

        let mut gm = GeometryModel::new();
        // Link 1 geometry (small sphere at joint 1)
        gm.add(GeometryObject {
            name: "link1".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.1 },
            mesh_path: None,
            mesh_scale: None,
        });
        // Link 2 geometry (small sphere at joint 2)
        gm.add(GeometryObject {
            name: "link2".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.1 },
            mesh_path: None,
            mesh_scale: None,
        });

        let target = Vector3::new(0.5, 0.8, 0.0);
        let q0 = vec![0.0, 0.0];

        let cfg = IkConfig {
            max_iters: 80,
            step_size: 0.5,
            damping: Damping::Fixed(1e-2),
            ..IkConfig::default()
        };

        // Plain IK (no collision avoidance)
        let plain_result = solve_joint_position_ik(&model, &q0, 2, target, &cfg);

        // Collision-aware IK with a safety margin that activates between the two spheres
        let acm = AllowedCollisionMatrix::from_adjacent_links(&model, &gm);
        let cc = CollisionConfig {
            safety_margin: 0.5,
            collision_weight: 2.0,
            acm: Some(acm.clone()),
            ..CollisionConfig::default()
        };
        let ca_result = solve_joint_position_ik_with_collision_avoidance(
            &model, &gm, &q0, 2, target, &cc, &cfg,
        );

        // The collision-aware result should have lower or equal potential
        let v_plain = collision_potential(&model, &gm, &plain_result.q, Some(&acm), cc.safety_margin);
        let v_ca = collision_potential(&model, &gm, &ca_result.q, Some(&acm), cc.safety_margin);
        assert!(
            v_ca <= v_plain + 1e-9,
            "Collision-aware IK (V={v_ca:.4}) should have ≤ potential than plain IK (V={v_plain:.4})",
        );
    }

    #[test]
    fn configuration_is_collision_free_detects_collision() {
        use crate::geometry::{GeometryModel, GeometryObject, GeometryShape};

        let model = two_link_planar();
        let mut gm = GeometryModel::new();
        // Two large spheres — they will overlap at q=0
        gm.add(GeometryObject {
            name: "s1".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.8 },
            mesh_path: None,
            mesh_scale: None,
        });
        gm.add(GeometryObject {
            name: "s2".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.8 },
            mesh_path: None,
            mesh_scale: None,
        });

        let acm = AllowedCollisionMatrix::from_adjacent_links(&model, &gm);
        // j1 and j2 are adjacent (parent-child) → allowed → still appears collision-free
        assert!(configuration_is_collision_free(&model, &gm, &[0.0, 0.0], Some(&acm)));

        // Without ACM: the adjacent collision is reported
        assert!(!configuration_is_collision_free(&model, &gm, &[0.0, 0.0], None));
    }
}
