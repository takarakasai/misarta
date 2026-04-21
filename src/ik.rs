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

/// Solver method for the differential IK step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMethod {
    /// Damped Least Squares:  δq = W⁻¹Jᵀ(JW⁻¹Jᵀ + λ²I)⁻¹ e
    DampedLeastSquares,
    /// Jacobian Transpose:   δq = α·W⁻¹·Jᵀ·e,  α = eᵀJJᵀe / ‖JJᵀe‖²
    JacobianTranspose,
}

impl Default for SolverMethod {
    fn default() -> Self { SolverMethod::DampedLeastSquares }
}

/// Per-joint cost weights for weighted pseudo-inverse.
///
/// Weights are in configuration space (one per DoF).  Larger weight means
/// the joint is "more expensive" to move and will be used less.
///
/// The weighted pseudo-inverse is:  J⁺_W = W⁻¹ Jᵀ (J W⁻¹ Jᵀ + λ²I)⁻¹
#[derive(Debug, Clone)]
pub struct JointWeights {
    /// Weight per DoF.  Length must equal `model.nv`.
    pub weights: Vec<f64>,
}

impl JointWeights {
    /// Create uniform (identity) weights.
    pub fn uniform(nv: usize) -> Self {
        Self { weights: vec![1.0; nv] }
    }

    /// Create weights from a gradient that prefers joints near the EE.
    ///
    /// Given a chain of joint indices (ordered root → EE), joint `i` gets
    /// weight `alpha^(n-1-i)`.  All other DoFs get weight `alpha^(n-1)`.
    pub fn ee_proximal(nv: usize, chain: &[usize], alpha: f64) -> Self {
        let n = chain.len();
        let mut w = vec![alpha.powi(n.max(1) as i32 - 1); nv];
        for (i, &vi) in chain.iter().enumerate() {
            if vi < nv {
                w[vi] = alpha.powi((n - 1 - i) as i32);
            }
        }
        w
            .iter_mut()
            .for_each(|x| *x = x.max(1e-6));
        Self { weights: w }
    }
}

#[derive(Debug, Clone)]
pub struct IkConfig {
    pub max_iters: usize,
    pub tol_error: f64,
    pub tol_step: f64,
    pub step_size: f64,
    pub damping: Damping,
    pub joint_limits: Option<JointLimits>,
    /// Solver method (DLS or JT).  Default: DLS.
    pub solver_method: SolverMethod,
    /// Per-joint cost weights.  `None` = uniform.
    pub joint_weights: Option<JointWeights>,
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
            solver_method: SolverMethod::DampedLeastSquares,
            joint_weights: None,
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

fn dls_step(j: &DMatrix<f64>, e: &DVector<f64>, damping: &Damping, weights: Option<&JointWeights>) -> Option<DVector<f64>> {
    let lambda = lambda_from_jacobian(j, damping);
    let m = j.nrows();
    let n = j.ncols();

    if let Some(w) = weights {
        // Weighted: J̃ = J · diag(1/w), then solve (J̃·Jᵀ + λ²I) y = e,
        // then δq = diag(1/w) · Jᵀ · y
        let mut jw = j.clone();
        for col in 0..n {
            let inv = 1.0 / w.weights.get(col).copied().unwrap_or(1.0).max(1e-12);
            for row in 0..m {
                jw[(row, col)] *= inv;
            }
        }
        let a = &jw * j.transpose() + DMatrix::<f64>::identity(m, m) * (lambda * lambda);
        let y = a.lu().solve(e)?;
        let mut dq = j.transpose() * y;
        for i in 0..n {
            dq[i] *= 1.0 / w.weights.get(i).copied().unwrap_or(1.0).max(1e-12);
        }
        Some(dq)
    } else {
        let a = j * j.transpose() + DMatrix::<f64>::identity(m, m) * (lambda * lambda);
        let y = a.lu().solve(e)?;
        Some(j.transpose() * y)
    }
}

fn dls_pseudoinverse(j: &DMatrix<f64>, damping: &Damping, weights: Option<&JointWeights>) -> Option<DMatrix<f64>> {
    let lambda = lambda_from_jacobian(j, damping);
    let m = j.nrows();
    let n = j.ncols();

    if let Some(w) = weights {
        let mut jw = j.clone();
        for col in 0..n {
            let inv = 1.0 / w.weights.get(col).copied().unwrap_or(1.0).max(1e-12);
            for row in 0..m {
                jw[(row, col)] *= inv;
            }
        }
        let a = &jw * j.transpose() + DMatrix::<f64>::identity(m, m) * (lambda * lambda);
        let a_inv = a.lu().solve(&DMatrix::<f64>::identity(m, m))?;
        let mut pinv = j.transpose() * a_inv;
        for row in 0..n {
            let inv = 1.0 / w.weights.get(row).copied().unwrap_or(1.0).max(1e-12);
            for col in 0..m {
                pinv[(row, col)] *= inv;
            }
        }
        Some(pinv)
    } else {
        let a = j * j.transpose() + DMatrix::<f64>::identity(m, m) * (lambda * lambda);
        let a_inv = a.lu().solve(&DMatrix::<f64>::identity(m, m))?;
        Some(j.transpose() * a_inv)
    }
}

/// Jacobian Transpose step: δq = α · W⁻¹ · Jᵀ · e
///
/// Optimal step size: α = eᵀ·J·Jᵀ·e / ‖J·Jᵀ·e‖²
fn jt_step(j: &DMatrix<f64>, e: &DVector<f64>, weights: Option<&JointWeights>) -> DVector<f64> {
    let jjt = j * j.transpose();
    let jjt_e = &jjt * e;
    let num = e.dot(&jjt_e);
    let den = jjt_e.norm_squared();
    let alpha = if den > 1e-30 { num / den } else { 1e-3 };
    let n = j.ncols();
    let mut dq = j.transpose() * e * alpha;
    if let Some(w) = weights {
        for i in 0..n {
            dq[i] *= 1.0 / w.weights.get(i).copied().unwrap_or(1.0).max(1e-12);
        }
    }
    dq
}

/// Compute one IK step using the configured solver method.
fn solver_step(j: &DMatrix<f64>, e: &DVector<f64>, config: &IkConfig) -> Option<DVector<f64>> {
    let wt = config.joint_weights.as_ref();
    match config.solver_method {
        SolverMethod::DampedLeastSquares => dls_step(j, e, &config.damping, wt),
        SolverMethod::JacobianTranspose => Some(jt_step(j, e, wt)),
    }
}

/// Compute pseudo-inverse using the configured weights.
fn solver_pseudoinverse(j: &DMatrix<f64>, config: &IkConfig) -> Option<DMatrix<f64>> {
    dls_pseudoinverse(j, &config.damping, config.joint_weights.as_ref())
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

        let Some(mut dq) = solver_step(&j, &e, config) else {
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

        let Some(dq_primary) = solver_step(&j, &e, config) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: e_norm,
                status: IkStatus::NumericalFailure,
            };
        };

        let Some(j_pinv) = solver_pseudoinverse(&j, config) else {
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

        let Some(dq1) = solver_step(&j1, &e1, config) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: last_error,
                status: IkStatus::NumericalFailure,
            };
        };

        let Some(j1_pinv) = solver_pseudoinverse(&j1, config) else {
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
        let dq2 = solver_step(&j2n, &rhs2, config).unwrap_or_else(|| DVector::zeros(model.nv));

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

        let Some(dq1) = solver_step(&j, &e_vec, config) else {
            return IkResult {
                q,
                iterations: iter,
                final_error_norm: last_error,
                status: IkStatus::NumericalFailure,
            };
        };

        // Null-space projector
        let Some(j_pinv) = solver_pseudoinverse(&j, config) else {
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

// ─── Differential (single-step) IK ───────────────────────────────────────────

/// Configuration for a single differential IK step.
///
/// Unlike [`IkConfig`] which drives an iterative solver to convergence,
/// this is for GUI-style "one step per frame" resolved-rate control.
#[derive(Debug, Clone)]
pub struct DiffIkConfig {
    /// Proportional gain ∈ (0, 1].  Fraction of position error to correct
    /// per step.  Small values (~0.2–0.3) yield smooth, stable tracking.
    pub gain: f64,
    /// Hard clamp on each |δqᵢ| (radians) as a safety net.
    pub max_joint_step: f64,
    /// Damping strategy to use.
    pub damping: Damping,
    /// Solver method (DLS or JT).
    pub solver_method: SolverMethod,
    /// Per-joint cost weights.  `None` = uniform.
    pub joint_weights: Option<JointWeights>,
    /// Optional task-space projection matrix (m×3).
    /// When `Some`, the 3D Jacobian and error are projected:
    ///   J_proj = P · J  (m×n),  Δx_proj = P · Δx  (m)
    /// Useful for 2-DoF screen-plane IK (m=2, P = [cam_right; cam_up]).
    pub task_projection: Option<DMatrix<f64>>,
}

impl Default for DiffIkConfig {
    fn default() -> Self {
        Self {
            gain: 0.3,
            max_joint_step: 0.15,
            damping: Damping::Fixed(0.05),
            solver_method: SolverMethod::DampedLeastSquares,
            joint_weights: None,
            task_projection: None,
        }
    }
}

/// Result of a single differential IK step.
#[derive(Debug, Clone)]
pub struct DiffIkResult {
    /// Joint-angle deltas (one per DoF in `chain`).
    pub dq: Vec<f64>,
    /// Position error norm *before* this step was applied.
    pub error_before: f64,
}

/// Compute one differential IK step (resolved-rate control).
///
/// Given a chain of velocity indices and a Jacobian (3×n, world frame),
/// computes δq to move the end-effector toward `target_pos`.
///
/// This is the primitive that GUI interactive IK should call once per frame.
///
/// # Parameters
/// - `jac_world` — 3×n positional Jacobian in world frame.
/// - `ee_pos` — current EE world position.
/// - `target_pos` — desired EE world position.
/// - `config` — solver parameters.
///
/// # Returns
/// [`DiffIkResult`] with joint deltas and the pre-step error norm.
pub fn differential_ik_step(
    jac_world: &DMatrix<f64>,
    ee_pos: &Vector3<f64>,
    target_pos: &Vector3<f64>,
    config: &DiffIkConfig,
) -> DiffIkResult {
    let n = jac_world.ncols();
    let dx3 = (target_pos - ee_pos) * config.gain;
    let error_before = (target_pos - ee_pos).norm();

    // Optionally project to lower-dimensional task space
    let (dx_vec, jac) = if let Some(ref proj) = config.task_projection {
        let dx_full = DVector::from_column_slice(&[dx3.x, dx3.y, dx3.z]);
        (proj * &dx_full, proj * jac_world)
    } else {
        (
            DVector::from_column_slice(&[dx3.x, dx3.y, dx3.z]),
            jac_world.clone(),
        )
    };
    let _m = jac.nrows();

    // Build a temporary IkConfig to reuse solver_step infrastructure
    let tmp_config = IkConfig {
        damping: config.damping.clone(),
        solver_method: config.solver_method,
        joint_weights: config.joint_weights.clone(),
        ..IkConfig::default()
    };

    let mut dq = solver_step(&jac, &dx_vec, &tmp_config)
        .unwrap_or_else(|| DVector::zeros(n));

    // Safety clamp
    for i in 0..n {
        dq[i] = dq[i].clamp(-config.max_joint_step, config.max_joint_step);
    }

    DiffIkResult {
        dq: (0..n).map(|i| dq[i]).collect(),
        error_before,
    }
}

// =====================================================================
//  Multi-constraint differential IK
// =====================================================================

/// A single equality constraint for multi-constraint differential IK.
///
/// Each constraint contributes rows to an augmented Jacobian system.
#[derive(Debug, Clone)]
pub struct DiffIkConstraint {
    /// Constraint Jacobian (m×n, same column space as primary Jacobian).
    pub jacobian: DMatrix<f64>,
    /// Constraint error to drive to zero (m-dimensional).
    /// Positive = overshoot.  The solver drives `error → 0`.
    pub error: DVector<f64>,
    /// Weight of this constraint relative to the primary task.
    /// Larger weight → harder constraint (but still soft).
    pub weight: f64,
}

/// Compute one differential IK step with equality constraints (augmented Jacobian).
///
/// Stacks the primary 3-DoF position task with arbitrary equality constraints:
///
/// $$\begin{bmatrix} J_{\text{task}} \\ w_1 J_{c_1} \\ w_2 J_{c_2} \\ \vdots
/// \end{bmatrix} \delta q
/// = \begin{bmatrix} \text{gain} \cdot \Delta x \\ -w_1 e_{c_1} \\ -w_2 e_{c_2} \\ \vdots
/// \end{bmatrix}$$
///
/// The combined system is solved with the configured solver method (DLS or JT)
/// and joint weights.
///
/// # Parameters
/// - `jac_world` — 3×n positional Jacobian for the primary task (world frame).
/// - `ee_pos` — current EE world position.
/// - `target_pos` — desired EE world position.
/// - `constraints` — equality constraints to enforce simultaneously.
/// - `config` — solver parameters (gain, damping, etc.).
pub fn differential_ik_step_with_constraints(
    jac_world: &DMatrix<f64>,
    ee_pos: &Vector3<f64>,
    target_pos: &Vector3<f64>,
    constraints: &[DiffIkConstraint],
    config: &DiffIkConfig,
) -> DiffIkResult {
    let n = jac_world.ncols();
    let dx3 = (target_pos - ee_pos) * config.gain;
    let error_before = (target_pos - ee_pos).norm();

    // Project primary task if requested (e.g. 2-DoF screen plane)
    let (dx_task, jac_task) = if let Some(ref proj) = config.task_projection {
        let dx_full = DVector::from_column_slice(&[dx3.x, dx3.y, dx3.z]);
        (proj * &dx_full, proj * jac_world)
    } else {
        (
            DVector::from_column_slice(&[dx3.x, dx3.y, dx3.z]),
            jac_world.clone(),
        )
    };

    // Compute total rows for augmented system
    let m_task = jac_task.nrows();
    let m_constraints: usize = constraints.iter().map(|c| c.jacobian.nrows()).sum();
    let m_total = m_task + m_constraints;

    // Build augmented Jacobian and RHS
    let mut jac_aug = DMatrix::<f64>::zeros(m_total, n);
    let mut rhs_aug = DVector::<f64>::zeros(m_total);

    // Primary task rows
    jac_aug.view_mut((0, 0), (m_task, n)).copy_from(&jac_task);
    rhs_aug.rows_mut(0, m_task).copy_from(&dx_task);

    // Constraint rows
    let mut row_offset = m_task;
    for c in constraints {
        let mc = c.jacobian.nrows();
        assert_eq!(c.jacobian.ncols(), n, "Constraint Jacobian column count must match primary");
        assert_eq!(c.error.len(), mc, "Constraint error dimension must match Jacobian rows");
        let w = c.weight;
        // Weighted constraint: w * J_c * dq = -w * e_c
        for r in 0..mc {
            for col in 0..n {
                jac_aug[(row_offset + r, col)] = w * c.jacobian[(r, col)];
            }
            rhs_aug[row_offset + r] = -w * c.error[r];
        }
        row_offset += mc;
    }

    // Solve augmented system with configured solver
    let tmp_config = IkConfig {
        damping: config.damping.clone(),
        solver_method: config.solver_method,
        joint_weights: config.joint_weights.clone(),
        ..IkConfig::default()
    };

    let mut dq = solver_step(&jac_aug, &rhs_aug, &tmp_config)
        .unwrap_or_else(|| DVector::zeros(n));

    // Safety clamp
    for i in 0..n {
        dq[i] = dq[i].clamp(-config.max_joint_step, config.max_joint_step);
    }

    DiffIkResult {
        dq: (0..n).map(|i| dq[i]).collect(),
        error_before,
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
            ..IkConfig::default()
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
            mesh_data: None,
            material: None,
        });
        // Link 2 geometry (small sphere at joint 2)
        gm.add(GeometryObject {
            name: "link2".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.1 },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
            material: None,
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
            mesh_data: None,
            material: None,
        });
        gm.add(GeometryObject {
            name: "s2".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.8 },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
            material: None,
        });

        let acm = AllowedCollisionMatrix::from_adjacent_links(&model, &gm);
        // j1 and j2 are adjacent (parent-child) → allowed → still appears collision-free
        assert!(configuration_is_collision_free(&model, &gm, &[0.0, 0.0], Some(&acm)));

        // Without ACM: the adjacent collision is reported
        assert!(!configuration_is_collision_free(&model, &gm, &[0.0, 0.0], None));
    }

    // ─── New feature tests ─────────────────────────────────────────────

    #[test]
    fn jt_solver_converges() {
        let model = two_link_planar();
        let q0 = vec![0.1, -0.3];
        let target = Vector3::new(0.8, 0.6, 0.0);
        let cfg = IkConfig {
            max_iters: 500,
            tol_error: 1e-4,
            step_size: 0.5,
            solver_method: SolverMethod::JacobianTranspose,
            ..IkConfig::default()
        };
        let result = solve_joint_position_ik(&model, &q0, 2, target, &cfg);
        assert_eq!(result.status, IkStatus::Converged);
    }

    #[test]
    fn weighted_ik_prefers_cheap_joints() {
        let model = two_link_planar();
        let q0 = vec![0.0, 0.0];
        let target = Vector3::new(0.8, 0.6, 0.0);

        // Uniform: both joints move freely
        let cfg_uniform = IkConfig {
            max_iters: 200,
            tol_error: 1e-6,
            step_size: 0.8,
            damping: Damping::Fixed(1e-3),
            ..IkConfig::default()
        };
        let res_uniform = solve_joint_position_ik(&model, &q0, 2, target, &cfg_uniform);

        // Weighted: joint 0 is very expensive (weight=100), joint 1 is cheap (weight=1)
        let cfg_weighted = IkConfig {
            joint_weights: Some(JointWeights { weights: vec![100.0, 1.0] }),
            ..cfg_uniform.clone()
        };
        let res_weighted = solve_joint_position_ik(&model, &q0, 2, target, &cfg_weighted);

        // Weighted solution should move joint 0 less
        assert!(
            res_weighted.q[0].abs() < res_uniform.q[0].abs(),
            "Weighted j0={:.4} should be smaller than uniform j0={:.4}",
            res_weighted.q[0].abs(), res_uniform.q[0].abs(),
        );
    }

    #[test]
    fn differential_ik_step_reduces_error() {
        let model = two_link_planar();
        let q = vec![0.3, -0.5];
        let data = forward_kinematics(&model, &q);
        let ee = se3::translation(&data.oMi[2]);
        let target = Vector3::new(0.8, 0.6, 0.0);

        let jac = compute_joint_jacobian(&model, &q, 2)
            .rows(3, 3).into_owned();

        let cfg = DiffIkConfig::default();
        let result = differential_ik_step(&jac, &ee, &target, &cfg);

        assert_eq!(result.dq.len(), 2);
        assert!(result.error_before > 0.0);

        // Apply step and check error decreased
        let q_new: Vec<f64> = q.iter().zip(&result.dq)
            .map(|(qi, dqi)| qi + dqi)
            .collect();
        let data2 = forward_kinematics(&model, &q_new);
        let ee2 = se3::translation(&data2.oMi[2]);
        let error_after = (target - ee2).norm();
        assert!(error_after < result.error_before);
    }

    #[test]
    fn differential_ik_step_with_projection() {
        let model = two_link_planar();
        let q = vec![0.3, -0.5];
        let data = forward_kinematics(&model, &q);
        let ee = se3::translation(&data.oMi[2]);
        let target = Vector3::new(0.8, 0.6, 0.0);

        let jac = compute_joint_jacobian(&model, &q, 2)
            .rows(3, 3).into_owned();

        // Project to XY plane (2-DoF)
        let mut proj = DMatrix::<f64>::zeros(2, 3);
        proj[(0, 0)] = 1.0; // X
        proj[(1, 1)] = 1.0; // Y

        let cfg = DiffIkConfig {
            task_projection: Some(proj),
            ..DiffIkConfig::default()
        };
        let result = differential_ik_step(&jac, &ee, &target, &cfg);
        assert_eq!(result.dq.len(), 2);
        assert!(result.error_before > 0.0);
    }

    #[test]
    fn differential_ik_step_with_weights() {
        let model = two_link_planar();
        let q = vec![0.0, 0.0];
        let data = forward_kinematics(&model, &q);
        let ee = se3::translation(&data.oMi[2]);
        let target = Vector3::new(0.8, 0.6, 0.0);

        let jac = compute_joint_jacobian(&model, &q, 2)
            .rows(3, 3).into_owned();

        // Without weights
        let res_unif = differential_ik_step(&jac, &ee, &target, &DiffIkConfig::default());

        // With weights: j0 very expensive
        let res_weighted = differential_ik_step(&jac, &ee, &target, &DiffIkConfig {
            joint_weights: Some(JointWeights { weights: vec![100.0, 1.0] }),
            ..DiffIkConfig::default()
        });

        assert!(
            res_weighted.dq[0].abs() < res_unif.dq[0].abs(),
            "Weighted dq0={:.6} should be smaller than uniform dq0={:.6}",
            res_weighted.dq[0].abs(), res_unif.dq[0].abs(),
        );
    }

    #[test]
    fn jt_differential_ik_step() {
        let model = two_link_planar();
        let q = vec![0.3, -0.5];
        let data = forward_kinematics(&model, &q);
        let ee = se3::translation(&data.oMi[2]);
        let target = Vector3::new(0.8, 0.6, 0.0);

        let jac = compute_joint_jacobian(&model, &q, 2)
            .rows(3, 3).into_owned();

        let cfg = DiffIkConfig {
            solver_method: SolverMethod::JacobianTranspose,
            ..DiffIkConfig::default()
        };
        let result = differential_ik_step(&jac, &ee, &target, &cfg);
        assert_eq!(result.dq.len(), 2);

        // Apply and check error decreased
        let q_new: Vec<f64> = q.iter().zip(&result.dq)
            .map(|(qi, dqi)| qi + dqi)
            .collect();
        let data2 = forward_kinematics(&model, &q_new);
        let ee2 = se3::translation(&data2.oMi[2]);
        assert!((target - ee2).norm() < result.error_before);
    }

    #[test]
    fn differential_ik_with_constraint_reduces_both_errors() {
        // 3-link planar chain: joint 1 → link 1, joint 2 → link 2 (EE)
        let model = two_link_planar();
        let q = vec![0.3, -0.5];
        let data = forward_kinematics(&model, &q);
        let ee = se3::translation(&data.oMi[2]);
        let mid = se3::translation(&data.oMi[1]);
        let target = Vector3::new(0.8, 0.6, 0.0);

        // Primary: move EE (joint 2) toward target
        let jac_ee = compute_joint_jacobian(&model, &q, 2)
            .rows(3, 3).into_owned();

        // Constraint: pin midpoint (joint 1) to its current position
        let jac_mid = compute_joint_jacobian(&model, &q, 1)
            .rows(3, 3).into_owned();
        let mid_error = mid - mid; // already at target → zero error
        let constraint = DiffIkConstraint {
            jacobian: jac_mid.clone(),
            error: DVector::from_column_slice(&[mid_error.x, mid_error.y, mid_error.z]),
            weight: 10.0,
        };

        let cfg = DiffIkConfig::default();
        let result = differential_ik_step_with_constraints(
            &jac_ee, &ee, &target, &[constraint], &cfg,
        );
        assert_eq!(result.dq.len(), 2);

        // EE error should decrease
        let q_new: Vec<f64> = q.iter().zip(&result.dq)
            .map(|(qi, dqi)| qi + dqi).collect();
        let data2 = forward_kinematics(&model, &q_new);
        let ee2 = se3::translation(&data2.oMi[2]);
        assert!((target - ee2).norm() < result.error_before,
            "EE should get closer to target");

        // Mid link should barely move (pinned with high weight)
        let mid2 = se3::translation(&data2.oMi[1]);
        let mid_drift = (mid2 - mid).norm();
        assert!(mid_drift < 0.02,
            "Pinned link drifted {mid_drift:.4}, expected < 0.02");
    }
}
