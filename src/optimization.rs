//! Optimization-oriented interfaces (Phase 5).
//!
//! This module provides compact, MPC-friendly linearization helpers:
//! - Kinematic task residual/Jacobian interfaces for FK-based tasks.
//! - Discrete dynamics linearization `(A, B)` for state-space MPC.
//!
//! The configuration manifold is handled through `manifold::integrate` and
//! `manifold::difference`, so free-flyer and revolute wrapping are respected.

use crate::aba::aba;
use crate::fk::forward_kinematics;
use crate::jacobian::compute_joint_jacobian;
use crate::limits::{self, JointLimits};
use crate::manifold;
use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, DVector, Vector3};

#[derive(Debug, Clone)]
pub struct ResidualLinearization {
    pub residual: DVector<f64>,
    pub jacobian: DMatrix<f64>,
}

#[derive(Debug, Clone)]
pub struct DiscreteDynamicsLinearization {
    /// State Jacobian in tangent coordinates.
    ///
    /// State is represented as `x = [δq (nv); v (nv)]`, so `A` is `2nv × 2nv`.
    pub a: DMatrix<f64>,
    /// Input Jacobian (`2nv × nv`) for torque input `tau`.
    pub b: DMatrix<f64>,
    /// Nominal next configuration from `(q, v, tau)`.
    pub q_next: Vec<f64>,
    /// Nominal next velocity from `(q, v, tau)`.
    pub v_next: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StageQuadraticApproximation {
    /// Scalar stage cost value at the expansion point.
    pub l0: f64,
    /// First derivative w.r.t. state tangent `x = [δq; v]` (size `2nv`).
    pub lx: DVector<f64>,
    /// First derivative w.r.t. input `u` (size `nv`).
    pub lu: DVector<f64>,
    /// Second derivative w.r.t. state (`2nv × 2nv`).
    pub lxx: DMatrix<f64>,
    /// Second derivative w.r.t. input (`nv × nv`).
    pub luu: DMatrix<f64>,
    /// Cross derivative (`nv × 2nv`) = ∂²l/∂u∂x.
    pub lux: DMatrix<f64>,
}

#[derive(Debug, Clone)]
pub struct TerminalQuadraticApproximation {
    /// Scalar terminal cost value.
    pub l0: f64,
    /// Terminal gradient w.r.t. tangent state (`2nv`).
    pub lx: DVector<f64>,
    /// Terminal Hessian w.r.t. tangent state (`2nv × 2nv`).
    pub lxx: DMatrix<f64>,
}

#[derive(Debug, Clone)]
pub struct StageLinearQuadraticModel {
    /// Linearized discrete dynamics around nominal `(q_k, v_k, u_k)`.
    pub dynamics: DiscreteDynamicsLinearization,
    /// Quadratic stage-cost approximation at `(q_k, v_k, u_k)`.
    pub cost: StageQuadraticApproximation,
}

#[derive(Debug, Clone)]
pub struct HorizonLinearQuadraticModel {
    /// Number of stages (control horizon length).
    pub horizon: usize,
    /// Nominal state trajectory of length `horizon + 1`.
    pub q_nominal: Vec<Vec<f64>>,
    pub v_nominal: Vec<Vec<f64>>,
    /// Per-stage linear/quadratic models, length `horizon`.
    pub stages: Vec<StageLinearQuadraticModel>,
    /// Terminal cost approximation at state `N`.
    pub terminal: TerminalQuadraticApproximation,
}

#[derive(Debug, Clone)]
pub struct IlqrConfig {
    pub max_iters: usize,
    pub tol_cost: f64,
    pub regularization: f64,
    pub reg_min: f64,
    pub reg_max: f64,
    pub alphas: Vec<f64>,
    /// Optional lower bounds for control input (size nv).
    pub u_min: Option<Vec<f64>>,
    /// Optional upper bounds for control input (size nv).
    pub u_max: Option<Vec<f64>>,
    /// Optional state limits projected during rollout.
    pub joint_limits: Option<JointLimits>,
}

impl Default for IlqrConfig {
    fn default() -> Self {
        Self {
            max_iters: 30,
            tol_cost: 1e-8,
            regularization: 1e-4,
            reg_min: 1e-8,
            reg_max: 1e8,
            alphas: vec![1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
            u_min: None,
            u_max: None,
            joint_limits: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IlqrResult {
    pub u_seq: Vec<Vec<f64>>,
    pub q_seq: Vec<Vec<f64>>,
    pub v_seq: Vec<Vec<f64>>,
    pub cost_initial: f64,
    pub cost_final: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn rollout_controls(
    model: &Model<f64>,
    q0: &[f64],
    v0: &[f64],
    u_seq: &[Vec<f64>],
    dt: f64,
    joint_limits: Option<&JointLimits>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = u_seq.len();
    let mut q_seq = Vec::with_capacity(n + 1);
    let mut v_seq = Vec::with_capacity(n + 1);

    let mut q_init = q0.to_vec();
    let mut v_init = v0.to_vec();
    if let Some(lim) = joint_limits {
        q_init = limits::clamp_configuration(model, &q_init, lim);
        v_init = limits::saturate_velocity(model, &v_init, lim);
    }

    q_seq.push(q_init);
    v_seq.push(v_init);

    for k in 0..n {
        assert_eq!(u_seq[k].len(), model.nv);
        let (mut q_next, mut v_next) =
            discrete_dynamics_step(model, &q_seq[k], &v_seq[k], &u_seq[k], dt);
        if let Some(lim) = joint_limits {
            q_next = limits::clamp_configuration(model, &q_next, lim);
            v_next = limits::saturate_velocity(model, &v_next, lim);
        }
        q_seq.push(q_next);
        v_seq.push(v_next);
    }

    (q_seq, v_seq)
}

fn clamp_control_in_place(u: &mut [f64], u_min: Option<&[f64]>, u_max: Option<&[f64]>) {
    if let Some(min_v) = u_min {
        assert_eq!(min_v.len(), u.len());
    }
    if let Some(max_v) = u_max {
        assert_eq!(max_v.len(), u.len());
    }
    for i in 0..u.len() {
        if let Some(min_v) = u_min {
            u[i] = u[i].max(min_v[i]);
        }
        if let Some(max_v) = u_max {
            u[i] = u[i].min(max_v[i]);
        }
    }
}

fn evaluate_trajectory_cost(
    model: &Model<f64>,
    q_seq: &[Vec<f64>],
    v_seq: &[Vec<f64>],
    u_seq: &[Vec<f64>],
    q_ref_seq: &[Vec<f64>],
    v_ref_seq: &[Vec<f64>],
    u_ref_seq: &[Vec<f64>],
    q_weight: &DMatrix<f64>,
    r_weight: &DMatrix<f64>,
    qf_weight: &DMatrix<f64>,
) -> f64 {
    let n = u_seq.len();
    let mut total = 0.0_f64;

    for k in 0..n {
        total += quadratic_stage_cost_approximation(
            model,
            &q_seq[k],
            &v_seq[k],
            &u_seq[k],
            &q_ref_seq[k],
            &v_ref_seq[k],
            &u_ref_seq[k],
            q_weight,
            r_weight,
        )
        .l0;
    }

    total
        + quadratic_terminal_cost_approximation(
            model,
            &q_seq[n],
            &v_seq[n],
            &q_ref_seq[n],
            &v_ref_seq[n],
            qf_weight,
        )
        .l0
}

/// Minimal iLQR solver built on this module's linearization/quadratization APIs.
///
/// This function optimizes a control sequence `u_seq` for the finite-horizon
/// problem defined by dynamics from [`discrete_dynamics_step`] and quadratic
/// tracking costs (`Q`, `R`, `Qf`) in tangent coordinates.
///
/// - Backward pass computes feedforward `k_k` and feedback `K_k` gains.
/// - Forward pass performs line-search rollout with `u_k = ū_k + α k_k + K_k δx_k`.
pub fn solve_ilqr(
    model: &Model<f64>,
    q0: &[f64],
    v0: &[f64],
    u_init: &[Vec<f64>],
    q_ref_seq: &[Vec<f64>],
    v_ref_seq: &[Vec<f64>],
    u_ref_seq: &[Vec<f64>],
    q_weight: &DMatrix<f64>,
    r_weight: &DMatrix<f64>,
    qf_weight: &DMatrix<f64>,
    dt: f64,
    eps: f64,
    config: &IlqrConfig,
) -> IlqrResult {
    let n = u_init.len();
    assert!(n > 0);
    assert_eq!(q_ref_seq.len(), n + 1);
    assert_eq!(v_ref_seq.len(), n + 1);
    assert_eq!(u_ref_seq.len(), n);

    let nx = 2 * model.nv;
    let nu = model.nv;

    if let Some(u_min) = &config.u_min {
        assert_eq!(u_min.len(), nu);
    }
    if let Some(u_max) = &config.u_max {
        assert_eq!(u_max.len(), nu);
    }
    if let (Some(u_min), Some(u_max)) = (&config.u_min, &config.u_max) {
        for i in 0..nu {
            assert!(u_min[i] <= u_max[i], "u_min[{}] > u_max[{}]", i, i);
        }
    }
    if let Some(lim) = &config.joint_limits {
        lim.validate(model);
    }

    let mut u_seq = u_init.to_vec();
    for u in &mut u_seq {
        assert_eq!(u.len(), nu);
        clamp_control_in_place(
            u,
            config.u_min.as_deref(),
            config.u_max.as_deref(),
        );
    }

    let (mut q_seq, mut v_seq) = rollout_controls(model, q0, v0, &u_seq, dt, config.joint_limits.as_ref());
    let mut cost = evaluate_trajectory_cost(
        model,
        &q_seq,
        &v_seq,
        &u_seq,
        q_ref_seq,
        v_ref_seq,
        u_ref_seq,
        q_weight,
        r_weight,
        qf_weight,
    );
    let cost_initial = cost;

    let mut reg = config.regularization;
    let mut converged = false;
    let mut iters_done = 0usize;

    for iter in 0..config.max_iters {
        iters_done = iter + 1;

        let lq = build_horizon_linear_quadratic_model(
            model,
            q0,
            v0,
            &u_seq,
            q_ref_seq,
            v_ref_seq,
            u_ref_seq,
            q_weight,
            r_weight,
            qf_weight,
            dt,
            eps,
        );

        let mut k_ff: Vec<DVector<f64>> = vec![DVector::zeros(nu); n];
        let mut k_fb: Vec<DMatrix<f64>> = vec![DMatrix::zeros(nu, nx); n];

        let mut vx = lq.terminal.lx.clone();
        let mut vxx = lq.terminal.lxx.clone();

        let mut backward_ok = true;

        for kk in (0..n).rev() {
            let a = &lq.stages[kk].dynamics.a;
            let b = &lq.stages[kk].dynamics.b;
            let c = &lq.stages[kk].cost;

            let qx = &c.lx + a.transpose() * &vx;
            let qu = &c.lu + b.transpose() * &vx;
            let qxx = &c.lxx + a.transpose() * &vxx * a;
            let mut quu = &c.luu + b.transpose() * &vxx * b;
            let qux = &c.lux + b.transpose() * &vxx * a;

            for i in 0..nu {
                quu[(i, i)] += reg;
            }

            let lu = quu.clone().lu();
            let Some(inv_qu) = lu.solve(&DMatrix::<f64>::identity(nu, nu)) else {
                backward_ok = false;
                break;
            };

            let k = -&inv_qu * qu.clone();
            let k_mat = -&inv_qu * qux.clone();

            let vx_new =
                qx + k_mat.transpose() * &quu * &k + k_mat.transpose() * &qu + qux.transpose() * &k;
            let mut vxx_new =
                qxx + k_mat.transpose() * &quu * &k_mat + k_mat.transpose() * &qux + qux.transpose() * &k_mat;
            vxx_new = 0.5 * (&vxx_new + vxx_new.transpose());

            k_ff[kk] = k;
            k_fb[kk] = k_mat;
            vx = vx_new;
            vxx = vxx_new;
        }

        if !backward_ok {
            reg = (reg * 10.0).min(config.reg_max);
            if reg >= config.reg_max {
                break;
            }
            continue;
        }

        let mut accepted = false;
        let mut best_cost = cost;
        let mut best_u = u_seq.clone();
        let mut best_q = q_seq.clone();
        let mut best_v = v_seq.clone();

        for &alpha in &config.alphas {
            let mut cand_u: Vec<Vec<f64>> = vec![vec![0.0; nu]; n];
            let mut cand_q: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
            let mut cand_v: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
            cand_q.push(q0.to_vec());
            cand_v.push(v0.to_vec());

            let mut dx = DVector::<f64>::zeros(nx);

            for k in 0..n {
                let du = alpha * &k_ff[k] + &k_fb[k] * &dx;
                for i in 0..nu {
                    cand_u[k][i] = u_seq[k][i] + du[i];
                }
                clamp_control_in_place(
                    &mut cand_u[k],
                    config.u_min.as_deref(),
                    config.u_max.as_deref(),
                );

                let (mut q_next, mut v_next) =
                    discrete_dynamics_step(model, &cand_q[k], &cand_v[k], &cand_u[k], dt);
                if let Some(lim) = config.joint_limits.as_ref() {
                    q_next = limits::clamp_configuration(model, &q_next, lim);
                    v_next = limits::saturate_velocity(model, &v_next, lim);
                }
                cand_q.push(q_next);
                cand_v.push(v_next);

                let dq_dev = manifold::difference(model, &lq.q_nominal[k + 1], &cand_q[k + 1]);
                dx = DVector::<f64>::zeros(nx);
                for i in 0..nu {
                    dx[i] = dq_dev[i];
                    dx[nu + i] = cand_v[k + 1][i] - lq.v_nominal[k + 1][i];
                }
            }

            let cand_cost = evaluate_trajectory_cost(
                model,
                &cand_q,
                &cand_v,
                &cand_u,
                q_ref_seq,
                v_ref_seq,
                u_ref_seq,
                q_weight,
                r_weight,
                qf_weight,
            );

            if cand_cost < best_cost {
                accepted = true;
                best_cost = cand_cost;
                best_u = cand_u;
                best_q = cand_q;
                best_v = cand_v;
            }
        }

        if accepted {
            let improvement = cost - best_cost;
            u_seq = best_u;
            q_seq = best_q;
            v_seq = best_v;
            cost = best_cost;
            reg = (reg * 0.5).max(config.reg_min);

            if improvement.abs() <= config.tol_cost {
                converged = true;
                break;
            }
        } else {
            reg = (reg * 10.0).min(config.reg_max);
            if reg >= config.reg_max {
                break;
            }
        }
    }

    IlqrResult {
        u_seq,
        q_seq,
        v_seq,
        cost_initial,
        cost_final: cost,
        iterations: iters_done,
        converged,
    }
}

/// Build tangent-state error `x_err = [q_ref ⊖ q; v - v_ref]`.
pub fn state_error_tangent(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    q_ref: &[f64],
    v_ref: &[f64],
) -> DVector<f64> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(q_ref.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(v_ref.len(), model.nv);

    let dq = manifold::difference(model, q_ref, q);
    let mut out = DVector::<f64>::zeros(2 * model.nv);
    for i in 0..model.nv {
        out[i] = dq[i];
        out[model.nv + i] = v[i] - v_ref[i];
    }
    out
}

/// Quadratic stage-cost approximation for MPC / iLQR style solvers.
///
/// Defines
///
/// `l(x,u) = 0.5 * x_err^T Q x_err + 0.5 * u_err^T R u_err`
///
/// where
///
/// - `x_err = [q_ref ⊖ q; v - v_ref]` in tangent coordinates (`2nv`)
/// - `u_err = u - u_ref` (`nv`)
///
/// and returns first/second derivatives:
///
/// - `l_x = Q x_err`
/// - `l_u = R u_err`
/// - `l_xx = Q`, `l_uu = R`, `l_ux = 0`
pub fn quadratic_stage_cost_approximation(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    u: &[f64],
    q_ref: &[f64],
    v_ref: &[f64],
    u_ref: &[f64],
    q_weight: &DMatrix<f64>,
    r_weight: &DMatrix<f64>,
) -> StageQuadraticApproximation {
    assert_eq!(u.len(), model.nv);
    assert_eq!(u_ref.len(), model.nv);

    let nx = 2 * model.nv;
    assert_eq!(q_weight.nrows(), nx);
    assert_eq!(q_weight.ncols(), nx);
    assert_eq!(r_weight.nrows(), model.nv);
    assert_eq!(r_weight.ncols(), model.nv);

    let x_err = state_error_tangent(model, q, v, q_ref, v_ref);
    let u_err = DVector::from_iterator(model.nv, (0..model.nv).map(|i| u[i] - u_ref[i]));

    let qx = q_weight * &x_err;
    let ru = r_weight * &u_err;

    let l0 = 0.5 * x_err.dot(&qx) + 0.5 * u_err.dot(&ru);
    let lx = qx;
    let lu = ru;
    let lxx = q_weight.clone();
    let luu = r_weight.clone();
    let lux = DMatrix::<f64>::zeros(model.nv, nx);

    StageQuadraticApproximation {
        l0,
        lx,
        lu,
        lxx,
        luu,
        lux,
    }
}

/// Quadratic terminal cost approximation (no control term):
///
/// `l_f(x_N) = 0.5 * x_err^T Q_f x_err`
pub fn quadratic_terminal_cost_approximation(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    q_ref: &[f64],
    v_ref: &[f64],
    qf_weight: &DMatrix<f64>,
) -> TerminalQuadraticApproximation {
    let nx = 2 * model.nv;
    assert_eq!(qf_weight.nrows(), nx);
    assert_eq!(qf_weight.ncols(), nx);

    let x_err = state_error_tangent(model, q, v, q_ref, v_ref);
    let qx = qf_weight * &x_err;
    let l0 = 0.5 * x_err.dot(&qx);

    TerminalQuadraticApproximation {
        l0,
        lx: qx,
        lxx: qf_weight.clone(),
    }
}

/// Build a horizon-wide linearized/quadratized model for iLQR/DDP-like solvers.
///
/// Given initial state `(q0, v0)` and a nominal control sequence `u_seq`, this
/// function rolls out nominal dynamics and returns:
///
/// - stage dynamics linearization `(A_k, B_k)`
/// - stage quadratic cost terms `(l0, l_x, l_u, l_xx, l_uu, l_ux)`
/// - terminal quadratic cost terms `(l0_f, l_x_f, l_xx_f)`
///
/// Sequence lengths must satisfy:
/// - `u_seq.len() = N`
/// - `q_ref_seq.len() = N + 1`
/// - `v_ref_seq.len() = N + 1`
/// - `u_ref_seq.len() = N`
pub fn build_horizon_linear_quadratic_model(
    model: &Model<f64>,
    q0: &[f64],
    v0: &[f64],
    u_seq: &[Vec<f64>],
    q_ref_seq: &[Vec<f64>],
    v_ref_seq: &[Vec<f64>],
    u_ref_seq: &[Vec<f64>],
    q_weight: &DMatrix<f64>,
    r_weight: &DMatrix<f64>,
    qf_weight: &DMatrix<f64>,
    dt: f64,
    eps: f64,
) -> HorizonLinearQuadraticModel {
    let n = u_seq.len();
    assert_eq!(q_ref_seq.len(), n + 1);
    assert_eq!(v_ref_seq.len(), n + 1);
    assert_eq!(u_ref_seq.len(), n);
    assert_eq!(q0.len(), model.nq);
    assert_eq!(v0.len(), model.nv);

    let nx = 2 * model.nv;
    assert_eq!(q_weight.nrows(), nx);
    assert_eq!(q_weight.ncols(), nx);
    assert_eq!(r_weight.nrows(), model.nv);
    assert_eq!(r_weight.ncols(), model.nv);
    assert_eq!(qf_weight.nrows(), nx);
    assert_eq!(qf_weight.ncols(), nx);

    let mut q_nominal = Vec::with_capacity(n + 1);
    let mut v_nominal = Vec::with_capacity(n + 1);
    q_nominal.push(q0.to_vec());
    v_nominal.push(v0.to_vec());

    let mut stages = Vec::with_capacity(n);

    for k in 0..n {
        let qk = q_nominal[k].clone();
        let vk = v_nominal[k].clone();
        let uk = &u_seq[k];

        assert_eq!(uk.len(), model.nv);
        assert_eq!(u_ref_seq[k].len(), model.nv);
        assert_eq!(q_ref_seq[k].len(), model.nq);
        assert_eq!(v_ref_seq[k].len(), model.nv);

        let dynamics = linearize_discrete_dynamics(model, &qk, &vk, uk, dt, eps);
        let cost = quadratic_stage_cost_approximation(
            model,
            &qk,
            &vk,
            uk,
            &q_ref_seq[k],
            &v_ref_seq[k],
            &u_ref_seq[k],
            q_weight,
            r_weight,
        );

        q_nominal.push(dynamics.q_next.clone());
        v_nominal.push(dynamics.v_next.clone());
        stages.push(StageLinearQuadraticModel { dynamics, cost });
    }

    let terminal = quadratic_terminal_cost_approximation(
        model,
        &q_nominal[n],
        &v_nominal[n],
        &q_ref_seq[n],
        &v_ref_seq[n],
        qf_weight,
    );

    HorizonLinearQuadraticModel {
        horizon: n,
        q_nominal,
        v_nominal,
        stages,
        terminal,
    }
}

/// Linearize a joint position tracking residual using analytical Jacobian.
///
/// Residual is `r(q) = target - p_joint(q)` (3D), and Jacobian is
/// `J_r = -J_lin` where `J_lin` is the linear (rows 3..5) part of the joint
/// geometric Jacobian.
pub fn linearize_joint_position_residual(
    model: &Model<f64>,
    q: &[f64],
    joint_idx: usize,
    target: Vector3<f64>,
) -> ResidualLinearization {
    assert!(joint_idx > 0 && joint_idx < model.joints.len());

    let data = forward_kinematics(model, q);
    let p = se3::translation(&data.oMi[joint_idx]);
    let residual_v = target - p;
    let residual = DVector::from_vec(vec![residual_v[0], residual_v[1], residual_v[2]]);

    let j_full = compute_joint_jacobian(model, q, joint_idx);
    let j_lin = j_full.rows(3, 3).into_owned();
    let jacobian = -j_lin;

    ResidualLinearization { residual, jacobian }
}

/// Linearize a joint position tracking residual using central finite differences.
///
/// Uses manifold-consistent perturbation:
/// `q± = integrate(q, ±eps * e_i, dt=1)`.
pub fn linearize_joint_position_residual_numerical(
    model: &Model<f64>,
    q: &[f64],
    joint_idx: usize,
    target: Vector3<f64>,
    eps: f64,
) -> ResidualLinearization {
    assert!(joint_idx > 0 && joint_idx < model.joints.len());

    let data = forward_kinematics(model, q);
    let p = se3::translation(&data.oMi[joint_idx]);
    let residual_v = target - p;
    let residual = DVector::from_vec(vec![residual_v[0], residual_v[1], residual_v[2]]);

    let mut jacobian = DMatrix::<f64>::zeros(3, model.nv);
    for col in 0..model.nv {
        let mut dq_plus = vec![0.0_f64; model.nv];
        let mut dq_minus = vec![0.0_f64; model.nv];
        dq_plus[col] = eps;
        dq_minus[col] = -eps;

        let q_plus = manifold::integrate(model, q, &dq_plus, 1.0);
        let q_minus = manifold::integrate(model, q, &dq_minus, 1.0);

        let p_plus = se3::translation(&forward_kinematics(model, &q_plus).oMi[joint_idx]);
        let p_minus = se3::translation(&forward_kinematics(model, &q_minus).oMi[joint_idx]);

        // r = target - p
        let dr = -(p_plus - p_minus) / (2.0 * eps);
        jacobian[(0, col)] = dr[0];
        jacobian[(1, col)] = dr[1];
        jacobian[(2, col)] = dr[2];
    }

    ResidualLinearization { residual, jacobian }
}

/// Semi-implicit Euler step used by the MPC linearization.
///
/// - `a = aba(model, q, v, tau)`
/// - `v_next = v + dt * a`
/// - `q_next = integrate(model, q, v_next, dt)`
pub fn discrete_dynamics_step(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    tau: &[f64],
    dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(tau.len(), model.nv);

    let a = aba(model, q, v, tau);
    let mut v_next = v.to_vec();
    for i in 0..model.nv {
        v_next[i] += dt * a[i];
    }
    let q_next = manifold::integrate(model, q, &v_next, dt);
    (q_next, v_next)
}

/// Linearize discrete dynamics around `(q, v, tau)` for MPC.
///
/// Returns tangent-space Jacobians:
///
/// - `δx_next = A δx + B δu`
/// - `δx = [δq; δv]` with `δq` in tangent coordinates (`nv` dims)
/// - `δu = δtau`
///
/// Both `A` and `B` are computed by central finite differences.
pub fn linearize_discrete_dynamics(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    tau: &[f64],
    dt: f64,
    eps: f64,
) -> DiscreteDynamicsLinearization {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(tau.len(), model.nv);

    let nv = model.nv;
    let nx = 2 * nv;

    let (q_next_nom, v_next_nom) = discrete_dynamics_step(model, q, v, tau, dt);
    let mut a_mat = DMatrix::<f64>::zeros(nx, nx);
    let mut b_mat = DMatrix::<f64>::zeros(nx, nv);

    // A = ∂x_next/∂x
    for col in 0..nx {
        let mut q_plus = q.to_vec();
        let mut q_minus = q.to_vec();
        let mut v_plus = v.to_vec();
        let mut v_minus = v.to_vec();

        if col < nv {
            // Perturb configuration along tangent basis direction.
            let mut dq = vec![0.0_f64; nv];
            dq[col] = eps;
            q_plus = manifold::integrate(model, q, &dq, 1.0);

            dq[col] = -eps;
            q_minus = manifold::integrate(model, q, &dq, 1.0);
        } else {
            let k = col - nv;
            v_plus[k] += eps;
            v_minus[k] -= eps;
        }

        let (q_np, v_np) = discrete_dynamics_step(model, &q_plus, &v_plus, tau, dt);
        let (q_nm, v_nm) = discrete_dynamics_step(model, &q_minus, &v_minus, tau, dt);

        let dq_np = manifold::difference(model, &q_next_nom, &q_np);
        let dq_nm = manifold::difference(model, &q_next_nom, &q_nm);

        for r in 0..nv {
            a_mat[(r, col)] = (dq_np[r] - dq_nm[r]) / (2.0 * eps);
            a_mat[(nv + r, col)] = (v_np[r] - v_nm[r]) / (2.0 * eps);
        }
    }

    // B = ∂x_next/∂u
    for col in 0..nv {
        let mut tau_plus = tau.to_vec();
        let mut tau_minus = tau.to_vec();
        tau_plus[col] += eps;
        tau_minus[col] -= eps;

        let (q_np, v_np) = discrete_dynamics_step(model, q, v, &tau_plus, dt);
        let (q_nm, v_nm) = discrete_dynamics_step(model, q, v, &tau_minus, dt);

        let dq_np = manifold::difference(model, &q_next_nom, &q_np);
        let dq_nm = manifold::difference(model, &q_next_nom, &q_nm);

        for r in 0..nv {
            b_mat[(r, col)] = (dq_np[r] - dq_nm[r]) / (2.0 * eps);
            b_mat[(nv + r, col)] = (v_np[r] - v_nm[r]) / (2.0 * eps);
        }
    }

    DiscreteDynamicsLinearization {
        a: a_mat,
        b: b_mat,
        q_next: q_next_nom,
        v_next: v_next_nom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::limits::JointLimits;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Rotation3};

    fn simple_revolute_model() -> Model<f64> {
        let inertia = LinkInertia {
            mass: 1.0,
            center_of_mass: Vector3::new(0.0, 0.0, 0.0),
            rotational_inertia: Matrix3::identity() * 0.1,
        };

        ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::revolute_z(),
                se3::from_rotation_and_translation(
                    &Rotation3::identity(),
                    &Vector3::new(0.0, 0.0, 0.0),
                ),
                inertia,
            )
            .build()
    }

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
                se3::from_rotation_and_translation(
                    &Rotation3::identity(),
                    &Vector3::new(1.0, 0.0, 0.0),
                ),
                LinkInertia::zero(),
            )
            .build()
    }

    #[test]
    fn joint_position_residual_analytic_matches_numerical() {
        let model = two_link_planar();
        let q = vec![0.3, -0.2];
        let target = Vector3::new(0.7, 0.5, 0.0);

        let a = linearize_joint_position_residual(&model, &q, 2, target);
        let n = linearize_joint_position_residual_numerical(&model, &q, 2, target, 1e-6);

        assert_relative_eq!(a.residual, n.residual, epsilon = 1e-12);
        for r in 0..3 {
            for c in 0..model.nv {
                assert_relative_eq!(a.jacobian[(r, c)], n.jacobian[(r, c)], epsilon = 5e-5);
            }
        }
    }

    #[test]
    fn dynamics_linearization_dimensions() {
        let model = simple_revolute_model();
        let q = vec![0.2];
        let v = vec![0.1];
        let tau = vec![0.0];

        let lin = linearize_discrete_dynamics(&model, &q, &v, &tau, 0.01, 1e-6);
        assert_eq!(lin.a.nrows(), 2);
        assert_eq!(lin.a.ncols(), 2);
        assert_eq!(lin.b.nrows(), 2);
        assert_eq!(lin.b.ncols(), 1);
        assert_eq!(lin.q_next.len(), model.nq);
        assert_eq!(lin.v_next.len(), model.nv);
    }

    #[test]
    fn dynamics_linearization_predicts_small_perturbation() {
        let model = simple_revolute_model();
        let q = vec![0.2];
        let v = vec![0.1];
        let tau = vec![0.05];

        let dt = 0.01;
        let eps = 1e-6;
        let lin = linearize_discrete_dynamics(&model, &q, &v, &tau, dt, eps);

        // Small perturbation in tangent state and input.
        let dx = DVector::from_vec(vec![1e-5, -2e-5]); // [δq, δv]
        let du = DVector::from_vec(vec![3e-5]);

        // Build perturbed inputs on manifold.
        let q_pert = manifold::integrate(&model, &q, &[dx[0]], 1.0);
        let v_pert = vec![v[0] + dx[1]];
        let tau_pert = vec![tau[0] + du[0]];

        let (q_nom_n, v_nom_n) = discrete_dynamics_step(&model, &q, &v, &tau, dt);
        let (q_pert_n, v_pert_n) =
            discrete_dynamics_step(&model, &q_pert, &v_pert, &tau_pert, dt);

        let dq_actual = manifold::difference(&model, &q_nom_n, &q_pert_n);
        let dv_actual = v_pert_n[0] - v_nom_n[0];
        let dx_actual = DVector::from_vec(vec![dq_actual[0], dv_actual]);

        let dx_pred = &lin.a * dx + &lin.b * du;

        assert_relative_eq!(dx_pred[0], dx_actual[0], epsilon = 2e-6);
        assert_relative_eq!(dx_pred[1], dx_actual[1], epsilon = 2e-6);
    }

    #[test]
    fn quadratic_stage_cost_shapes_and_hessians() {
        let model = simple_revolute_model();
        let q = vec![0.2];
        let v = vec![-0.3];
        let u = vec![0.1];
        let q_ref = vec![0.0];
        let v_ref = vec![0.0];
        let u_ref = vec![0.0];

        let q_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![10.0, 1.0]));
        let r_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![0.5]));

        let qa = quadratic_stage_cost_approximation(
            &model, &q, &v, &u, &q_ref, &v_ref, &u_ref, &q_weight, &r_weight,
        );

        assert_eq!(qa.lx.len(), 2);
        assert_eq!(qa.lu.len(), 1);
        assert_eq!(qa.lxx.nrows(), 2);
        assert_eq!(qa.lxx.ncols(), 2);
        assert_eq!(qa.luu.nrows(), 1);
        assert_eq!(qa.luu.ncols(), 1);
        assert_eq!(qa.lux.nrows(), 1);
        assert_eq!(qa.lux.ncols(), 2);

        assert_relative_eq!(qa.lxx[(0, 0)], q_weight[(0, 0)], epsilon = 1e-12);
        assert_relative_eq!(qa.lxx[(1, 1)], q_weight[(1, 1)], epsilon = 1e-12);
        assert_relative_eq!(qa.luu[(0, 0)], r_weight[(0, 0)], epsilon = 1e-12);
        assert_relative_eq!(qa.lux[(0, 0)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(qa.lux[(0, 1)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn quadratic_stage_cost_gradients_match_finite_difference() {
        let model = simple_revolute_model();
        let q = vec![0.25];
        let v = vec![0.15];
        let u = vec![-0.07];
        let q_ref = vec![0.05];
        let v_ref = vec![0.0];
        let u_ref = vec![0.0];

        let q_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![4.0, 2.0]));
        let r_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![3.0]));

        let qa = quadratic_stage_cost_approximation(
            &model, &q, &v, &u, &q_ref, &v_ref, &u_ref, &q_weight, &r_weight,
        );

        let eps = 1e-7;

        // Finite-difference for x = [δq, δv]
        // x[0] direction perturbs configuration on manifold.
        let q_plus = manifold::integrate(&model, &q, &[eps], 1.0);
        let q_minus = manifold::integrate(&model, &q, &[-eps], 1.0);
        let l_plus_q = quadratic_stage_cost_approximation(
            &model, &q_plus, &v, &u, &q_ref, &v_ref, &u_ref, &q_weight, &r_weight,
        )
        .l0;
        let l_minus_q = quadratic_stage_cost_approximation(
            &model, &q_minus, &v, &u, &q_ref, &v_ref, &u_ref, &q_weight, &r_weight,
        )
        .l0;
        let gx_q_fd = (l_plus_q - l_minus_q) / (2.0 * eps);

        // x[1] direction perturbs velocity.
        let l_plus_v = quadratic_stage_cost_approximation(
            &model,
            &q,
            &[v[0] + eps],
            &u,
            &q_ref,
            &v_ref,
            &u_ref,
            &q_weight,
            &r_weight,
        )
        .l0;
        let l_minus_v = quadratic_stage_cost_approximation(
            &model,
            &q,
            &[v[0] - eps],
            &u,
            &q_ref,
            &v_ref,
            &u_ref,
            &q_weight,
            &r_weight,
        )
        .l0;
        let gx_v_fd = (l_plus_v - l_minus_v) / (2.0 * eps);

        // u direction
        let l_plus_u = quadratic_stage_cost_approximation(
            &model,
            &q,
            &v,
            &[u[0] + eps],
            &q_ref,
            &v_ref,
            &u_ref,
            &q_weight,
            &r_weight,
        )
        .l0;
        let l_minus_u = quadratic_stage_cost_approximation(
            &model,
            &q,
            &v,
            &[u[0] - eps],
            &q_ref,
            &v_ref,
            &u_ref,
            &q_weight,
            &r_weight,
        )
        .l0;
        let gu_fd = (l_plus_u - l_minus_u) / (2.0 * eps);

        assert_relative_eq!(qa.lx[0], gx_q_fd, epsilon = 1e-6);
        assert_relative_eq!(qa.lx[1], gx_v_fd, epsilon = 1e-6);
        assert_relative_eq!(qa.lu[0], gu_fd, epsilon = 1e-6);
    }

    #[test]
    fn horizon_lq_model_shapes_and_rollout_length() {
        let model = simple_revolute_model();
        let q0 = vec![0.0];
        let v0 = vec![0.0];

        let u_seq = vec![vec![0.1], vec![0.0], vec![-0.1]];
        let n = u_seq.len();

        let q_ref_seq = vec![vec![0.0]; n + 1];
        let v_ref_seq = vec![vec![0.0]; n + 1];
        let u_ref_seq = vec![vec![0.0]; n];

        let q_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![5.0, 1.0]));
        let r_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![0.2]));
        let qf_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![20.0, 2.0]));

        let h = build_horizon_linear_quadratic_model(
            &model,
            &q0,
            &v0,
            &u_seq,
            &q_ref_seq,
            &v_ref_seq,
            &u_ref_seq,
            &q_weight,
            &r_weight,
            &qf_weight,
            0.01,
            1e-6,
        );

        assert_eq!(h.horizon, n);
        assert_eq!(h.stages.len(), n);
        assert_eq!(h.q_nominal.len(), n + 1);
        assert_eq!(h.v_nominal.len(), n + 1);

        for s in &h.stages {
            assert_eq!(s.dynamics.a.nrows(), 2);
            assert_eq!(s.dynamics.a.ncols(), 2);
            assert_eq!(s.dynamics.b.nrows(), 2);
            assert_eq!(s.dynamics.b.ncols(), 1);
            assert_eq!(s.cost.lx.len(), 2);
            assert_eq!(s.cost.lu.len(), 1);
            assert_eq!(s.cost.lxx.nrows(), 2);
            assert_eq!(s.cost.luu.nrows(), 1);
            assert_eq!(s.cost.lux.nrows(), 1);
            assert_eq!(s.cost.lux.ncols(), 2);
        }

        assert_eq!(h.terminal.lx.len(), 2);
        assert_eq!(h.terminal.lxx.nrows(), 2);
        assert_eq!(h.terminal.lxx.ncols(), 2);
    }

    #[test]
    fn horizon_nominal_matches_stage_dynamics_next_state() {
        let model = simple_revolute_model();
        let q0 = vec![0.1];
        let v0 = vec![0.2];

        let u_seq = vec![vec![0.03], vec![0.01]];
        let n = u_seq.len();

        let q_ref_seq = vec![vec![0.0]; n + 1];
        let v_ref_seq = vec![vec![0.0]; n + 1];
        let u_ref_seq = vec![vec![0.0]; n];

        let q_weight = DMatrix::<f64>::identity(2, 2);
        let r_weight = DMatrix::<f64>::identity(1, 1);
        let qf_weight = DMatrix::<f64>::identity(2, 2);

        let h = build_horizon_linear_quadratic_model(
            &model,
            &q0,
            &v0,
            &u_seq,
            &q_ref_seq,
            &v_ref_seq,
            &u_ref_seq,
            &q_weight,
            &r_weight,
            &qf_weight,
            0.02,
            1e-6,
        );

        for k in 0..n {
            let q_from_stage = &h.stages[k].dynamics.q_next;
            let v_from_stage = &h.stages[k].dynamics.v_next;
            let q_from_rollout = &h.q_nominal[k + 1];
            let v_from_rollout = &h.v_nominal[k + 1];

            assert_relative_eq!(q_from_stage[0], q_from_rollout[0], epsilon = 1e-12);
            assert_relative_eq!(v_from_stage[0], v_from_rollout[0], epsilon = 1e-12);
        }
    }

    #[test]
    fn ilqr_reduces_cost_on_revolute_tracking() {
        let model = simple_revolute_model();
        let q0 = vec![0.8];
        let v0 = vec![0.0];

        let n = 25usize;
        let u_init = vec![vec![0.0]; n];
        let q_ref_seq = vec![vec![0.0]; n + 1];
        let v_ref_seq = vec![vec![0.0]; n + 1];
        let u_ref_seq = vec![vec![0.0]; n];

        let q_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![30.0, 1.0]));
        let r_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![0.05]));
        let qf_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![80.0, 3.0]));

        let cfg = IlqrConfig {
            max_iters: 20,
            tol_cost: 1e-10,
            ..IlqrConfig::default()
        };

        let result = solve_ilqr(
            &model,
            &q0,
            &v0,
            &u_init,
            &q_ref_seq,
            &v_ref_seq,
            &u_ref_seq,
            &q_weight,
            &r_weight,
            &qf_weight,
            0.02,
            1e-6,
            &cfg,
        );

        assert!(
            result.cost_final < result.cost_initial,
            "iLQR should reduce cost: initial={} final={}",
            result.cost_initial,
            result.cost_final
        );
        assert_eq!(result.u_seq.len(), n);
        assert_eq!(result.q_seq.len(), n + 1);
        assert_eq!(result.v_seq.len(), n + 1);
    }

    #[test]
    fn ilqr_output_dimensions_match_model() {
        let model = simple_revolute_model();
        let q0 = vec![0.2];
        let v0 = vec![-0.1];
        let n = 8usize;

        let u_init = vec![vec![0.0]; n];
        let q_ref_seq = vec![vec![0.0]; n + 1];
        let v_ref_seq = vec![vec![0.0]; n + 1];
        let u_ref_seq = vec![vec![0.0]; n];

        let q_weight = DMatrix::<f64>::identity(2, 2);
        let r_weight = DMatrix::<f64>::identity(1, 1);
        let qf_weight = DMatrix::<f64>::identity(2, 2);

        let result = solve_ilqr(
            &model,
            &q0,
            &v0,
            &u_init,
            &q_ref_seq,
            &v_ref_seq,
            &u_ref_seq,
            &q_weight,
            &r_weight,
            &qf_weight,
            0.01,
            1e-6,
            &IlqrConfig::default(),
        );

        assert_eq!(result.u_seq.len(), n);
        assert_eq!(result.q_seq.len(), n + 1);
        assert_eq!(result.v_seq.len(), n + 1);
        for u in &result.u_seq {
            assert_eq!(u.len(), model.nv);
        }
        for q in &result.q_seq {
            assert_eq!(q.len(), model.nq);
        }
        for v in &result.v_seq {
            assert_eq!(v.len(), model.nv);
        }
    }

    #[test]
    fn ilqr_respects_input_bounds() {
        let model = simple_revolute_model();
        let q0 = vec![1.0];
        let v0 = vec![0.0];
        let n = 20usize;

        // Intentionally out-of-bounds initializer to verify internal clipping.
        let u_init = vec![vec![2.0]; n];
        let q_ref_seq = vec![vec![0.0]; n + 1];
        let v_ref_seq = vec![vec![0.0]; n + 1];
        let u_ref_seq = vec![vec![0.0]; n];

        let q_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![40.0, 2.0]));
        let r_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![0.01]));
        let qf_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![80.0, 3.0]));

        let cfg = IlqrConfig {
            max_iters: 10,
            u_min: Some(vec![-0.05]),
            u_max: Some(vec![0.05]),
            ..IlqrConfig::default()
        };

        let result = solve_ilqr(
            &model,
            &q0,
            &v0,
            &u_init,
            &q_ref_seq,
            &v_ref_seq,
            &u_ref_seq,
            &q_weight,
            &r_weight,
            &qf_weight,
            0.02,
            1e-6,
            &cfg,
        );

        for uk in &result.u_seq {
            assert!(uk[0] >= -0.05 - 1e-12);
            assert!(uk[0] <= 0.05 + 1e-12);
        }
    }

    #[test]
    fn ilqr_respects_state_limits_projection() {
        let model = simple_revolute_model();
        let q0 = vec![0.6];
        let v0 = vec![0.5];
        let n = 15usize;

        let u_init = vec![vec![0.2]; n];
        let q_ref_seq = vec![vec![0.0]; n + 1];
        let v_ref_seq = vec![vec![0.0]; n + 1];
        let u_ref_seq = vec![vec![0.0]; n];

        let q_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![30.0, 2.0]));
        let r_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![0.02]));
        let qf_weight = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![60.0, 3.0]));

        let mut lim = JointLimits::unbounded(&model);
        lim.q_min[0] = -0.1;
        lim.q_max[0] = 0.1;
        lim.v_max[0] = 0.03;

        let cfg = IlqrConfig {
            max_iters: 8,
            joint_limits: Some(lim.clone()),
            ..IlqrConfig::default()
        };

        let result = solve_ilqr(
            &model,
            &q0,
            &v0,
            &u_init,
            &q_ref_seq,
            &v_ref_seq,
            &u_ref_seq,
            &q_weight,
            &r_weight,
            &qf_weight,
            0.02,
            1e-6,
            &cfg,
        );

        for q in &result.q_seq {
            assert!(q[0] >= lim.q_min[0] - 1e-12);
            assert!(q[0] <= lim.q_max[0] + 1e-12);
        }
        for v in &result.v_seq {
            assert!(v[0].abs() <= lim.v_max[0] + 1e-12);
        }
    }
}
