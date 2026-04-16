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
}
