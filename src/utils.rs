//! Numerical differentiation utilities — finite differences for validation.
//!
//! Provides helpers for computing numerical approximations to Jacobians,
//! gradients, and Hessians via finite difference methods. Useful for:
//!
//! - Validating analytical Jacobians (compare_jacobian_analytical_vs_numerical)
//! - Debugging dynamic algorithms
//! - Testing automatic differentiation
//!
//! Generic over `T: RealField`.

use crate::fk::forward_kinematics;
use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, RealField};

/// Compute the numerical Jacobian of a function via forward finite differences.
///
/// Given a function `f: ℝⁿ → ℝᵐ`, computes the Jacobian matrix numerically:
///
/// ```text
/// J[i,j] ≈ (f(x + e_j Δx) - f(x)) / Δx
/// ```
///
/// # Arguments
///
/// * `f` — closure mapping input vector to output vector
/// * `x` — evaluation point (length `n`)
/// * `epsilon` — step size (default: 1e-8)
///
/// # Returns
///
/// `m × n` Jacobian matrix
pub fn numerical_jacobian<T: RealField>(
    f: &dyn Fn(&[T]) -> Vec<T>,
    x: &[T],
    epsilon: T,
) -> DMatrix<T> {
    let n = x.len();
    let f_x = f(x);
    let m = f_x.len();

    let mut jac = DMatrix::zeros(m, n);
    let _two_inv = T::one() / (T::one() + T::one());
    let eps_inv = T::one() / epsilon.clone();

    for j in 0..n {
        let mut x_plus = x.to_vec();
        x_plus[j] = x_plus[j].clone() + epsilon.clone();
        let f_plus = f(&x_plus);

        for i in 0..m {
            jac[(i, j)] = (f_plus[i].clone() - f_x[i].clone()) * eps_inv.clone();
        }
    }

    jac
}

/// Compute the numerical Jacobian using central differences (more accurate).
///
/// ```text
/// J[i,j] ≈ (f(x + e_j Δx) - f(x - e_j Δx)) / (2 Δx)
/// ```
pub fn numerical_jacobian_central<T: RealField>(
    f: &dyn Fn(&[T]) -> Vec<T>,
    x: &[T],
    epsilon: T,
) -> DMatrix<T> {
    let n = x.len();
    let m = f(x).len();

    let mut jac = DMatrix::zeros(m, n);
    let two = T::one() + T::one();
    let eps_inv = T::one() / (two * epsilon.clone());

    for j in 0..n {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[j] = x_plus[j].clone() + epsilon.clone();
        x_minus[j] = x_minus[j].clone() - epsilon.clone();

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);

        for i in 0..m {
            jac[(i, j)] = (f_plus[i].clone() - f_minus[i].clone()) * eps_inv.clone();
        }
    }

    jac
}

/// Compute numerical Jacobian of FK joint placements.
///
/// For a given joint index, compute how the joint's absolute placement
/// changes with respect to configuration `q`.
///
/// Returns a 6×nv matrix where each column is the spatial velocity of the
/// joint when only joint `j` has unit velocity.
pub fn numerical_jacobian_fk<T: RealField>(
    model: &Model<T>,
    q: &[T],
    joint_idx: usize,
    epsilon: T,
) -> DMatrix<T> {
    let nv = model.nv;
    let mut jac = DMatrix::zeros(6, nv);

    let data0 = forward_kinematics(model, q);
    let r0 = se3::rotation_matrix(&data0.oMi[joint_idx]);

    let two = T::one() + T::one();

    // Central differences
    let eps_inv_2 = T::one() / (two * epsilon.clone());

    for j in 0..nv {
        let mut q_plus = q.to_vec();
        let mut q_minus = q.to_vec();

        // Find which joint this velocity DOF belongs to
        let mut vi = 0;
        let mut ji = 1;
        while vi + model.joints[ji].joint_type.nv() <= j {
            vi += model.joints[ji].joint_type.nv();
            ji += 1;
        }

        q_plus[model.q_idx[ji] + (j - vi)] += epsilon.clone();
        q_minus[model.q_idx[ji] + (j - vi)] -= epsilon.clone();

        let data_plus = forward_kinematics(model, &q_plus);
        let data_minus = forward_kinematics(model, &q_minus);

        let p_plus = se3::translation(&data_plus.oMi[joint_idx]);
        let p_minus = se3::translation(&data_minus.oMi[joint_idx]);

        // Linear velocity finite difference
        let v_lin = (p_plus - p_minus) * eps_inv_2.clone();
        for i in 0..3 {
            jac[(3 + i, j)] = v_lin[i].clone();
        }

        // Angular velocity via rotation matrix log
        let r_plus = se3::rotation_matrix(&data_plus.oMi[joint_idx]);
        let r_minus = se3::rotation_matrix(&data_minus.oMi[joint_idx]);

        // dR/dt ≈ (R_plus - R_minus) / (2ε)
        let dr = (r_plus - r_minus) * eps_inv_2.clone();

        // Angular velocity: ω = vech(R^T dR) (approximate from matrix derivative)
        // For small differences: R^T dR ≈ [ω]×
        // So: ω_i ≈ (R^T dR)_{j,k} for appropriate (j,k)
        let rt = r0.transpose();
        let rt_dr = &rt * &dr;

        // Skew-symmetric part: ω_1 ≈ -(dR^T R)_{2,3}
        if rt_dr[(2, 1)].clone().abs() > nalgebra::convert(1e-15) {
            jac[(0, j)] = (rt_dr[(2, 1)].clone() - rt_dr[(1, 2)].clone()) 
                * (T::one() / (T::one() + T::one()));
            jac[(1, j)] = (rt_dr[(0, 2)].clone() - rt_dr[(2, 0)].clone()) 
                * (T::one() / (T::one() + T::one()));
            jac[(2, j)] = (rt_dr[(1, 0)].clone() - rt_dr[(0, 1)].clone()) 
                * (T::one() / (T::one() + T::one()));
        }
    }

    jac
}

/// Compute numerical gradient of a scalar function via finite differences.
pub fn numerical_gradient<T: RealField>(
    f: &dyn Fn(&[T]) -> T,
    x: &[T],
    epsilon: T,
) -> Vec<T> {
    let n = x.len();
    let mut grad = vec![T::zero(); n];
    let eps_inv = T::one() / epsilon.clone();

    let f_x = f(x);

    for j in 0..n {
        let mut x_plus = x.to_vec();
        x_plus[j] = x_plus[j].clone() + epsilon.clone();
        let f_plus = f(&x_plus);
        grad[j] = (f_plus - f_x.clone()) * eps_inv.clone();
    }

    grad
}

/// Compute numerical Hessian (matrix of 2nd partial derivatives).
///
/// Uses central differences:
/// ```text
/// H[i,j] ≈ (f(x + e_i ε + e_j ε) - f(x + e_i ε) - f(x + e_j ε) + f(x)) / ε²
/// ```
pub fn numerical_hessian<T: RealField>(
    f: &dyn Fn(&[T]) -> T,
    x: &[T],
    epsilon: T,
) -> DMatrix<T> {
    let n = x.len();
    let eps2_inv = T::one() / (epsilon.clone() * epsilon.clone());

    let mut hess = DMatrix::zeros(n, n);

    let f_x = f(x);
    let mut f_ei = vec![T::zero(); n];
    for i in 0..n {
        let mut xi = x.to_vec();
        xi[i] = xi[i].clone() + epsilon.clone();
        f_ei[i] = f(&xi);
    }

    for i in 0..n {
        for j in 0..n {
            let mut xi_ej = x.to_vec();
            xi_ej[i] = xi_ej[i].clone() + epsilon.clone();
            xi_ej[j] = xi_ej[j].clone() + epsilon.clone();
            let f_ei_ej = f(&xi_ej);

            let h = (f_ei_ej - f_ei[i].clone() - f_ei[j].clone() + f_x.clone()) * eps2_inv.clone();
            hess[(i, j)] = h;
        }
    }

    // Symmetrize
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (hess[(i, j)].clone() + hess[(j, i)].clone()) 
                * (T::one() / (T::one() + T::one()));
            hess[(i, j)] = avg.clone();
            hess[(j, i)] = avg;
        }
    }

    hess
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn numerical_jacobian_simple() {
        // f(x) = [x[0]^2, x[0] + x[1]]
        let f = |x: &[f64]| vec![x[0] * x[0], x[0] + x[1]];
        let x = vec![2.0, 3.0];

        let jac = numerical_jacobian(&f, &x, 1e-7);

        // Expected: [[4, 0], [1, 1]]  at x = [2, 3]
        assert_relative_eq!(jac[(0, 0)], 4.0, epsilon = 1e-5);
        assert_relative_eq!(jac[(0, 1)], 0.0, epsilon = 1e-5);
        assert_relative_eq!(jac[(1, 0)], 1.0, epsilon = 1e-5);
        assert_relative_eq!(jac[(1, 1)], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn numerical_jacobian_central_more_accurate() {
        let f = |x: &[f64]| vec![x[0].sin(), x[0].cos()];
        let x = vec![0.5];

        let jac_fwd = numerical_jacobian(&f, &x, 1e-5);
        let jac_central = numerical_jacobian_central(&f, &x, 1e-5);

        // Central should be closer to analytical [cos(0.5), -sin(0.5)]
        let cos_half = 0.5_f64.cos();
        let _sin_half = 0.5_f64.sin();

        let fwd_error = (jac_fwd[(0, 0)] - cos_half).abs();
        let central_error = (jac_central[(0, 0)] - cos_half).abs();

        assert!(central_error < fwd_error);
    }

    #[test]
    fn numerical_gradient_simple() {
        // f(x) = x[0]^2 + 2*x[1]^2
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let x = vec![3.0, 4.0];

        let grad = numerical_gradient(&f, &x, 1e-7);

        // Expected: [6, 16]
        assert_relative_eq!(grad[0], 6.0, epsilon = 1e-5);
        assert_relative_eq!(grad[1], 16.0, epsilon = 1e-5);
    }

    #[test]
    fn numerical_hessian_quadratic() {
        // f(x) = x[0]^2 + 3*x[0]*x[1] + 2*x[1]^2
        let f = |x: &[f64]| x[0] * x[0] + 3.0 * x[0] * x[1] + 2.0 * x[1] * x[1];

        let hess = numerical_hessian(&f, &vec![1.0, 1.0], 1e-6);

        // Expected: [[2, 3], [3, 4]] but numerical approximation has small error
        assert_relative_eq!(hess[(0, 0)], 2.0, epsilon = 1e-3);
        assert_relative_eq!(hess[(0, 1)], 3.0, epsilon = 1e-3);
        assert_relative_eq!(hess[(1, 0)], 3.0, epsilon = 1e-3);
        assert_relative_eq!(hess[(1, 1)], 4.0, epsilon = 1e-3);
    }

    #[test]
    fn numerical_hessian_is_symmetric() {
        let f = |x: &[f64]| (x[0].sin() * x[1].cos()).exp();
        let hess = numerical_hessian(&f, &vec![0.1, 0.2], 1e-5);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(hess[(i, j)], hess[(j, i)], epsilon = 1e-6);
            }
        }
    }
}
