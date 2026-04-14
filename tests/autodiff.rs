//! Automatic differentiation integration tests.
//!
//! Demonstrates that a pure FK position function, written generically over a
//! `DualNum` scalar type, can be automatically differentiated using `num-dual`
//! to obtain exact Jacobians — matching the analytical Jacobian from misarta.
//!
//! This validates the **referential transparency** design: because FK is a pure
//! function `q → position`, it composes cleanly with AD.

use approx::assert_relative_eq;
use misarta::jacobian::compute_joint_jacobian;
use misarta::joint;
use misarta::model::{LinkInertia, ModelBuilder};
use misarta::se3;
use nalgebra::{SMatrix, Vector3};
use num_dual::{Dual64, DualNum};

/// 2-link planar arm end-effector position, generic over scalar type.
///
/// This is the key pattern for AD: write the forward map as a pure function
/// `ℝⁿ → ℝᵐ` using generic scalars, then num-dual can differentiate it.
///
/// x = cos(q₀) + cos(q₀ + q₁)
/// y = sin(q₀) + sin(q₀ + q₁)
/// z = 0
fn end_effector_2link<D: DualNum<f64> + Copy>(q: &[D; 2]) -> [D; 3] {
    let q0 = q[0];
    let q1 = q[1];

    let s0 = q0.sin();
    let c0 = q0.cos();
    let sum = q0 + q1;
    let s01 = sum.sin();
    let c01 = sum.cos();

    [c0 + c01, s0 + s01, D::from(0.0)]
}

/// Compute Jacobian via forward-mode AD, one column at a time.
///
/// For column j, we seed q[j] with dual part = 1 and extract ∂f/∂qⱼ.
fn compute_ad_jacobian(q_vals: [f64; 2]) -> SMatrix<f64, 3, 2> {
    let mut jac = SMatrix::<f64, 3, 2>::zeros();

    for j in 0..2 {
        let mut q_dual = [Dual64::from(q_vals[0]), Dual64::from(q_vals[1])];
        q_dual[j] = Dual64::new(q_vals[j], 1.0); // seed derivative direction

        let result = end_effector_2link(&q_dual);
        for i in 0..3 {
            jac[(i, j)] = result[i].eps;
        }
    }

    jac
}

#[test]
fn autodiff_matches_analytical_jacobian() {
    let q_vals = [0.3_f64, -0.5];

    // ── Analytical Jacobian via misarta ──────────────────────────────────
    let offset = se3::from_rotation_and_translation(
        &nalgebra::Rotation3::identity(),
        &Vector3::new(1.0, 0.0, 0.0),
    );
    let model = ModelBuilder::new()
        .add_joint(
            "shoulder",
            0,
            joint::revolute_z(),
            se3::identity(),
            LinkInertia::zero(),
        )
        .add_joint("elbow", 1, joint::revolute_z(), offset.clone(), LinkInertia::zero())
        .add_joint("tip", 2, misarta::joint::JointType::Fixed, offset, LinkInertia::zero())
        .build();

    let q_full = vec![q_vals[0], q_vals[1]];
    let jac_analytical = compute_joint_jacobian(&model, &q_full, 3);
    let j_pos_analytical = jac_analytical.rows(3, 3);

    // ── AD Jacobian ─────────────────────────────────────────────────────
    let jac_ad = compute_ad_jacobian(q_vals);

    // ── Compare ─────────────────────────────────────────────────────────
    for i in 0..3 {
        for j in 0..2 {
            assert_relative_eq!(
                j_pos_analytical[(i, j)],
                jac_ad[(i, j)],
                epsilon = 1e-10,
            );
        }
    }
}

#[test]
fn autodiff_at_zero_config() {
    let q_vals = [0.0_f64, 0.0];
    let jac_ad = compute_ad_jacobian(q_vals);

    // ∂x/∂q0 = -sin(0) - sin(0) = 0
    assert_relative_eq!(jac_ad[(0, 0)], 0.0, epsilon = 1e-14);
    // ∂y/∂q0 = cos(0) + cos(0) = 2
    assert_relative_eq!(jac_ad[(1, 0)], 2.0, epsilon = 1e-14);
    // ∂x/∂q1 = -sin(0) = 0
    assert_relative_eq!(jac_ad[(0, 1)], 0.0, epsilon = 1e-14);
    // ∂y/∂q1 = cos(0) = 1
    assert_relative_eq!(jac_ad[(1, 1)], 1.0, epsilon = 1e-14);
}

#[test]
fn autodiff_value_matches_f64() {
    // Verify that the dual-number computation gives the same *value* as plain f64.
    let q_vals = [0.7_f64, -0.2];

    let f64_result = end_effector_2link(&q_vals);

    let dual_q = [
        Dual64::from(q_vals[0]),
        Dual64::from(q_vals[1]),
    ];
    let dual_result = end_effector_2link(&dual_q);

    for i in 0..3 {
        assert_relative_eq!(f64_result[i], dual_result[i].re, epsilon = 1e-14);
    }
}

#[test]
fn autodiff_pure_deterministic() {
    let q_vals = [0.7_f64, -0.2];
    let j1 = compute_ad_jacobian(q_vals);
    let j2 = compute_ad_jacobian(q_vals);
    assert_eq!(j1, j2);
}
