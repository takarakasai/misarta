//! Analytical derivatives of ABA (forward dynamics).
//!
//! Computes ∂q̈/∂q, ∂q̈/∂v, ∂q̈/∂τ for the Articulated Body Algorithm.
//!
//! Uses the **indirect method** (Carpentier & Mansard, RSS 2018):
//!   1. Compute q̈ = ABA(q, v, τ)
//!   2. Compute RNEA derivatives at (q, v, q̈)
//!   3. ∂q̈/∂x = −M⁻¹ · ∂τ/∂x  for x ∈ {q, v}
//!   4. ∂q̈/∂τ = M⁻¹
//!
//! This relies on the implicit function theorem applied to
//! τ = M(q)q̈ + h(q, q̇):
//!   ∂q̈/∂q = −M⁻¹ ∂τ/∂q |_{a=q̈}
//!   ∂q̈/∂v = −M⁻¹ ∂τ/∂v |_{a=q̈}
//!
//! **Pure function**: `(model, q, v, tau) → AbaDerivatives`.

use crate::aba::aba;
use crate::model::Model;
use crate::rnea_derivatives::compute_rnea_derivatives;
use nalgebra::{DMatrix, RealField};

/// Output of ABA analytical derivatives.
#[derive(Debug, Clone)]
pub struct AbaDerivatives<T: RealField> {
    /// ∂q̈/∂q — nv × nv matrix.
    pub ddq_dq: DMatrix<T>,
    /// ∂q̈/∂v — nv × nv matrix.
    pub ddq_dv: DMatrix<T>,
    /// ∂q̈/∂τ — nv × nv matrix (equals M⁻¹(q)).
    pub ddq_dtau: DMatrix<T>,
}

/// Compute the analytical derivatives of ABA (forward dynamics).
///
/// Returns ∂q̈/∂q, ∂q̈/∂v, ∂q̈/∂τ.
///
/// # Arguments
///
/// * `model` — robot model
/// * `q`     — configuration vector (length `nq`)
/// * `v`     — velocity vector (length `nv`)
/// * `tau`   — applied joint torques (length `nv`)
pub fn compute_aba_derivatives(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    tau: &[f64],
) -> AbaDerivatives<f64> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(tau.len(), model.nv);

    // 1. Compute q̈ = ABA(q, v, τ)
    let qdd = aba(model, q, v, tau);
    let qdd_slice: Vec<f64> = qdd.as_slice().to_vec();

    // 2. Compute RNEA derivatives at (q, v, q̈)
    //    dtau_dq, dtau_dv, dtau_da = M
    let rnea_derivs = compute_rnea_derivatives(model, q, v, &qdd_slice);

    // 3. M⁻¹ via Cholesky (M is symmetric positive definite)
    let m = rnea_derivs.dtau_da; // = M(q)
    let m_chol = m.clone().cholesky().expect("Mass matrix must be positive definite");
    let m_inv = m_chol.inverse();

    // 4. ∂q̈/∂q = −M⁻¹ ∂τ/∂q
    let ddq_dq = -&m_inv * &rnea_derivs.dtau_dq;

    // 5. ∂q̈/∂v = −M⁻¹ ∂τ/∂v
    let ddq_dv = -&m_inv * &rnea_derivs.dtau_dv;

    // 6. ∂q̈/∂τ = M⁻¹
    let ddq_dtau = m_inv;

    AbaDerivatives {
        ddq_dq,
        ddq_dv,
        ddq_dtau,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aba::{aba, compute_minv};
    use crate::joint;
    use crate::model::{LinkInertia, Model, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Vector3};

    fn two_link_arm() -> Model<f64> {
        let offset1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let offset2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_y(), offset1,
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.02,
                    ),
                },
            )
            .add_joint(
                "j2", 1, joint::revolute_y(), offset2,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.01,
                    ),
                },
            )
            .build()
    }

    fn three_link_arm() -> Model<f64> {
        let offset1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.3),
        );
        let offset2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let offset3 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.4),
        );
        ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_y(), offset1,
                LinkInertia {
                    mass: 3.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.25),
                    rotational_inertia: Matrix3::new(
                        0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.03,
                    ),
                },
            )
            .add_joint(
                "j2", 1, joint::revolute_z(), offset2,
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.2),
                    rotational_inertia: Matrix3::new(
                        0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.02,
                    ),
                },
            )
            .add_joint(
                "j3", 2, joint::revolute_x(), offset3,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.15),
                    rotational_inertia: Matrix3::new(
                        0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.01,
                    ),
                },
            )
            .build()
    }

    /// Finite-difference: ∂q̈/∂q.
    fn fd_ddq_dq(model: &Model<f64>, q: &[f64], v: &[f64], tau: &[f64], eps: f64) -> DMatrix<f64> {
        let nv = model.nv;
        let mut result = DMatrix::zeros(nv, nv);
        for k in 0..nv {
            let mut q_p = q.to_vec();
            let mut q_m = q.to_vec();
            q_p[k] += eps;
            q_m[k] -= eps;
            let qdd_p = aba(model, &q_p, v, tau);
            let qdd_m = aba(model, &q_m, v, tau);
            let col = (&qdd_p - &qdd_m) / (2.0 * eps);
            for r in 0..nv { result[(r, k)] = col[r]; }
        }
        result
    }

    /// Finite-difference: ∂q̈/∂v.
    fn fd_ddq_dv(model: &Model<f64>, q: &[f64], v: &[f64], tau: &[f64], eps: f64) -> DMatrix<f64> {
        let nv = model.nv;
        let mut result = DMatrix::zeros(nv, nv);
        for k in 0..nv {
            let mut v_p = v.to_vec();
            let mut v_m = v.to_vec();
            v_p[k] += eps;
            v_m[k] -= eps;
            let qdd_p = aba(model, q, &v_p, tau);
            let qdd_m = aba(model, q, &v_m, tau);
            let col = (&qdd_p - &qdd_m) / (2.0 * eps);
            for r in 0..nv { result[(r, k)] = col[r]; }
        }
        result
    }

    // ── ddq_dtau = M⁻¹ ─────────────────────────────────────────────────

    #[test]
    fn ddq_dtau_equals_minv_two_link() {
        let model = two_link_arm();
        let q = [0.3, -0.5]; let v = [1.0, -0.5]; let tau = [5.0, -2.0];
        let derivs = compute_aba_derivatives(&model, &q, &v, &tau);
        let minv = compute_minv(&model, &q);
        assert_relative_eq!(derivs.ddq_dtau, minv, epsilon = 1e-8);
    }

    #[test]
    fn ddq_dtau_equals_minv_three_link() {
        let model = three_link_arm();
        let q = [0.4, -0.3, 0.2]; let v = [1.0, -0.5, 0.8]; let tau = [3.0, -1.0, 0.5];
        let derivs = compute_aba_derivatives(&model, &q, &v, &tau);
        let minv = compute_minv(&model, &q);
        assert_relative_eq!(derivs.ddq_dtau, minv, epsilon = 1e-8);
    }

    // ── ddq_dq vs finite differences ────────────────────────────────────

    #[test]
    fn ddq_dq_two_link_vs_fd() {
        let model = two_link_arm();
        let q = [0.3, -0.5]; let v = [1.0, -0.5]; let tau = [5.0, -2.0];
        let derivs = compute_aba_derivatives(&model, &q, &v, &tau);
        let fd = fd_ddq_dq(&model, &q, &v, &tau, 1e-7);
        assert_relative_eq!(derivs.ddq_dq, fd, epsilon = 1e-4);
    }

    #[test]
    fn ddq_dq_three_link_vs_fd() {
        let model = three_link_arm();
        let q = [0.4, -0.3, 0.2]; let v = [1.0, -0.5, 0.8]; let tau = [3.0, -1.0, 0.5];
        let derivs = compute_aba_derivatives(&model, &q, &v, &tau);
        let fd = fd_ddq_dq(&model, &q, &v, &tau, 1e-7);
        assert_relative_eq!(derivs.ddq_dq, fd, epsilon = 1e-4);
    }

    // ── ddq_dv vs finite differences ────────────────────────────────────

    #[test]
    fn ddq_dv_two_link_vs_fd() {
        let model = two_link_arm();
        let q = [0.3, -0.5]; let v = [1.0, -0.5]; let tau = [5.0, -2.0];
        let derivs = compute_aba_derivatives(&model, &q, &v, &tau);
        let fd = fd_ddq_dv(&model, &q, &v, &tau, 1e-7);
        assert_relative_eq!(derivs.ddq_dv, fd, epsilon = 1e-4);
    }

    #[test]
    fn ddq_dv_three_link_vs_fd() {
        let model = three_link_arm();
        let q = [0.4, -0.3, 0.2]; let v = [1.0, -0.5, 0.8]; let tau = [3.0, -1.0, 0.5];
        let derivs = compute_aba_derivatives(&model, &q, &v, &tau);
        let fd = fd_ddq_dv(&model, &q, &v, &tau, 1e-7);
        assert_relative_eq!(derivs.ddq_dv, fd, epsilon = 1e-4);
    }

    // ── Special cases ───────────────────────────────────────────────────

    #[test]
    fn ddq_dq_zero_velocity_zero_torque() {
        // Free-fall: tau=0, v=0 → q̈ = -g projection. Check derivative.
        let model = two_link_arm();
        let q = [0.3, -0.5]; let z = [0.0, 0.0];
        let derivs = compute_aba_derivatives(&model, &q, &z, &z);
        let fd = fd_ddq_dq(&model, &q, &z, &z, 1e-7);
        assert_relative_eq!(derivs.ddq_dq, fd, epsilon = 1e-4);
    }

    #[test]
    fn ddq_dv_zero_velocity_zero_torque() {
        let model = two_link_arm();
        let q = [0.3, -0.5]; let z = [0.0, 0.0];
        let derivs = compute_aba_derivatives(&model, &q, &z, &z);
        let fd = fd_ddq_dv(&model, &q, &z, &z, 1e-7);
        assert_relative_eq!(derivs.ddq_dv, fd, epsilon = 1e-4);
    }
}
