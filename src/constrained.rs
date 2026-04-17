//! Constrained dynamics — forward dynamics with contacts / loop constraints,
//! and impact dynamics for impulse resolution.
//!
//! # Constrained forward dynamics
//!
//! Solves the KKT system for joint accelerations subject to holonomic
//! constraints:
//!
//! ```text
//! [ M  Jcᵀ ] [ q̈ ] = [ τ − h ]
//! [ Jc  0  ] [ λ  ]   [ −γ    ]
//! ```
//!
//! where:
//! - `M` = joint-space inertia matrix
//! - `h = C(q,v)v + g(q)` = nonlinear effects (Coriolis + gravity)
//! - `Jc` = constraint Jacobian (nc × nv)
//! - `γ = Ȧc v` = constraint acceleration drift
//! - `λ` = constraint forces (Lagrange multipliers)
//!
//! # Impact dynamics
//!
//! Solves for post-impact velocity given a rigid contact (Newton's impact law):
//!
//! ```text
//! [ M  Jcᵀ ] [ v⁺ ] = [ M v⁻ ]
//! [ Jc  0  ] [ Λ  ]   [ −e Jc v⁻ ]
//! ```
//!
//! where `e ∈ [0, 1]` is the coefficient of restitution.

use nalgebra::{DMatrix, DVector};

use crate::crba::crba;
use crate::model::Model;
use crate::rnea;

/// Result of constrained forward dynamics.
#[derive(Debug, Clone)]
pub struct ConstrainedDynamicsResult {
    /// Joint accelerations q̈ ∈ ℝⁿᵛ.
    pub qdd: DVector<f64>,
    /// Constraint forces (Lagrange multipliers) λ ∈ ℝⁿᶜ.
    pub lambda: DVector<f64>,
}

/// Solve constrained forward dynamics via the KKT system.
///
/// Given:
/// - `model` — robot model
/// - `q` — configuration (nq)
/// - `v` — velocity (nv)
/// - `tau` — applied torques (nv)
/// - `jc` — constraint Jacobian (nc × nv)
/// - `gamma` — constraint acceleration drift `γ = Ȧc v` (nc).
///   Pass a zero vector if drift compensation is not needed.
///
/// Returns joint accelerations `q̈` and constraint forces `λ`.
///
/// # Algorithm
///
/// Assembles and solves the KKT system:
///
/// ```text
/// [ M   Jcᵀ ] [ q̈ ] = [ τ − h ]
/// [ Jc   0  ] [ λ  ]   [ −γ    ]
/// ```
pub fn constrained_forward_dynamics(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    tau: &[f64],
    jc: &DMatrix<f64>,
    gamma: &DVector<f64>,
) -> ConstrainedDynamicsResult {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(tau.len(), model.nv);
    let nc = jc.nrows();
    assert_eq!(jc.ncols(), model.nv);
    assert_eq!(gamma.len(), nc);

    let nv = model.nv;

    // Compute M and h
    let m = crba(model, q);
    let h = rnea::nonlinear_effects(model, q, v);

    // Build KKT matrix [(nv+nc) × (nv+nc)]
    let dim = nv + nc;
    let mut kkt = DMatrix::zeros(dim, dim);

    // Upper-left: M
    kkt.view_mut((0, 0), (nv, nv)).copy_from(&m);
    // Upper-right: Jcᵀ
    kkt.view_mut((0, nv), (nv, nc)).copy_from(&jc.transpose());
    // Lower-left: Jc
    kkt.view_mut((nv, 0), (nc, nv)).copy_from(jc);
    // Lower-right: 0 (already zero)

    // Build RHS
    let tau_vec = DVector::from_column_slice(tau);
    let mut rhs = DVector::zeros(dim);
    rhs.rows_mut(0, nv).copy_from(&(tau_vec - h));
    rhs.rows_mut(nv, nc).copy_from(&(-gamma));

    // Solve via LU
    let lu = kkt.lu();
    let sol = lu.solve(&rhs).expect("KKT system is singular");

    let qdd = sol.rows(0, nv).into_owned();
    let lambda = sol.rows(nv, nc).into_owned();

    ConstrainedDynamicsResult { qdd, lambda }
}

/// Result of impact dynamics.
#[derive(Debug, Clone)]
pub struct ImpactResult {
    /// Post-impact velocity v⁺ ∈ ℝⁿᵛ.
    pub v_post: DVector<f64>,
    /// Impulse magnitudes (Lagrange multipliers) Λ ∈ ℝⁿᶜ.
    pub impulse: DVector<f64>,
}

/// Solve impact dynamics (impulse resolution).
///
/// Given a set of rigid contacts with Jacobian `jc` and coefficient of
/// restitution `restitution_coeff` (0 = perfectly plastic, 1 = perfectly
/// elastic), computes the post-impact velocity.
///
/// # Algorithm
///
/// Solves:
///
/// ```text
/// [ M   Jcᵀ ] [ v⁺ ] = [ M v⁻   ]
/// [ Jc   0  ] [ Λ  ]   [ −e Jc v⁻ ]
/// ```
pub fn impact_dynamics(
    model: &Model<f64>,
    q: &[f64],
    v_pre: &[f64],
    jc: &DMatrix<f64>,
    restitution_coeff: f64,
) -> ImpactResult {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v_pre.len(), model.nv);
    let nc = jc.nrows();
    assert_eq!(jc.ncols(), model.nv);

    let nv = model.nv;

    // Compute M
    let m = crba(model, q);

    // Build KKT matrix
    let dim = nv + nc;
    let mut kkt = DMatrix::zeros(dim, dim);
    kkt.view_mut((0, 0), (nv, nv)).copy_from(&m);
    kkt.view_mut((0, nv), (nv, nc)).copy_from(&jc.transpose());
    kkt.view_mut((nv, 0), (nc, nv)).copy_from(jc);

    // Build RHS
    let v_pre_vec = DVector::from_column_slice(v_pre);
    let m_v = &m * &v_pre_vec;
    let jc_v = jc * &v_pre_vec;

    let mut rhs = DVector::zeros(dim);
    rhs.rows_mut(0, nv).copy_from(&m_v);
    rhs.rows_mut(nv, nc).copy_from(&(-restitution_coeff * &jc_v));

    // Solve via LU
    let lu = kkt.lu();
    let sol = lu.solve(&rhs).expect("Impact KKT system is singular");

    let v_post = sol.rows(0, nv).into_owned();
    let impulse = sol.rows(nv, nc).into_owned();

    ImpactResult { v_post, impulse }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aba::aba;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Vector3};

    fn two_link_arm() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::revolute_y(),
                offset.clone(),
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::identity() * 0.2,
                },
            )
            .add_joint(
                "j2",
                1,
                joint::revolute_y(),
                offset,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::identity() * 0.1,
                },
            )
            .build()
    }

    #[test]
    fn unconstrained_matches_aba() {
        // With no constraints (empty Jc), constrained dynamics should
        // match the unconstrained ABA result.
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let tau = vec![5.0, -2.0];

        let jc = DMatrix::zeros(0, model.nv);
        let gamma = DVector::zeros(0);

        let result = constrained_forward_dynamics(&model, &q, &v, &tau, &jc, &gamma);
        let qdd_aba = aba(&model, &q, &v, &tau);

        assert_relative_eq!(result.qdd, qdd_aba, epsilon = 1e-8);
    }

    #[test]
    fn constraint_is_satisfied() {
        // Fix joint 1 via a constraint: Jc = [1, 0], meaning q̈₁ = 0.
        let model = two_link_arm();
        let q = vec![0.0, 0.0];
        let v = vec![0.0, 0.0];
        let tau = vec![0.0, 0.0];

        let mut jc = DMatrix::zeros(1, model.nv);
        jc[(0, 0)] = 1.0; // constrain DOF 0
        let gamma = DVector::zeros(1);

        let result = constrained_forward_dynamics(&model, &q, &v, &tau, &jc, &gamma);

        // q̈₁ should be 0 (constrained)
        assert_relative_eq!(result.qdd[0], 0.0, epsilon = 1e-10);
        // λ should be non-zero (gravity force must be balanced by constraint)
        // The constraint force balances gravity on DOF 0
    }

    #[test]
    fn constraint_force_direction() {
        // Constrain both DOFs: system is fully locked.
        // λ should equal the negative of the unconstrained rhs for those DOFs.
        let model = two_link_arm();
        let q = vec![0.3, -0.2];
        let v = vec![0.0, 0.0];
        let tau = vec![0.0, 0.0];

        let jc = DMatrix::identity(model.nv, model.nv);
        let gamma = DVector::zeros(model.nv);

        let result = constrained_forward_dynamics(&model, &q, &v, &tau, &jc, &gamma);

        // Both accelerations must be zero
        assert_relative_eq!(result.qdd[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.qdd[1], 0.0, epsilon = 1e-10);

        // From KKT: Jcᵀ λ = τ - h → λ = τ - h = -h (since Jc = I, τ = 0)
        let g = crate::rnea::compute_gravity(&model, &q);
        assert_relative_eq!(result.lambda[0], -g[0], epsilon = 1e-8);
        assert_relative_eq!(result.lambda[1], -g[1], epsilon = 1e-8);
    }

    #[test]
    fn impact_plastic_stops_motion() {
        // Perfectly plastic impact (e=0) with full-rank contact Jacobian
        // should stop all constrained velocity.
        let model = two_link_arm();
        let q = vec![0.0, 0.0];
        let v_pre = vec![1.0, -0.5];

        let mut jc = DMatrix::zeros(1, model.nv);
        jc[(0, 0)] = 1.0; // constraint on DOF 0

        let result = impact_dynamics(&model, &q, &v_pre, &jc, 0.0);

        // Post-impact: Jc * v⁺ = 0
        let constraint_vel = &jc * &result.v_post;
        assert_relative_eq!(constraint_vel[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn impact_elastic_reverses_velocity() {
        // Perfectly elastic impact (e=1): Jc * v⁺ = -Jc * v⁻
        let model = two_link_arm();
        let q = vec![0.0, 0.0];
        let v_pre = vec![1.0, 0.0];

        let mut jc = DMatrix::zeros(1, model.nv);
        jc[(0, 0)] = 1.0;

        let result = impact_dynamics(&model, &q, &v_pre, &jc, 1.0);

        let jc_v_pre = &jc * &DVector::from_column_slice(&v_pre);
        let jc_v_post = &jc * &result.v_post;
        assert_relative_eq!(jc_v_post[0], -jc_v_pre[0], epsilon = 1e-10);
    }

    #[test]
    fn impact_conserves_momentum_zero_restitution() {
        // For e=0, M v⁺ = M v⁻ + Jcᵀ Λ
        let model = two_link_arm();
        let q = vec![0.3, -0.2];
        let v_pre = vec![2.0, -1.0];

        let mut jc = DMatrix::zeros(1, model.nv);
        jc[(0, 0)] = 1.0;

        let result = impact_dynamics(&model, &q, &v_pre, &jc, 0.0);

        // From KKT: M v⁺ + Jcᵀ Λ = M v⁻ → M v⁺ = M v⁻ − Jcᵀ Λ
        let m = crba(&model, &q);
        let v_pre_vec = DVector::from_column_slice(&v_pre);
        let lhs = &m * &result.v_post;
        let rhs = &m * &v_pre_vec - jc.transpose() * &result.impulse;
        assert_relative_eq!(lhs, rhs, epsilon = 1e-10);
    }
}
