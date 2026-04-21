//! Coriolis matrix — C(q, q̇) ∈ ℝ^{nv × nv}.
//!
//! The Coriolis/centrifugal matrix satisfies:
//!
//! ```text
//! C(q, q̇) q̇ = nle(q, q̇) − g(q)
//! ```
//!
//! where `nle` is the non-linear effects vector (Coriolis + centrifugal + gravity)
//! and `g` is the gravity torque.
//!
//! **Algorithm**: RNEA column-expansion method.  Column `j` is computed as:
//!
//! ```text
//! C(:, j) = rnea(q, v, eⱼ) − g(q)
//! ```
//!
//! where `eⱼ` is the `j`-th unit vector.  This exploits the fact that
//! `rnea(q, v, a) = M(q) a + C(q,v) v + g(q)`, so `rnea(q, v, eⱼ) − g(q)`
//! gives the `j`-th column of `M(q)` plus `C(q,v) v`.  Wait — that yields
//! `M(:,j) + nle` which is NOT what we want.
//!
//! **Correct approach**: We use the RNEA identity:
//!
//! ```text
//! rnea(q, v, a) = M(q) a + nle(q, v)
//! ```
//!
//! so:
//!
//! ```text
//! rnea(q, eⱼ, 0) = M(q) · 0 + C(q, eⱼ) eⱼ + g(q)  ← NO, nle depends on v!
//! ```
//!
//! The standard approach uses Christoffel symbols or the RNEA difference:
//!
//! ```text
//! C(:, j) = (rnea(q, v, eⱼ) − rnea(q, v, 0)) − M(:, j)  ← but this is 0!
//! ```
//!
//! Actually, the correct identity is that `τ = M a + C v + g`, and `rnea(q, v, 0) = C v + g`.
//! So `rnea(q, v, eⱼ) - rnea(q, v, 0) = M(:,j)`, which doesn't help.
//!
//! **Correct method (Christoffel symbols from RNEA)**:
//!
//! We compute C using the relation C(q,v)v = nle(q,v) - g(q), and exploit
//! bilinearity of C:  C(q, v) is bilinear in v, i.e.
//! C(q, α u + β w) = α C(q, u) + β C(q, w) ... no, C depends on v
//! non-trivially through the velocity-dependent terms.
//!
//! The most practical approach:
//!
//! ```text
//! C(q, v)(:, j) = rnea(q, eⱼ, 0) − g(q)
//!     − 0.5 * [ rnea(q, eⱼ+eₖ, 0) − rnea(q, eⱼ, 0) − rnea(q, eₖ, 0) + g(q) ]  ...
//! ```
//!
//! Forget the manual derivation — we use RNEA expansion directly:
//!
//! For each column j of C, set v = eⱼ (unit vector), compute nle(q, eⱼ) − g(q).
//! But this gives C(q,eⱼ)·eⱼ, a single column dotted with eⱼ, which is just
//! column j of C.  Wait: C(q, eⱼ) eⱼ = C_j, the j-th diagonal? No!
//!
//! **Key insight**: `nle(q, v) - g(q) = C(q, v) v`.  If we define
//! `h(v) := nle(q, v) - g(q) = C(q,v) v`, then h is quadratic in v, and:
//!
//! ```text
//! C_jk = ∂h_j / ∂v_k  evaluated appropriately
//! ```
//!
//! But C(q,v) depends on v itself (it has Christoffel symbols multiplied by v).
//! Actually, C(q,v) v is quadratic in v, so C(q,v) is linear in v.  Therefore:
//!
//! ```text
//! C(q, v)(:,j) * v_j summed = h(v)
//! ```
//!
//! Since C(q,v) is linear in v, we can extract column k of C(q,v) as:
//!
//! ```text
//! C(q, v)(:, k) = [ nle(q, v + ε eₖ) − nle(q, v − ε eₖ) ] / (2ε)  − M(:,k)·0
//! ```
//!
//! But actually, since C(q,v)v is quadratic in v, the matrix C(q,v) (Christoffel form)
//! can be obtained from:
//!
//!   C(q,v)(:,k) = rnea(q, v, eₖ) − g(q) − M(:,k)
//!
//! Because rnea(q, v, eₖ) = M eₖ + C(q,v) v + g(q), so:
//!   rnea(q, v, eₖ) − g(q) − M(:,k) = C(q,v) v
//!
//! That gives C(q,v) v for every k, not C(:,k).
//!
//! **Final correct approach**: Use the Christoffel identity.  The Coriolis matrix
//! satisfying the skew-symmetry property N = Ṁ − 2C is uniquely defined by:
//!
//! ```text
//! C_ij(q, v) = Σ_k Γ_ijk(q) v_k
//! ```
//!
//! where Γ_ijk = 0.5 (∂M_ij/∂q_k + ∂M_ik/∂q_j − ∂M_jk/∂q_i) are the
//! Christoffel symbols of the first kind.
//!
//! Implementation: compute dM/dq via numerical differentiation, then assemble C.

use crate::crba;
use crate::model::Model;
use nalgebra::DMatrix;

/// Compute the Coriolis/centrifugal matrix C(q, v) ∈ ℝ^{nv × nv}.
///
/// Uses the Christoffel symbols of the first kind:
///
/// ```text
/// C_ij = Σ_k Γ_ijk v_k,   Γ_ijk = 0.5 (∂M_ij/∂q_k + ∂M_ik/∂q_j − ∂M_jk/∂q_i)
/// ```
///
/// The mass matrix partial derivatives are computed by central finite differences.
///
/// This matrix satisfies:
/// - `C(q,v) v = nle(q,v) - g(q)`
/// - Skew-symmetry: `Ṁ - 2C` is skew-symmetric (passivity property)
///
/// # Arguments
///
/// * `model` — robot model
/// * `q`     — configuration vector (length `nq`)
/// * `v`     — velocity vector (length `nv`)
pub fn compute_coriolis_matrix(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
) -> DMatrix<f64> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);

    let nv = model.nv;
    let eps = 1e-8;

    // Compute ∂M/∂q_k for each k via central differences
    let mut dm_dq: Vec<DMatrix<f64>> = Vec::with_capacity(nv);

    for k in 0..nv {
        let mut q_plus = q.to_vec();
        let mut q_minus = q.to_vec();

        // For revolute/prismatic joints, q_idx == v_idx (nq == nv == 1)
        // For FreeFlyer we'd need manifold-aware perturbation, but for
        // 1-DOF joints we can perturb q directly.
        // Find which q index corresponds to v index k.
        let q_k = find_q_index_for_v(model, k);

        q_plus[q_k] += eps;
        q_minus[q_k] -= eps;

        let m_plus = crba::crba(model, &q_plus);
        let m_minus = crba::crba(model, &q_minus);

        dm_dq.push((&m_plus - &m_minus) / (2.0 * eps));
    }

    // Assemble C using Christoffel symbols
    let mut c = DMatrix::zeros(nv, nv);

    for i in 0..nv {
        for j in 0..nv {
            let mut c_ij = 0.0;
            for k in 0..nv {
                // Γ_ijk = 0.5 * (∂M_ij/∂q_k + ∂M_ik/∂q_j - ∂M_jk/∂q_i)
                let gamma = 0.5 * (
                    dm_dq[k][(i, j)] + dm_dq[j][(i, k)] - dm_dq[i][(j, k)]
                );
                c_ij += gamma * v[k];
            }
            c[(i, j)] = c_ij;
        }
    }

    c
}

/// Find the q-vector index corresponding to a given v-vector index.
///
/// For 1-DOF joints (Revolute, Prismatic), q_idx == v_idx.
/// For Fixed joints (0-DOF), there's no v entry.
/// For FreeFlyer, mapping is nontrivial (nq=7, nv=6).
fn find_q_index_for_v(model: &Model<f64>, v_index: usize) -> usize {
    for i in 1..model.joints.len() {
        let vi = model.v_idx[i];
        let nv_j = model.joints[i].joint_type.nv();
        if v_index >= vi && v_index < vi + nv_j {
            let local = v_index - vi;
            return model.q_idx[i] + local;
        }
    }
    panic!("v_index {} out of range", v_index);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::rnea;
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Matrix3, Vector3};

    fn pendulum() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_y(), offset, LinkInertia {
                mass: 1.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                rotational_inertia: Matrix3::new(
                    0.1, 0.0, 0.0,
                    0.0, 0.1, 0.0,
                    0.0, 0.0, 0.01,
                ),
            })
            .build()
    }

    fn two_link() -> Model<f64> {
        let off1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let off2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_y(), off1, LinkInertia {
                mass: 2.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                rotational_inertia: Matrix3::new(
                    0.2, 0.01, 0.0,
                    0.01, 0.2, 0.0,
                    0.0, 0.0, 0.02,
                ),
            })
            .add_joint("j2", 1, joint::revolute_y(), off2, LinkInertia {
                mass: 1.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                rotational_inertia: Matrix3::new(
                    0.1, 0.0, 0.005,
                    0.0, 0.1, 0.0,
                    0.005, 0.0, 0.01,
                ),
            })
            .build()
    }

    fn three_link() -> Model<f64> {
        let off1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.3),
        );
        let off2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let off3 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.4),
        );
        ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), off1, LinkInertia {
                mass: 3.0,
                center_of_mass: Vector3::new(0.1, 0.0, -0.15),
                rotational_inertia: Matrix3::new(
                    0.3, 0.01, 0.0,
                    0.01, 0.3, 0.02,
                    0.0, 0.02, 0.05,
                ),
            })
            .add_joint("j2", 1, joint::revolute_y(), off2, LinkInertia {
                mass: 2.0,
                center_of_mass: Vector3::new(0.0, 0.05, -0.25),
                rotational_inertia: Matrix3::new(
                    0.2, 0.0, 0.01,
                    0.0, 0.2, 0.0,
                    0.01, 0.0, 0.03,
                ),
            })
            .add_joint("j3", 2, joint::revolute_x(), off3, LinkInertia {
                mass: 1.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.2),
                rotational_inertia: Matrix3::new(
                    0.1, 0.0, 0.0,
                    0.0, 0.1, 0.0,
                    0.0, 0.0, 0.01,
                ),
            })
            .build()
    }

    /// C(q,v) v should equal nle(q,v) - g(q)
    #[test]
    fn coriolis_times_v_equals_nle_minus_gravity_pendulum() {
        let model = pendulum();
        let q = vec![0.7];
        let v = vec![2.0];

        let c = compute_coriolis_matrix(&model, &q, &v);
        let cv = &c * DVector::from_column_slice(&v);

        let nle = rnea::nonlinear_effects(&model, &q, &v);
        let g = rnea::compute_gravity(&model, &q);
        let expected = &nle - &g;

        assert_relative_eq!(cv, expected, epsilon = 1e-6);
    }

    /// C(q,v) v should equal nle(q,v) - g(q) for a 2-link arm
    #[test]
    fn coriolis_times_v_equals_nle_minus_gravity_two_link() {
        let model = two_link();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];

        let c = compute_coriolis_matrix(&model, &q, &v);
        let cv = &c * DVector::from_column_slice(&v);

        let nle = rnea::nonlinear_effects(&model, &q, &v);
        let g = rnea::compute_gravity(&model, &q);
        let expected = &nle - &g;

        assert_relative_eq!(cv, expected, epsilon = 1e-6);
    }

    /// C(q,v) v should equal nle(q,v) - g(q) for a 3-link arm
    #[test]
    fn coriolis_times_v_equals_nle_minus_gravity_three_link() {
        let model = three_link();
        let q = vec![0.5, -0.3, 1.2];
        let v = vec![0.8, -1.0, 0.3];

        let c = compute_coriolis_matrix(&model, &q, &v);
        let cv = &c * DVector::from_column_slice(&v);

        let nle = rnea::nonlinear_effects(&model, &q, &v);
        let g = rnea::compute_gravity(&model, &q);
        let expected = &nle - &g;

        assert_relative_eq!(cv, expected, epsilon = 1e-6);
    }

    /// Skew-symmetry: Ṁ − 2C should be skew-symmetric (passivity property).
    /// We test: v^T (Ṁ − 2C) v = 0 for arbitrary v.
    #[test]
    fn skew_symmetry_property_two_link() {
        let model = two_link();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let eps = 1e-8;

        // Ṁ ≈ (M(q + v*dt) - M(q - v*dt)) / (2*dt)
        let mut q_plus = q.clone();
        let mut q_minus = q.clone();
        for i in 0..model.nv {
            q_plus[i] += v[i] * eps;
            q_minus[i] -= v[i] * eps;
        }
        let m_plus = crba::crba(&model, &q_plus);
        let m_minus = crba::crba(&model, &q_minus);
        let m_dot = (&m_plus - &m_minus) / (2.0 * eps);

        let c = compute_coriolis_matrix(&model, &q, &v);
        let n_mat = &m_dot - 2.0 * &c;

        // v^T N v should be zero
        let vv = DVector::from_column_slice(&v);
        let vtNv = vv.dot(&(&n_mat * &vv));
        assert!(vtNv.abs() < 1e-5, "v^T (Ṁ-2C) v = {} (should be ~0)", vtNv);
    }

    /// Skew-symmetry for 3-link
    #[test]
    fn skew_symmetry_property_three_link() {
        let model = three_link();
        let q = vec![0.5, -0.3, 1.2];
        let v = vec![0.8, -1.0, 0.3];
        let eps = 1e-8;

        let mut q_plus = q.clone();
        let mut q_minus = q.clone();
        for i in 0..model.nv {
            q_plus[i] += v[i] * eps;
            q_minus[i] -= v[i] * eps;
        }
        let m_plus = crba::crba(&model, &q_plus);
        let m_minus = crba::crba(&model, &q_minus);
        let m_dot = (&m_plus - &m_minus) / (2.0 * eps);

        let c = compute_coriolis_matrix(&model, &q, &v);
        let n_mat = &m_dot - 2.0 * &c;

        let vv = DVector::from_column_slice(&v);
        let vtNv = vv.dot(&(&n_mat * &vv));
        assert!(vtNv.abs() < 1e-5, "v^T (Ṁ-2C) v = {} (should be ~0)", vtNv);
    }

    /// C should be zero when v is zero
    #[test]
    fn coriolis_zero_velocity() {
        let model = two_link();
        let q = vec![0.3, -0.5];
        let v = vec![0.0, 0.0];

        let c = compute_coriolis_matrix(&model, &q, &v);
        assert_relative_eq!(c, DMatrix::zeros(2, 2), epsilon = 1e-10);
    }

    /// C is linear in v: C(q, αv) = α C(q, v)
    #[test]
    fn coriolis_linearity_in_velocity() {
        let model = two_link();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let alpha = 3.0;
        let v_scaled: Vec<f64> = v.iter().map(|x| x * alpha).collect();

        let c = compute_coriolis_matrix(&model, &q, &v);
        let c_scaled = compute_coriolis_matrix(&model, &q, &v_scaled);

        assert_relative_eq!(c_scaled, c * alpha, epsilon = 1e-5);
    }

    /// Shape should be nv × nv
    #[test]
    fn coriolis_shape() {
        let model = three_link();
        let q = vec![0.0; 3];
        let v = vec![0.0; 3];
        let c = compute_coriolis_matrix(&model, &q, &v);
        assert_eq!(c.nrows(), 3);
        assert_eq!(c.ncols(), 3);
    }
}
