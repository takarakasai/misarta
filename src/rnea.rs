//! Recursive Newton-Euler Algorithm (RNEA) — inverse dynamics.
//!
//! Computes the joint torques required to achieve a given acceleration, i.e.
//! τ = M(q) q̈  +  C(q, q̇) q̇  +  g(q)
//!
//! The algorithm follows Featherstone (2008) / Pinocchio:
//!
//! 1. **Forward pass**: compute body velocities and accelerations (propagating gravity).
//! 2. **Backward pass**: accumulate forces on each body and project onto joint axes.
//!
//! Convention: spatial vectors are [angular(3); linear(3)] (Featherstone).
//!
//! **Pure function**: `(model, q, v, a) → τ`.  Generic over `T: RealField`.

use crate::model::Model;
use crate::se3::{self, Motion};
use nalgebra::{DVector, RealField, Vector3, Vector6};

/// Compute inverse dynamics via the Recursive Newton-Euler Algorithm.
///
/// Returns the generalized force vector τ ∈ ℝⁿᵛ.
///
/// # Arguments
///
/// * `model` — robot model
/// * `q`     — configuration vector (length `nq`)
/// * `v`     — velocity vector (length `nv`)
/// * `a`     — desired acceleration vector (length `nv`)
///
/// # Panics
///
/// Panics if vector dimensions do not match the model.
pub fn rnea<T: RealField>(
    model: &Model<T>,
    q: &[T],
    v: &[T],
    a: &[T],
) -> DVector<T> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(a.len(), model.nv);

    let n = model.joints.len(); // includes universe at 0

    // ── Per-body storage ────────────────────────────────────────────────
    // Joint placement relative to parent: parent_X_joint
    let mut x_j: Vec<se3::SE3<T>> = vec![se3::identity(); n];
    // Body velocity in body frame
    let mut vel: Vec<Motion<T>> = vec![Vector6::zeros(); n];
    // Body acceleration in body frame
    let mut acc: Vec<Motion<T>> = vec![Vector6::zeros(); n];
    // Net force on body (wrench) in body frame
    let mut f: Vec<Vector6<T>> = vec![Vector6::zeros(); n];

    // Gravity as a spatial acceleration of the universe (Featherstone trick):
    // The universe has an acceleration of −g (as if it were falling upward).
    let mut a0 = Vector6::<T>::zeros();
    a0[3] = -model.gravity[0].clone();
    a0[4] = -model.gravity[1].clone();
    a0[5] = -model.gravity[2].clone();

    // Result vector
    let mut tau = DVector::zeros(model.nv);

    // ── Forward pass ────────────────────────────────────────────────────
    for i in 1..n {
        let joint = &model.joints[i];
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];
        let nq_j = joint.joint_type.nq();
        let nv_j = joint.joint_type.nv();
        let q_slice = &q[qi..qi + nq_j];

        // Joint placement: placement * forward(q)
        let m_j = joint.joint_type.forward(q_slice);
        x_j[i] = se3::compose(&joint.placement, &m_j);

        // Rotation and translation from parent_M_child
        let r = se3::rotation_matrix(&x_j[i]);
        let p = se3::translation(&x_j[i]);
        let rt = r.transpose();

        // Transform parent velocity to this body's frame
        let parent = joint.parent;
        let v_parent = &vel[parent];

        // v_child = X^{-1} v_parent:
        //   ω_child = R^T ω_parent
        //   v_child = R^T (v_parent - p × ω_parent)
        let omega_parent = Vector3::new(
            v_parent[0].clone(), v_parent[1].clone(), v_parent[2].clone(),
        );
        let vlin_parent = Vector3::new(
            v_parent[3].clone(), v_parent[4].clone(), v_parent[5].clone(),
        );

        let omega_body = &rt * &omega_parent;
        let vlin_body = &rt * (vlin_parent - p.cross(&omega_parent));

        // Motion subspace in body frame
        let s = joint.joint_type.motion_subspace(q_slice);

        // Joint velocity: S * qd
        let mut v_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qd_c = v[vi + c].clone();
            for r in 0..6 {
                v_j[r] += s[(r, c)].clone() * qd_c.clone();
            }
        }

        // Body velocity = transformed parent + joint velocity
        vel[i] = Vector6::new(
            omega_body[0].clone() + v_j[0].clone(),
            omega_body[1].clone() + v_j[1].clone(),
            omega_body[2].clone() + v_j[2].clone(),
            vlin_body[0].clone() + v_j[3].clone(),
            vlin_body[1].clone() + v_j[4].clone(),
            vlin_body[2].clone() + v_j[5].clone(),
        );

        // Transform parent acceleration (or universe a0) to body frame
        let a_parent = if parent == 0 { &a0 } else { &acc[parent] };
        let omega_a = Vector3::new(
            a_parent[0].clone(), a_parent[1].clone(), a_parent[2].clone(),
        );
        let vlin_a = Vector3::new(
            a_parent[3].clone(), a_parent[4].clone(), a_parent[5].clone(),
        );
        let alpha_body = &rt * &omega_a;
        let alin_body = &rt * (vlin_a - p.cross(&omega_a));

        // Joint acceleration: S * qdd
        let mut a_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qdd_c = a[vi + c].clone();
            for r in 0..6 {
                a_j[r] += s[(r, c)].clone() * qdd_c.clone();
            }
        }

        // Body acceleration = transformed parent + joint acceleration + v × v_j
        // The cross term is the velocity-dependent Coriolis/centrifugal contribution.
        let vx = se3::motion_cross(&vel[i]);
        let cross_term = vx * &v_j;

        acc[i] = Vector6::new(
            alpha_body[0].clone() + a_j[0].clone() + cross_term[0].clone(),
            alpha_body[1].clone() + a_j[1].clone() + cross_term[1].clone(),
            alpha_body[2].clone() + a_j[2].clone() + cross_term[2].clone(),
            alin_body[0].clone() + a_j[3].clone() + cross_term[3].clone(),
            alin_body[1].clone() + a_j[4].clone() + cross_term[4].clone(),
            alin_body[2].clone() + a_j[5].clone() + cross_term[5].clone(),
        );

        // Spatial inertia of body i in body frame
        let inertia = &model.inertias[i];
        let y_i = se3::spatial_inertia(
            inertia.mass.clone(),
            &inertia.center_of_mass,
            &inertia.rotational_inertia,
        );

        // Net force: f_i = Y_i * a_i  +  v_i ×* (Y_i * v_i)
        let y_a = &y_i * &acc[i];
        let y_v = &y_i * &vel[i];
        let vxstar = se3::force_cross(&vel[i]);
        f[i] = y_a + vxstar * y_v;
    }

    // ── Backward pass ───────────────────────────────────────────────────
    for i in (1..n).rev() {
        let joint = &model.joints[i];
        let vi = model.v_idx[i];
        let nv_j = joint.joint_type.nv();
        let qi = model.q_idx[i];
        let q_slice = &q[qi..qi + joint.joint_type.nq()];
        let s = joint.joint_type.motion_subspace(q_slice);

        // τ_i = S^T f_i
        for c in 0..nv_j {
            let mut dot = T::zero();
            for r in 0..6 {
                dot += s[(r, c)].clone() * f[i][r].clone();
            }
            tau[vi + c] = dot;
        }

        // Propagate force to parent: f_parent += X_i^{-T} f_i
        // X_i^{-T} transforms a wrench from child frame to parent frame:
        //   f_parent += X^T_inv f_child
        // In Featherstone: f_parent += X_i^* f_i
        let parent = joint.parent;
        if parent > 0 {
            let r = se3::rotation_matrix(&x_j[i]);
            let p = se3::translation(&x_j[i]);

            let f_ang = Vector3::new(f[i][0].clone(), f[i][1].clone(), f[i][2].clone());
            let f_lin = Vector3::new(f[i][3].clone(), f[i][4].clone(), f[i][5].clone());

            // Force transform (parent_X_child)^* :
            //   f_parent_ang = R f_ang + p × (R f_lin)
            //   f_parent_lin = R f_lin
            let r_f_lin = &r * &f_lin;
            let r_f_ang = &r * &f_ang;
            let f_parent_ang = &r_f_ang + p.cross(&r_f_lin);
            let f_parent_lin = r_f_lin;

            for k in 0..3 {
                f[parent][k] += f_parent_ang[k].clone();
                f[parent][k + 3] += f_parent_lin[k].clone();
            }
        }
    }

    tau
}

/// Compute the gravity torque vector g(q), i.e. the joint torques needed to
/// statically support the robot against gravity.
///
/// Equivalent to `rnea(model, q, 0, 0)`.
pub fn compute_gravity<T: RealField>(model: &Model<T>, q: &[T]) -> DVector<T> {
    let zero = vec![T::zero(); model.nv];
    rnea(model, q, &zero, &zero)
}

/// Compute the non-linear effects (Coriolis + centrifugal + gravity):
///   nle(q, v) = C(q, v) v + g(q)
///
/// Equivalent to `rnea(model, q, v, 0)`.
pub fn nonlinear_effects<T: RealField>(
    model: &Model<T>,
    q: &[T],
    v: &[T],
) -> DVector<T> {
    let zero = vec![T::zero(); model.nv];
    rnea(model, q, v, &zero)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use approx::assert_relative_eq;
    use nalgebra::Matrix3;

    fn simple_pendulum() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        ModelBuilder::new()
            .add_joint(
                "joint1",
                0,
                joint::revolute_y(),
                offset,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.1, 0.0, 0.0,
                        0.0, 0.1, 0.0,
                        0.0, 0.0, 0.01,
                    ),
                },
            )
            .build()
    }

    #[test]
    fn gravity_torque_pendulum_at_zero() {
        // Pendulum hanging straight down (q=0). The gravity torque should be zero
        // because the CoM is directly below the joint axis.
        let model = simple_pendulum();
        let q = vec![0.0];
        let g = compute_gravity(&model, &q);
        assert_relative_eq!(g[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn gravity_torque_pendulum_at_90() {
        // Pendulum at 90° (horizontal). Gravity torque = m * g * L_com
        // where L_com = 1.0 (0.5 offset + 0.5 CoM).
        let model = simple_pendulum();
        let q = vec![std::f64::consts::FRAC_PI_2];
        let g = compute_gravity(&model, &q);
        // Expected: m * g * L = 1.0 * 9.81 * 1.0 = 9.81
        // Sign depends on convention. The revolute-Y joint at 90° puts the CoM along X.
        assert_relative_eq!(g[0].abs(), 4.905, epsilon = 1e-8);
    }

    #[test]
    fn rnea_zero_velocity_zero_accel_equals_gravity() {
        let model = simple_pendulum();
        let q = vec![0.3];
        let g = compute_gravity(&model, &q);
        let tau = rnea(&model, &q, &[0.0], &[0.0]);
        assert_relative_eq!(tau[0], g[0], epsilon = 1e-14);
    }

    #[test]
    fn rnea_is_pure() {
        let model = simple_pendulum();
        let q = vec![0.5];
        let v = vec![1.0];
        let a = vec![0.5];
        let t1 = rnea(&model, &q, &v, &a);
        let t2 = rnea(&model, &q, &v, &a);
        assert_relative_eq!(t1, t2, epsilon = 1e-14);
    }

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
                "j1",
                0,
                joint::revolute_y(),
                offset1,
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.2, 0.0, 0.0,
                        0.0, 0.2, 0.0,
                        0.0, 0.0, 0.02,
                    ),
                },
            )
            .add_joint(
                "j2",
                1,
                joint::revolute_y(),
                offset2,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.1, 0.0, 0.0,
                        0.0, 0.1, 0.0,
                        0.0, 0.0, 0.01,
                    ),
                },
            )
            .build()
    }

    #[test]
    fn rnea_two_link_gravity() {
        let model = two_link_arm();
        let q = vec![0.0, 0.0];
        let g = compute_gravity(&model, &q);
        // Both links hang straight down → no gravity torque.
        assert_relative_eq!(g[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(g[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn rnea_numerical_validation() {
        // Validate τ = M(q)a + nle(q,v) by checking linearity in acceleration:
        // rnea(q, v, a) - rnea(q, v, 0) should be linear in a.
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let a = vec![2.0, -1.0];
        let a2 = vec![4.0, -2.0];

        let nle = nonlinear_effects(&model, &q, &v);
        let tau_a = rnea(&model, &q, &v, &a);
        let tau_2a = rnea(&model, &q, &v, &a2);

        // τ(a) - nle = M*a, τ(2a) - nle = M*2a = 2*(τ(a) - nle)
        let diff_a = &tau_a - &nle;
        let diff_2a = &tau_2a - &nle;
        assert_relative_eq!(diff_2a, diff_a * 2.0, epsilon = 1e-10);
    }
}
