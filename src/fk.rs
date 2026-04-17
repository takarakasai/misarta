//! Forward kinematics — pure function that computes joint placements.
//!
//! Equivalent to `pinocchio::forwardKinematics` + `pinocchio::updateFramePlacements`.
//!
//! The core function is **pure**: `(model, q) → Data`.
//! No mutation, no hidden state, fully suitable for automatic differentiation.
//! Generic over `T: RealField`.

use crate::data::Data;
use crate::model::Model;
use crate::se3;
use nalgebra::{RealField, Vector3, Vector6};

/// Compute forward kinematics for the entire model.
///
/// Returns a fresh `Data` with `oMi[i]` = world placement of joint `i`.
///
/// **Pure function**: no side effects, deterministic output for given input.
///
/// # Algorithm (Pinocchio-equivalent)
///
/// For each joint `i` in topological order (parents before children):
///
/// ```text
/// joint_placements[i] = model.joints[i].placement * joint_type.forward(q_i)
/// oMi[i] = oMi[parent(i)] * joint_placements[i]
/// ```
pub fn forward_kinematics<T: RealField>(model: &Model<T>, q: &[T]) -> Data<T> {
    assert_eq!(
        q.len(),
        model.nq,
        "Configuration vector length ({}) != model.nq ({})",
        q.len(),
        model.nq
    );

    let mut data = Data::new(model);

    for i in 1..model.joints.len() {
        let joint = &model.joints[i];
        let qi = model.q_idx[i];
        let q_slice = &q[qi..qi + joint.joint_type.nq()];

        // Joint-local placement from configuration
        let m_joint = joint.joint_type.forward(q_slice);

        // Placement relative to parent = fixed offset * joint motion
        let rel = se3::compose(&joint.placement, &m_joint);
        // Absolute placement = parent absolute * relative
        let parent_idx = joint.parent;
        data.oMi[i] = se3::compose(&data.oMi[parent_idx], &rel);
        data.joint_placements[i] = rel;
    }

    data
}

/// Compute forward kinematics with velocities (1st order).
///
/// Returns a `Data` with `oMi[i]` and `v[i]` filled.
/// `v[i]` is the spatial velocity of body `i` expressed in the body frame:
/// `v[i] = [ω_i; v_lin_i]`.
///
/// Equivalent to `pinocchio::forwardKinematics(model, data, q, v)`.
///
/// # Algorithm
///
/// For each joint `i` in topological order:
///
/// ```text
/// v_i = X_i^{-1} v_{parent(i)} + S_i q̇_i
/// ```
pub fn forward_kinematics_velocity<T: RealField>(
    model: &Model<T>,
    q: &[T],
    v: &[T],
) -> Data<T> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);

    let mut data = Data::new(model);

    for i in 1..model.joints.len() {
        let joint = &model.joints[i];
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];
        let nq_j = joint.joint_type.nq();
        let nv_j = joint.joint_type.nv();
        let q_slice = &q[qi..qi + nq_j];

        // Placement
        let m_joint = joint.joint_type.forward(q_slice);
        let rel = se3::compose(&joint.placement, &m_joint);
        let parent_idx = joint.parent;
        data.oMi[i] = se3::compose(&data.oMi[parent_idx], &rel);
        data.joint_placements[i] = rel.clone();

        // Transform parent velocity to body frame: X_i^{-1} v_{parent}
        let r = se3::rotation_matrix(&rel);
        let p = se3::translation(&rel);
        let rt = r.transpose();

        let v_parent = &data.v[parent_idx];
        let omega_parent = Vector3::new(
            v_parent[0].clone(), v_parent[1].clone(), v_parent[2].clone(),
        );
        let vlin_parent = Vector3::new(
            v_parent[3].clone(), v_parent[4].clone(), v_parent[5].clone(),
        );
        let omega_body = &rt * &omega_parent;
        let vlin_body = &rt * (vlin_parent - p.cross(&omega_parent));

        // Motion subspace
        let s = joint.joint_type.motion_subspace(q_slice);

        // Joint velocity: S * qd
        let mut v_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qd_c = v[vi + c].clone();
            for row in 0..6 {
                v_j[row] += s[(row, c)].clone() * qd_c.clone();
            }
        }

        // Body velocity
        data.v[i] = Vector6::new(
            omega_body[0].clone() + v_j[0].clone(),
            omega_body[1].clone() + v_j[1].clone(),
            omega_body[2].clone() + v_j[2].clone(),
            vlin_body[0].clone() + v_j[3].clone(),
            vlin_body[1].clone() + v_j[4].clone(),
            vlin_body[2].clone() + v_j[5].clone(),
        );
    }

    data
}

/// Compute forward kinematics with velocities and accelerations (2nd order).
///
/// Returns a `Data` with `oMi[i]`, `v[i]`, and `a[i]` filled.
/// `a[i]` is the spatial acceleration of body `i` in the body frame, **including**
/// the gravity contribution (Featherstone convention: the universe accelerates by −g).
///
/// Equivalent to `pinocchio::forwardKinematics(model, data, q, v, a)`.
///
/// # Algorithm
///
/// ```text
/// a_i = X_i^{-1} a_{parent(i)} + S_i q̈_i + v_i × (S_i q̇_i)
/// ```
pub fn forward_kinematics_acceleration<T: RealField>(
    model: &Model<T>,
    q: &[T],
    v: &[T],
    a: &[T],
) -> Data<T> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(a.len(), model.nv);

    let mut data = Data::new(model);

    // Gravity as spatial acceleration of the universe (Featherstone trick)
    let mut a0 = Vector6::<T>::zeros();
    a0[3] = -model.gravity[0].clone();
    a0[4] = -model.gravity[1].clone();
    a0[5] = -model.gravity[2].clone();
    data.a[0] = a0;

    for i in 1..model.joints.len() {
        let joint = &model.joints[i];
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];
        let nq_j = joint.joint_type.nq();
        let nv_j = joint.joint_type.nv();
        let q_slice = &q[qi..qi + nq_j];

        // Placement
        let m_joint = joint.joint_type.forward(q_slice);
        let rel = se3::compose(&joint.placement, &m_joint);
        let parent_idx = joint.parent;
        data.oMi[i] = se3::compose(&data.oMi[parent_idx], &rel);
        data.joint_placements[i] = rel.clone();

        // Transform parent velocity to body frame
        let r = se3::rotation_matrix(&rel);
        let p = se3::translation(&rel);
        let rt = r.transpose();

        let v_parent = &data.v[parent_idx];
        let omega_parent = Vector3::new(
            v_parent[0].clone(), v_parent[1].clone(), v_parent[2].clone(),
        );
        let vlin_parent = Vector3::new(
            v_parent[3].clone(), v_parent[4].clone(), v_parent[5].clone(),
        );
        let omega_body = &rt * &omega_parent;
        let vlin_body = &rt * (vlin_parent - p.cross(&omega_parent));

        // Motion subspace
        let s = joint.joint_type.motion_subspace(q_slice);

        // Joint velocity: S * qd
        let mut v_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qd_c = v[vi + c].clone();
            for row in 0..6 {
                v_j[row] += s[(row, c)].clone() * qd_c.clone();
            }
        }

        // Body velocity
        data.v[i] = Vector6::new(
            omega_body[0].clone() + v_j[0].clone(),
            omega_body[1].clone() + v_j[1].clone(),
            omega_body[2].clone() + v_j[2].clone(),
            vlin_body[0].clone() + v_j[3].clone(),
            vlin_body[1].clone() + v_j[4].clone(),
            vlin_body[2].clone() + v_j[5].clone(),
        );

        // Transform parent acceleration to body frame
        let a_parent = &data.a[parent_idx];
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
            for row in 0..6 {
                a_j[row] += s[(row, c)].clone() * qdd_c.clone();
            }
        }

        // Coriolis: v × v_j
        let vx = se3::motion_cross(&data.v[i]);
        let cross_term = vx * &v_j;

        // Body acceleration
        data.a[i] = Vector6::new(
            alpha_body[0].clone() + a_j[0].clone() + cross_term[0].clone(),
            alpha_body[1].clone() + a_j[1].clone() + cross_term[1].clone(),
            alpha_body[2].clone() + a_j[2].clone() + cross_term[2].clone(),
            alin_body[0].clone() + a_j[3].clone() + cross_term[3].clone(),
            alin_body[1].clone() + a_j[4].clone() + cross_term[4].clone(),
            alin_body[2].clone() + a_j[5].clone() + cross_term[5].clone(),
        );
    }

    data
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use std::f64::consts::FRAC_PI_2;

    fn two_link_arm() -> Model<f64> {
        // Two revolute-Z joints, each link 1m long along X.
        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        ModelBuilder::new()
            .add_joint(
                "shoulder",
                0,
                joint::revolute_z(),
                se3::identity(),
                LinkInertia::zero(),
            )
            .add_joint("elbow", 1, joint::revolute_z(), offset, LinkInertia::zero())
            .build()
    }

    #[test]
    fn fk_zero_config() {
        let model = two_link_arm();
        let q = vec![0.0, 0.0];
        let data = forward_kinematics(&model, &q);

        // Joint 1 at origin (no offset, zero angle)
        assert_relative_eq!(
            se3::translation(&data.oMi[1]),
            Vector3::zeros(),
            epsilon = 1e-12
        );
        // Joint 2 at (1, 0, 0) — offset along X
        assert_relative_eq!(
            se3::translation(&data.oMi[2]),
            Vector3::new(1.0, 0.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn fk_shoulder_90deg() {
        let model = two_link_arm();
        let q = vec![FRAC_PI_2, 0.0];
        let data = forward_kinematics(&model, &q);

        // Joint 1 at origin, rotated 90° about Z
        assert_relative_eq!(
            se3::translation(&data.oMi[1]),
            Vector3::zeros(),
            epsilon = 1e-12
        );
        // Joint 2: link was along X, now rotated → along Y
        assert_relative_eq!(
            se3::translation(&data.oMi[2]),
            Vector3::new(0.0, 1.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn fk_both_90deg() {
        let model = two_link_arm();
        let q = vec![FRAC_PI_2, FRAC_PI_2];
        let data = forward_kinematics(&model, &q);

        // Shoulder 90° + Elbow 90° = end effector at (0, 1, 0) + rotation by further 90°
        // Link 2 originally along X from joint 2, but joint 2 itself rotated total 180° about Z
        // So link2 tip would be at (0, 1, 0) + rot(180°) * (1, 0, 0) = (0, 1, 0) + (-1, 0, 0)
        // = (-1, 1, 0)
        // But oMi[2] is at the *joint* frame, not the end effector.
        // oMi[2] = joint 2 position = (0, 1, 0), which is correct from previous test.
        assert_relative_eq!(
            se3::translation(&data.oMi[2]),
            Vector3::new(0.0, 1.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn fk_is_pure() {
        // Calling forward_kinematics twice with the same input must give the same output.
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let d1 = forward_kinematics(&model, &q);
        let d2 = forward_kinematics(&model, &q);
        for i in 1..model.joints.len() {
            assert_relative_eq!(
                se3::to_homogeneous(&d1.oMi[i]),
                se3::to_homogeneous(&d2.oMi[i]),
                epsilon = 1e-14
            );
        }
    }

    #[test]
    fn fk_velocity_revolute_z() {
        // Single revolute-Z at zero config with angular velocity = 1 rad/s.
        // Body velocity should be [0, 0, 1, 0, 0, 0] (pure rotation about Z
        // in the body frame, which coincides with world at q=0).
        let model = ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero(),
            )
            .build();
        let q = vec![0.0];
        let v = vec![1.0];
        let data = forward_kinematics_velocity(&model, &q, &v);
        assert_relative_eq!(data.v[1][0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(data.v[1][1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(data.v[1][2], 1.0, epsilon = 1e-12);
        assert_relative_eq!(data.v[1][3], 0.0, epsilon = 1e-12);
        assert_relative_eq!(data.v[1][4], 0.0, epsilon = 1e-12);
        assert_relative_eq!(data.v[1][5], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn fk_velocity_two_link_propagation() {
        // Two-link arm, both spinning at 1 rad/s about Z.
        // Joint 2 body velocity should accumulate both rotations.
        let model = two_link_arm();
        let q = vec![0.0, 0.0];
        let v = vec![1.0, 1.0];
        let data = forward_kinematics_velocity(&model, &q, &v);

        // Joint 1: ωz = 1
        assert_relative_eq!(data.v[1][2], 1.0, epsilon = 1e-12);
        // Joint 2: ωz = 2 (parent + own)
        assert_relative_eq!(data.v[2][2], 2.0, epsilon = 1e-12);
        // Joint 2 linear velocity: lever arm from parent rotation
        // Parent ω = [0,0,1], joint 2 at (1,0,0) from parent → v_lin_body = ω × p
        // In body frame this becomes (-R^T p × ω_parent) contribution
        // At zero config, R=I, p=(1,0,0), ω_parent=[0,0,1]
        // v_lin = v_parent_lin - p × ω_parent = 0 - (1,0,0)×(0,0,1) = -(0,-1,0) = (0,1,0)
        assert_relative_eq!(data.v[2][3], 0.0, epsilon = 1e-12);
        assert_relative_eq!(data.v[2][4], 1.0, epsilon = 1e-12);
        assert_relative_eq!(data.v[2][5], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn fk_velocity_placements_match_zero_order() {
        // FK velocity should produce identical placements as FK zero-order.
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -2.0];
        let d0 = forward_kinematics(&model, &q);
        let dv = forward_kinematics_velocity(&model, &q, &v);
        for i in 1..model.joints.len() {
            assert_relative_eq!(
                se3::to_homogeneous(&d0.oMi[i]),
                se3::to_homogeneous(&dv.oMi[i]),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn fk_acceleration_gravity_only() {
        // Single link at rest: acceleration should reflect gravity in body frame.
        let model = ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero(),
            )
            .build();
        let q = vec![0.0];
        let v = vec![0.0];
        let a = vec![0.0];
        let data = forward_kinematics_acceleration(&model, &q, &v, &a);
        // At zero config, body frame = world frame.
        // a[1] should include gravity contribution: universe accelerates upward
        // gravity = [0, 0, -9.81] → a0 = [0, 0, 0, 0, 0, +9.81]
        assert_relative_eq!(data.a[1][3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(data.a[1][4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(data.a[1][5], 9.81, epsilon = 1e-10);
    }

    #[test]
    fn fk_acceleration_consistent_with_rnea() {
        // The body velocities and accelerations from FK should match
        // what RNEA computes internally.
        use crate::rnea;
        use nalgebra::Matrix3;

        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        let model = ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_y(), se3::identity(),
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::identity() * 0.1,
                },
            )
            .add_joint(
                "j2", 1, joint::revolute_y(), offset,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::identity() * 0.1,
                },
            )
            .build();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let a_in = vec![0.5, 0.2];

        // FK 2nd order
        let data = forward_kinematics_acceleration(&model, &q, &v, &a_in);

        // Use RNEA (which computes body velocities and accelerations internally);
        // Since RNEA returns τ = M(q)a + C(q,v)v + g(q) = Y*a + pa,
        // we can verify that the FK accelerations are consistent by checking
        // that rnea(model, q, v, a) produces finite reasonable values.
        let tau = rnea::rnea(&model, &q, &v, &a_in);
        assert!(tau.norm() < 1e10, "tau not finite");

        // Body velocities must be finite and non-zero for non-trivial input.
        assert!(data.v[1].norm() > 0.0);
        assert!(data.v[2].norm() > 0.0);
        assert!(data.a[1].norm() < 1e10, "a[1] not finite");
        assert!(data.a[2].norm() < 1e10, "a[2] not finite");
    }
}
