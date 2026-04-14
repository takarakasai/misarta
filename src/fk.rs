//! Forward kinematics — pure function that computes joint placements.
//!
//! Equivalent to `pinocchio::forwardKinematics` + `pinocchio::updateFramePlacements`.
//!
//! The core function is **pure**: `(model, q) → Data`.
//! No mutation, no hidden state, fully suitable for automatic differentiation.

use crate::data::Data;
use crate::model::Model;
use crate::se3;

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
pub fn forward_kinematics(model: &Model, q: &[f64]) -> Data {
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
        data.joint_placements[i] = rel;

        // Absolute placement = parent absolute * relative
        let parent_idx = joint.parent;
        data.oMi[i] = se3::compose(&data.oMi[parent_idx], &rel);
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

    fn two_link_arm() -> Model {
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
}
