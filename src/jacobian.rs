//! Jacobian computation — pure function mapping (model, q) → Jacobian matrix.
//!
//! Computes the geometric Jacobian of each joint frame expressed in the world frame,
//! equivalent to `pinocchio::computeJointJacobians`.
//!
//! # Convention
//!
//! The Jacobian J is 6×nv, where column j corresponds to velocity DOF j.
//! The top 3 rows are angular velocity, the bottom 3 are linear velocity
//! (Pinocchio / Featherstone convention).

use crate::data::Data;
use crate::fk::forward_kinematics;
use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, Vector3};

/// Compute the world-frame geometric Jacobian for a specific joint.
///
/// **Pure function**: `(model, q, joint_idx) → 6×nv DMatrix`.
///
/// The Jacobian maps the full velocity vector q̇ to the spatial velocity of
/// joint `joint_idx` expressed in the world frame.
pub fn compute_joint_jacobian(model: &Model, q: &[f64], joint_idx: usize) -> DMatrix<f64> {
    assert!(joint_idx > 0 && joint_idx < model.joints.len());

    let data = forward_kinematics(model, q);
    compute_joint_jacobian_from_data(model, q, &data, joint_idx)
}

/// Same as `compute_joint_jacobian` but takes pre-computed FK data.
///
/// Useful when you already have FK results and want to avoid recomputing them.
pub fn compute_joint_jacobian_from_data(
    model: &Model,
    q: &[f64],
    data: &Data,
    joint_idx: usize,
) -> DMatrix<f64> {
    let mut jac = DMatrix::zeros(6, model.nv);

    // Walk from the target joint back to the root, accumulating columns.
    let target_pos = se3::translation(&data.oMi[joint_idx]);

    let mut current = joint_idx;
    while current > 0 {
        let joint = &model.joints[current];
        let vi = model.v_idx[current];
        let nv = joint.joint_type.nv();

        if nv > 0 {
            // Get joint axis in world frame
            let _qi = model.q_idx[current];
            let s_local = joint.joint_type.motion_subspace(q_slice(model, q, current));
            let r = se3::rotation_matrix(&data.oMi[current]);
            let p_joint = se3::translation(&data.oMi[current]);

            for col in 0..nv {
                // Angular part: R * s_angular
                let s_ang = Vector3::new(s_local[(0, col)], s_local[(1, col)], s_local[(2, col)]);
                let s_lin = Vector3::new(s_local[(3, col)], s_local[(4, col)], s_local[(5, col)]);

                let w = r * s_ang; // angular velocity axis in world
                let v_lin = r * s_lin; // linear velocity of joint frame

                // For revolute: linear velocity at target = ω × (p_target - p_joint)
                let lever = target_pos - p_joint;
                let v_at_target = v_lin + w.cross(&lever);

                jac[(0, vi + col)] = w[0];
                jac[(1, vi + col)] = w[1];
                jac[(2, vi + col)] = w[2];
                jac[(3, vi + col)] = v_at_target[0];
                jac[(4, vi + col)] = v_at_target[1];
                jac[(5, vi + col)] = v_at_target[2];
            }
        }

        current = joint.parent;
    }

    jac
}

/// Helper: extract the configuration slice for joint `i`.
fn q_slice<'a>(model: &Model, q: &'a [f64], i: usize) -> &'a [f64] {
    let qi = model.q_idx[i];
    &q[qi..qi + model.joints[i].joint_type.nq()]
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

    fn two_link_arm() -> Model {
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
    fn jacobian_two_link_zero_config() {
        let model = two_link_arm();
        let q = vec![0.0, 0.0];
        let jac = compute_joint_jacobian(&model, &q, 2);

        // At q = [0, 0], joint 2 is at (1, 0, 0).
        // ∂p/∂q1: shoulder rotation about Z → velocity = ω × (1,0,0) = (0,0,1)×(1,0,0) = (0,1,0)
        // But lever = target - shoulder = (1,0,0) - (0,0,0) = (1,0,0)
        // v = (0,0,1) × (1,0,0) = (0,1,0) - skipping the linear part as it's zero for revolute
        assert_relative_eq!(jac[(2, 0)], 1.0, epsilon = 1e-12); // angular z from shoulder
        assert_relative_eq!(jac[(4, 0)], 1.0, epsilon = 1e-12); // linear y from shoulder

        // ∂p/∂q2: elbow rotation about Z, joint 2 at (1,0,0), lever = (0,0,0)
        assert_relative_eq!(jac[(2, 1)], 1.0, epsilon = 1e-12); // angular z from elbow
        assert_relative_eq!(jac[(3, 1)], 0.0, epsilon = 1e-12); // no linear (zero lever)
        assert_relative_eq!(jac[(4, 1)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn jacobian_numerical_validation() {
        // Validate Jacobian via finite differences.
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let jac = compute_joint_jacobian(&model, &q, 2);
        let joint_idx = 2;

        let eps = 1e-8;
        let data_ref = crate::fk::forward_kinematics(&model, &q);
        let p_ref = se3::translation(&data_ref.oMi[joint_idx]);

        // Check linear part (rows 3-5) via finite differences on position
        for j in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[j] += eps;
            let data_plus = crate::fk::forward_kinematics(&model, &q_plus);
            let p_plus = se3::translation(&data_plus.oMi[joint_idx]);

            let dp = (p_plus - p_ref) / eps;
            assert_relative_eq!(jac[(3, j)], dp[0], epsilon = 1e-5);
            assert_relative_eq!(jac[(4, j)], dp[1], epsilon = 1e-5);
            assert_relative_eq!(jac[(5, j)], dp[2], epsilon = 1e-5);
        }
    }

    #[test]
    fn jacobian_is_pure() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let j1 = compute_joint_jacobian(&model, &q, 2);
        let j2 = compute_joint_jacobian(&model, &q, 2);
        assert_relative_eq!(j1, j2, epsilon = 1e-14);
    }
}
