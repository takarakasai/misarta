//! Constraint error computation.
//!
//! Evaluates the stacked error vector for all constraints in a
//! [`ConstraintModel`].

use crate::data::Data;
use crate::fk::forward_kinematics;
use crate::frames::compute_frame_placement_from_data;
use crate::model::Model;
use crate::se3::{self, SE3};
use nalgebra::{DVector, Vector3};

use super::{ConstraintModel, ConstraintType, ReferenceFrame};

// NOTE on sign convention:
//
// The constraint error is defined as e = actual − desired (e.g. p2 − expected).
// The constraint Jacobian is J_c = de/dq ≈ J2 − J1.
// To drive e → 0 we need dq = −J_c⁺ e (negative sign).

/// Compute the stacked constraint error vector.
///
/// For each constraint, the error depends on the type:
///
/// - **Contact6D** (world frame): $e = \log(M_1^{-1} M_2 M_{\text{des}}^{-1})$
///   — the 6-D pose error expressed in the world frame.
/// - **Contact3D** (world frame): $e = t_2 - t_1 - R_1 t_{\text{des}}$
///   — the 3-D position error in the world frame.
///
/// Returns a `DVector<f64>` of length equal to `cm.total_dim()`.
pub fn compute_constraint_error(
    model: &Model<f64>,
    q: &[f64],
    cm: &ConstraintModel<f64>,
) -> DVector<f64> {
    let data = forward_kinematics(model, q);
    compute_constraint_error_from_data(&data, cm)
}

/// Same as [`compute_constraint_error`] but with pre-computed FK data.
pub fn compute_constraint_error_from_data(
    data: &Data<f64>,
    cm: &ConstraintModel<f64>,
) -> DVector<f64> {
    let total = cm.total_dim();
    let mut err = DVector::zeros(total);
    let mut row = 0;

    for c in &cm.constraints {
        let m1 = compute_frame_placement_from_data(data, &c.frame1);
        let m2 = compute_frame_placement_from_data(data, &c.frame2);

        match c.constraint_type {
            ConstraintType::Contact6D => {
                let e = compute_pose_error_6d(&m1, &m2, &c.desired_relative_placement, c.reference_frame);
                for i in 0..6 {
                    err[row + i] = e[i];
                }
                row += 6;
            }
            ConstraintType::Contact3D => {
                let e = compute_position_error_3d(&m1, &m2, &c.desired_relative_placement, c.reference_frame);
                for i in 0..3 {
                    err[row + i] = e[i];
                }
                row += 3;
            }
        }
    }

    err
}

/// 6-D pose error: $\log(M_{\text{des}}^{-1} M_1^{-1} M_2)$
fn compute_pose_error_6d(
    m1: &SE3<f64>,
    m2: &SE3<f64>,
    m_des: &SE3<f64>,
    reference_frame: ReferenceFrame,
) -> nalgebra::Vector6<f64> {
    // Relative placement: M_1^{-1} * M_2
    let m_rel = se3::compose(&se3::inverse(m1), m2);
    // Error: M_rel * M_des^{-1}  (== identity when constraint is satisfied)
    let m_err = se3::compose(&m_rel, &se3::inverse(m_des));
    let log_err = se3::log(&m_err);

    match reference_frame {
        ReferenceFrame::World => {
            // Rotate error from frame1 to world: R1 * log_err
            let r1 = se3::rotation_matrix(m1);
            let omega = Vector3::new(log_err[0], log_err[1], log_err[2]);
            let v = Vector3::new(log_err[3], log_err[4], log_err[5]);
            let omega_w = &r1 * omega;
            let v_w = &r1 * v;
            nalgebra::Vector6::new(omega_w[0], omega_w[1], omega_w[2], v_w[0], v_w[1], v_w[2])
        }
        ReferenceFrame::Local => log_err,
    }
}

/// 3-D position error
fn compute_position_error_3d(
    m1: &SE3<f64>,
    m2: &SE3<f64>,
    m_des: &SE3<f64>,
    reference_frame: ReferenceFrame,
) -> Vector3<f64> {
    let p1 = se3::translation(m1);
    let p2 = se3::translation(m2);
    let p_des = se3::translation(m_des);

    // Expected position of frame2 in world: p1 + R1 * p_des
    let r1 = se3::rotation_matrix(m1);
    let expected = &p1 + &r1 * &p_des;
    let err_world = p2 - expected;

    match reference_frame {
        ReferenceFrame::World => err_world,
        ReferenceFrame::Local => r1.transpose() * err_world,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::Frame;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{Rotation3, Vector3};

    use super::super::RigidConstraint;

    fn y_tree() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        ModelBuilder::new()
            .name("y_tree")
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j2", 1, joint::revolute_x(), offset.clone(), LinkInertia::zero())
            .add_joint("j3", 1, joint::revolute_y(), offset, LinkInertia::zero())
            .build()
    }

    fn frame_at_joint(name: &str, joint_idx: usize) -> Frame<f64> {
        Frame {
            name: name.to_string(),
            parent_joint: joint_idx,
            placement: se3::identity(),
        }
    }

    #[test]
    fn error_zero_when_frames_coincide() {
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);
        let err = compute_constraint_error(&model, &q, &cm);
        assert_relative_eq!(err.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn error_nonzero_when_frames_differ() {
        let model = y_tree();
        let q = vec![0.0, 0.5, -0.5];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2, f3),
        ]);
        let err = compute_constraint_error(&model, &q, &cm);
        assert!(err.norm() > 0.01, "error should be nonzero when frames differ");
    }

    #[test]
    fn error_with_desired_offset() {
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let desired = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.5, 0.0, 0.0),
        );

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3).with_desired_placement(desired),
        ]);

        let err = compute_constraint_error(&model, &q, &cm);
        assert_relative_eq!(err[0], -0.5, epsilon = 1e-12);
        assert_relative_eq!(err[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(err[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn local_frame_error_rotates() {
        let model = y_tree();
        let q_nonzero = vec![std::f64::consts::FRAC_PI_4, 0.2, -0.3];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        // World-frame constraint
        let cm_world = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2.clone(), f3.clone()),
        ]);
        let e_world = compute_constraint_error(&model, &q_nonzero, &cm_world);

        // Local-frame constraint
        let cm_local = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2.clone(), f3.clone())
                .with_reference_frame(ReferenceFrame::Local),
        ]);
        let e_local = compute_constraint_error(&model, &q_nonzero, &cm_local);

        // Both should have the same norm (rotation preserves length)
        assert_relative_eq!(e_world.norm(), e_local.norm(), epsilon = 1e-10);

        // In general the components should differ (non-trivial rotation)
        let diff = (&e_world - &e_local).norm();
        assert!(diff > 1e-6, "world and local errors should differ in components");
    }
}
