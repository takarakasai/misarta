//! Constraint Jacobian computation.
//!
//! Evaluates the stacked Jacobian $J_c = J_2 - J_1$ for all constraints in a
//! [`ConstraintModel`].

use crate::data::Data;
use crate::fk::forward_kinematics;
use crate::frames::{compute_frame_jacobian_from_data, compute_frame_placement_from_data};
use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, Vector3};

use super::{ConstraintModel, ConstraintType, ReferenceFrame};

/// Compute the stacked constraint Jacobian.
///
/// For each constraint:
///
/// $$J_c = J_{\text{frame2}} - J_{\text{frame1}}$$
///
/// (with appropriate row extraction for 3-D constraints).
///
/// Returns a `DMatrix<f64>` of shape `(total_dim, nv)`.
pub fn compute_constraint_jacobian(
    model: &Model<f64>,
    q: &[f64],
    cm: &ConstraintModel<f64>,
) -> DMatrix<f64> {
    let data = forward_kinematics(model, q);
    compute_constraint_jacobian_from_data(model, q, &data, cm)
}

/// Same as [`compute_constraint_jacobian`] but with pre-computed FK data.
pub fn compute_constraint_jacobian_from_data(
    model: &Model<f64>,
    q: &[f64],
    data: &Data<f64>,
    cm: &ConstraintModel<f64>,
) -> DMatrix<f64> {
    let total = cm.total_dim();
    let nv = model.nv;
    let mut jc = DMatrix::zeros(total, nv);
    let mut row = 0;

    for c in &cm.constraints {
        // Compute frame Jacobians (6 × nv each)
        let j1 = if c.frame1.parent_joint == 0 {
            // Frame1 is anchored to world → zero Jacobian
            DMatrix::zeros(6, nv)
        } else {
            compute_frame_jacobian_from_data(model, q, data, &c.frame1)
        };

        let j2 = if c.frame2.parent_joint == 0 {
            DMatrix::zeros(6, nv)
        } else {
            compute_frame_jacobian_from_data(model, q, data, &c.frame2)
        };

        // Relative Jacobian: J_c = J2 - J1
        let j_rel = &j2 - &j1;

        match c.constraint_type {
            ConstraintType::Contact6D => {
                match c.reference_frame {
                    ReferenceFrame::World => {
                        // Use as-is (world-frame Jacobian)
                        jc.view_mut((row, 0), (6, nv)).copy_from(&j_rel);
                    }
                    ReferenceFrame::Local => {
                        // Rotate to frame1's local frame
                        let m1 = compute_frame_placement_from_data(data, &c.frame1);
                        let r1 = se3::rotation_matrix(&m1);
                        let r1t = r1.transpose();
                        for col in 0..nv {
                            let w = Vector3::new(j_rel[(0, col)], j_rel[(1, col)], j_rel[(2, col)]);
                            let v = Vector3::new(j_rel[(3, col)], j_rel[(4, col)], j_rel[(5, col)]);
                            let w_l = &r1t * w;
                            let v_l = &r1t * v;
                            jc[(row, col)] = w_l[0];
                            jc[(row + 1, col)] = w_l[1];
                            jc[(row + 2, col)] = w_l[2];
                            jc[(row + 3, col)] = v_l[0];
                            jc[(row + 4, col)] = v_l[1];
                            jc[(row + 5, col)] = v_l[2];
                        }
                    }
                }
                row += 6;
            }
            ConstraintType::Contact3D => {
                match c.reference_frame {
                    ReferenceFrame::World => {
                        // Extract linear rows (rows 3-5)
                        jc.view_mut((row, 0), (3, nv))
                            .copy_from(&j_rel.view((3, 0), (3, nv)));
                    }
                    ReferenceFrame::Local => {
                        let m1 = compute_frame_placement_from_data(data, &c.frame1);
                        let r1 = se3::rotation_matrix(&m1);
                        let r1t = r1.transpose();
                        for col in 0..nv {
                            let v = Vector3::new(j_rel[(3, col)], j_rel[(4, col)], j_rel[(5, col)]);
                            let v_l = &r1t * v;
                            jc[(row, col)] = v_l[0];
                            jc[(row + 1, col)] = v_l[1];
                            jc[(row + 2, col)] = v_l[2];
                        }
                    }
                }
                row += 3;
            }
        }
    }

    jc
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::error::compute_constraint_error;
    use super::super::RigidConstraint;
    use crate::frames::Frame;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{Rotation3, Vector3};

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

    fn dual_arm() -> Model<f64> {
        let shoulder_y = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.3, 0.0),
        );
        let shoulder_ny = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, -0.3, 0.0),
        );
        let forearm = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        ModelBuilder::new()
            .name("dual_arm")
            .add_joint("base", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("left_shoulder", 1, joint::revolute_x(), shoulder_y, LinkInertia::zero())
            .add_joint("left_elbow", 2, joint::revolute_x(), forearm.clone(), LinkInertia::zero())
            .add_joint("right_shoulder", 1, joint::revolute_x(), shoulder_ny, LinkInertia::zero())
            .add_joint("right_elbow", 4, joint::revolute_x(), forearm, LinkInertia::zero())
            .build()
    }

    fn frame_at_joint(name: &str, joint_idx: usize) -> Frame<f64> {
        Frame {
            name: name.to_string(),
            parent_joint: joint_idx,
            placement: se3::identity(),
        }
    }

    fn frame_with_offset(name: &str, joint_idx: usize, offset: Vector3<f64>) -> Frame<f64> {
        Frame {
            name: name.to_string(),
            parent_joint: joint_idx,
            placement: se3::from_rotation_and_translation(&Rotation3::identity(), &offset),
        }
    }

    #[test]
    fn jacobian_shape() {
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        assert_eq!(jc.nrows(), 3);
        assert_eq!(jc.ncols(), model.nv);
    }

    #[test]
    fn jacobian_6d_shape() {
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        assert_eq!(jc.nrows(), 6);
        assert_eq!(jc.ncols(), model.nv);
    }

    #[test]
    fn jacobian_multiple_constraints_stacked() {
        let model = dual_arm();
        let q = vec![0.0; model.nq];

        let left_tip = frame_at_joint("left_tip", 3);
        let right_tip = frame_at_joint("right_tip", 5);
        let world_anchor = Frame {
            name: "world".into(),
            parent_joint: 0,
            placement: se3::from_rotation_and_translation(
                &Rotation3::identity(),
                &Vector3::new(0.0, 0.0, -1.0),
            ),
        };

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(left_tip.clone(), right_tip),
            RigidConstraint::pose(left_tip, world_anchor),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        assert_eq!(jc.nrows(), 9);
        assert_eq!(jc.ncols(), model.nv);
    }

    #[test]
    fn jacobian_finite_diff_validation_3d() {
        let model = y_tree();
        let q = vec![0.3, -0.2, 0.5];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let e0 = compute_constraint_error(&model, &q, &cm);

        let eps = 1e-7;
        for col in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[col] += eps;
            let e_plus = compute_constraint_error(&model, &q_plus, &cm);
            let de = (&e_plus - &e0) / eps;

            for row in 0..3 {
                assert_relative_eq!(jc[(row, col)], de[row], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn jacobian_finite_diff_validation_6d() {
        let model = y_tree();
        let q = vec![0.001, 0.001, -0.001];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::pose(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let e0 = compute_constraint_error(&model, &q, &cm);

        let eps = 1e-7;
        for col in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[col] += eps;
            let e_plus = compute_constraint_error(&model, &q_plus, &cm);
            let de = (&e_plus - &e0) / eps;

            for row in 0..6 {
                assert_relative_eq!(jc[(row, col)], de[row], epsilon = 5e-3);
            }
        }
    }

    #[test]
    fn jacobian_world_anchor_single_branch() {
        let model = y_tree();
        let q = vec![0.2, 0.0, 0.0];

        let f2 = frame_at_joint("left", 2);
        let world_origin = Frame {
            name: "world_origin".into(),
            parent_joint: 0,
            placement: se3::identity(),
        };

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(world_origin, f2),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let data = forward_kinematics(&model, &q);
        let j_f2 = compute_frame_jacobian_from_data(&model, &q, &data, &frame_at_joint("left", 2));
        let j_f2_lin = j_f2.rows(3, 3);

        assert_relative_eq!(jc, j_f2_lin.into_owned(), epsilon = 1e-12);
    }

    #[test]
    fn cross_branch_jacobian_nonzero_both_branches() {
        let model = y_tree();
        let q = vec![0.3, 0.2, -0.4];

        let f2 = frame_with_offset("left_tip", 2, Vector3::new(0.0, 0.0, 0.5));
        let f3 = frame_with_offset("right_tip", 3, Vector3::new(0.0, 0.0, 0.5));

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);

        let col1_norm = jc.column(1).norm();
        let col2_norm = jc.column(2).norm();
        assert!(col1_norm > 1e-6, "j2 column should be nonzero: {col1_norm}");
        assert!(col2_norm > 1e-6, "j3 column should be nonzero: {col2_norm}");
    }

    #[test]
    fn cross_branch_jacobian_common_ancestor_column() {
        let model = y_tree();
        let q = vec![0.0; model.nq];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let col0 = jc.column(0);
        assert_relative_eq!(col0.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn local_jacobian_finite_diff_validation() {
        let model = y_tree();
        let q = vec![0.2, 0.3, -0.1];

        let f2 = frame_at_joint("left", 2);
        let f3 = frame_at_joint("right", 3);

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3)
                .with_reference_frame(super::super::ReferenceFrame::Local),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let e0 = compute_constraint_error(&model, &q, &cm);

        let eps = 1e-7;
        for col in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[col] += eps;
            let e_plus = compute_constraint_error(&model, &q_plus, &cm);
            let de = (&e_plus - &e0) / eps;

            for row in 0..3 {
                assert_relative_eq!(jc[(row, col)], de[row], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn constraint_jacobian_in_kkt() {
        use crate::constrained::constrained_forward_dynamics;

        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        let inertia = LinkInertia {
            mass: 1.0,
            center_of_mass: Vector3::new(0.5, 0.0, 0.0),
            rotational_inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01)),
        };
        let model = ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), inertia.clone())
            .add_joint("j2", 1, joint::revolute_x(), offset.clone(), inertia.clone())
            .add_joint("j3", 1, joint::revolute_y(), offset, inertia.clone())
            .build();

        let q = vec![0.1, 0.2, -0.2];
        let v = vec![0.0; model.nv];
        let tau = vec![0.0; model.nv];

        let f2 = frame_with_offset("left_tip", 2, Vector3::new(0.5, 0.0, 0.0));
        let f3 = frame_with_offset("right_tip", 3, Vector3::new(0.5, 0.0, 0.0));

        let cm = ConstraintModel::from_constraints(vec![
            RigidConstraint::position(f2, f3),
        ]);

        let jc = compute_constraint_jacobian(&model, &q, &cm);
        let gamma = nalgebra::DVector::zeros(jc.nrows());

        let result = constrained_forward_dynamics(&model, &q, &v, &tau, &jc, &gamma);
        assert_eq!(result.qdd.len(), model.nv);
        assert_eq!(result.lambda.len(), 3);
    }
}
