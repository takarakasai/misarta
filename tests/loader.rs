//! Integration tests for URDF/SDF loading.
//!
//! Loads the actual test fixture files from the articara workspace and
//! validates kinematic results.

use approx::assert_relative_eq;
use misarta::fk::forward_kinematics;
use misarta::model::{LinkInertia, ModelBuilder};
use misarta::se3;
use misarta::joint;
use nalgebra::{Rotation3, Vector3};
use std::path::PathBuf;

fn fixture_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("model");
    p.push(rel);
    p
}

// ─── URDF fixture tests ────────────────────────────────────────────────────

#[test]
fn load_fixture_urdf() {
    let path = fixture_path("urdf/test_robot.urdf");
    let model = misarta::urdf::load_urdf(&path).expect("failed to load fixture URDF");

    // test_robot.urdf has: base_link, link1, link2, fixed_part
    // Joints: joint1 (revolute), joint2 (revolute), fixed_joint (fixed)
    assert_eq!(model.num_joints(), 3);
    // 2 revolute + 1 fixed = 2 DOF
    assert_eq!(model.nq, 2);
    assert_eq!(model.nv, 2);
}

#[test]
fn urdf_fixture_fk_zero() {
    let path = fixture_path("urdf/test_robot.urdf");
    let model = misarta::urdf::load_urdf(&path).unwrap();

    let q = model.neutral_q();
    let data = forward_kinematics(&model, &q);

    // At zero config, joint1 is at (0, 0, 0.05) from origin
    // (that's the <origin xyz="0 0 0.05"/> from the URDF)
    let t1 = se3::translation(&data.oMi[1]);
    assert_relative_eq!(t1[2], 0.05, epsilon = 1e-10);
}

#[test]
fn urdf_fixture_fk_nonzero() {
    let path = fixture_path("urdf/test_robot.urdf");
    let model = misarta::urdf::load_urdf(&path).unwrap();

    // Apply some joint angles and verify FK runs without error
    let q = vec![0.5, -0.3];
    let data = forward_kinematics(&model, &q);

    // Joint 2 should be offset from joint 1
    let t1 = se3::translation(&data.oMi[1]);
    let t2 = se3::translation(&data.oMi[2]);
    // They should not be exactly the same position
    assert!((t2 - t1).norm() > 0.01);
}

#[test]
fn urdf_fixture_jacobian() {
    let path = fixture_path("urdf/test_robot.urdf");
    let model = misarta::urdf::load_urdf(&path).unwrap();

    let q = vec![0.3, -0.5];
    // Validate Jacobian via finite differences on joint 2
    let jac = misarta::jacobian::compute_joint_jacobian(&model, &q, 2);
    let data_ref = forward_kinematics(&model, &q);
    let p_ref = se3::translation(&data_ref.oMi[2]);

    let eps = 1e-8;
    for j in 0..model.nv {
        let mut q_plus = q.clone();
        q_plus[j] += eps;
        let data_plus = forward_kinematics(&model, &q_plus);
        let p_plus = se3::translation(&data_plus.oMi[2]);
        let dp = (p_plus - p_ref) / eps;

        assert_relative_eq!(jac[(3, j)], dp[0], epsilon = 1e-5);
        assert_relative_eq!(jac[(4, j)], dp[1], epsilon = 1e-5);
        assert_relative_eq!(jac[(5, j)], dp[2], epsilon = 1e-5);
    }
}

// ─── SDF fixture tests ─────────────────────────────────────────────────────

#[test]
fn load_fixture_sdf() {
    let path = fixture_path("sdf/test_robot.sdf");
    let model = misarta::sdf::load_sdf(&path).expect("failed to load fixture SDF");

    // test_robot.sdf: base_link, link1, link2, fixed_part
    // Joints: joint1 (revolute), joint2 (revolute), fixed_joint (fixed)
    assert_eq!(model.num_joints(), 3);
    // 2 revolute + 1 fixed = 2 DOF
    assert_eq!(model.nq, 2);
    assert_eq!(model.nv, 2);
}

#[test]
fn sdf_fixture_fk_zero() {
    let path = fixture_path("sdf/test_robot.sdf");
    let model = misarta::sdf::load_sdf(&path).unwrap();

    let q = model.neutral_q();
    let data = forward_kinematics(&model, &q);

    let t1 = se3::translation(&data.oMi[1]);
    assert_relative_eq!(t1[2], 0.05, epsilon = 1e-10);
}

#[test]
fn sdf_fixture_jacobian() {
    let path = fixture_path("sdf/test_robot.sdf");
    let model = misarta::sdf::load_sdf(&path).unwrap();

    let q = vec![0.3, -0.5];
    let jac = misarta::jacobian::compute_joint_jacobian(&model, &q, 2);
    let data_ref = forward_kinematics(&model, &q);
    let p_ref = se3::translation(&data_ref.oMi[2]);

    let eps = 1e-8;
    for j in 0..model.nv {
        let mut q_plus = q.clone();
        q_plus[j] += eps;
        let data_plus = forward_kinematics(&model, &q_plus);
        let p_plus = se3::translation(&data_plus.oMi[2]);
        let dp = (p_plus - p_ref) / eps;

        assert_relative_eq!(jac[(3, j)], dp[0], epsilon = 1e-5);
        assert_relative_eq!(jac[(4, j)], dp[1], epsilon = 1e-5);
        assert_relative_eq!(jac[(5, j)], dp[2], epsilon = 1e-5);
    }
}

// ─── URDF ↔ SDF cross-validation ───────────────────────────────────────────

#[test]
fn urdf_and_sdf_produce_same_fk() {
    // The fixture URDF and SDF describe the same 2-DOF robot (ignoring the
    // extra fixed joint in the URDF). Compare FK for matching joint names.
    let urdf_model = misarta::urdf::load_urdf(&fixture_path("urdf/test_robot.urdf")).unwrap();
    let sdf_model = misarta::sdf::load_sdf(&fixture_path("sdf/test_robot.sdf")).unwrap();

    let q = vec![0.5, -0.3];
    let data_urdf = forward_kinematics(&urdf_model, &q);
    let data_sdf = forward_kinematics(&sdf_model, &q);

    // Match joints by name ("joint1", "joint2" exist in both)
    for target_name in ["joint1", "joint2"] {
        let urdf_idx = urdf_model
            .joints
            .iter()
            .position(|j| j.name == target_name)
            .unwrap();
        let sdf_idx = sdf_model
            .joints
            .iter()
            .position(|j| j.name == target_name)
            .unwrap();
        assert_relative_eq!(
            se3::to_homogeneous(&data_urdf.oMi[urdf_idx]),
            se3::to_homogeneous(&data_sdf.oMi[sdf_idx]),
            epsilon = 1e-10,
        );
    }
}

// ─── Model structural equality tests ────────────────────────────────────────

#[test]
fn urdf_reload_produces_identical_model() {
    // Loading the same URDF file twice must yield structurally equal models.
    let path = fixture_path("urdf/test_robot.urdf");
    let m1 = misarta::urdf::load_urdf(&path).unwrap();
    let m2 = misarta::urdf::load_urdf(&path).unwrap();
    assert!(m1.approx_eq(&m2, 1e-14));
}

#[test]
fn sdf_reload_produces_identical_model() {
    let path = fixture_path("sdf/test_robot.sdf");
    let m1 = misarta::sdf::load_sdf(&path).unwrap();
    let m2 = misarta::sdf::load_sdf(&path).unwrap();
    assert!(m1.approx_eq(&m2, 1e-14));
}

#[test]
fn urdf_matches_hand_built_model() {
    // Build the same 2-revolute-Y + 1-fixed robot by hand and compare.
    let urdf_model = misarta::urdf::load_urdf(&fixture_path("urdf/test_robot.urdf")).unwrap();

    let offset1 = se3::from_rotation_and_translation(
        &Rotation3::identity(),
        &Vector3::new(0.0, 0.0, 0.05),
    );
    let offset2 = se3::from_rotation_and_translation(
        &Rotation3::identity(),
        &Vector3::new(0.0, 0.0, 0.2),
    );
    let offset_fixed = se3::from_rotation_and_translation(
        &Rotation3::identity(),
        &Vector3::new(0.1, 0.0, 0.0),
    );

    let hand = ModelBuilder::<f64>::new()
        .name("test_robot")
        .add_joint(
            "joint1", 0,
            misarta::joint::JointType::Revolute { axis: Vector3::y() },
            offset1,
            LinkInertia { mass: 0.5, center_of_mass: Vector3::new(0.0, 0.0, 0.1) },
        )
        .add_joint(
            "fixed_joint", 0,
            misarta::joint::JointType::Fixed,
            offset_fixed,
            LinkInertia { mass: 0.1, center_of_mass: Vector3::zeros() },
        )
        .add_joint(
            "joint2", 1,
            misarta::joint::JointType::Revolute { axis: Vector3::y() },
            offset2,
            LinkInertia { mass: 0.3, center_of_mass: Vector3::new(0.0, 0.0, 0.075) },
        )
        .build();

    assert!(urdf_model.approx_eq(&hand, 1e-12));
}

#[test]
fn urdf_sdf_approx_eq_by_name() {
    // Both URDF and SDF now have the same 3 joints (joint1, joint2, fixed_joint).
    // approx_eq_by_name should find 3 matching joints with no mismatches.
    let urdf_model = misarta::urdf::load_urdf(&fixture_path("urdf/test_robot.urdf")).unwrap();
    let sdf_model = misarta::sdf::load_sdf(&fixture_path("sdf/test_robot.sdf")).unwrap();

    let (matching, mismatches) = urdf_model.approx_eq_by_name(&sdf_model, 1e-10);
    assert_eq!(matching, 3, "should match joint1, joint2, and fixed_joint");
    assert!(
        mismatches.is_empty(),
        "no mismatches expected, got: {:?}",
        mismatches
    );
}

#[test]
fn urdf_sdf_full_approx_eq() {
    // URDF and SDF now describe the exact same robot (same name, joints, inertias).
    // Full structural equality via approx_eq should pass.
    let urdf_model = misarta::urdf::load_urdf(&fixture_path("urdf/test_robot.urdf")).unwrap();
    let sdf_model = misarta::sdf::load_sdf(&fixture_path("sdf/test_robot.sdf")).unwrap();
    assert!(
        urdf_model.approx_eq(&sdf_model, 1e-10),
        "URDF and SDF models should be structurally equal",
    );
}

#[test]
fn approx_eq_by_name_detects_mismatch() {
    // Build two models with same joint name but different axis.
    let a = ModelBuilder::<f64>::new()
        .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
        .build();
    let b = ModelBuilder::<f64>::new()
        .add_joint("j1", 0, joint::revolute_x(), se3::identity(), LinkInertia::zero())
        .build();

    let (matching, mismatches) = a.approx_eq_by_name(&b, 1e-12);
    assert_eq!(matching, 0);
    assert_eq!(mismatches.len(), 1);
    assert_eq!(mismatches[0].0, "j1");
    assert!(mismatches[0].1.contains("joint_type"));
}
