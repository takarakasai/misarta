//! Integration tests for misarta kinematics.
//!
//! Tests multi-link chains, branched trees, and validates the Jacobian
//! against finite-difference approximations.

use approx::assert_relative_eq;
use misarta::fk::forward_kinematics;
use misarta::jacobian::compute_joint_jacobian;
use misarta::joint;
use misarta::model::{LinkInertia, ModelBuilder};
use misarta::se3;
use nalgebra::{Rotation3, Vector3};
use std::f64::consts::{FRAC_PI_2, PI};

// ─── Helper ─────────────────────────────────────────────────────────────────

fn link_offset(x: f64, y: f64, z: f64) -> misarta::se3::SE3 {
    se3::from_rotation_and_translation(&Rotation3::identity(), &Vector3::new(x, y, z))
}

// ─── 3-DOF planar arm ──────────────────────────────────────────────────────

fn three_link_planar() -> misarta::model::Model {
    ModelBuilder::new()
        .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
        .add_joint("j2", 1, joint::revolute_z(), link_offset(1.0, 0.0, 0.0), LinkInertia::zero())
        .add_joint("j3", 2, joint::revolute_z(), link_offset(1.0, 0.0, 0.0), LinkInertia::zero())
        .build()
}

#[test]
fn three_link_fk_straight() {
    let model = three_link_planar();
    let q = vec![0.0, 0.0, 0.0];
    let data = forward_kinematics(&model, &q);

    assert_relative_eq!(se3::translation(&data.oMi[1]), Vector3::zeros(), epsilon = 1e-12);
    assert_relative_eq!(
        se3::translation(&data.oMi[2]),
        Vector3::new(1.0, 0.0, 0.0),
        epsilon = 1e-12,
    );
    assert_relative_eq!(
        se3::translation(&data.oMi[3]),
        Vector3::new(2.0, 0.0, 0.0),
        epsilon = 1e-12,
    );
}

#[test]
fn three_link_fk_folded() {
    let model = three_link_planar();
    // Each joint 90°: forms an L-shape
    let q = vec![0.0, FRAC_PI_2, 0.0];
    let data = forward_kinematics(&model, &q);

    // Joint 3 is 1m along X then 1m along Y (after 90° elbow bend)
    assert_relative_eq!(
        se3::translation(&data.oMi[3]),
        Vector3::new(1.0, 1.0, 0.0),
        epsilon = 1e-12,
    );
}

#[test]
fn three_link_full_fold() {
    let model = three_link_planar();
    // j1=0, j2=π, j3=0 → folds back on itself
    let q = vec![0.0, PI, 0.0];
    let data = forward_kinematics(&model, &q);

    // Joint 3 at (1 + (-1), 0, 0) = (0, 0, 0)
    assert_relative_eq!(
        se3::translation(&data.oMi[3]),
        Vector3::zeros(),
        epsilon = 1e-10,
    );
}

// ─── Jacobian numerical validation (3-DOF) ─────────────────────────────────

#[test]
fn jacobian_three_link_finite_diff() {
    let model = three_link_planar();
    let q = vec![0.5, -0.3, 0.8];
    let eps = 1e-8;

    for target in 1..=3 {
        let jac = compute_joint_jacobian(&model, &q, target);
        let data_ref = forward_kinematics(&model, &q);
        let p_ref = se3::translation(&data_ref.oMi[target]);

        for j in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[j] += eps;
            let data_plus = forward_kinematics(&model, &q_plus);
            let p_plus = se3::translation(&data_plus.oMi[target]);
            let dp = (p_plus - p_ref) / eps;

            assert_relative_eq!(jac[(3, j)], dp[0], epsilon = 1e-5);
            assert_relative_eq!(jac[(4, j)], dp[1], epsilon = 1e-5);
            assert_relative_eq!(jac[(5, j)], dp[2], epsilon = 1e-5);
        }
    }
}

// ─── Mixed joint types ─────────────────────────────────────────────────────

#[test]
fn prismatic_plus_revolute() {
    // Prismatic along Z, then revolute about Z
    let model = ModelBuilder::new()
        .add_joint("slide", 0, joint::prismatic_z(), se3::identity(), LinkInertia::zero())
        .add_joint("rot", 1, joint::revolute_z(), link_offset(1.0, 0.0, 0.0), LinkInertia::zero())
        .build();

    let q = vec![2.0, FRAC_PI_2]; // slide 2m up, then rotate 90°
    let data = forward_kinematics(&model, &q);

    // Joint 1 (prismatic): at (0, 0, 2)
    assert_relative_eq!(
        se3::translation(&data.oMi[1]),
        Vector3::new(0.0, 0.0, 2.0),
        epsilon = 1e-12,
    );
    // Joint 2 (revolute after offset): at (1, 0, 2), then rotation doesn't move the joint
    assert_relative_eq!(
        se3::translation(&data.oMi[2]),
        Vector3::new(1.0, 0.0, 2.0),
        epsilon = 1e-12,
    );
}

// ─── Branched tree ──────────────────────────────────────────────────────────

#[test]
fn branched_tree() {
    // Root → j1 (revolute Z) → j2 (revolute Z, offset X+1)
    //                        → j3 (revolute Z, offset Y+1)   (branch from j1)
    let model = ModelBuilder::new()
        .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
        .add_joint("j2", 1, joint::revolute_z(), link_offset(1.0, 0.0, 0.0), LinkInertia::zero())
        .add_joint("j3", 1, joint::revolute_z(), link_offset(0.0, 1.0, 0.0), LinkInertia::zero())
        .build();

    let q = vec![FRAC_PI_2, 0.0, 0.0];
    let data = forward_kinematics(&model, &q);

    // j1 at origin, rotated 90° about Z
    // j2 offset was (1,0,0) → rotated to (0,1,0)
    assert_relative_eq!(
        se3::translation(&data.oMi[2]),
        Vector3::new(0.0, 1.0, 0.0),
        epsilon = 1e-12,
    );
    // j3 offset was (0,1,0) → rotated to (-1,0,0)
    assert_relative_eq!(
        se3::translation(&data.oMi[3]),
        Vector3::new(-1.0, 0.0, 0.0),
        epsilon = 1e-12,
    );
}
