//! Jacobian computation — pure functions mapping (model, q) → Jacobian matrix.
//!
//! Computes the geometric Jacobian of each joint frame expressed in the world frame,
//! equivalent to `pinocchio::computeJointJacobians`.
//!
//! # Convention
//!
//! The Jacobian J is 6×nv, where column j corresponds to velocity DOF j.
//! The top 3 rows are angular velocity, the bottom 3 are linear velocity
//! (Pinocchio / Featherstone convention).
//!
//! # Extended API
//!
//! - [`compute_joint_jacobian`] — world-frame Jacobian for a single joint (root→joint)
//! - [`compute_relative_jacobian`] — Jacobian of `ee_idx` expressed in the frame of `base_idx`
//! - [`compute_masked_jacobian`] — like `compute_joint_jacobian` but with a joint mask
//! - [`compute_relative_masked_jacobian`] — relative Jacobian with a joint mask
//!
//! Generic over `T: RealField`.

use crate::data::Data;
use crate::fk::forward_kinematics;
use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, RealField, Vector3};

// ─── Core: world-frame Jacobian ─────────────────────────────────────────────

/// Compute the world-frame geometric Jacobian for a specific joint.
///
/// **Pure function**: `(model, q, joint_idx) → 6×nv DMatrix`.
///
/// The Jacobian maps the full velocity vector q̇ to the spatial velocity of
/// joint `joint_idx` expressed in the world frame.
pub fn compute_joint_jacobian<T: RealField>(
    model: &Model<T>,
    q: &[T],
    joint_idx: usize,
) -> DMatrix<T> {
    assert!(joint_idx > 0 && joint_idx < model.joints.len());

    let data = forward_kinematics(model, q);
    compute_joint_jacobian_from_data(model, q, &data, joint_idx)
}

/// Same as [`compute_joint_jacobian`] but takes pre-computed FK data.
///
/// Useful when you already have FK results and want to avoid recomputing them.
pub fn compute_joint_jacobian_from_data<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    joint_idx: usize,
) -> DMatrix<T> {
    let mut jac = DMatrix::zeros(6, model.nv);
    write_chain_columns(model, q, data, joint_idx, joint_idx, &mut jac);
    jac
}

// ─── Relative Jacobian (between any two joints) ────────────────────────────

/// Compute the geometric Jacobian of `ee_idx` **relative to** `base_idx`.
///
/// Returns a 6×nv matrix equal to `J(ee) − J(base)`, where each J is the
/// standard world-frame Jacobian. This correctly handles:
///
/// - **Serial chain** (base is an ancestor of ee): common-ancestor columns
///   cancel in angular and produce a lever-arm difference in linear.
/// - **Branched tree** (base and ee on different branches): base-only joints
///   contribute with negated sign.
///
/// The result is expressed in the **world frame**.
///
/// ## Panics
///
/// Panics if `base_idx` or `ee_idx` is 0 or out of range.
pub fn compute_relative_jacobian<T: RealField>(
    model: &Model<T>,
    q: &[T],
    base_idx: usize,
    ee_idx: usize,
) -> DMatrix<T> {
    assert!(base_idx > 0 && base_idx < model.joints.len());
    assert!(ee_idx > 0 && ee_idx < model.joints.len());

    let data = forward_kinematics(model, q);
    compute_relative_jacobian_from_data(model, q, &data, base_idx, ee_idx)
}

/// Same as [`compute_relative_jacobian`] but takes pre-computed FK data.
pub fn compute_relative_jacobian_from_data<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    base_idx: usize,
    ee_idx: usize,
) -> DMatrix<T> {
    let j_ee = compute_joint_jacobian_from_data(model, q, data, ee_idx);
    let j_base = compute_joint_jacobian_from_data(model, q, data, base_idx);
    j_ee - j_base
}

// ─── Masked Jacobian (exclude joints) ───────────────────────────────────────

/// Compute the world-frame Jacobian for `joint_idx`, zeroing out columns
/// for joints whose index appears in `mask` (disabled joints).
///
/// `mask` is a slice of **joint indices** (1-based) that should be locked /
/// excluded from the Jacobian. Their columns will be zero.
pub fn compute_masked_jacobian<T: RealField>(
    model: &Model<T>,
    q: &[T],
    joint_idx: usize,
    mask: &[usize],
) -> DMatrix<T> {
    let data = forward_kinematics(model, q);
    compute_masked_jacobian_from_data(model, q, &data, joint_idx, mask)
}

/// Same as [`compute_masked_jacobian`] but takes pre-computed FK data.
pub fn compute_masked_jacobian_from_data<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    joint_idx: usize,
    mask: &[usize],
) -> DMatrix<T> {
    let mut jac = DMatrix::zeros(6, model.nv);
    let mask_set: std::collections::HashSet<usize> = mask.iter().copied().collect();
    write_chain_columns_filtered(
        model,
        q,
        data,
        joint_idx,
        joint_idx,
        &mask_set,
        &mut jac,
    );
    jac
}

// ─── Relative + Masked (combined) ───────────────────────────────────────────

/// Compute the relative Jacobian (ee relative to base) with disabled joints.
///
/// Combines relative Jacobian logic with a joint mask.
pub fn compute_relative_masked_jacobian<T: RealField>(
    model: &Model<T>,
    q: &[T],
    base_idx: usize,
    ee_idx: usize,
    mask: &[usize],
) -> DMatrix<T> {
    let data = forward_kinematics(model, q);
    compute_relative_masked_jacobian_from_data(model, q, &data, base_idx, ee_idx, mask)
}

/// Same as [`compute_relative_masked_jacobian`] but takes pre-computed FK data.
pub fn compute_relative_masked_jacobian_from_data<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    base_idx: usize,
    ee_idx: usize,
    mask: &[usize],
) -> DMatrix<T> {
    let j_ee = compute_masked_jacobian_from_data(model, q, data, ee_idx, mask);
    let j_base = compute_masked_jacobian_from_data(model, q, data, base_idx, mask);
    j_ee - j_base
}

// ─── Internal column writers ────────────────────────────────────────────────

/// Write Jacobian columns for joints from `start` up to the root,
/// computing lever arms relative to the world position of `target_joint`.
fn write_chain_columns<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    start: usize,
    target_joint: usize,
    jac: &mut DMatrix<T>,
) {
    let target_pos = se3::translation(&data.oMi[target_joint]);
    let mut current = start;
    while current > 0 {
        write_joint_column(model, q, data, current, &target_pos, T::one(), jac);
        current = model.joints[current].parent;
    }
}

/// Write Jacobian columns from `start` up to root, skipping masked joints.
fn write_chain_columns_filtered<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    start: usize,
    target_joint: usize,
    mask: &std::collections::HashSet<usize>,
    jac: &mut DMatrix<T>,
) {
    let target_pos = se3::translation(&data.oMi[target_joint]);
    let mut current = start;
    while current > 0 {
        if !mask.contains(&current) {
            write_joint_column(model, q, data, current, &target_pos, T::one(), jac);
        }
        current = model.joints[current].parent;
    }
}

/// Write the Jacobian columns for a single joint into `jac`.
fn write_joint_column<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    joint_idx: usize,
    target_pos: &Vector3<T>,
    sign: T,
    jac: &mut DMatrix<T>,
) {
    let joint = &model.joints[joint_idx];
    let vi = model.v_idx[joint_idx];
    let nv = joint.joint_type.nv();
    if nv == 0 {
        return;
    }

    let s_local = joint.joint_type.motion_subspace(q_slice(model, q, joint_idx));
    let r = se3::rotation_matrix(&data.oMi[joint_idx]);
    let p_joint = se3::translation(&data.oMi[joint_idx]);

    for col in 0..nv {
        let s_ang = Vector3::new(
            s_local[(0, col)].clone(),
            s_local[(1, col)].clone(),
            s_local[(2, col)].clone(),
        );
        let s_lin = Vector3::new(
            s_local[(3, col)].clone(),
            s_local[(4, col)].clone(),
            s_local[(5, col)].clone(),
        );

        let w = &r * s_ang;
        let v_lin = &r * s_lin;
        let lever = target_pos - &p_joint;
        let v_at_target = v_lin + w.cross(&lever);

        jac[(0, vi + col)] = sign.clone() * w[0].clone();
        jac[(1, vi + col)] = sign.clone() * w[1].clone();
        jac[(2, vi + col)] = sign.clone() * w[2].clone();
        jac[(3, vi + col)] = sign.clone() * v_at_target[0].clone();
        jac[(4, vi + col)] = sign.clone() * v_at_target[1].clone();
        jac[(5, vi + col)] = sign.clone() * v_at_target[2].clone();
    }
}

/// Helper: extract the configuration slice for joint `i`.
fn q_slice<'a, T: RealField>(model: &Model<T>, q: &'a [T], i: usize) -> &'a [T] {
    let qi = model.q_idx[i];
    &q[qi..qi + model.joints[i].joint_type.nq()]
}

// ─── Local-frame (body-frame) Jacobian ──────────────────────────────────────

/// Compute the body-frame (local) geometric Jacobian for a specific joint.
///
/// Returns a 6×nv matrix where spatial velocities are expressed in the
/// frame of `joint_idx` rather than the world frame.
///
/// Equivalent to `pinocchio::computeJointJacobian` with `LOCAL` reference frame.
///
/// Relationship: `J_local = Ad_{oMi[i]}^{-1} * J_world`, applied column-wise as
/// a rotation of both angular and linear parts.
pub fn compute_joint_jacobian_local<T: RealField>(
    model: &Model<T>,
    q: &[T],
    joint_idx: usize,
) -> DMatrix<T> {
    let data = forward_kinematics(model, q);
    compute_joint_jacobian_local_from_data(model, q, &data, joint_idx)
}

/// Same as [`compute_joint_jacobian_local`] but takes pre-computed FK data.
pub fn compute_joint_jacobian_local_from_data<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    joint_idx: usize,
) -> DMatrix<T> {
    let j_world = compute_joint_jacobian_from_data(model, q, data, joint_idx);
    let r = se3::rotation_matrix(&data.oMi[joint_idx]);
    let rt = r.transpose();
    let p = se3::translation(&data.oMi[joint_idx]);

    let mut j_local = DMatrix::zeros(6, model.nv);
    for c in 0..model.nv {
        let w = Vector3::new(
            j_world[(0, c)].clone(),
            j_world[(1, c)].clone(),
            j_world[(2, c)].clone(),
        );
        let v = Vector3::new(
            j_world[(3, c)].clone(),
            j_world[(4, c)].clone(),
            j_world[(5, c)].clone(),
        );
        // Rotate to local frame:
        // ω_local = R^T ω_world
        // v_local = R^T (v_world − p × ω_world)
        let w_local = &rt * &w;
        let v_local = &rt * (v - p.cross(&w));
        j_local[(0, c)] = w_local[0].clone();
        j_local[(1, c)] = w_local[1].clone();
        j_local[(2, c)] = w_local[2].clone();
        j_local[(3, c)] = v_local[0].clone();
        j_local[(4, c)] = v_local[1].clone();
        j_local[(5, c)] = v_local[2].clone();
    }
    j_local
}

// ─── Jacobian time derivative ───────────────────────────────────────────────

/// Compute the time derivative of the world-frame geometric Jacobian: dJ/dt.
///
/// Returns a 6×nv matrix such that the spatial acceleration contribution
/// from the changing Jacobian is `dJ/dt * v`.
///
/// This is computed via central finite differences of the Jacobian with
/// respect to configuration, weighted by the velocity:
///
/// `Ȧ ≈ Σ_k  v_k * (J(q + ε e_k) − J(q − ε e_k)) / (2ε)`
///
/// Equivalent to `pinocchio::computeJointJacobiansTimeVariation`.
pub fn compute_joint_jacobian_time_derivative(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    joint_idx: usize,
) -> DMatrix<f64> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert!(joint_idx > 0 && joint_idx < model.joints.len());

    let eps = 1e-8;
    let mut dj = DMatrix::zeros(6, model.nv);

    for k in 0..model.nv {
        if v[k].abs() < 1e-30 {
            continue;
        }
        let mut q_plus = q.to_vec();
        let mut q_minus = q.to_vec();
        q_plus[k] += eps;
        q_minus[k] -= eps;

        let j_plus = compute_joint_jacobian(model, &q_plus, joint_idx);
        let j_minus = compute_joint_jacobian(model, &q_minus, joint_idx);
        let dj_dqk = (&j_plus - &j_minus) / (2.0 * eps);
        dj += v[k] * dj_dqk;
    }

    dj
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

    fn two_link_arm() -> Model<f64> {
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

    /// Three-link arm:  root → j1(Z) → j2(Z) → j3(Z), each offset 1m along X.
    fn three_link_arm() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::revolute_z(),
                se3::identity(),
                LinkInertia::zero(),
            )
            .add_joint("j2", 1, joint::revolute_z(), offset.clone(), LinkInertia::zero())
            .add_joint("j3", 2, joint::revolute_z(), offset, LinkInertia::zero())
            .build()
    }

    /// Branched tree:  root → j1 → j2 (chain), root → j3 (branch).
    fn branched_arm() -> Model<f64> {
        let offset_x = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        let offset_y = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 1.0, 0.0),
        );
        ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j2", 1, joint::revolute_z(), offset_x, LinkInertia::zero())
            .add_joint("j3", 0, joint::revolute_z(), offset_y, LinkInertia::zero())
            .build()
    }

    // ── Original tests (preserved) ──────────────────────────────────────

    #[test]
    fn jacobian_two_link_zero_config() {
        let model = two_link_arm();
        let q = vec![0.0, 0.0];
        let jac = compute_joint_jacobian(&model, &q, 2);

        assert_relative_eq!(jac[(2, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(jac[(4, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(jac[(2, 1)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(jac[(3, 1)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(jac[(4, 1)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn jacobian_numerical_validation() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let jac = compute_joint_jacobian(&model, &q, 2);

        let eps = 1e-8;
        let data_ref = crate::fk::forward_kinematics(&model, &q);
        let p_ref = se3::translation(&data_ref.oMi[2]);

        for j in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[j] += eps;
            let data_plus = crate::fk::forward_kinematics(&model, &q_plus);
            let p_plus = se3::translation(&data_plus.oMi[2]);

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

    // ── Relative Jacobian tests ─────────────────────────────────────────

    #[test]
    fn relative_jacobian_serial_chain() {
        // base=j1, ee=j3 in three_link_arm (serial chain).
        // J_rel = J(j3) - J(j1).
        // j1 is a common ancestor → angular cancels, linear = ω × (p_ee - p_base).
        // j2, j3 appear only in J(j3) → standard columns.
        let model = three_link_arm();
        let q = vec![0.3, -0.5, 0.8];

        let jac_full_ee = compute_joint_jacobian(&model, &q, 3);
        let jac_full_base = compute_joint_jacobian(&model, &q, 1);
        let jac_rel = compute_relative_jacobian(&model, &q, 1, 3);

        // Should equal J(ee) - J(base)
        let expected = &jac_full_ee - &jac_full_base;
        assert_relative_eq!(jac_rel, expected, epsilon = 1e-14);
    }

    #[test]
    fn relative_jacobian_numerical_validation() {
        // Validate relative Jacobian via finite differences.
        // relative_pos = oMi[base]^{-1} * oMi[ee].translation
        let model = three_link_arm();
        let q = vec![0.3, -0.5, 0.8];
        let jac = compute_relative_jacobian(&model, &q, 1, 3);

        let eps = 1e-8;
        let data_ref = crate::fk::forward_kinematics(&model, &q);
        let p_ee = se3::translation(&data_ref.oMi[3]);
        let p_base = se3::translation(&data_ref.oMi[1]);
        let rel_ref = &p_ee - &p_base;

        for j in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[j] += eps;
            let data_plus = crate::fk::forward_kinematics(&model, &q_plus);
            let p_ee_p = se3::translation(&data_plus.oMi[3]);
            let p_base_p = se3::translation(&data_plus.oMi[1]);
            let rel_plus = &p_ee_p - &p_base_p;

            let dp = (&rel_plus - &rel_ref) / eps;
            assert_relative_eq!(jac[(3, j)], dp[0], epsilon = 1e-4);
            assert_relative_eq!(jac[(4, j)], dp[1], epsilon = 1e-4);
            assert_relative_eq!(jac[(5, j)], dp[2], epsilon = 1e-4);
        }
    }

    #[test]
    fn relative_jacobian_branched() {
        // base=j2 (on chain), ee=j3 (on branch). LCA = universe (0).
        let model = branched_arm();
        let q = vec![0.4, -0.2, 0.6];
        let jac = compute_relative_jacobian(&model, &q, 2, 3);

        // Validate numerically
        let eps = 1e-8;
        let data_ref = crate::fk::forward_kinematics(&model, &q);
        let p_ee = se3::translation(&data_ref.oMi[3]);
        let p_base = se3::translation(&data_ref.oMi[2]);
        let rel_ref = &p_ee - &p_base;

        for j in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[j] += eps;
            let data_plus = crate::fk::forward_kinematics(&model, &q_plus);
            let p_ee_p = se3::translation(&data_plus.oMi[3]);
            let p_base_p = se3::translation(&data_plus.oMi[2]);
            let rel_plus = &p_ee_p - &p_base_p;

            let dp = (&rel_plus - &rel_ref) / eps;
            assert_relative_eq!(jac[(3, j)], dp[0], epsilon = 1e-4);
            assert_relative_eq!(jac[(4, j)], dp[1], epsilon = 1e-4);
            assert_relative_eq!(jac[(5, j)], dp[2], epsilon = 1e-4);
        }
    }

    // ── Masked Jacobian tests ───────────────────────────────────────────

    #[test]
    fn masked_jacobian_excludes_joint() {
        let model = three_link_arm();
        let q = vec![0.3, -0.5, 0.8];

        // Mask out j2 (index 2)
        let jac = compute_masked_jacobian(&model, &q, 3, &[2]);
        let jac_full = compute_joint_jacobian(&model, &q, 3);

        // j2 columns should be zero
        for row in 0..6 {
            assert_relative_eq!(jac[(row, 1)], 0.0, epsilon = 1e-14);
        }
        // j1 and j3 should be identical to full
        for row in 0..6 {
            assert_relative_eq!(jac[(row, 0)], jac_full[(row, 0)], epsilon = 1e-14);
            assert_relative_eq!(jac[(row, 2)], jac_full[(row, 2)], epsilon = 1e-14);
        }
    }

    #[test]
    fn masked_jacobian_empty_mask_equals_full() {
        let model = three_link_arm();
        let q = vec![0.3, -0.5, 0.8];

        let jac_full = compute_joint_jacobian(&model, &q, 3);
        let jac_masked = compute_masked_jacobian(&model, &q, 3, &[]);
        assert_relative_eq!(jac_full, jac_masked, epsilon = 1e-14);
    }

    #[test]
    fn masked_jacobian_mask_all_gives_zero() {
        let model = three_link_arm();
        let q = vec![0.3, -0.5, 0.8];

        let jac = compute_masked_jacobian(&model, &q, 3, &[1, 2, 3]);
        assert_relative_eq!(jac, DMatrix::zeros(6, 3), epsilon = 1e-14);
    }

    #[test]
    fn masked_jacobian_numerical_validation() {
        // Mask out j2, verify only j1 and j3 produce the expected finite-diff.
        let model = three_link_arm();
        let q = vec![0.3, -0.5, 0.8];
        let mask = &[2usize];
        let jac = compute_masked_jacobian(&model, &q, 3, mask);

        let eps = 1e-8;
        let data_ref = crate::fk::forward_kinematics(&model, &q);
        let p_ref = se3::translation(&data_ref.oMi[3]);

        // Only non-masked joints should match finite diff
        for j in [0usize, 2] {
            // DOF indices 0 (j1) and 2 (j3)
            let mut q_plus = q.clone();
            q_plus[j] += eps;
            let data_plus = crate::fk::forward_kinematics(&model, &q_plus);
            let p_plus = se3::translation(&data_plus.oMi[3]);

            let dp = (p_plus - &p_ref) / eps;
            assert_relative_eq!(jac[(3, j)], dp[0], epsilon = 1e-5);
            assert_relative_eq!(jac[(4, j)], dp[1], epsilon = 1e-5);
            assert_relative_eq!(jac[(5, j)], dp[2], epsilon = 1e-5);
        }
    }

    // ── Combined: relative + masked ─────────────────────────────────────

    #[test]
    fn relative_masked_jacobian_combined() {
        let model = three_link_arm();
        let q = vec![0.3, -0.5, 0.8];

        // Relative j1→j3 with j2 masked
        let jac = compute_relative_masked_jacobian(&model, &q, 1, 3, &[2]);

        // j2 col = 0 (masked)
        for row in 0..6 {
            assert_relative_eq!(jac[(row, 1)], 0.0, epsilon = 1e-14);
        }
        // j3 column should be non-zero
        let j3_norm = (0..6)
            .map(|r| jac[(r, 2)] * jac[(r, 2)])
            .sum::<f64>()
            .sqrt();
        assert!(j3_norm > 0.01, "j3 column should be non-zero");
    }

    #[test]
    fn relative_masked_numerical_validation() {
        let model = three_link_arm();
        let q = vec![0.3, -0.5, 0.8];
        let mask = &[2usize];
        let jac = compute_relative_masked_jacobian(&model, &q, 1, 3, mask);

        let eps = 1e-8;
        let data_ref = crate::fk::forward_kinematics(&model, &q);
        let p_ee = se3::translation(&data_ref.oMi[3]);
        let p_base = se3::translation(&data_ref.oMi[1]);
        let rel_ref = &p_ee - &p_base;

        // j3 (DOF index 2) should match the relative finite difference
        let j = 2usize;
        let mut q_plus = q.clone();
        q_plus[j] += eps;
        let data_plus = crate::fk::forward_kinematics(&model, &q_plus);
        let p_ee_p = se3::translation(&data_plus.oMi[3]);
        let p_base_p = se3::translation(&data_plus.oMi[1]);
        let rel_plus = &p_ee_p - &p_base_p;
        let dp = (&rel_plus - &rel_ref) / eps;

        assert_relative_eq!(jac[(3, j)], dp[0], epsilon = 1e-4);
        assert_relative_eq!(jac[(4, j)], dp[1], epsilon = 1e-4);
        assert_relative_eq!(jac[(5, j)], dp[2], epsilon = 1e-4);
    }

    // ── Local-frame Jacobian tests ──────────────────────────────────────

    #[test]
    fn local_jacobian_zero_config_matches_world() {
        // At zero config of a revolute-Z at origin, body frame = world frame.
        // So J_local should equal J_world.
        let model = ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero(),
            )
            .build();
        let q = vec![0.0];
        let j_world = compute_joint_jacobian(&model, &q, 1);
        let j_local = compute_joint_jacobian_local(&model, &q, 1);
        assert_relative_eq!(j_world, j_local, epsilon = 1e-12);
    }

    #[test]
    fn local_jacobian_rotation_invariance() {
        // The body-frame angular Jacobian column for the own joint's revolute axis
        // should always be [0,0,1,0,0,0] regardless of configuration, since the
        // motion subspace is constant in the body frame.
        let model = ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero(),
            )
            .build();
        for &angle in &[0.0, 0.7, -1.3, std::f64::consts::PI] {
            let q = vec![angle];
            let j_local = compute_joint_jacobian_local(&model, &q, 1);
            // Angular part of column 0 should be [0, 0, 1]
            assert_relative_eq!(j_local[(0, 0)], 0.0, epsilon = 1e-12);
            assert_relative_eq!(j_local[(1, 0)], 0.0, epsilon = 1e-12);
            assert_relative_eq!(j_local[(2, 0)], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn local_jacobian_transform_consistency() {
        // J_local * v should give body-frame velocity, which should match
        // R^T * J_world * v for the angular part.
        let model = two_link_arm();
        let q = vec![0.5, -0.3];
        let v = nalgebra::DVector::from_column_slice(&[1.0, -0.5]);

        let data = crate::fk::forward_kinematics(&model, &q);
        let j_world = compute_joint_jacobian_from_data(&model, &q, &data, 2);
        let j_local = compute_joint_jacobian_local_from_data(&model, &q, &data, 2);

        let vel_world = &j_world * &v;
        let vel_local = &j_local * &v;

        // R^T * vel_world_angular should == vel_local_angular
        let r = se3::rotation_matrix(&data.oMi[2]);
        let rt = r.transpose();
        let w_world = Vector3::new(vel_world[0], vel_world[1], vel_world[2]);
        let w_local_expected = &rt * w_world;
        assert_relative_eq!(vel_local[0], w_local_expected[0], epsilon = 1e-10);
        assert_relative_eq!(vel_local[1], w_local_expected[1], epsilon = 1e-10);
        assert_relative_eq!(vel_local[2], w_local_expected[2], epsilon = 1e-10);
    }

    // ── Jacobian time derivative tests ──────────────────────────────────

    #[test]
    fn dj_dt_zero_velocity_is_zero() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let v = vec![0.0, 0.0];
        let dj = compute_joint_jacobian_time_derivative(&model, &q, &v, 2);
        assert_relative_eq!(dj, DMatrix::zeros(6, 2), epsilon = 1e-10);
    }

    #[test]
    fn dj_dt_finite_difference_validation() {
        // Validate dJ/dt numerically: for small dt,
        // J(q + v*dt) ≈ J(q) + dJ/dt * dt
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let dj = compute_joint_jacobian_time_derivative(&model, &q, &v, 2);

        let dt = 1e-6;
        let q_fwd: Vec<f64> = q.iter().zip(v.iter()).map(|(qi, vi)| qi + vi * dt).collect();
        let j_fwd = compute_joint_jacobian(&model, &q_fwd, 2);
        let j_cur = compute_joint_jacobian(&model, &q, 2);
        let dj_fd = (&j_fwd - &j_cur) / dt;

        assert_relative_eq!(dj, dj_fd, epsilon = 1e-4);
    }
}
