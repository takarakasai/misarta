//! Centroidal dynamics utilities.
//!
//! This module provides functions to compute centroidal quantities for a rigid
//! body system:
//!
//! - **Center of Mass (CoM)**: mass-weighted mean of link CoM positions.
//! - **CoM Jacobian** (3×nv): maps generalized velocity to CoM linear velocity.
//! - **Centroidal Momentum Matrix** (6×nv): maps generalized velocity to the
//!   world-frame centroidal momentum `h = [angular; linear]`.
//! - **Centroidal Inertia** (6×6): composite rigid body inertia expressed at
//!   the CoM frame (Inertioid / spatial inertia of the whole robot).
//! - **Momentum**: `h = CMM * v`.
//! - **CoM velocity**: `ṗ_com = J_com * v`.
//!
//! # Coordinate conventions
//!
//! All quantities are expressed in the **world frame**.
//! The momentum vector is ordered `[angular (3); linear (3)]`, matching the
//! Jacobian row convention used elsewhere in `misarta` (rows 0-2 angular,
//! rows 3-5 linear).

use nalgebra::{DMatrix, DVector, Matrix3, Matrix6, Vector3, Vector6};

use crate::{
    fk::forward_kinematics,
    jacobian::compute_joint_jacobian_from_data,
    model::Model,
    se3,
};

// ─── helpers ──────────────────────────────────────────────────────────────────

/// 3×3 skew-symmetric (cross-product) matrix of `v`.
///
/// `skew(v) * w  ==  v × w`
#[inline]
fn skew3(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

// ─── public API ───────────────────────────────────────────────────────────────

/// Total mass of the robot.
pub fn total_mass(model: &Model<f64>) -> f64 {
    model.inertias.iter().map(|i| i.mass).sum()
}

/// World-frame center of mass (CoM) at configuration `q`.
///
/// `p_com = (1/M) Σ_i  m_i * (R_i * c_i + t_i)`
///
/// where `c_i` is the CoM of link `i` in its body frame and `R_i`, `t_i` are
/// the rotation and translation of joint `i` in the world frame.
pub fn compute_com(model: &Model<f64>, q: &[f64]) -> Vector3<f64> {
    let data = forward_kinematics(model, q);
    let mut p_com = Vector3::zeros();
    let mut total_m = 0.0_f64;

    for (i, inertia) in model.inertias.iter().enumerate() {
        if inertia.mass == 0.0 {
            continue;
        }
        let r = se3::rotation_matrix(&data.oMi[i]);
        let t = se3::translation(&data.oMi[i]);
        let p_link = r * &inertia.center_of_mass + t;
        p_com += inertia.mass * p_link;
        total_m += inertia.mass;
    }

    if total_m > 0.0 {
        p_com / total_m
    } else {
        Vector3::zeros()
    }
}

/// CoM Jacobian: a 3×nv matrix `J_com` such that `ṗ_com = J_com * v̇`.
///
/// Computed as  `J_com = (1/M) Σ_i  m_i * J_lin_com_i`
///
/// where `J_lin_com_i` is the linear Jacobian of the CoM point of link `i`:
///
/// `J_lin_com_i = J_lin_i − skew(r_i) * J_ang_i`
///
/// and `r_i = p_com_i_world − p_joint_i_world`.
pub fn compute_com_jacobian(model: &Model<f64>, q: &[f64]) -> DMatrix<f64> {
    let data = forward_kinematics(model, q);
    let nv = model.nv;
    let mut j_com = DMatrix::zeros(3, nv);
    let mut total_m = 0.0_f64;

    for (i, inertia) in model.inertias.iter().enumerate() {
        if inertia.mass == 0.0 || i == 0 {
            // index 0 is the universe/root; skip zero-mass links
            continue;
        }
        // world-frame translation and rotation of joint i
        let r_mat = se3::rotation_matrix(&data.oMi[i]);
        let t_joint = se3::translation(&data.oMi[i]);

        // world position of this link's CoM
        let p_com_link = r_mat * &inertia.center_of_mass + t_joint;

        // offset from joint origin to link CoM (world frame)
        let r = p_com_link - t_joint; // == r_mat * c_i

        // full 6×nv Jacobian at joint i
        let j_full = compute_joint_jacobian_from_data(model, q, &data, i);
        let j_ang = j_full.rows(0, 3); // rows 0-2: angular
        let j_lin = j_full.rows(3, 3); // rows 3-5: linear

        // shift linear Jacobian to the CoM point of this link
        // v_com = v_joint + ω × r  →  J_lin_com = J_lin − skew(r) * J_ang
        let j_lin_com = j_lin - skew3(&r) * j_ang;

        j_com += inertia.mass * j_lin_com;
        total_m += inertia.mass;
    }

    if total_m > 0.0 {
        j_com / total_m
    } else {
        j_com
    }
}

/// CoM velocity `ṗ_com = J_com * v`.
pub fn compute_com_velocity(model: &Model<f64>, q: &[f64], v: &[f64]) -> Vector3<f64> {
    let j = compute_com_jacobian(model, q);
    let v_vec = DVector::from_column_slice(v);
    (j * v_vec).fixed_rows::<3>(0).into()
}

/// Centroidal Momentum Matrix (CMM) A_G: a 6×nv matrix such that
///
/// `h = [h_ang; h_lin] = A_G * v`
///
/// - `h_lin = M * ṗ_com = (Σ m_i * J_lin_com_i) * v`
/// - `h_ang = Σ_i [ I_world_i * J_ang_i + m_i * skew(d_i) * J_lin_com_i ] * v`
///
/// where `d_i = p_com_link_i − p_com_robot` and `I_world_i = R_i * I_body_i * R_i^T`.
pub fn compute_centroidal_momentum_matrix(model: &Model<f64>, q: &[f64]) -> DMatrix<f64> {
    let data = forward_kinematics(model, q);
    let nv = model.nv;

    // first pass: compute robot CoM
    let mut p_com_robot = Vector3::zeros();
    let mut total_m = 0.0_f64;
    for (i, inertia) in model.inertias.iter().enumerate() {
        if inertia.mass == 0.0 {
            continue;
        }
        let r_mat = se3::rotation_matrix(&data.oMi[i]);
        let t = se3::translation(&data.oMi[i]);
        p_com_robot += inertia.mass * (r_mat * &inertia.center_of_mass + t);
        total_m += inertia.mass;
    }
    if total_m > 0.0 {
        p_com_robot /= total_m;
    }

    // second pass: assemble CMM
    let mut a_lin = DMatrix::zeros(3, nv); // rows 3-5
    let mut a_ang = DMatrix::zeros(3, nv); // rows 0-2

    for (i, inertia) in model.inertias.iter().enumerate() {
        if inertia.mass == 0.0 || i == 0 {
            continue;
        }
        let r_mat = se3::rotation_matrix(&data.oMi[i]);
        let t_joint = se3::translation(&data.oMi[i]);

        let p_com_link = r_mat * &inertia.center_of_mass + t_joint;
        let r = p_com_link - t_joint; // offset joint→link CoM (world)
        let d = p_com_link - p_com_robot; // offset robot CoM→link CoM (world)

        let j_full = compute_joint_jacobian_from_data(model, q, &data, i);
        let j_ang = j_full.rows(0, 3);
        let j_lin = j_full.rows(3, 3);
        let j_lin_com = j_lin - skew3(&r) * j_ang.clone_owned();

        // linear momentum contribution
        a_lin += inertia.mass * &j_lin_com;

        // angular momentum contribution about robot CoM
        // h_ang_i = I_world_i * ω_i + m_i * d_i × v_com_i
        //         = (I_world_i * J_ang_i + m_i * skew(d_i) * J_lin_com_i) * v
        let i_world = r_mat * &inertia.rotational_inertia * r_mat.transpose();
        a_ang += i_world * j_ang.clone_owned() + inertia.mass * skew3(&d) * &j_lin_com;
    }

    let mut cmm = DMatrix::zeros(6, nv);
    cmm.rows_mut(0, 3).copy_from(&a_ang);
    cmm.rows_mut(3, 3).copy_from(&a_lin);
    cmm
}

/// Centroidal momentum `h = [h_ang (3); h_lin (3)]` at `(q, v)`.
pub fn compute_momentum(model: &Model<f64>, q: &[f64], v: &[f64]) -> Vector6<f64> {
    let cmm = compute_centroidal_momentum_matrix(model, q);
    let v_vec = DVector::from_column_slice(v);
    let h = cmm * v_vec;
    Vector6::new(h[0], h[1], h[2], h[3], h[4], h[5])
}

/// Centroidal composite rigid body inertia (CCRBI): a 6×6 spatial inertia
/// tensor of the whole robot expressed at its CoM frame.
///
/// This is the "locked" inertia — it encodes how the robot would behave as
/// a single rigid body with all joints locked.
///
/// Block structure:
///
/// ```text
/// [ I_c          m * skew(0) ]   =  [ I_c   0 ]
/// [ m * skew(0)  m * I_3     ]      [ 0     m*I ]
/// ```
///
/// Actually for the composite body, the off-diagonal blocks vanish when
/// expressed at the CoM:
///
/// `Φ = [ Σ(I_world_i + m_i * (|d_i|² I − d_i d_i^T))   0 ]`
/// `    [                          0                     M*I ]`
pub fn compute_centroidal_inertia(model: &Model<f64>, q: &[f64]) -> Matrix6<f64> {
    let data = forward_kinematics(model, q);

    let mut p_com = Vector3::zeros();
    let mut total_m = 0.0_f64;
    for (i, inertia) in model.inertias.iter().enumerate() {
        if inertia.mass == 0.0 {
            continue;
        }
        let r_mat = se3::rotation_matrix(&data.oMi[i]);
        let t = se3::translation(&data.oMi[i]);
        p_com += inertia.mass * (r_mat * &inertia.center_of_mass + t);
        total_m += inertia.mass;
    }
    if total_m > 0.0 {
        p_com /= total_m;
    }

    let mut i_rot = Matrix3::zeros(); // upper-left block (rotational)
    for (i, inertia) in model.inertias.iter().enumerate() {
        if inertia.mass == 0.0 {
            continue;
        }
        let r_mat = se3::rotation_matrix(&data.oMi[i]);
        let t = se3::translation(&data.oMi[i]);
        let p_com_link = r_mat * &inertia.center_of_mass + t;
        let d = p_com_link - p_com; // displacement from robot CoM to link CoM

        // Rotate body inertia to world frame
        let i_world = r_mat * &inertia.rotational_inertia * r_mat.transpose();

        // Steiner (parallel axis) theorem to shift to robot CoM
        let dd = d.norm_squared();
        let steiner = inertia.mass * (Matrix3::identity() * dd - d * d.transpose());

        i_rot += i_world + steiner;
    }

    let i_lin = Matrix3::identity() * total_m; // lower-right block

    let mut phi = Matrix6::zeros();
    phi.fixed_view_mut::<3, 3>(0, 0).copy_from(&i_rot);
    phi.fixed_view_mut::<3, 3>(3, 3).copy_from(&i_lin);
    phi
}

/// Centroidal Momentum Matrix time derivative: dA_G/dt (6×nv).
///
/// Returns the matrix `Ȧ_G` such that the rate of change of centroidal
/// momentum is:
///
/// `ḣ = A_G q̈ + Ȧ_G q̇`
///
/// This is needed for momentum-rate control and centroidal dynamics.
///
/// Computed via central finite differences:
///
/// `Ȧ_G ≈ Σ_k v_k * (A_G(q + ε e_k) − A_G(q − ε e_k)) / (2ε)`
///
/// Equivalent to `pinocchio::dccrba` / `pinocchio::computeCentroidalMapTimeVariation`.
pub fn compute_centroidal_momentum_matrix_time_derivative(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
) -> DMatrix<f64> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);

    let eps = 1e-8;
    let nv = model.nv;
    let mut da = DMatrix::zeros(6, nv);

    for k in 0..nv {
        if v[k].abs() < 1e-30 {
            continue;
        }
        let mut q_plus = q.to_vec();
        let mut q_minus = q.to_vec();
        q_plus[k] += eps;
        q_minus[k] -= eps;

        let a_plus = compute_centroidal_momentum_matrix(model, &q_plus);
        let a_minus = compute_centroidal_momentum_matrix(model, &q_minus);
        let da_dqk = (&a_plus - &a_minus) / (2.0 * eps);
        da += v[k] * da_dqk;
    }

    da
}

/// Centroidal momentum rate: `ḣ = A_G q̈ + Ȧ_G q̇`.
///
/// Computes the full time derivative of centroidal momentum given
/// generalized velocity and acceleration.
pub fn compute_momentum_rate(
    model: &Model<f64>,
    q: &[f64],
    v: &[f64],
    a: &[f64],
) -> Vector6<f64> {
    let ag = compute_centroidal_momentum_matrix(model, q);
    let dag = compute_centroidal_momentum_matrix_time_derivative(model, q, v);
    let v_vec = DVector::from_column_slice(v);
    let a_vec = DVector::from_column_slice(a);
    let h_dot = ag * a_vec + dag * v_vec;
    Vector6::new(h_dot[0], h_dot[1], h_dot[2], h_dot[3], h_dot[4], h_dot[5])
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;

    use crate::{
        joint,
        model::{LinkInertia, ModelBuilder},
        se3 as se3_mod,
    };
    use nalgebra::Rotation3;

    /// Build a simple 2-link planar arm.
    ///
    ///  universe ─[rev_z, L=1]─ link1 ─[rev_z, L=1]─ link2
    ///
    /// Both links have mass 1 kg, CoM at (0.5, 0, 0) in body frame.
    fn two_link_arm() -> (crate::model::Model<f64>, Vec<f64>) {
        let link_len = 1.0_f64;
        let half = Vector3::new(0.5, 0.0, 0.0);
        let inertia = LinkInertia {
            mass: 1.0,
            center_of_mass: half,
            rotational_inertia: Matrix3::identity() * 0.1,
        };
        let placement = se3_mod::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(link_len, 0.0, 0.0),
        );
        let model = ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), se3_mod::identity(), inertia.clone())
            .add_joint("j2", 1, joint::revolute_z(), placement, inertia.clone())
            .build();
        let q = vec![0.0, 0.0]; // straight configuration
        (model, q)
    }

    #[test]
    fn total_mass_two_links() {
        let (model, _) = two_link_arm();
        assert_abs_diff_eq!(total_mass(&model), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn com_straight_arm_at_zero() {
        let (model, q) = two_link_arm();
        // j1 at origin, link1 CoM at (0.5, 0, 0)
        // j2 at (1.0, 0, 0), link2 CoM at (1.5, 0, 0)
        // total CoM = ((0.5 + 1.5) / 2, 0, 0) = (1.0, 0, 0)
        let com = compute_com(&model, &q);
        assert_abs_diff_eq!(com.x, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(com.y, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(com.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn com_folded_arm_90_degrees() {
        // q1 = 0, q2 = π/2  →  link2 points upward
        // j2 is at (1, 0, 0); link2 CoM is at (1, 0.5, 0) after rotation
        let (model, _) = two_link_arm();
        let q = vec![0.0, std::f64::consts::FRAC_PI_2];
        let com = compute_com(&model, &q);
        // link1 CoM: (0.5, 0, 0)
        // link2 CoM: j2 at (1,0,0) + R(π/2) * (0.5,0,0) = (1,0,0) + (0,0.5,0) = (1, 0.5, 0)
        // mean: ((0.5+1)/2, (0+0.5)/2, 0) = (0.75, 0.25, 0)
        assert_abs_diff_eq!(com.x, 0.75, epsilon = 1e-10);
        assert_abs_diff_eq!(com.y, 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(com.z, 0.0,  epsilon = 1e-10);
    }

    #[test]
    fn com_jacobian_matches_finite_difference() {
        let (model, q) = two_link_arm();
        let j_com = compute_com_jacobian(&model, &q);
        let eps = 1e-6_f64;

        for col in 0..model.nv {
            let mut q_plus = q.clone();
            q_plus[col] += eps;
            let mut q_minus = q.clone();
            q_minus[col] -= eps;
            let dp = (compute_com(&model, &q_plus) - compute_com(&model, &q_minus)) / (2.0 * eps);
            let j_col = j_com.column(col);
            assert_abs_diff_eq!(j_col[0], dp.x, epsilon = 1e-6);
            assert_abs_diff_eq!(j_col[1], dp.y, epsilon = 1e-6);
            assert_abs_diff_eq!(j_col[2], dp.z, epsilon = 1e-6);
        }
    }

    #[test]
    fn com_velocity_matches_com_jacobian_times_v() {
        let (model, q) = two_link_arm();
        let v = vec![1.0, 2.0];
        let vel = compute_com_velocity(&model, &q, &v);
        let j = compute_com_jacobian(&model, &q);
        let v_vec = DVector::from_column_slice(&v);
        let expected = j * v_vec;
        assert_abs_diff_eq!(vel.x, expected[0], epsilon = 1e-12);
        assert_abs_diff_eq!(vel.y, expected[1], epsilon = 1e-12);
    }

    #[test]
    fn linear_momentum_equals_mass_times_com_velocity() {
        // h_lin = M * ṗ_com
        let (model, q) = two_link_arm();
        let v = vec![1.5, -0.5];
        let h = compute_momentum(&model, &q, &v);
        let p_dot = compute_com_velocity(&model, &q, &v);
        let m = total_mass(&model);
        assert_abs_diff_eq!(h[3], m * p_dot.x, epsilon = 1e-10);
        assert_abs_diff_eq!(h[4], m * p_dot.y, epsilon = 1e-10);
        assert_abs_diff_eq!(h[5], m * p_dot.z, epsilon = 1e-10);
    }

    #[test]
    fn centroidal_inertia_positive_definite_rotation_block() {
        let (model, q) = two_link_arm();
        let phi = compute_centroidal_inertia(&model, &q);
        // Top-left 3×3 must be positive definite (all eigenvalues > 0)
        let i_rot = phi.fixed_view::<3, 3>(0, 0).into_owned();
        let sym = (i_rot.clone() + i_rot.transpose()) * 0.5;
        let eig = sym.symmetric_eigen();
        for &ev in eig.eigenvalues.iter() {
            assert!(ev > 0.0, "Eigenvalue {ev} not positive");
        }
    }

    #[test]
    fn centroidal_inertia_linear_block_is_mass_identity() {
        let (model, q) = two_link_arm();
        let phi = compute_centroidal_inertia(&model, &q);
        let mass = total_mass(&model);
        let i_lin = phi.fixed_view::<3, 3>(3, 3).into_owned();
        assert_abs_diff_eq!(i_lin, Matrix3::identity() * mass, epsilon = 1e-12);
    }

    #[test]
    fn cmm_angular_momentum_finite_difference() {
        // Check the CMM angular rows via finite difference of angular momentum
        // Using a single revolute joint for simplicity
        let inertia = LinkInertia {
            mass: 2.0,
            center_of_mass: Vector3::new(0.3, 0.0, 0.0),
            rotational_inertia: Matrix3::identity() * 0.5,
        };
        let model = ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), se3_mod::identity(), inertia)
            .build();
        let q = vec![0.3];
        let v = vec![1.0];
        let h = compute_momentum(&model, &q, &v);

        // The angular momentum about the world CoM should be non-zero
        // (body spinning about z with ω=1 rad/s)
        assert!(h[2].abs() > 0.0, "Angular z-momentum should be nonzero");
        // Linear momentum: M * ṗ_com  (body CoM is revolving)
        let p_dot = compute_com_velocity(&model, &q, &v);
        let m = total_mass(&model);
        assert_abs_diff_eq!(h[3], m * p_dot.x, epsilon = 1e-10);
        assert_abs_diff_eq!(h[4], m * p_dot.y, epsilon = 1e-10);
    }

    #[test]
    fn cmm_time_derivative_zero_velocity_is_zero() {
        let (model, q) = two_link_arm();
        let v = vec![0.0, 0.0];
        let da = compute_centroidal_momentum_matrix_time_derivative(&model, &q, &v);
        assert_abs_diff_eq!(da, DMatrix::zeros(6, model.nv), epsilon = 1e-10);
    }

    #[test]
    fn cmm_time_derivative_finite_difference() {
        // Validate Ȧ_G via: A_G(q + v*dt) ≈ A_G(q) + Ȧ_G * dt
        let (model, q) = two_link_arm();
        let v = vec![1.0, -0.5];
        let da = compute_centroidal_momentum_matrix_time_derivative(&model, &q, &v);

        let dt = 1e-6;
        let q_fwd: Vec<f64> = q.iter().zip(v.iter()).map(|(qi, vi)| qi + vi * dt).collect();
        let a_fwd = compute_centroidal_momentum_matrix(&model, &q_fwd);
        let a_cur = compute_centroidal_momentum_matrix(&model, &q);
        let da_fd = (&a_fwd - &a_cur) / dt;

        assert_abs_diff_eq!(da, da_fd, epsilon = 1e-4);
    }

    #[test]
    fn momentum_rate_matches_finite_difference() {
        // ḣ ≈ (h(q+v*dt, v+a*dt) − h(q, v)) / dt
        let (model, q) = two_link_arm();
        let v = vec![1.0, -0.5];
        let a = vec![0.5, 0.2];
        let h_dot = compute_momentum_rate(&model, &q, &v, &a);

        let dt = 1e-6;
        let q_fwd: Vec<f64> = q.iter().zip(v.iter()).map(|(qi, vi)| qi + vi * dt).collect();
        let v_fwd: Vec<f64> = v.iter().zip(a.iter()).map(|(vi, ai)| vi + ai * dt).collect();

        let h0 = compute_momentum(&model, &q, &v);
        let h1 = compute_momentum(&model, &q_fwd, &v_fwd);
        let h_dot_fd = (h1 - h0) / dt;

        assert_abs_diff_eq!(h_dot[0], h_dot_fd[0], epsilon = 1e-3);
        assert_abs_diff_eq!(h_dot[1], h_dot_fd[1], epsilon = 1e-3);
        assert_abs_diff_eq!(h_dot[2], h_dot_fd[2], epsilon = 1e-3);
        assert_abs_diff_eq!(h_dot[3], h_dot_fd[3], epsilon = 1e-3);
        assert_abs_diff_eq!(h_dot[4], h_dot_fd[4], epsilon = 1e-3);
        assert_abs_diff_eq!(h_dot[5], h_dot_fd[5], epsilon = 1e-3);
    }
}
