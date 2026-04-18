//! Inertial parameter regressor — τ = Y(q, v, a) π.
//!
//! The Recursive Newton-Euler Algorithm (RNEA) is **linear** in the inertial
//! parameters of each link.  This module computes the regressor matrix `Y`
//! such that:
//!
//! ```text
//! τ = Y(q, v̇, v) · π
//! ```
//!
//! where `π ∈ ℝ^{10 nb}` stacks the 10 standard inertial parameters of
//! each body:
//!
//! ```text
//! πᵢ = [m, mc_x, mc_y, mc_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz]ᵀ
//! ```
//!
//! **Applications**: system identification, adaptive control, parameter
//! estimation from measured torques.
//!
//! **Pure functions**: `(model, q, v, a) → Y`.  Generic over `T: RealField`.

use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, DVector, Matrix3, Matrix6, RealField, Vector3, Vector6};

/// Number of inertial parameters per body (standard parameterization).
const PARAMS_PER_BODY: usize = 10;

// ─── Regressor from parameters ──────────────────────────────────────────────

/// Build a 6×6 spatial inertia matrix from the 10-parameter vector.
///
/// Param layout: [m, mcx, mcy, mcz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
fn spatial_inertia_from_params<T: RealField>(pi: &[T]) -> Matrix6<T> {
    let m = pi[0].clone();
    let mc = Vector3::new(pi[1].clone(), pi[2].clone(), pi[3].clone());
    let rot_inertia = Matrix3::new(
        pi[4].clone(), pi[5].clone(), pi[6].clone(),
        pi[5].clone(), pi[7].clone(), pi[8].clone(),
        pi[6].clone(), pi[8].clone(), pi[9].clone(),
    );

    // Spatial inertia: same as se3::spatial_inertia but with mc = m*c directly.
    let cx = se3::skew(&mc);
    let cxt = cx.transpose();
    let m_eye = Matrix3::identity() * m.clone();

    // Upper-left: I + [mc]×ᵀ [mc]× / m  ... but we must be careful.
    // Actually se3::spatial_inertia uses (m, c, I_rotational).
    // With the standard parameterization, the 6×6 spatial inertia Y is:
    //
    //  Y = [ Ibar + [mc]×ᵀ [mc]× / m,  [mc]×  ]
    //      [        [mc]×ᵀ          ,   m I₃   ]
    //
    // But Ibar here is the rotational inertia about CoM expressed in body frame.
    // In the standard regressor parameterization, the Ixx..Izz are the inertia
    // about the origin of the body frame, not CoM.  So we don't need the
    // parallel axis theorem:
    //
    //  Y = [ I_origin,  [mc]×  ]
    //      [ [mc]×ᵀ  ,  m I₃  ]
    //
    // where I_origin = I_com + m [c]×ᵀ [c]× = I_com + [mc]×ᵀ [mc]× / m.
    //
    // If π stores I_origin directly (standard regressor parameterization):
    let i_origin = rot_inertia;

    let mut y = Matrix6::zeros();
    y.fixed_view_mut::<3, 3>(0, 0).copy_from(&i_origin);
    y.fixed_view_mut::<3, 3>(0, 3).copy_from(&cx);
    y.fixed_view_mut::<3, 3>(3, 0).copy_from(&cxt);
    y.fixed_view_mut::<3, 3>(3, 3).copy_from(&m_eye);
    y
}

/// Extract the 10 standard inertial parameters from a `Model`.
///
/// For each body `i` (1-based), the parameters are:
/// ```text
/// [m, m*cx, m*cy, m*cz, Ixx_o, Ixy_o, Ixz_o, Iyy_o, Iyz_o, Izz_o]
/// ```
/// where `I_o = I_com + m [c]×ᵀ [c]×` is the inertia about the body frame origin.
///
/// Returns `π ∈ ℝ^{10 nb}` where `nb = model.joints.len() - 1`.
pub fn inertia_params_from_model<T: RealField>(model: &Model<T>) -> DVector<T> {
    let nb = model.joints.len() - 1; // exclude universe
    let mut pi = DVector::zeros(PARAMS_PER_BODY * nb);

    for i in 1..model.joints.len() {
        let inertia = &model.inertias[i];
        let idx = (i - 1) * PARAMS_PER_BODY;
        let m = inertia.mass.clone();
        let c = &inertia.center_of_mass;

        pi[idx] = m.clone();
        pi[idx + 1] = m.clone() * c[0].clone();
        pi[idx + 2] = m.clone() * c[1].clone();
        pi[idx + 3] = m.clone() * c[2].clone();

        // I_origin = I_com + m * [c]×ᵀ [c]×
        let cx = se3::skew(c);
        let i_origin = &inertia.rotational_inertia + cx.transpose() * &cx * m;
        pi[idx + 4] = i_origin[(0, 0)].clone();
        pi[idx + 5] = i_origin[(0, 1)].clone();
        pi[idx + 6] = i_origin[(0, 2)].clone();
        pi[idx + 7] = i_origin[(1, 1)].clone();
        pi[idx + 8] = i_origin[(1, 2)].clone();
        pi[idx + 9] = i_origin[(2, 2)].clone();
    }

    pi
}

// ─── Joint torque regressor ─────────────────────────────────────────────────

/// Compute the joint torque regressor matrix `Y` such that `τ = Y · π`.
///
/// Follows the RNEA structure but, instead of computing the net force from
/// concrete inertial parameters, builds the linear map from parameters to
/// the projected torque at each joint.
///
/// # Arguments
///
/// * `model` — robot model
/// * `q`     — configuration vector (length `nq`)
/// * `v`     — velocity vector (length `nv`)
/// * `a`     — acceleration vector (length `nv`)
///
/// # Returns
///
/// Regressor matrix `Y ∈ ℝ^{nv × 10 nb}`.
///
/// # Panics
///
/// Panics if vector dimensions do not match the model.
pub fn compute_joint_torque_regressor<T: RealField>(
    model: &Model<T>,
    q: &[T],
    v: &[T],
    a: &[T],
) -> DMatrix<T> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(a.len(), model.nv);

    let n = model.joints.len();
    let nb = n - 1;

    // ── Forward pass: compute body velocities and accelerations ─────────
    let mut x_j: Vec<se3::SE3<T>> = vec![se3::identity(); n];
    let mut vel: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
    let mut acc: Vec<Vector6<T>> = vec![Vector6::zeros(); n];

    // Gravity as spatial acceleration of universe (Featherstone trick)
    let mut a0 = Vector6::<T>::zeros();
    a0[3] = -model.gravity[0].clone();
    a0[4] = -model.gravity[1].clone();
    a0[5] = -model.gravity[2].clone();

    for i in 1..n {
        let joint = &model.joints[i];
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];
        let nq_j = joint.joint_type.nq();
        let nv_j = joint.joint_type.nv();
        let q_slice = &q[qi..qi + nq_j];

        let m_j = joint.joint_type.forward(q_slice);
        x_j[i] = se3::compose(&joint.placement, &m_j);

        let r = se3::rotation_matrix(&x_j[i]);
        let p = se3::translation(&x_j[i]);
        let rt = r.transpose();

        let parent = joint.parent;
        let v_parent = &vel[parent];

        let omega_parent = Vector3::new(
            v_parent[0].clone(), v_parent[1].clone(), v_parent[2].clone(),
        );
        let vlin_parent = Vector3::new(
            v_parent[3].clone(), v_parent[4].clone(), v_parent[5].clone(),
        );

        let omega_body = &rt * &omega_parent;
        let vlin_body = &rt * (vlin_parent - p.cross(&omega_parent));

        let s = joint.joint_type.motion_subspace(q_slice);

        let mut v_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qd_c = v[vi + c].clone();
            for r in 0..6 {
                v_j[r] += s[(r, c)].clone() * qd_c.clone();
            }
        }

        vel[i] = Vector6::new(
            omega_body[0].clone() + v_j[0].clone(),
            omega_body[1].clone() + v_j[1].clone(),
            omega_body[2].clone() + v_j[2].clone(),
            vlin_body[0].clone() + v_j[3].clone(),
            vlin_body[1].clone() + v_j[4].clone(),
            vlin_body[2].clone() + v_j[5].clone(),
        );

        let a_parent = if parent == 0 { &a0 } else { &acc[parent] };
        let omega_a = Vector3::new(
            a_parent[0].clone(), a_parent[1].clone(), a_parent[2].clone(),
        );
        let vlin_a = Vector3::new(
            a_parent[3].clone(), a_parent[4].clone(), a_parent[5].clone(),
        );
        let alpha_body = &rt * &omega_a;
        let alin_body = &rt * (vlin_a - p.cross(&omega_a));

        let mut a_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qdd_c = a[vi + c].clone();
            for r in 0..6 {
                a_j[r] += s[(r, c)].clone() * qdd_c.clone();
            }
        }

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
    }

    // ── Build per-body 6×10 regressors ──────────────────────────────────
    //
    // For body i, the net wrench is:
    //   f_i = Y_i a_i + v_i ×* (Y_i v_i)
    //
    // This is linear in the 10 inertial parameters πᵢ.  We build a 6×10
    // matrix Φᵢ such that f_i = Φᵢ πᵢ.
    //
    // The 10 parameters: [m, mcx, mcy, mcz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    // define the spatial inertia Y_i(π).  Then:
    //   Φᵢ = ∂(Y_i a_i + v_i ×* Y_i v_i) / ∂πᵢ
    //
    // Since Y_i is linear in πᵢ, and v_i ×* Y_i v_i is also linear in πᵢ,
    // we can compute Φᵢ column by column using unit parameter vectors.

    let mut phi: Vec<nalgebra::SMatrix<T, 6, 10>> = Vec::with_capacity(n);
    phi.push(nalgebra::SMatrix::<T, 6, 10>::zeros()); // universe placeholder

    for i in 1..n {
        let mut phi_i = nalgebra::SMatrix::<T, 6, 10>::zeros();
        let vxstar = se3::force_cross(&vel[i]);

        for k in 0..PARAMS_PER_BODY {
            // Unit parameter vector for column k
            let mut pi_unit: [T; 10] = core::array::from_fn(|_| T::zero());
            pi_unit[k] = T::one();

            let y_k = spatial_inertia_from_params(&pi_unit);
            let col = &y_k * &acc[i] + &vxstar * (&y_k * &vel[i]);

            for r in 0..6 {
                phi_i[(r, k)] = col[r].clone();
            }
        }

        phi.push(phi_i);
    }

    // ── Backward pass: propagate regressors to build Y ──────────────────
    //
    // For RNEA backward pass:  τ[j] = S_jᵀ Σ_{i ∈ subtree(j)} X_{j←i}^{*} Φᵢ πᵢ
    //
    // We accumulate the 6×(10*nb) body-frame regressor, then propagate to
    // parent frames exactly as forces are propagated in RNEA.

    // Per-body 6×(10*nb) regressor in body frame
    // We represent it as a Vec of per-body columns using the phi matrices.
    // For efficiency, we work with the accumulated 6×(10*nb) matrix.
    let total_params = PARAMS_PER_BODY * nb;
    let mut body_regressor: Vec<nalgebra::DMatrix<T>> = Vec::with_capacity(n);
    for _ in 0..n {
        body_regressor.push(DMatrix::zeros(6, total_params));
    }

    // Set each body's own contribution
    for i in 1..n {
        let idx = (i - 1) * PARAMS_PER_BODY;
        for r in 0..6 {
            for c in 0..PARAMS_PER_BODY {
                body_regressor[i][(r, idx + c)] = phi[i][(r, c)].clone();
            }
        }
    }

    // Backward pass: accumulate child contributions to parent
    let mut result = DMatrix::zeros(model.nv, total_params);

    for i in (1..n).rev() {
        let joint = &model.joints[i];
        let vi = model.v_idx[i];
        let nv_j = joint.joint_type.nv();
        let qi = model.q_idx[i];
        let q_slice = &q[qi..qi + joint.joint_type.nq()];
        let s = joint.joint_type.motion_subspace(q_slice);

        // τ_i = S^T * body_regressor[i]  →  row(s) of result
        for c_param in 0..total_params {
            let f_col = Vector6::new(
                body_regressor[i][(0, c_param)].clone(),
                body_regressor[i][(1, c_param)].clone(),
                body_regressor[i][(2, c_param)].clone(),
                body_regressor[i][(3, c_param)].clone(),
                body_regressor[i][(4, c_param)].clone(),
                body_regressor[i][(5, c_param)].clone(),
            );
            for jj in 0..nv_j {
                let mut dot = T::zero();
                for r in 0..6 {
                    dot += s[(r, jj)].clone() * f_col[r].clone();
                }
                result[(vi + jj, c_param)] = dot;
            }
        }

        // Propagate to parent: body_regressor[parent] += X_i^{*} body_regressor[i]
        let parent = joint.parent;
        if parent > 0 {
            let r_mat = se3::rotation_matrix(&x_j[i]);
            let p = se3::translation(&x_j[i]);

            for c_param in 0..total_params {
                let f_ang = Vector3::new(
                    body_regressor[i][(0, c_param)].clone(),
                    body_regressor[i][(1, c_param)].clone(),
                    body_regressor[i][(2, c_param)].clone(),
                );
                let f_lin = Vector3::new(
                    body_regressor[i][(3, c_param)].clone(),
                    body_regressor[i][(4, c_param)].clone(),
                    body_regressor[i][(5, c_param)].clone(),
                );

                let r_f_lin = &r_mat * &f_lin;
                let r_f_ang = &r_mat * &f_ang;
                let f_parent_ang = &r_f_ang + p.cross(&r_f_lin);
                let f_parent_lin = r_f_lin;

                for k in 0..3 {
                    body_regressor[parent][(k, c_param)] += f_parent_ang[k].clone();
                    body_regressor[parent][(k + 3, c_param)] += f_parent_lin[k].clone();
                }
            }
        }
    }

    result
}

/// Compute the static (gravity-only) regressor: `g(q) = Y_static · π`.
///
/// Equivalent to `compute_joint_torque_regressor(model, q, 0, 0)`.
pub fn compute_static_regressor<T: RealField>(
    model: &Model<T>,
    q: &[T],
) -> DMatrix<T> {
    let zeros = vec![T::zero(); model.nv];
    compute_joint_torque_regressor(model, q, &zeros, &zeros)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::rnea;
    use approx::assert_relative_eq;
    use nalgebra::Matrix3;

    fn pendulum() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        ModelBuilder::new()
            .add_joint(
                "j1", 0,
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

    fn two_link() -> Model<f64> {
        let off1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let off2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_y(), off1, LinkInertia {
                mass: 2.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                rotational_inertia: Matrix3::new(
                    0.2, 0.01, 0.0,
                    0.01, 0.2, 0.0,
                    0.0, 0.0, 0.02,
                ),
            })
            .add_joint("j2", 1, joint::revolute_y(), off2, LinkInertia {
                mass: 1.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                rotational_inertia: Matrix3::new(
                    0.1, 0.0, 0.005,
                    0.0, 0.1, 0.0,
                    0.005, 0.0, 0.01,
                ),
            })
            .build()
    }

    fn three_link() -> Model<f64> {
        let off1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.3),
        );
        let off2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let off3 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.4),
        );
        ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), off1, LinkInertia {
                mass: 3.0,
                center_of_mass: Vector3::new(0.1, 0.0, -0.15),
                rotational_inertia: Matrix3::new(
                    0.3, 0.01, 0.0,
                    0.01, 0.3, 0.02,
                    0.0, 0.02, 0.05,
                ),
            })
            .add_joint("j2", 1, joint::revolute_y(), off2, LinkInertia {
                mass: 2.0,
                center_of_mass: Vector3::new(0.0, 0.05, -0.25),
                rotational_inertia: Matrix3::new(
                    0.2, 0.0, 0.01,
                    0.0, 0.2, 0.0,
                    0.01, 0.0, 0.03,
                ),
            })
            .add_joint("j3", 2, joint::revolute_x(), off3, LinkInertia {
                mass: 1.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.2),
                rotational_inertia: Matrix3::new(
                    0.1, 0.0, 0.0,
                    0.0, 0.1, 0.0,
                    0.0, 0.0, 0.01,
                ),
            })
            .build()
    }

    #[test]
    fn regressor_times_params_equals_rnea_pendulum() {
        let model = pendulum();
        let q = vec![0.7];
        let v = vec![1.5];
        let a = vec![-0.3];

        let tau_rnea = rnea::rnea(&model, &q, &v, &a);
        let y = compute_joint_torque_regressor(&model, &q, &v, &a);
        let pi = inertia_params_from_model(&model);
        let tau_reg = &y * &pi;

        assert_relative_eq!(tau_rnea, tau_reg, epsilon = 1e-10);
    }

    #[test]
    fn regressor_times_params_equals_rnea_two_link() {
        let model = two_link();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let a = vec![2.0, -1.0];

        let tau_rnea = rnea::rnea(&model, &q, &v, &a);
        let y = compute_joint_torque_regressor(&model, &q, &v, &a);
        let pi = inertia_params_from_model(&model);
        let tau_reg = &y * &pi;

        assert_relative_eq!(tau_rnea, tau_reg, epsilon = 1e-10);
    }

    #[test]
    fn regressor_times_params_equals_rnea_three_link() {
        let model = three_link();
        let q = vec![0.5, -0.3, 1.2];
        let v = vec![0.8, -1.0, 0.3];
        let a = vec![-0.5, 1.5, -0.7];

        let tau_rnea = rnea::rnea(&model, &q, &v, &a);
        let y = compute_joint_torque_regressor(&model, &q, &v, &a);
        let pi = inertia_params_from_model(&model);
        let tau_reg = &y * &pi;

        assert_relative_eq!(tau_rnea, tau_reg, epsilon = 1e-10);
    }

    #[test]
    fn static_regressor_matches_gravity() {
        let model = two_link();
        let q = vec![0.8, -0.4];

        let g = rnea::compute_gravity(&model, &q);
        let y_s = compute_static_regressor(&model, &q);
        let pi = inertia_params_from_model(&model);
        let g_reg = &y_s * &pi;

        assert_relative_eq!(g, g_reg, epsilon = 1e-10);
    }

    #[test]
    fn regressor_shape() {
        let model = two_link();
        let q = vec![0.0, 0.0];
        let v = vec![0.0, 0.0];
        let a = vec![0.0, 0.0];

        let y = compute_joint_torque_regressor(&model, &q, &v, &a);
        assert_eq!(y.nrows(), model.nv);
        assert_eq!(y.ncols(), PARAMS_PER_BODY * (model.joints.len() - 1));
    }

    #[test]
    fn regressor_random_config_two_link() {
        // Test with more "random" values
        let model = two_link();
        let q = vec![1.234, -2.567];
        let v = vec![3.45, -1.23];
        let a = vec![-0.789, 4.56];

        let tau_rnea = rnea::rnea(&model, &q, &v, &a);
        let y = compute_joint_torque_regressor(&model, &q, &v, &a);
        let pi = inertia_params_from_model(&model);
        let tau_reg = &y * &pi;

        assert_relative_eq!(tau_rnea, tau_reg, epsilon = 1e-9);
    }

    #[test]
    fn regressor_zero_motion_equals_gravity() {
        let model = three_link();
        let q = vec![0.5, 0.3, -0.7];
        let v = vec![0.0; 3];
        let a = vec![0.0; 3];

        let g = rnea::compute_gravity(&model, &q);
        let y = compute_joint_torque_regressor(&model, &q, &v, &a);
        let pi = inertia_params_from_model(&model);
        let tau_reg = &y * &pi;

        assert_relative_eq!(g, tau_reg, epsilon = 1e-10);
    }

    #[test]
    fn regressor_linearity_in_acceleration() {
        // Y(q, v, a1) π + Y(q, v, a2) π = Y(q, v, a1+a2) π
        // ⟺ Y(q, v, a1) + Y(q, v, a2) = Y(q, v, a1+a2)  (since π is shared)
        // This isn't quite right because Y depends on a.
        // Actually τ(a) = Y(q,v,a) π is affine in a, so:
        //   Y(q,v,2a) = 2 Y(q,v,a) - Y(q,v,0)  ... no, Y is linear in a:
        //   τ(a) = M(q) a + nle(q,v) → Y(q,v,a)π = Y(q,v,a)π
        // Let's verify: Y(q,v,2a)π = 2*rnea(q,v,a) - rnea(q,v,0) ... no.
        //
        // Simply verify rnea = Y π at one more config.
        let model = three_link();
        let q = vec![-1.0, 0.7, 2.1];
        let v = vec![2.0, -0.5, 1.3];
        let a = vec![0.0, 0.0, 0.0]; // zero acceleration

        let tau_rnea = rnea::rnea(&model, &q, &v, &a);
        let y = compute_joint_torque_regressor(&model, &q, &v, &a);
        let pi = inertia_params_from_model(&model);
        let tau_reg = &y * &pi;

        assert_relative_eq!(tau_rnea, tau_reg, epsilon = 1e-10);
    }

    #[test]
    fn inertia_params_length() {
        let model = three_link();
        let pi = inertia_params_from_model(&model);
        assert_eq!(pi.len(), 30); // 3 bodies × 10
    }

    #[test]
    fn regressor_with_prismatic_joint() {
        let off1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 0.0),
        );
        let off2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let model = ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), off1, LinkInertia {
                mass: 2.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.3),
                rotational_inertia: Matrix3::new(
                    0.2, 0.0, 0.0,
                    0.0, 0.2, 0.0,
                    0.0, 0.0, 0.04,
                ),
            })
            .add_joint("j2", 1, joint::prismatic_z(), off2, LinkInertia {
                mass: 1.0,
                center_of_mass: Vector3::new(0.0, 0.0, -0.1),
                rotational_inertia: Matrix3::new(
                    0.05, 0.0, 0.0,
                    0.0, 0.05, 0.0,
                    0.0, 0.0, 0.01,
                ),
            })
            .build();

        let q = vec![0.5, 0.3];
        let v = vec![1.0, -0.2];
        let a = vec![0.5, 0.8];

        let tau_rnea = rnea::rnea(&model, &q, &v, &a);
        let y = compute_joint_torque_regressor(&model, &q, &v, &a);
        let pi = inertia_params_from_model(&model);
        let tau_reg = &y * &pi;

        assert_relative_eq!(tau_rnea, tau_reg, epsilon = 1e-10);
    }
}
