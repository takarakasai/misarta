//! Articulated Body Algorithm (ABA) — forward dynamics.
//!
//! Computes joint accelerations q̈ given (q, q̇, τ):
//!   q̈ = M(q)⁻¹ (τ − C(q,q̇)q̇ − g(q))
//!
//! Uses the O(n) Articulated Body Algorithm (Featherstone 2008, §7.3).
//!
//! **Pure function**: `(model, q, v, tau) → q̈`.  Generic over `T: RealField`.

use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, DVector, Matrix6, RealField, Vector3, Vector6};

/// Compute forward dynamics via the Articulated Body Algorithm.
///
/// Returns the joint acceleration vector q̈ ∈ ℝⁿᵛ.
///
/// # Arguments
///
/// * `model` — robot model
/// * `q`     — configuration vector (length `nq`)
/// * `v`     — velocity vector (length `nv`)
/// * `tau`   — applied joint torques (length `nv`)
///
/// # Panics
///
/// Panics if vector dimensions do not match the model.
pub fn aba<T: RealField>(
    model: &Model<T>,
    q: &[T],
    v: &[T],
    tau: &[T],
) -> DVector<T> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(tau.len(), model.nv);

    let n = model.joints.len();

    // ── Per-body storage ────────────────────────────────────────────────
    let mut x_j: Vec<se3::SE3<T>> = vec![se3::identity(); n];
    let mut vel: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
    let mut c_vec: Vec<Vector6<T>> = vec![Vector6::zeros(); n]; // Coriolis acceleration
    let mut pa: Vec<Vector6<T>> = vec![Vector6::zeros(); n];     // bias force
    let mut ia: Vec<Matrix6<T>> = vec![Matrix6::zeros(); n];     // articulated inertia

    // Per-joint: S, U = Ia*S, D = S^T*U, u = tau - S^T*pa
    let mut s_store: Vec<nalgebra::Matrix6xX<T>> = Vec::with_capacity(n);
    for _ in 0..n {
        s_store.push(nalgebra::Matrix6xX::zeros(0));
    }
    let mut u_store: Vec<Vec<Vector6<T>>> = vec![vec![]; n];
    let mut d_store: Vec<nalgebra::DMatrix<T>> = Vec::with_capacity(n);
    for _ in 0..n {
        d_store.push(nalgebra::DMatrix::zeros(0, 0));
    }
    let mut u_scalar: Vec<DVector<T>> = Vec::with_capacity(n);
    for _ in 0..n {
        u_scalar.push(DVector::zeros(0));
    }

    // Gravity as spatial acceleration
    let mut a0 = Vector6::<T>::zeros();
    a0[3] = -model.gravity[0].clone();
    a0[4] = -model.gravity[1].clone();
    a0[5] = -model.gravity[2].clone();

    // ── Pass 1: Forward — velocities, bias forces, body inertias ────────

    for i in 1..n {
        let joint = &model.joints[i];
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];
        let nq_j = joint.joint_type.nq();
        let nv_j = joint.joint_type.nv();
        let q_slice = &q[qi..qi + nq_j];

        // Joint placement
        let m_j = joint.joint_type.forward(q_slice);
        x_j[i] = se3::compose(&joint.placement, &m_j);

        // Rotation and translation from parent_M_child
        let r = se3::rotation_matrix(&x_j[i]);
        let p = se3::translation(&x_j[i]);
        let rt = r.transpose();

        // Transform parent velocity to body frame
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

        // Motion subspace
        let s = joint.joint_type.motion_subspace(q_slice);

        // Joint velocity
        let mut v_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qd = v[vi + c].clone();
            for r in 0..6 {
                v_j[r] += s[(r, c)].clone() * qd.clone();
            }
        }

        // Body velocity
        vel[i] = Vector6::new(
            omega_body[0].clone() + v_j[0].clone(),
            omega_body[1].clone() + v_j[1].clone(),
            omega_body[2].clone() + v_j[2].clone(),
            vlin_body[0].clone() + v_j[3].clone(),
            vlin_body[1].clone() + v_j[4].clone(),
            vlin_body[2].clone() + v_j[5].clone(),
        );

        // Coriolis acceleration: c = v × v_j
        let vx = se3::motion_cross(&vel[i]);
        c_vec[i] = vx * &v_j;

        // Body spatial inertia
        let inertia = &model.inertias[i];
        ia[i] = se3::spatial_inertia(
            inertia.mass.clone(),
            &inertia.center_of_mass,
            &inertia.rotational_inertia,
        );

        // Bias force: pa = v ×* (Ia v)
        let ia_v = &ia[i] * &vel[i];
        let vxstar = se3::force_cross(&vel[i]);
        pa[i] = vxstar * ia_v;

        s_store[i] = s;
    }

    // ── Pass 2: Backward — articulated inertias and bias forces ─────────

    for i in (1..n).rev() {
        let joint = &model.joints[i];
        let vi = model.v_idx[i];
        let nv_j = joint.joint_type.nv();

        if nv_j > 0 {
            // U = Ia * S
            let mut u_cols: Vec<Vector6<T>> = Vec::with_capacity(nv_j);
            for c in 0..nv_j {
                u_cols.push(&ia[i] * s_store[i].column(c).into_owned());
            }

            // D = S^T * U  (nv_j × nv_j)
            let mut d = nalgebra::DMatrix::zeros(nv_j, nv_j);
            for ci in 0..nv_j {
                for ri in 0..nv_j {
                    d[(ri, ci)] = s_store[i].column(ri).dot(&u_cols[ci]);
                }
            }

            // u = tau - S^T * pa
            let mut u_vec = DVector::zeros(nv_j);
            for c in 0..nv_j {
                let st_pa = s_store[i].column(c).dot(&pa[i]);
                u_vec[c] = tau[vi + c].clone() - st_pa;
            }

            // Ia -= U D^{-1} U^T
            // pa += Ia * c  +  U D^{-1} u
            if nv_j == 1 {
                // Scalar case (most common)
                let d_inv = T::one() / d[(0, 0)].clone();
                // Ia -= (1/D) * U * U^T
                let u0 = &u_cols[0];
                for r in 0..6 {
                    for c in 0..6 {
                        ia[i][(r, c)] -= d_inv.clone() * u0[r].clone() * u0[c].clone();
                    }
                }
                // pa += Ia * c  +  U * (u / D)
                let ia_c = &ia[i] * &c_vec[i];
                let u_d_inv_u = u0 * (d_inv * u_vec[0].clone());
                pa[i] = &pa[i] + ia_c + u_d_inv_u;
            } else {
                // General case: use LU decomposition of D
                let d_lu = d.clone().lu();
                // Ia -= U D^{-1} U^T
                for ci in 0..nv_j {
                    let e_c = DVector::from_fn(nv_j, |r, _| {
                        if r == ci { T::one() } else { T::zero() }
                    });
                    let d_inv_col = d_lu.solve(&e_c).unwrap();
                    for r in 0..6 {
                        for cc in 0..6 {
                            let mut val = T::zero();
                            for k in 0..nv_j {
                                val += u_cols[ci][r].clone() * d_inv_col[k].clone() * u_cols[k][cc].clone();
                            }
                            // Only subtract once per (ci)
                            if ci == 0 {
                                // We'll compute the full U D^{-1} U^T separately
                            }
                            // Actually let's just compute U * D^{-1} * U^T directly
                        }
                    }
                }

                // Simpler: compute D^{-1}, then U D^{-1} U^T
                let d_inv = d_lu.solve(&nalgebra::DMatrix::identity(nv_j, nv_j)).unwrap();

                // Build U as 6×nv_j matrix
                let mut u_mat = nalgebra::Matrix6xX::zeros(nv_j);
                for c in 0..nv_j {
                    for r in 0..6 {
                        u_mat[(r, c)] = u_cols[c][r].clone();
                    }
                }

                // U D^{-1} U^T  (6×6)
                let ud = &u_mat * &d_inv;
                let udu = &ud * u_mat.transpose();
                ia[i] -= udu;

                // pa += Ia * c  +  U * D^{-1} * u
                let ia_c = &ia[i] * &c_vec[i];
                let d_inv_u = d_lu.solve(&u_vec.clone()).unwrap();
                let mut u_dinv_u = Vector6::zeros();
                for c in 0..nv_j {
                    for r in 0..6 {
                        u_dinv_u[r] += u_cols[c][r].clone() * d_inv_u[c].clone();
                    }
                }
                pa[i] = &pa[i] + ia_c + u_dinv_u;
            }

            u_store[i] = u_cols;
            d_store[i] = d;
            u_scalar[i] = u_vec;
        } else {
            // Fixed joint: just propagate Ia*c to pa
            let ia_c = &ia[i] * &c_vec[i];
            pa[i] = &pa[i] + ia_c;
        }

        // Propagate to parent: Ia_parent += X^{-T} Ia_child X^{-1}
        //                       pa_parent += X^{*} pa_child
        let parent = joint.parent;
        if parent > 0 {
            let r = se3::rotation_matrix(&x_j[i]);
            let p = se3::translation(&x_j[i]);

            // Transform articulated inertia
            let transformed_ia = transform_spatial_inertia_6x6(&ia[i], &r, &p);
            ia[parent] = &ia[parent] + &transformed_ia;

            // Transform bias force
            let pa_ang = Vector3::new(pa[i][0].clone(), pa[i][1].clone(), pa[i][2].clone());
            let pa_lin = Vector3::new(pa[i][3].clone(), pa[i][4].clone(), pa[i][5].clone());
            let r_pa_lin = &r * &pa_lin;
            let r_pa_ang = &r * &pa_ang;
            let pa_p_ang = &r_pa_ang + p.cross(&r_pa_lin);
            for k in 0..3 {
                pa[parent][k] += pa_p_ang[k].clone();
                pa[parent][k + 3] += r_pa_lin[k].clone();
            }
        }
    }

    // ── Pass 3: Forward — compute accelerations ─────────────────────────

    let mut acc: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
    let mut qdd = DVector::zeros(model.nv);

    for i in 1..n {
        let joint = &model.joints[i];
        let vi = model.v_idx[i];
        let nv_j = joint.joint_type.nv();
        let parent = joint.parent;

        // Transform parent acceleration to body frame
        let r = se3::rotation_matrix(&x_j[i]);
        let p = se3::translation(&x_j[i]);
        let rt = r.transpose();

        let a_parent = if parent == 0 { &a0 } else { &acc[parent] };
        let omega_a = Vector3::new(
            a_parent[0].clone(), a_parent[1].clone(), a_parent[2].clone(),
        );
        let vlin_a = Vector3::new(
            a_parent[3].clone(), a_parent[4].clone(), a_parent[5].clone(),
        );
        let alpha_body = &rt * &omega_a;
        let alin_body = &rt * (vlin_a - p.cross(&omega_a));

        let a_parent_body = Vector6::new(
            alpha_body[0].clone(),
            alpha_body[1].clone(),
            alpha_body[2].clone(),
            alin_body[0].clone(),
            alin_body[1].clone(),
            alin_body[2].clone(),
        );

        if nv_j > 0 {
            // qdd_i = D^{-1} (u - U^T * (a_parent_body + c))
            let a_plus_c = &a_parent_body + &c_vec[i];

            let mut ut_apc = DVector::zeros(nv_j);
            for c in 0..nv_j {
                ut_apc[c] = u_store[i][c].dot(&a_plus_c);
            }

            let rhs = &u_scalar[i] - &ut_apc;

            if nv_j == 1 {
                let d_inv = T::one() / d_store[i][(0, 0)].clone();
                qdd[vi] = d_inv * rhs[0].clone();
            } else {
                let d_lu = d_store[i].clone().lu();
                let qdd_j = d_lu.solve(&rhs).unwrap();
                for c in 0..nv_j {
                    qdd[vi + c] = qdd_j[c].clone();
                }
            }

            // a_i = a_parent_body + c + S * qdd
            acc[i] = a_plus_c;
            for c in 0..nv_j {
                for r in 0..6 {
                    acc[i][r] += s_store[i][(r, c)].clone() * qdd[vi + c].clone();
                }
            }
        } else {
            acc[i] = &a_parent_body + &c_vec[i];
        }
    }

    qdd
}

/// Compute M(q)⁻¹ τ using O(n) ABA forward dynamics.
///
/// Equivalent to solving `M(q) x = τ` for `x` without forming M.
/// This is the preferred way to apply the inverse mass matrix to a single
/// vector, as it avoids the O(n²) CRBA + O(n³) factorization path.
///
/// Uses ABA with zero velocity and zero gravity to isolate the `M⁻¹ τ` term.
pub fn compute_minv_times_vec<T: RealField>(
    model: &Model<T>,
    q: &[T],
    tau: &[T],
) -> DVector<T> {
    // M⁻¹ τ = aba(model_no_gravity, q, 0, τ)
    // We temporarily override gravity by using a zero-velocity, zero-gravity call.
    let mut model_no_g = model.clone();
    model_no_g.gravity = Vector3::zeros();
    let v_zero = vec![T::zero(); model.nv];
    aba(&model_no_g, q, &v_zero, tau)
}

/// Compute the full inverse mass matrix M(q)⁻¹ as an nv×nv matrix.
///
/// Uses `compute_minv_times_vec` for each column (unit vector).
/// Overall complexity is O(n²·nv) which equals O(n³), but each column
/// benefits from the O(n) ABA structure rather than dense factorization.
///
/// For applying M⁻¹ to a single vector, prefer [`compute_minv_times_vec`].
pub fn compute_minv(model: &Model<f64>, q: &[f64]) -> DMatrix<f64> {
    let nv = model.nv;
    let mut minv = DMatrix::zeros(nv, nv);

    for col in 0..nv {
        let mut e_col = vec![0.0; nv];
        e_col[col] = 1.0;
        let col_vec = compute_minv_times_vec(model, q, &e_col);
        for row in 0..nv {
            minv[(row, col)] = col_vec[row];
        }
    }

    minv
}

/// Transform a 6×6 spatial inertia from child frame to parent frame.
fn transform_spatial_inertia_6x6<T: RealField>(
    ic: &Matrix6<T>,
    r: &nalgebra::Matrix3<T>,
    p: &Vector3<T>,
) -> Matrix6<T> {
    let px = se3::skew(p);
    let rt = r.transpose();

    let mut x_inv = Matrix6::<T>::zeros();
    x_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&rt);
    let neg_rt_px = -&rt * &px;
    x_inv.fixed_view_mut::<3, 3>(3, 0).copy_from(&neg_rt_px);
    x_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&rt);

    let x_inv_t = x_inv.transpose();
    &x_inv_t * ic * &x_inv
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crba::crba;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::rnea;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Vector3};

    fn two_link_arm() -> Model<f64> {
        let offset1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let offset2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::revolute_y(),
                offset1,
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.2, 0.0, 0.0,
                        0.0, 0.2, 0.0,
                        0.0, 0.0, 0.02,
                    ),
                },
            )
            .add_joint(
                "j2",
                1,
                joint::revolute_y(),
                offset2,
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

    #[test]
    fn aba_zero_torque_gravity_free() {
        // No gravity, no torque → zero acceleration
        let model = ModelBuilder::new()
            .gravity(Vector3::zeros())
            .add_joint(
                "j1",
                0,
                joint::revolute_z(),
                se3::identity(),
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::zeros(),
                    rotational_inertia: Matrix3::identity() * 0.1,
                },
            )
            .build();
        let q = vec![0.0];
        let v = vec![0.0];
        let tau = vec![0.0];
        let qdd = aba(&model, &q, &v, &tau);
        assert_relative_eq!(qdd[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn aba_consistent_with_crba_rnea() {
        // ABA should produce: qdd = M^{-1} (tau - nle)
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let tau_input = vec![5.0, -2.0];

        let qdd_aba = aba(&model, &q, &v, &tau_input);

        // Compute via CRBA + RNEA
        let m = crba(&model, &q);
        let nle = rnea::nonlinear_effects(&model, &q, &v);
        let rhs = DVector::from_column_slice(&tau_input) - nle;
        let m_lu = m.lu();
        let qdd_expected = m_lu.solve(&rhs).unwrap();

        assert_relative_eq!(qdd_aba, qdd_expected, epsilon = 1e-8);
    }

    #[test]
    fn aba_is_pure() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let v = vec![1.0, -0.5];
        let tau = vec![5.0, -2.0];
        let a1 = aba(&model, &q, &v, &tau);
        let a2 = aba(&model, &q, &v, &tau);
        assert_relative_eq!(a1, a2, epsilon = 1e-14);
    }

    #[test]
    fn aba_free_fall() {
        // A single body with gravity should accelerate at g when tau=0
        let model = ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::prismatic_z(),
                se3::identity(),
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::zeros(),
                    rotational_inertia: Matrix3::zeros(),
                },
            )
            .build();
        let q = vec![0.0];
        let v = vec![0.0];
        let tau = vec![0.0];
        let qdd = aba(&model, &q, &v, &tau);
        // Should get -9.81 (falling in -Z)
        assert_relative_eq!(qdd[0], -9.81, epsilon = 1e-8);
    }

    #[test]
    fn minv_times_vec_matches_crba_inverse() {
        // M⁻¹ τ computed via ABA should match M⁻¹ τ computed via CRBA + LU.
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let tau = vec![5.0, -2.0];

        let result_aba = compute_minv_times_vec(&model, &q, &tau);

        let m = crba(&model, &q);
        let m_lu = m.lu();
        let tau_vec = DVector::from_column_slice(&tau);
        let result_crba = m_lu.solve(&tau_vec).unwrap();

        assert_relative_eq!(result_aba, result_crba, epsilon = 1e-8);
    }

    #[test]
    fn compute_minv_matches_crba_inverse() {
        // Full M⁻¹ should satisfy M * M⁻¹ = I.
        let model = two_link_arm();
        let q = vec![0.3, -0.5];

        let m = crba(&model, &q);
        let minv = compute_minv(&model, &q);

        let product = &m * &minv;
        let eye = DMatrix::identity(model.nv, model.nv);
        assert_relative_eq!(product, eye, epsilon = 1e-8);
    }

    #[test]
    fn minv_is_symmetric() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let minv = compute_minv(&model, &q);
        assert_relative_eq!(minv.clone(), minv.transpose(), epsilon = 1e-10);
    }
}
