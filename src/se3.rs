//! SE(3) Lie group utilities — referentially transparent pure functions.
//!
//! All functions are pure: they take immutable references and return new values.
//! This makes them safe for automatic differentiation and easy to test.

use nalgebra::{
    Isometry3, Matrix3, Matrix4, Matrix6, Rotation3, Translation3, UnitQuaternion, Vector3,
    Vector6,
};

// ─── Type aliases ────────────────────────────────────────────────────────────

/// A rigid-body placement in 3-D space (rotation + translation).
pub type SE3 = Isometry3<f64>;

/// Spatial motion vector (twist): [angular; linear] ∈ ℝ⁶.
pub type Motion = Vector6<f64>;

/// Spatial force vector (wrench): [torque; force] ∈ ℝ⁶.
pub type Force = Vector6<f64>;

// ─── Construction ────────────────────────────────────────────────────────────

/// Identity placement.
#[inline]
pub fn identity() -> SE3 {
    SE3::identity()
}

/// Build an SE(3) from a rotation matrix and a translation vector.
#[inline]
pub fn from_rotation_and_translation(rot: &Rotation3<f64>, trans: &Vector3<f64>) -> SE3 {
    SE3::from_parts(Translation3::from(*trans), UnitQuaternion::from_rotation_matrix(rot))
}

/// Build an SE(3) from a 4×4 homogeneous matrix (top-left 3×3 must be orthonormal).
pub fn from_homogeneous(m: &Matrix4<f64>) -> SE3 {
    let rot = Rotation3::from_matrix_unchecked(m.fixed_view::<3, 3>(0, 0).into());
    let t = Vector3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]);
    from_rotation_and_translation(&rot, &t)
}

// ─── Conversions ─────────────────────────────────────────────────────────────

/// Convert placement to a 4×4 homogeneous matrix.
#[inline]
pub fn to_homogeneous(se3: &SE3) -> Matrix4<f64> {
    se3.to_homogeneous()
}

/// Extract the rotation matrix from a placement.
#[inline]
pub fn rotation_matrix(se3: &SE3) -> Matrix3<f64> {
    *se3.rotation.to_rotation_matrix().matrix()
}

/// Extract the translation vector from a placement.
#[inline]
pub fn translation(se3: &SE3) -> Vector3<f64> {
    se3.translation.vector
}

// ─── Composition (pure) ─────────────────────────────────────────────────────

/// Compose two placements: result = a * b  (apply b then a).
#[inline]
pub fn compose(a: &SE3, b: &SE3) -> SE3 {
    a * b
}

/// Inverse of a placement.
#[inline]
pub fn inverse(se3: &SE3) -> SE3 {
    se3.inverse()
}

/// Transform a point by a placement.
#[inline]
pub fn act_on_point(se3: &SE3, point: &Vector3<f64>) -> Vector3<f64> {
    se3.transform_point(&nalgebra::Point3::from(*point)).coords
}

// ─── Exponential / Logarithm (Lie algebra ↔ Lie group) ──────────────────────

/// Exponential map: se(3) twist → SE(3) placement.
///
/// twist = [ω; v] where ω is rotation (axis * angle) and v is translation part.
/// Uses Rodrigues' formula.
pub fn exp(twist: &Motion) -> SE3 {
    let omega = Vector3::new(twist[0], twist[1], twist[2]);
    let v = Vector3::new(twist[3], twist[4], twist[5]);
    let theta = omega.norm();

    if theta < 1e-12 {
        // Pure translation
        return SE3::from_parts(Translation3::from(v), UnitQuaternion::identity());
    }

    let _axis = omega / theta;
    let (sin_t, cos_t) = (theta.sin(), theta.cos());

    // Rodrigues rotation
    let rot = Rotation3::new(omega);

    // V matrix for translation: V = I + (1 - cos θ)/θ² [ω]× + (θ - sin θ)/θ³ [ω]×²
    let omega_cross = skew(&omega);
    let omega_cross_sq = omega_cross * omega_cross;
    let theta_sq = theta * theta;
    let v_mat = Matrix3::identity()
        + omega_cross * ((1.0 - cos_t) / theta_sq)
        + omega_cross_sq * ((theta - sin_t) / (theta_sq * theta));
    let t = v_mat * v;

    from_rotation_and_translation(&rot, &t)
}

/// Logarithmic map: SE(3) → se(3) twist.
pub fn log(se3: &SE3) -> Motion {
    let rot = se3.rotation.to_rotation_matrix();
    let t = translation(se3);

    let angle_axis = rot.scaled_axis();
    let theta = angle_axis.norm();

    if theta < 1e-12 {
        return Motion::new(0.0, 0.0, 0.0, t[0], t[1], t[2]);
    }

    let _axis = angle_axis / theta;
    let _half_theta = theta / 2.0;
    let omega_cross = skew(&angle_axis);
    let omega_cross_sq = omega_cross * omega_cross;
    let theta_sq = theta * theta;

    // V_inv = I - 0.5 [ω]× + (1/θ² - (1+cos θ)/(2 θ sin θ)) [ω]×²
    let coeff = 1.0 / theta_sq - (1.0 + theta.cos()) / (2.0 * theta * theta.sin());
    let v_inv = Matrix3::identity() - omega_cross * 0.5 + omega_cross_sq * coeff;
    let v = v_inv * t;

    Motion::new(
        angle_axis[0],
        angle_axis[1],
        angle_axis[2],
        v[0],
        v[1],
        v[2],
    )
}

// ─── Skew-symmetric matrix ──────────────────────────────────────────────────

/// Skew-symmetric matrix [v]× such that [v]× u = v × u.
#[inline]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

// ─── Spatial algebra (Featherstone conventions) ─────────────────────────────

/// 6×6 spatial motion transform: adjoint of SE(3).
///
/// Maps a motion vector from frame B to frame A, given placement A_M_B.
///
/// ```text
///     [ R    0 ]
/// X = [ [p]×R R ]
/// ```
pub fn motion_cross_matrix(se3: &SE3) -> Matrix6<f64> {
    let r = rotation_matrix(se3);
    let p = translation(se3);
    let px_r = skew(&p) * r;

    let mut x = Matrix6::zeros();
    // Top-left 3×3: R
    x.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    // Bottom-left 3×3: [p]× R
    x.fixed_view_mut::<3, 3>(3, 0).copy_from(&px_r);
    // Bottom-right 3×3: R
    x.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    x
}

/// 6×6 spatial force transform (dual of motion transform).
///
/// ```text
///      [ R    [p]×R ]
/// X* = [ 0    R     ]
/// ```
pub fn force_cross_matrix(se3: &SE3) -> Matrix6<f64> {
    let r = rotation_matrix(se3);
    let p = translation(se3);
    let px_r = skew(&p) * r;

    let mut x = Matrix6::zeros();
    x.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    x.fixed_view_mut::<3, 3>(0, 3).copy_from(&px_r);
    x.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    x
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn identity_compose() {
        let id = identity();
        let se3 = exp(&Motion::new(0.0, 0.0, PI / 4.0, 1.0, 2.0, 3.0));
        let result = compose(&id, &se3);
        assert_relative_eq!(to_homogeneous(&result), to_homogeneous(&se3), epsilon = 1e-12);
    }

    #[test]
    fn inverse_roundtrip() {
        let se3 = exp(&Motion::new(0.1, 0.2, 0.3, 1.0, 2.0, 3.0));
        let inv = inverse(&se3);
        let product = compose(&se3, &inv);
        assert_relative_eq!(to_homogeneous(&product), Matrix4::identity(), epsilon = 1e-10);
    }

    #[test]
    fn exp_log_roundtrip() {
        let twist = Motion::new(0.1, -0.2, 0.3, 0.5, -0.1, 0.7);
        let se3 = exp(&twist);
        let recovered = log(&se3);
        assert_relative_eq!(twist, recovered, epsilon = 1e-10);
    }

    #[test]
    fn exp_pure_translation() {
        let twist = Motion::new(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
        let se3 = exp(&twist);
        assert_relative_eq!(translation(&se3), Vector3::new(1.0, 2.0, 3.0), epsilon = 1e-12);
    }

    #[test]
    fn skew_cross_product() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);
        assert_relative_eq!(skew(&a) * b, a.cross(&b), epsilon = 1e-14);
    }
}
