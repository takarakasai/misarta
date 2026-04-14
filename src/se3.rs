//! SE(3) Lie group utilities — referentially transparent pure functions.
//!
//! All functions are pure: they take immutable references and return new values.
//! This makes them safe for automatic differentiation and easy to test.
//!
//! All types and functions are generic over `T: RealField`, enabling use with
//! `f64`, `Dual64` (automatic differentiation), or other scalar types.

use nalgebra::{
    Isometry3, Matrix3, Matrix4, Matrix6, RealField, Rotation3, Translation3, UnitQuaternion,
    Vector3, Vector6,
};

// ─── Type aliases ────────────────────────────────────────────────────────────

/// A rigid-body placement in 3-D space (rotation + translation).
pub type SE3<T> = Isometry3<T>;

/// Spatial motion vector (twist): [angular; linear] ∈ ℝ⁶.
pub type Motion<T> = Vector6<T>;

/// Spatial force vector (wrench): [torque; force] ∈ ℝ⁶.
pub type Force<T> = Vector6<T>;

// ─── Construction ────────────────────────────────────────────────────────────

/// Identity placement.
#[inline]
pub fn identity<T: RealField>() -> SE3<T> {
    SE3::identity()
}

/// Build an SE(3) from a rotation matrix and a translation vector.
#[inline]
pub fn from_rotation_and_translation<T: RealField>(
    rot: &Rotation3<T>,
    trans: &Vector3<T>,
) -> SE3<T> {
    SE3::from_parts(
        Translation3::from(trans.clone()),
        UnitQuaternion::from_rotation_matrix(rot),
    )
}

/// Build an SE(3) from a 4×4 homogeneous matrix (top-left 3×3 must be orthonormal).
pub fn from_homogeneous<T: RealField>(m: &Matrix4<T>) -> SE3<T> {
    let rot = Rotation3::from_matrix_unchecked(m.fixed_view::<3, 3>(0, 0).into_owned());
    let t = Vector3::new(m[(0, 3)].clone(), m[(1, 3)].clone(), m[(2, 3)].clone());
    from_rotation_and_translation(&rot, &t)
}

// ─── Conversions ─────────────────────────────────────────────────────────────

/// Convert placement to a 4×4 homogeneous matrix.
#[inline]
pub fn to_homogeneous<T: RealField>(se3: &SE3<T>) -> Matrix4<T> {
    se3.to_homogeneous()
}

/// Extract the rotation matrix from a placement.
#[inline]
pub fn rotation_matrix<T: RealField>(se3: &SE3<T>) -> Matrix3<T> {
    se3.rotation.clone().to_rotation_matrix().matrix().clone()
}

/// Extract the translation vector from a placement.
#[inline]
pub fn translation<T: RealField>(se3: &SE3<T>) -> Vector3<T> {
    se3.translation.vector.clone()
}

// ─── Composition (pure) ─────────────────────────────────────────────────────

/// Compose two placements: result = a * b  (apply b then a).
#[inline]
pub fn compose<T: RealField>(a: &SE3<T>, b: &SE3<T>) -> SE3<T> {
    a * b
}

/// Inverse of a placement.
#[inline]
pub fn inverse<T: RealField>(se3: &SE3<T>) -> SE3<T> {
    se3.inverse()
}

/// Transform a point by a placement.
#[inline]
pub fn act_on_point<T: RealField>(se3: &SE3<T>, point: &Vector3<T>) -> Vector3<T> {
    se3.transform_point(&nalgebra::Point3::from(point.clone())).coords
}

// ─── Exponential / Logarithm (Lie algebra ↔ Lie group) ──────────────────────

/// Exponential map: se(3) twist → SE(3) placement.
///
/// twist = [ω; v] where ω is rotation (axis * angle) and v is translation part.
/// Uses Rodrigues' formula.
pub fn exp<T: RealField>(twist: &Motion<T>) -> SE3<T> {
    let omega = Vector3::new(twist[0].clone(), twist[1].clone(), twist[2].clone());
    let v = Vector3::new(twist[3].clone(), twist[4].clone(), twist[5].clone());
    let theta = omega.norm();

    let eps: T = nalgebra::convert(1e-12);
    if theta < eps {
        // Pure translation
        return SE3::from_parts(Translation3::from(v), UnitQuaternion::identity());
    }

    let (sin_t, cos_t) = (theta.clone().sin(), theta.clone().cos());

    // Rodrigues rotation
    let rot = Rotation3::new(omega.clone());

    // V matrix for translation: V = I + (1 - cos θ)/θ² [ω]× + (θ - sin θ)/θ³ [ω]×²
    let omega_cross = skew(&omega);
    let omega_cross_sq = &omega_cross * &omega_cross;
    let theta_sq = theta.clone() * theta.clone();
    let one: T = nalgebra::convert(1.0);
    let v_mat = Matrix3::identity()
        + &omega_cross * ((one.clone() - cos_t) / theta_sq.clone())
        + omega_cross_sq * ((theta.clone() - sin_t) / (theta_sq * theta));
    let t = v_mat * v;

    from_rotation_and_translation(&rot, &t)
}

/// Logarithmic map: SE(3) → se(3) twist.
pub fn log<T: RealField>(se3: &SE3<T>) -> Motion<T> {
    let rot = se3.rotation.clone().to_rotation_matrix();
    let t = translation(se3);

    let angle_axis = rot.scaled_axis();
    let theta = angle_axis.norm();

    let eps: T = nalgebra::convert(1e-12);
    if theta < eps {
        return Motion::new(
            T::zero(),
            T::zero(),
            T::zero(),
            t[0].clone(),
            t[1].clone(),
            t[2].clone(),
        );
    }

    let omega_cross = skew(&angle_axis);
    let omega_cross_sq = &omega_cross * &omega_cross;
    let theta_sq = theta.clone() * theta.clone();

    // V_inv = I - 0.5 [ω]× + (1/θ² - (1+cos θ)/(2 θ sin θ)) [ω]×²
    let half: T = nalgebra::convert(0.5);
    let one: T = nalgebra::convert(1.0);
    let two: T = nalgebra::convert(2.0);
    let coeff = one.clone() / theta_sq
        - (one + theta.clone().cos()) / (two * theta.clone() * theta.sin());
    let v_inv = Matrix3::identity() - &omega_cross * half + omega_cross_sq * coeff;
    let v = v_inv * t;

    Motion::new(
        angle_axis[0].clone(),
        angle_axis[1].clone(),
        angle_axis[2].clone(),
        v[0].clone(),
        v[1].clone(),
        v[2].clone(),
    )
}

// ─── Skew-symmetric matrix ──────────────────────────────────────────────────

/// Skew-symmetric matrix [v]× such that [v]× u = v × u.
#[inline]
pub fn skew<T: RealField>(v: &Vector3<T>) -> Matrix3<T> {
    let (x, y, z) = (v[0].clone(), v[1].clone(), v[2].clone());
    Matrix3::new(
        T::zero(),
        -z.clone(),
        y.clone(),
        z,
        T::zero(),
        -x.clone(),
        -y,
        x,
        T::zero(),
    )
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
pub fn motion_cross_matrix<T: RealField>(se3: &SE3<T>) -> Matrix6<T> {
    let r = rotation_matrix(se3);
    let p = translation(se3);
    let px_r = skew(&p) * &r;

    let mut x = Matrix6::zeros();
    x.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    x.fixed_view_mut::<3, 3>(3, 0).copy_from(&px_r);
    x.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    x
}

/// 6×6 spatial force transform (dual of motion transform).
///
/// ```text
///      [ R    [p]×R ]
/// X* = [ 0    R     ]
/// ```
pub fn force_cross_matrix<T: RealField>(se3: &SE3<T>) -> Matrix6<T> {
    let r = rotation_matrix(se3);
    let p = translation(se3);
    let px_r = skew(&p) * &r;

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
        let id = identity::<f64>();
        let se3 = exp(&Motion::new(0.0, 0.0, PI / 4.0, 1.0, 2.0, 3.0));
        let result = compose(&id, &se3);
        assert_relative_eq!(to_homogeneous(&result), to_homogeneous(&se3), epsilon = 1e-12);
    }

    #[test]
    fn inverse_roundtrip() {
        let se3 = exp(&Motion::new(0.1, 0.2, 0.3, 1.0, 2.0, 3.0));
        let inv = inverse(&se3);
        let product = compose(&se3, &inv);
        assert_relative_eq!(
            to_homogeneous(&product),
            Matrix4::identity(),
            epsilon = 1e-10
        );
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
        assert_relative_eq!(
            translation(&se3),
            Vector3::new(1.0, 2.0, 3.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn skew_cross_product() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);
        assert_relative_eq!(skew(&a) * b, a.cross(&b), epsilon = 1e-14);
    }
}
