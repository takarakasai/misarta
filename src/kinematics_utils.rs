//! Kinematics utilities — convenience functions for operational space control.
//!
//! Higher-level wrappers for common kinematics tasks:
//! - Frame-to-frame Jacobians and placements
//! - Numerical geometry functions (distance, closest points)
//! - API helpers for operational space control
//!
//! All functions are generic over `T: RealField`.

use crate::fk::forward_kinematics;
use crate::jacobian::compute_joint_jacobian;
use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, Isometry3, Point3, RealField, Vector3};

/// Compute the Jacobian of one frame with respect to another.
///
/// This is useful for operational space control: computing the Jacobian
/// of an end-effector relative to an intermediate frame.
///
/// Returns the 6×nv Jacobian of `target_frame` with respect to `reference_frame`.
pub fn frame_to_frame_jacobian<T: RealField>(
    model: &Model<T>,
    q: &[T],
    reference_joint_idx: usize,
    target_joint_idx: usize,
) -> DMatrix<T> {
    let nv = model.nv;
    let data = forward_kinematics(model, q);

    // Jacobian of target in world frame
    let j_target = compute_joint_jacobian(model, q, target_joint_idx);

    // Jacobian of reference in world frame
    let j_ref = compute_joint_jacobian(model, q, reference_joint_idx);

    // Transform to reference frame
    let ref_m_world = data.oMi[reference_joint_idx].inverse();
    let r_ref = se3::rotation_matrix(&ref_m_world);

    let mut j_relative = DMatrix::zeros(6, nv);

    // Rotate Jacobians to reference frame
    for col in 0..nv {
        let j_target_col = j_target.column(col).clone_owned();
        let j_ref_col = j_ref.column(col).clone_owned();

        // Transform to reference frame
        let mut v_target = Vector3::zeros();
        let mut w_target = Vector3::zeros();
        for i in 0..3 {
            w_target[i] = j_target_col[i].clone();
            v_target[i] = j_target_col[3 + i].clone();
        }

        let mut w_ref = Vector3::zeros();
        let mut v_ref = Vector3::zeros();
        for i in 0..3 {
            w_ref[i] = j_ref_col[i].clone();
            v_ref[i] = j_ref_col[3 + i].clone();
        }

        // Rotate
        let w_rot = &r_ref * &w_target;
        let v_rot = &r_ref * &v_target;
        let w_ref_rot = &r_ref * &w_ref;
        let v_ref_rot = &r_ref * &v_ref;

        // Relative Jacobian
        let w_rel = w_rot - w_ref_rot;
        let v_rel = v_rot - v_ref_rot;

        for i in 0..3 {
            j_relative[(i, col)] = w_rel[i].clone();
            j_relative[(3 + i, col)] = v_rel[i].clone();
        }
    }

    j_relative
}

/// Compute the absolute placement of one frame with respect to another.
///
/// Returns the SE(3) transform from reference frame to target frame.
pub fn frame_to_frame_placement<T: RealField>(
    model: &Model<T>,
    q: &[T],
    reference_joint_idx: usize,
    target_joint_idx: usize,
) -> Isometry3<T> {
    let data = forward_kinematics(model, q);
    let ref_m_world = data.oMi[reference_joint_idx].inverse();
    ref_m_world * data.oMi[target_joint_idx].clone()
}

/// Compute the point-to-point distance between two joints/frames.
///
/// Returns the Euclidean distance between the origins of two reference frames.
pub fn frame_distance<T: RealField>(
    model: &Model<T>,
    q: &[T],
    frame1_idx: usize,
    frame2_idx: usize,
) -> T {
    let data = forward_kinematics(model, q);
    let p1 = se3::translation(&data.oMi[frame1_idx]);
    let p2 = se3::translation(&data.oMi[frame2_idx]);
    (p2 - p1).norm()
}

/// Compute point-to-plane distance (useful for contact geometry).
///
/// Plane defined by a point on the plane and a normal vector (unit norm assumed).
///
/// Distance = |n · (p - p_plane)|
pub fn point_to_plane_distance<T: RealField>(
    point: &Point3<T>,
    plane_point: &Point3<T>,
    plane_normal: &Vector3<T>,
) -> T {
    let rel = point - plane_point;
    (rel.transpose() * plane_normal)[0].clone().abs()
}

/// Find the closest point on a line segment to a given point.
///
/// Returns the closest point on segment [p0, p1] to point `p`,
/// and the parameter t ∈ [0, 1] indicating position along segment.
pub fn closest_point_on_segment<T: RealField>(
    p: &Point3<T>,
    p0: &Point3<T>,
    p1: &Point3<T>,
) -> (Point3<T>, T) {
    let seg = p1 - p0;
    let rel = p - p0;

    let seg_len_sq = seg.dot(&seg);

    // Avoid division by zero
    if seg_len_sq < nalgebra::convert::<f64, T>(1e-15) {
        return (p0.clone(), T::zero());
    }

    let t = rel.dot(&seg) / seg_len_sq;
    let t_clamped = if t < T::zero() {
        T::zero()
    } else if t > T::one() {
        T::one()
    } else {
        t.clone()
    };

    let closest = p0 + seg * t_clamped.clone();
    (closest, t_clamped)
}

/// Compute the closest point between two line segments (4 parameters).
///
/// Returns (point_on_seg1, point_on_seg2, distance, s, t)
/// where s, t ∈ [0, 1] parametrize the two segments.
pub fn closest_points_between_segments<T: RealField>(
    p0: &Point3<T>,
    p1: &Point3<T>,
    q0: &Point3<T>,
    q1: &Point3<T>,
) -> (Point3<T>, Point3<T>, T) {
    let seg1 = p1 - p0;
    let seg2 = q1 - q0;
    let cross = q0 - p0;

    let a = seg1.dot(&seg1);
    let b = seg1.dot(&seg2);
    let c = seg2.dot(&seg2);
    let d = seg1.dot(&cross);
    let e = seg2.dot(&cross);

    let denom = a.clone() * c.clone() - b.clone() * b.clone();

    let (s, t) = if denom.clone().abs() < nalgebra::convert::<f64, T>(1e-15) {
        // Segments are parallel
        (T::zero(), T::zero())
    } else {
        let s = (b.clone() * e.clone() - c.clone() * d.clone()) / denom.clone();
        let t = (a.clone() * e - b * d) / denom.clone();

        let s_clamped = if s < T::zero() {
            T::zero()
        } else if s > T::one() {
            T::one()
        } else {
            s
        };

        let t_clamped = if t < T::zero() {
            T::zero()
        } else if t > T::one() {
            T::one()
        } else {
            t
        };

        (s_clamped, t_clamped)
    };

    let pt1 = p0 + seg1 * s.clone();
    let pt2 = q0 + seg2 * t.clone();
    let dist = (&pt2 - &pt1).norm();

    (pt1, pt2, dist)
}

/// Compute signed distance from a point to an axis-aligned bounding box.
///
/// Negative inside, positive outside.
pub fn point_to_aabb_distance<T: RealField>(
    point: &Point3<T>,
    aabb_min: &Point3<T>,
    aabb_max: &Point3<T>,
) -> T {
    let mut dist_sq = T::zero();

    for i in 0..3 {
        if point[i] < aabb_min[i] {
            let d = aabb_min[i].clone() - point[i].clone();
            dist_sq = dist_sq.clone() + d.clone() * d;
        } else if point[i] > aabb_max[i] {
            let d = point[i].clone() - aabb_max[i].clone();
            dist_sq = dist_sq.clone() + d.clone() * d;
        }
    }

    if dist_sq < nalgebra::convert::<f64, T>(1e-15) {
        // Inside or on boundary — compute penetration depth
        let mut min_pen = T::from_f64(f64::INFINITY).unwrap();
        for i in 0..3 {
            let pen_min = point[i].clone() - aabb_min[i].clone();
            let pen_max = aabb_max[i].clone() - point[i].clone();
            let pen = if pen_min < pen_max {
                pen_min
            } else {
                pen_max
            };
            if pen < min_pen {
                min_pen = pen;
            }
        }
        -min_pen
    } else {
        dist_sq.sqrt()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_simple_model() -> Model<f64> {
        use crate::joint::revolute_z;
        use crate::model::{ModelBuilder, LinkInertia};

        // Simple 2-DOF arm
        ModelBuilder::new()
            .add_joint(
                "joint0",
                0,
                revolute_z(),
                nalgebra::Isometry3::identity(),
                LinkInertia::zero(),
            )
            .add_joint(
                "joint1",
                1,
                revolute_z(),
                nalgebra::Isometry3::new(Vector3::new(1.0, 0.0, 0.0), Default::default()),
                LinkInertia::zero(),
            )
            .build()
    }

    #[test]
    fn point_to_plane_distance_on_plane() {
        let point = Point3::new(1.0, 2.0, 3.0);
        let plane_point = Point3::new(0.0, 0.0, 0.0);
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);

        let dist = point_to_plane_distance(&point, &plane_point, &plane_normal);
        assert_relative_eq!(dist, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn point_to_plane_distance_zero() {
        let point = Point3::new(1.0, 2.0, 0.0);
        let plane_point = Point3::new(0.0, 0.0, 0.0);
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);

        let dist = point_to_plane_distance(&point, &plane_point, &plane_normal);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn closest_point_on_segment_at_start() {
        let p = Point3::new(0.0, 0.0, 0.0);
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        let (closest, t) = closest_point_on_segment(&p, &p0, &p1);
        assert_relative_eq!(t, 0.0, epsilon = 1e-10);
        assert_relative_eq!(closest[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn closest_point_on_segment_at_end() {
        let p = Point3::new(1.0, 0.0, 0.0);
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        let (closest, t) = closest_point_on_segment(&p, &p0, &p1);
        assert_relative_eq!(t, 1.0, epsilon = 1e-10);
        assert_relative_eq!(closest[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn closest_point_on_segment_midpoint() {
        let p = Point3::new(0.5, 0.0, 0.0);
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        let (closest, t) = closest_point_on_segment(&p, &p0, &p1);
        assert_relative_eq!(t, 0.5, epsilon = 1e-10);
        assert_relative_eq!(closest[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn closest_point_on_segment_perpendicular() {
        let p = Point3::new(0.5, 1.0, 0.0);
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);

        let (closest, t) = closest_point_on_segment(&p, &p0, &p1);
        assert_relative_eq!(t, 0.5, epsilon = 1e-10);
        assert_relative_eq!(closest[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(closest[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn point_to_aabb_distance_outside() {
        let point = Point3::new(5.0, 5.0, 5.0);
        let aabb_min = Point3::new(0.0, 0.0, 0.0);
        let aabb_max = Point3::new(1.0, 1.0, 1.0);

        let dist = point_to_aabb_distance(&point, &aabb_min, &aabb_max);
        // Distance to corner (1,1,1) is sqrt((5-1)^2 + (5-1)^2 + (5-1)^2) = 4*sqrt(3)
        let expected = 4.0 * 3.0_f64.sqrt();
        assert_relative_eq!(dist, expected, epsilon = 1e-10);
    }

    #[test]
    fn point_to_aabb_distance_inside() {
        let point = Point3::new(0.5, 0.5, 0.5);
        let aabb_min = Point3::new(0.0, 0.0, 0.0);
        let aabb_max = Point3::new(1.0, 1.0, 1.0);

        let dist = point_to_aabb_distance(&point, &aabb_min, &aabb_max);
        // Inside — distance is negative (penetration = 0.5)
        assert!(dist < 0.0);
        assert_relative_eq!(dist, -0.5, epsilon = 1e-10);
    }

    #[test]
    fn frame_distance_simple() {
        let model = make_simple_model();
        let q = vec![0.0, 0.0];

        // Distance between joint 1 (at origin) and joint 2 (at [1,0,0])
        let dist = frame_distance(&model, &q, 1, 2);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn closest_points_segments_clearly_distant() {
        // Two segments that are clearly distant
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let q0 = Point3::new(0.0, 2.0, 0.0);
        let q1 = Point3::new(1.0, 2.0, 0.0);

        let (pt1, pt2, dist) = closest_points_between_segments(&p0, &p1, &q0, &q1);
        // Parallel segments 2 units apart
        assert_relative_eq!(dist, 2.0, epsilon = 1e-10);
    }
}
