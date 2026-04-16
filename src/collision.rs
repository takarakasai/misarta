//! Collision detection based on `parry3d`.
//!
//! Provides collision and distance queries for a `GeometryModel` attached to a
//! kinematic `Model<f64>` at a given configuration `q`.

use crate::fk::forward_kinematics;
use crate::geometry::{GeometryModel, GeometryShape};
use crate::model::Model;

/// Returns `true` if the two joint indices are the same or are direct parent–child.
///
/// This is used to skip geometry pairs whose links are physically connected
/// and thus always interpenetrate at the joint.
fn are_adjacent_joints(model: &Model<f64>, a: usize, b: usize) -> bool {
    if a == b {
        return true;
    }
    // a is direct parent of b
    if b < model.joints.len() && model.joints[b].parent == a {
        return true;
    }
    // b is direct parent of a
    if a < model.joints.len() && model.joints[a].parent == b {
        return true;
    }
    false
}
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use parry3d::query;
use parry3d::shape::{Ball, Capsule, Cone, Cuboid, Cylinder, SharedShape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollisionPair {
    pub a: usize,
    pub b: usize,
}

#[derive(Debug, Clone)]
struct CollisionObject {
    index: usize,
    parent_joint: usize,
    world_pose: Isometry3<f64>,
    shape: SharedShape,
}

fn z_axis_to_y_axis_rotation() -> UnitQuaternion<f64> {
    UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2)
}

fn shape_to_parry(shape: &GeometryShape) -> Option<(SharedShape, Isometry3<f64>)> {
    match shape {
        GeometryShape::Box { x, y, z } => {
            let hx = x * 0.5;
            let hy = y * 0.5;
            let hz = z * 0.5;
            Some((
                SharedShape::new(Cuboid::new(Vector3::new(hx, hy, hz))),
                Isometry3::identity(),
            ))
        }
        GeometryShape::Sphere { radius } => {
            Some((SharedShape::new(Ball::new(*radius)), Isometry3::identity()))
        }
        GeometryShape::Cylinder { radius, length } => {
            let half = length * 0.5;
            let rot = z_axis_to_y_axis_rotation();
            Some((
                SharedShape::new(Cylinder::new(half, *radius)),
                Isometry3::from_parts(Translation3::identity(), rot),
            ))
        }
        GeometryShape::Capsule { radius, length } => {
            let half = length * 0.5;
            let rot = z_axis_to_y_axis_rotation();
            Some((
                SharedShape::new(Capsule::new_y(half, *radius)),
                Isometry3::from_parts(Translation3::identity(), rot),
            ))
        }
        GeometryShape::Cone { radius, length } => {
            let half = length * 0.5;
            let rot = z_axis_to_y_axis_rotation();
            Some((
                SharedShape::new(Cone::new(half, *radius)),
                Isometry3::from_parts(Translation3::identity(), rot),
            ))
        }
        GeometryShape::Mesh { .. } => {
            // GeometryShape::Mesh stores only a file path; mesh vertices are not available.
            None
        }
    }
}

fn build_collision_objects(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q: &[f64],
) -> Vec<CollisionObject> {
    let data = forward_kinematics(model, q);
    let mut objects = Vec::new();

    for (i, obj) in gmodel.objects.iter().enumerate() {
        if obj.parent_joint >= data.oMi.len() {
            continue;
        }
        if let Some((shape, local_shape_pose)) = shape_to_parry(&obj.shape) {
            let world_pose = data.oMi[obj.parent_joint] * obj.placement * local_shape_pose;
            objects.push(CollisionObject {
                index: i,
                parent_joint: obj.parent_joint,
                world_pose,
                shape,
            });
        }
    }
    objects
}

/// Return all colliding geometry object index pairs.
///
/// `ignore_adjacent_links` — when `true`, skips pairs whose parent joints are
/// the same joint **or** are directly connected (parent–child). This prevents
/// false positives from geometries that are physically touching at a joint.
pub fn collision_pairs(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q: &[f64],
    ignore_adjacent_links: bool,
) -> Vec<CollisionPair> {
    let objects = build_collision_objects(model, gmodel, q);
    let mut out = Vec::new();

    for i in 0..objects.len() {
        for j in (i + 1)..objects.len() {
            let a = &objects[i];
            let b = &objects[j];

            if ignore_adjacent_links && are_adjacent_joints(model, a.parent_joint, b.parent_joint) {
                continue;
            }

            if query::intersection_test(&a.world_pose, &*a.shape, &b.world_pose, &*b.shape)
                .unwrap_or(false)
            {
                out.push(CollisionPair {
                    a: a.index,
                    b: b.index,
                });
            }
        }
    }
    out
}

/// Returns `true` if any two non-adjacent geometry objects overlap.
pub fn has_collision(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q: &[f64],
    ignore_adjacent_links: bool,
) -> bool {
    !collision_pairs(model, gmodel, q, ignore_adjacent_links).is_empty()
}

/// Return the minimum separation distance between any two non-adjacent geometry pairs.
/// Returns `None` if there are fewer than two comparable objects.
pub fn minimum_distance(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q: &[f64],
    ignore_adjacent_links: bool,
) -> Option<f64> {
    let objects = build_collision_objects(model, gmodel, q);
    let mut min_d: Option<f64> = None;

    for i in 0..objects.len() {
        for j in (i + 1)..objects.len() {
            let a = &objects[i];
            let b = &objects[j];

            if ignore_adjacent_links && are_adjacent_joints(model, a.parent_joint, b.parent_joint) {
                continue;
            }

            let d = query::distance(&a.world_pose, &*a.shape, &b.world_pose, &*b.shape)
                .unwrap_or(f64::INFINITY);

            min_d = Some(match min_d {
                Some(v) => v.min(d),
                None => d,
            });
        }
    }

    min_d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{GeometryModel, GeometryObject, GeometryShape};
    use crate::joint;
    use crate::joint::JointType;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{Rotation3, Vector3};

    fn two_joint_model() -> Model<f64> {
        ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::revolute_z(),
                se3::identity(),
                LinkInertia::zero(),
            )
            .add_joint(
                "j2",
                1,
                JointType::Fixed,
                se3::from_rotation_and_translation(
                    &Rotation3::identity(),
                    &Vector3::new(1.5, 0.0, 0.0),
                ),
                LinkInertia::zero(),
            )
            .build()
    }

    #[test]
    fn overlapping_spheres_collide() {
        let model = two_joint_model();
        let mut gm = GeometryModel::new();

        gm.add(GeometryObject {
            name: "s1".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 1.0 },
            mesh_path: None,
            mesh_scale: None,
        });

        gm.add(GeometryObject {
            name: "s2".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 1.0 },
            mesh_path: None,
            mesh_scale: None,
        });

        let q = vec![0.0];
        assert!(has_collision(&model, &gm, &q, false));
        assert_eq!(collision_pairs(&model, &gm, &q, false), vec![CollisionPair { a: 0, b: 1 }]);
    }

    #[test]
    fn separated_spheres_no_collision() {
        let model = two_joint_model();
        let mut gm = GeometryModel::new();

        gm.add(GeometryObject {
            name: "s1".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.4 },
            mesh_path: None,
            mesh_scale: None,
        });

        gm.add(GeometryObject {
            name: "s2".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.4 },
            mesh_path: None,
            mesh_scale: None,
        });

        let q = vec![0.0];
        assert!(!has_collision(&model, &gm, &q, false));
        assert!(collision_pairs(&model, &gm, &q, false).is_empty());
    }

    #[test]
    fn minimum_distance_matches_expected_for_spheres() {
        let model = two_joint_model();
        let mut gm = GeometryModel::new();

        gm.add(GeometryObject {
            name: "s1".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.5 },
            mesh_path: None,
            mesh_scale: None,
        });

        gm.add(GeometryObject {
            name: "s2".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.5 },
            mesh_path: None,
            mesh_scale: None,
        });

        let q = vec![0.0];
        // center distance = 1.5, radii sum = 1.0 => distance 0.5
        let d = minimum_distance(&model, &gm, &q, false).unwrap();
        assert_relative_eq!(d, 0.5, epsilon = 1e-9);
    }

    /// Same joint index → always adjacent, excluded when flag is set.
    #[test]
    fn ignore_adjacent_excludes_same_joint() {
        let model = two_joint_model();
        let mut gm = GeometryModel::new();

        gm.add(GeometryObject {
            name: "a".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 1.0 },
            mesh_path: None,
            mesh_scale: None,
        });
        gm.add(GeometryObject {
            name: "b".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 1.0 },
            mesh_path: None,
            mesh_scale: None,
        });

        let q = vec![0.0];
        assert!(has_collision(&model, &gm, &q, false));
        assert!(!has_collision(&model, &gm, &q, true));
    }

    /// Parent joint (1) and child joint (2) are adjacent → excluded even though
    /// the spheres overlap, because the links are physically connected.
    #[test]
    fn ignore_adjacent_excludes_parent_child_joint() {
        // Spheres on joint 1 and joint 2; joint 2's parent is joint 1.
        // With ignore=false the overlap is reported; with ignore=true it is skipped.
        let model = two_joint_model();
        let mut gm = GeometryModel::new();

        gm.add(GeometryObject {
            name: "link1_geom".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 1.0 },
            mesh_path: None,
            mesh_scale: None,
        });
        gm.add(GeometryObject {
            name: "link2_geom".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 1.0 },
            mesh_path: None,
            mesh_scale: None,
        });

        let q = vec![0.0];
        // Centers are 1.5 m apart, each radius 1.0 m → overlap, reported without filter
        assert!(has_collision(&model, &gm, &q, false));
        // Parent–child pair → excluded with filter
        assert!(!has_collision(&model, &gm, &q, true));
    }

    /// Non-adjacent joints (e.g. joints 1 and 2 when a 3rd joint exists between
    /// them and the target) are NOT excluded even with the flag set.
    #[test]
    fn non_adjacent_joints_still_detected() {
        // 3-link chain: j1 → j2 → j3. Geometries on j1 and j3 are not adjacent.
        let model = ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint(
                "j2", 1, JointType::Fixed,
                se3::from_rotation_and_translation(&Rotation3::identity(), &Vector3::new(0.5, 0.0, 0.0)),
                LinkInertia::zero(),
            )
            .add_joint(
                "j3", 2, JointType::Fixed,
                se3::from_rotation_and_translation(&Rotation3::identity(), &Vector3::new(0.5, 0.0, 0.0)),
                LinkInertia::zero(),
            )
            .build();

        let mut gm = GeometryModel::new();
        gm.add(GeometryObject {
            name: "a".into(),
            parent_joint: 1,  // j1
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.8 },
            mesh_path: None,
            mesh_scale: None,
        });
        gm.add(GeometryObject {
            name: "b".into(),
            parent_joint: 3,  // j3, 1.0 m from j1 center, radius sum 1.6 → overlap
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.8 },
            mesh_path: None,
            mesh_scale: None,
        });

        let q = vec![0.0];
        // j1 and j3 are NOT direct parent-child → still reported even with flag
        assert!(has_collision(&model, &gm, &q, true));
    }
}
