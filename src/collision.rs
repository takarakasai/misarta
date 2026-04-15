//! Collision detection based on `parry3d`.
//!
//! Provides collision and distance queries for a `GeometryModel` attached to a
//! kinematic `Model<f64>` at a given configuration `q`.

use crate::fk::forward_kinematics;
use crate::geometry::{GeometryModel, GeometryShape};
use crate::model::Model;
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

pub fn collision_pairs(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q: &[f64],
    ignore_same_parent_joint: bool,
) -> Vec<CollisionPair> {
    let objects = build_collision_objects(model, gmodel, q);
    let mut out = Vec::new();

    for i in 0..objects.len() {
        for j in (i + 1)..objects.len() {
            let a = &objects[i];
            let b = &objects[j];

            if ignore_same_parent_joint && a.parent_joint == b.parent_joint {
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

pub fn has_collision(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q: &[f64],
    ignore_same_parent_joint: bool,
) -> bool {
    !collision_pairs(model, gmodel, q, ignore_same_parent_joint).is_empty()
}

pub fn minimum_distance(
    model: &Model<f64>,
    gmodel: &GeometryModel,
    q: &[f64],
    ignore_same_parent_joint: bool,
) -> Option<f64> {
    let objects = build_collision_objects(model, gmodel, q);
    let mut min_d: Option<f64> = None;

    for i in 0..objects.len() {
        for j in (i + 1)..objects.len() {
            let a = &objects[i];
            let b = &objects[j];

            if ignore_same_parent_joint && a.parent_joint == b.parent_joint {
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

    #[test]
    fn ignore_same_parent_joint_works() {
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
}
