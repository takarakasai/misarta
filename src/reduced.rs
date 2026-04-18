//! Model reduction — lock a subset of joints to produce a smaller model.
//!
//! Pinocchio's `buildReducedModel` equivalent.  Given a full model, a list of
//! joint indices to **lock**, and the configuration values at which to lock
//! them, this module creates a new model whose degrees of freedom are reduced
//! by the locked joints' DOFs.
//!
//! Locked joints become `Fixed` and their configuration-dependent placement
//! $M_J(q_{\text{lock}})$ is absorbed into the **child** joint's `placement`.
//! Link inertias of locked joints are merged into their nearest unlocked
//! ancestor using the spatial inertia transformation.
//!
//! # Example
//!
//! ```
//! use misarta::{model::*, joint, se3, reduced};
//!
//! let model = ModelBuilder::<f64>::new()
//!     .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
//!     .add_joint("j2", 1, joint::revolute_x(), se3::identity(), LinkInertia::zero())
//!     .add_joint("j3", 2, joint::revolute_y(), se3::identity(), LinkInertia::zero())
//!     .build();
//! assert_eq!(model.nv, 3);
//!
//! // Lock the middle joint at q=0
//! let q = vec![0.0; model.nq];
//! let reduced = reduced::build_reduced_model(&model, &[2], &q);
//! assert_eq!(reduced.nv, 2);   // j1 + j3 remain active
//! ```

use crate::frames::FrameModel;
use crate::geometry::{GeometryModel, GeometryObject};
use crate::model::{LinkInertia, Model, ModelBuilder};
use crate::se3::{self, SE3};
use nalgebra::{Matrix3, RealField, Vector3};
use std::collections::HashSet;

/// Build a reduced model by locking the specified joints.
///
/// # Arguments
///
/// * `model` — the full (source) model.
/// * `joints_to_lock` — joint indices (1-based) to lock.  Index 0 (universe)
///   is silently ignored.
/// * `q_lock` — full configuration vector at which locked joints are frozen.
///   Must have length `model.nq`.
///
/// # Returns
///
/// A new `Model` with fewer DOFs.  Locked joints are removed from the tree;
/// their fixed transforms are absorbed into the child placement, and their
/// inertias are combined with the nearest unlocked ancestor.
///
/// # Panics
///
/// Panics if `q_lock.len() != model.nq` or if any index in `joints_to_lock`
/// is out of range.
pub fn build_reduced_model<T: RealField + Copy>(
    model: &Model<T>,
    joints_to_lock: &[usize],
    q_lock: &[T],
) -> Model<T> {
    assert_eq!(
        q_lock.len(),
        model.nq,
        "q_lock length {} != model.nq {}",
        q_lock.len(),
        model.nq
    );

    let lock_set: HashSet<usize> = joints_to_lock.iter().copied().filter(|&i| i > 0).collect();

    // Validate indices
    for &idx in &lock_set {
        assert!(
            idx < model.joints.len(),
            "joint index {} out of range (model has {} joints including universe)",
            idx,
            model.joints.len()
        );
    }

    // ── Step 1: Compute the accumulated fixed transform for locked joints ──
    //
    // For each locked joint i, its configuration-dependent placement is:
    //   T_lock[i] = placement[i] * M_J(q_lock[i])
    //
    // When a chain of consecutive joints is locked, the transforms accumulate.
    // We compute, for every joint in the original model, the total fixed
    // transform from its nearest unlocked ancestor to itself.

    let n = model.joints.len();

    // old_to_new[i] = new joint index, or usize::MAX if locked/removed.
    let mut old_to_new = vec![usize::MAX; n];

    // For each joint, the transform from the nearest unlocked ancestor's
    // frame to this joint's frame, accounting for all locked joints in between.
    let mut accumulated_placement: Vec<SE3<T>> = Vec::with_capacity(n);
    for _ in 0..n {
        accumulated_placement.push(se3::identity());
    }

    // The nearest unlocked ancestor (in old indices) for each joint.
    let mut unlocked_ancestor = vec![0usize; n];

    // Universe is always kept
    old_to_new[0] = 0;
    unlocked_ancestor[0] = 0;
    accumulated_placement[0] = se3::identity();

    // Process joints in topological order (they are stored parent-before-child)
    for i in 1..n {
        let parent = model.joints[i].parent;
        let qi = model.q_idx[i];
        let nqi = model.joints[i].joint_type.nq();
        let q_slice = &q_lock[qi..qi + nqi];

        // This joint's fixed transform: placement * M_J(q_lock)
        let m_j = model.joints[i].joint_type.forward(q_slice);
        let this_placement = se3::compose(&model.joints[i].placement, &m_j);

        if lock_set.contains(&i) {
            // Locked — accumulate transform
            unlocked_ancestor[i] = unlocked_ancestor[parent];
            accumulated_placement[i] =
                se3::compose(&accumulated_placement[parent], &this_placement);
        } else {
            // Kept — record the accumulated placement from its unlocked
            // ancestor, composed with this joint's own placement.
            // But only the *locked prefix* is absorbed; this joint's own M_J
            // should NOT be baked in (it will be computed at runtime).
            // So the new placement = accumulated_from_parent * placement_i.
            unlocked_ancestor[i] = i;
            accumulated_placement[i] = se3::identity();
        }
    }

    // ── Step 2: Build the new model ────────────────────────────────────────

    let mut builder = ModelBuilder::<T>::new()
        .name(model.name.clone())
        .root_link_name(model.link_names[0].clone())
        .gravity(model.gravity);

    // Assign new index to universe
    let mut new_idx = 1usize; // next available new joint index

    for i in 1..n {
        if lock_set.contains(&i) {
            continue; // skip locked joints
        }

        let parent_old = model.joints[i].parent;

        // The new parent is the nearest unlocked ancestor
        // ... but if parent itself is unlocked, ancestor_old == parent_old
        // We need to handle the case where parent is locked
        let new_parent = if lock_set.contains(&parent_old) {
            // parent is locked, find the unlocked ancestor
            old_to_new[unlocked_ancestor[parent_old]]
        } else {
            old_to_new[parent_old]
        };

        // New placement absorbs locked intermediate transforms
        let new_placement = if lock_set.contains(&parent_old) {
            se3::compose(&accumulated_placement[parent_old], &model.joints[i].placement)
        } else {
            model.joints[i].placement
        };

        old_to_new[i] = new_idx;
        new_idx += 1;

        builder = builder.add_joint_with_link(
            model.joints[i].name.clone(),
            new_parent,
            model.joints[i].joint_type.clone(),
            new_placement,
            model.inertias[i].clone(),
            model.link_names[i].clone(),
        );
    }

    // ── Step 3: Merge locked-joint inertias into their unlocked ancestors ──

    let mut reduced = builder.build();

    for i in 1..n {
        if !lock_set.contains(&i) {
            continue;
        }

        let ancestor_old = unlocked_ancestor[model.joints[i].parent];
        let ancestor_new = old_to_new[ancestor_old];

        // Transform from the unlocked ancestor frame to the locked joint frame
        let x = if ancestor_old == model.joints[i].parent {
            // Direct child of an unlocked joint — use placement * M_J(q_lock)
            let qi = model.q_idx[i];
            let nqi = model.joints[i].joint_type.nq();
            let m_j = model.joints[i].joint_type.forward(&q_lock[qi..qi + nqi]);
            se3::compose(&model.joints[i].placement, &m_j)
        } else {
            // Multiple locked joints in between — use accumulated placement
            // from the unlocked ancestor, up through the locked parent,
            // then to this locked joint.
            let qi = model.q_idx[i];
            let nqi = model.joints[i].joint_type.nq();
            let m_j = model.joints[i].joint_type.forward(&q_lock[qi..qi + nqi]);
            let parent_acc = &accumulated_placement[model.joints[i].parent];
            se3::compose(parent_acc, &se3::compose(&model.joints[i].placement, &m_j))
        };

        // Merge inertia: transform locked joint's inertia to ancestor frame
        // and add it to the ancestor's inertia.
        let locked_inertia = &model.inertias[i];
        if locked_inertia.mass > T::zero() {
            merge_inertia(&mut reduced.inertias[ancestor_new], locked_inertia, &x);
        }
    }

    reduced
}

/// Build a reduced model together with reduced geometry models.
///
/// Geometry objects attached to locked joints are re-parented to the nearest
/// unlocked ancestor, with their placement adjusted accordingly.
///
/// # Arguments
///
/// * `model` — the full model.
/// * `visual_model` — visual geometry model.
/// * `collision_model` — collision geometry model.
/// * `joints_to_lock` — joint indices to lock.
/// * `q_lock` — configuration vector at which to lock.
///
/// # Returns
///
/// `(reduced_model, reduced_visual, reduced_collision)`.
pub fn build_reduced_model_with_geometry(
    model: &Model<f64>,
    visual_model: &GeometryModel,
    collision_model: &GeometryModel,
    joints_to_lock: &[usize],
    q_lock: &[f64],
) -> (Model<f64>, GeometryModel, GeometryModel) {
    let lock_set: HashSet<usize> = joints_to_lock.iter().copied().filter(|&i| i > 0).collect();
    let n = model.joints.len();

    // Compute mapping and accumulated placement (same logic as build_reduced_model)
    let mut old_to_new = vec![usize::MAX; n];
    let mut accumulated_placement: Vec<SE3<f64>> = vec![se3::identity::<f64>(); n];
    let mut unlocked_ancestor = vec![0usize; n];

    old_to_new[0] = 0;

    for i in 1..n {
        let parent = model.joints[i].parent;
        let qi = model.q_idx[i];
        let nqi = model.joints[i].joint_type.nq();
        let q_slice = &q_lock[qi..qi + nqi];
        let m_j = model.joints[i].joint_type.forward(q_slice);
        let this_placement = se3::compose(&model.joints[i].placement, &m_j);

        if lock_set.contains(&i) {
            unlocked_ancestor[i] = unlocked_ancestor[parent];
            accumulated_placement[i] =
                se3::compose(&accumulated_placement[parent], &this_placement);
        } else {
            unlocked_ancestor[i] = i;
            accumulated_placement[i] = se3::identity();
        }
    }

    // Build new joint index map
    let mut new_idx = 1usize;
    for i in 1..n {
        if !lock_set.contains(&i) {
            old_to_new[i] = new_idx;
            new_idx += 1;
        }
    }

    // Build the reduced dynamics model
    let reduced_model = build_reduced_model(model, joints_to_lock, q_lock);

    // Remap geometry
    let reduced_visual = remap_geometry(
        visual_model,
        &old_to_new,
        &unlocked_ancestor,
        &accumulated_placement,
        &lock_set,
        model,
        q_lock,
    );
    let reduced_collision = remap_geometry(
        collision_model,
        &old_to_new,
        &unlocked_ancestor,
        &accumulated_placement,
        &lock_set,
        model,
        q_lock,
    );

    (reduced_model, reduced_visual, reduced_collision)
}

/// Remap a `GeometryModel` to use the reduced model's joint indices.
fn remap_geometry(
    geom: &GeometryModel,
    old_to_new: &[usize],
    unlocked_ancestor: &[usize],
    accumulated_placement: &[SE3<f64>],
    lock_set: &HashSet<usize>,
    model: &Model<f64>,
    q_lock: &[f64],
) -> GeometryModel {
    let mut result = GeometryModel::new();

    for obj in &geom.objects {
        let old_joint = obj.parent_joint;

        if old_joint >= old_to_new.len() {
            // Out of range — skip
            continue;
        }

        let (new_joint, new_placement) = if lock_set.contains(&old_joint) {
            // Parent joint is locked — re-parent to nearest unlocked ancestor
            let ancestor_old = unlocked_ancestor[old_joint];
            let ancestor_new = old_to_new[ancestor_old];

            // Compute transform from ancestor to this locked joint
            let qi = model.q_idx[old_joint];
            let nqi = model.joints[old_joint].joint_type.nq();
            let m_j = model.joints[old_joint].joint_type.forward(&q_lock[qi..qi + nqi]);
            let joint_transform = se3::compose(&model.joints[old_joint].placement, &m_j);

            // Total transform from unlocked ancestor to the geometry
            let acc_to_joint = if accumulated_placement[model.joints[old_joint].parent]
                != se3::identity()
            {
                se3::compose(
                    &accumulated_placement[model.joints[old_joint].parent],
                    &joint_transform,
                )
            } else {
                if old_joint > 0 && lock_set.contains(&model.joints[old_joint].parent) {
                    se3::compose(
                        &accumulated_placement[model.joints[old_joint].parent],
                        &joint_transform,
                    )
                } else {
                    joint_transform
                }
            };

            let adjusted_placement = se3::compose(&acc_to_joint, &obj.placement);
            (ancestor_new, adjusted_placement)
        } else {
            // Parent joint is kept — just remap the index
            let new_j = old_to_new[old_joint];
            (new_j, obj.placement)
        };

        result.add(GeometryObject {
            name: obj.name.clone(),
            parent_joint: new_joint,
            placement: new_placement,
            shape: obj.shape.clone(),
            mesh_path: obj.mesh_path.clone(),
            mesh_scale: obj.mesh_scale,
            mesh_data: obj.mesh_data.clone(),
            material: obj.material.clone(),
        });
    }

    result
}

/// Reduce a `FrameModel` to use the reduced model's joint indices.
///
/// Frames attached to locked joints are re-parented to the nearest unlocked
/// ancestor, with their placement adjusted to remain in the same world-frame
/// position for the given `q_lock`.
pub fn reduce_frame_model<T: RealField + Copy>(
    frame_model: &FrameModel<T>,
    model: &Model<T>,
    joints_to_lock: &[usize],
    q_lock: &[T],
) -> FrameModel<T> {
    let lock_set: HashSet<usize> = joints_to_lock.iter().copied().filter(|&i| i > 0).collect();
    let n = model.joints.len();

    // Recompute mapping
    let mut old_to_new = vec![usize::MAX; n];
    let mut accumulated_placement: Vec<SE3<T>> = Vec::with_capacity(n);
    for _ in 0..n {
        accumulated_placement.push(se3::identity());
    }
    let mut unlocked_ancestor = vec![0usize; n];
    old_to_new[0] = 0;

    for i in 1..n {
        let parent = model.joints[i].parent;
        let qi = model.q_idx[i];
        let nqi = model.joints[i].joint_type.nq();
        let q_slice = &q_lock[qi..qi + nqi];
        let m_j = model.joints[i].joint_type.forward(q_slice);
        let this_placement = se3::compose(&model.joints[i].placement, &m_j);

        if lock_set.contains(&i) {
            unlocked_ancestor[i] = unlocked_ancestor[parent];
            accumulated_placement[i] =
                se3::compose(&accumulated_placement[parent], &this_placement);
        } else {
            unlocked_ancestor[i] = i;
            accumulated_placement[i] = se3::identity();
        }
    }

    let mut new_idx = 1usize;
    for i in 1..n {
        if !lock_set.contains(&i) {
            old_to_new[i] = new_idx;
            new_idx += 1;
        }
    }

    let mut result = FrameModel::new();
    for frame in &frame_model.frames {
        let old_joint = frame.parent_joint;
        if old_joint >= n {
            continue;
        }

        let (new_joint, new_placement) = if lock_set.contains(&old_joint) {
            let ancestor_old = unlocked_ancestor[old_joint];
            let ancestor_new = old_to_new[ancestor_old];
            let adjusted = se3::compose(&accumulated_placement[old_joint], &frame.placement);
            (ancestor_new, adjusted)
        } else {
            (old_to_new[old_joint], frame.placement)
        };

        result.add_frame(frame.name.clone(), new_joint, new_placement);
    }

    result
}

// ─── Inertia merging helper ─────────────────────────────────────────────────

/// Merge `child_inertia` (expressed in the child's body frame) into
/// `parent_inertia` (in the parent's body frame), given the SE(3) transform
/// `x` from the parent frame to the child frame.
///
/// Uses the parallel axis theorem (Steiner's theorem) for rotational inertia.
fn merge_inertia<T: RealField + Copy>(parent: &mut LinkInertia<T>, child: &LinkInertia<T>, x: &SE3<T>) {
    let r = se3::rotation_matrix(x);
    let t = se3::translation(x);

    let m_p = parent.mass;
    let m_c = child.mass;
    let m_total = m_p + m_c;

    if m_total <= T::zero() {
        return;
    }

    // Child's CoM in parent frame
    let child_com_parent = r * child.center_of_mass + t;

    // New combined CoM
    let new_com = (parent.center_of_mass * m_p + child_com_parent * m_c) / m_total;

    // Parallel axis theorem for both bodies about the new CoM
    // I_new = I_parent_about_new_com + I_child_about_new_com

    // Parent's inertia about new CoM (shift from parent.com to new_com)
    let d_p = parent.center_of_mass - new_com;
    let i_p = parent.rotational_inertia + parallel_axis_shift(m_p, &d_p);

    // Child's inertia rotated into parent frame, then shifted to new CoM
    let i_c_rotated = r * child.rotational_inertia * r.transpose();
    let d_c = child_com_parent - new_com;
    let i_c = i_c_rotated + parallel_axis_shift(m_c, &d_c);

    parent.mass = m_total;
    parent.center_of_mass = new_com;
    parent.rotational_inertia = i_p + i_c;
}

/// Parallel axis theorem: $\Delta I = m (|d|^2 I_3 - d d^T)$
fn parallel_axis_shift<T: RealField + Copy>(mass: T, d: &Vector3<T>) -> Matrix3<T> {
    let d2 = d.dot(d);
    Matrix3::from_diagonal(&Vector3::new(d2, d2, d2)) * mass - d * d.transpose() * mass
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::{aba, crba, fk, rnea};
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Rotation3, Vector3};

    /// Helper: build a 4-link planar chain with non-trivial inertias.
    fn build_4link_chain() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 1.0),
        );
        let inertia = LinkInertia {
            mass: 1.0,
            center_of_mass: Vector3::new(0.0, 0.0, 0.5),
            rotational_inertia: Matrix3::from_diagonal(&Vector3::new(0.1, 0.1, 0.01)),
        };
        ModelBuilder::<f64>::new()
            .name("4link")
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), inertia.clone())
            .add_joint("j2", 1, joint::revolute_z(), offset, inertia.clone())
            .add_joint("j3", 2, joint::revolute_z(), offset, inertia.clone())
            .add_joint("j4", 3, joint::revolute_z(), offset, inertia.clone())
            .build()
    }

    #[test]
    fn lock_nothing_preserves_model() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[], &q);

        assert_eq!(reduced.num_joints(), model.num_joints());
        assert_eq!(reduced.nq, model.nq);
        assert_eq!(reduced.nv, model.nv);
    }

    #[test]
    fn lock_one_middle_joint() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq]; // lock at zero config

        // Lock joint 2 (j2)
        let reduced = build_reduced_model(&model, &[2], &q);

        // 4 joints → 3 joints remain (j1, j3, j4)
        assert_eq!(reduced.num_joints(), 3);
        assert_eq!(reduced.nq, 3);
        assert_eq!(reduced.nv, 3);

        // Joint names preserved
        assert_eq!(reduced.joints[1].name, "j1");
        assert_eq!(reduced.joints[2].name, "j3");
        assert_eq!(reduced.joints[3].name, "j4");

        // j3's parent should now be j1 (index 1 in reduced)
        assert_eq!(reduced.joints[2].parent, 1);
        // j4's parent should be j3 (index 2 in reduced)
        assert_eq!(reduced.joints[3].parent, 2);
    }

    #[test]
    fn lock_multiple_joints() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];

        // Lock joints 2 and 3
        let reduced = build_reduced_model(&model, &[2, 3], &q);

        // 4 → 2 joints remain (j1, j4)
        assert_eq!(reduced.num_joints(), 2);
        assert_eq!(reduced.nq, 2);
        assert_eq!(reduced.nv, 2);
        assert_eq!(reduced.joints[1].name, "j1");
        assert_eq!(reduced.joints[2].name, "j4");

        // j4 should now be child of j1
        assert_eq!(reduced.joints[2].parent, 1);
    }

    #[test]
    fn lock_all_joints() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];

        // Lock all 4 joints
        let reduced = build_reduced_model(&model, &[1, 2, 3, 4], &q);

        assert_eq!(reduced.num_joints(), 0);
        assert_eq!(reduced.nq, 0);
        assert_eq!(reduced.nv, 0);
    }

    #[test]
    fn lock_first_joint() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];

        // Lock joint 1 (j1)
        let reduced = build_reduced_model(&model, &[1], &q);

        assert_eq!(reduced.num_joints(), 3);
        // j2's parent should be universe (0)
        assert_eq!(reduced.joints[1].parent, 0);
        assert_eq!(reduced.joints[1].name, "j2");
    }

    #[test]
    fn lock_last_joint() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];

        // Lock joint 4 (j4)
        let reduced = build_reduced_model(&model, &[4], &q);

        assert_eq!(reduced.num_joints(), 3);
        assert_eq!(reduced.nq, 3);
        assert_eq!(reduced.joints[3].name, "j3");
    }

    #[test]
    fn link_names_preserved() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[2], &q);

        // link_names[0] = root, link_names[1] = link of j1, etc.
        assert_eq!(reduced.link_names[0], model.link_names[0]);
        assert_eq!(reduced.link_names[1], model.link_names[1]); // j1's link
        assert_eq!(reduced.link_names[2], model.link_names[3]); // j3's link
        assert_eq!(reduced.link_names[3], model.link_names[4]); // j4's link
    }

    #[test]
    fn fk_consistency_lock_at_zero() {
        // Lock at zero config: FK of the reduced model at zero should match
        // FK of the full model at zero for shared joints.
        let model = build_4link_chain();
        let q_full = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[2], &q_full);
        let q_reduced = vec![0.0; reduced.nq];

        let data_full = fk::forward_kinematics(&model, &q_full);
        let data_reduced = fk::forward_kinematics(&reduced, &q_reduced);

        // j1 is at index 1 in both
        assert_relative_eq!(
            se3::to_homogeneous(&data_full.oMi[1]),
            se3::to_homogeneous(&data_reduced.oMi[1]),
            epsilon = 1e-12
        );

        // j4 is at index 4 in full, index 3 in reduced
        assert_relative_eq!(
            se3::to_homogeneous(&data_full.oMi[4]),
            se3::to_homogeneous(&data_reduced.oMi[3]),
            epsilon = 1e-12
        );
    }

    #[test]
    fn fk_consistency_lock_at_nonzero() {
        // Lock j2 at 0.5 rad, then check FK matches.
        let model = build_4link_chain();
        let q_full = vec![0.3, 0.5, 0.7, 0.9]; // j1=0.3, j2=0.5(locked), j3=0.7, j4=0.9
        let reduced = build_reduced_model(&model, &[2], &q_full);
        let q_reduced = vec![0.3, 0.7, 0.9]; // j1, j3, j4

        let data_full = fk::forward_kinematics(&model, &q_full);
        let data_reduced = fk::forward_kinematics(&reduced, &q_reduced);

        // j1 (full: 1, reduced: 1) should match
        assert_relative_eq!(
            se3::to_homogeneous(&data_full.oMi[1]),
            se3::to_homogeneous(&data_reduced.oMi[1]),
            epsilon = 1e-12
        );

        // j3 (full: 3, reduced: 2) should match
        assert_relative_eq!(
            se3::to_homogeneous(&data_full.oMi[3]),
            se3::to_homogeneous(&data_reduced.oMi[2]),
            epsilon = 1e-12
        );

        // j4 (full: 4, reduced: 3) should match
        assert_relative_eq!(
            se3::to_homogeneous(&data_full.oMi[4]),
            se3::to_homogeneous(&data_reduced.oMi[3]),
            epsilon = 1e-12
        );
    }

    #[test]
    fn fk_consistency_lock_two_consecutive() {
        // Lock j2 and j3 at nonzero values
        let model = build_4link_chain();
        let q_full = vec![0.1, 0.2, 0.3, 0.4];
        let reduced = build_reduced_model(&model, &[2, 3], &q_full);
        let q_reduced = vec![0.1, 0.4]; // j1, j4

        let data_full = fk::forward_kinematics(&model, &q_full);
        let data_reduced = fk::forward_kinematics(&reduced, &q_reduced);

        // j1 (full:1, reduced:1)
        assert_relative_eq!(
            se3::to_homogeneous(&data_full.oMi[1]),
            se3::to_homogeneous(&data_reduced.oMi[1]),
            epsilon = 1e-12
        );

        // j4 (full:4, reduced:2)
        assert_relative_eq!(
            se3::to_homogeneous(&data_full.oMi[4]),
            se3::to_homogeneous(&data_reduced.oMi[2]),
            epsilon = 1e-12
        );
    }

    #[test]
    fn inertia_merged_for_locked_joint() {
        // When we lock a joint with mass, the mass should be merged into
        // the nearest unlocked ancestor.
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 1.0),
        );
        let model = ModelBuilder::<f64>::new()
            .add_joint(
                "j1", 0, joint::revolute_z(), se3::identity(),
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::zeros(),
                    rotational_inertia: Matrix3::zeros(),
                },
            )
            .add_joint(
                "j2", 1, joint::revolute_z(), offset,
                LinkInertia {
                    mass: 3.0,
                    center_of_mass: Vector3::zeros(),
                    rotational_inertia: Matrix3::zeros(),
                },
            )
            .build();

        let q = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[2], &q);

        // j2's mass (3.0) merged into j1's (2.0) → total 5.0
        assert_relative_eq!(reduced.inertias[1].mass, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn total_mass_preserved() {
        let model = build_4link_chain();
        let q = vec![0.2, 0.3, 0.4, 0.5];
        let reduced = build_reduced_model(&model, &[2, 3], &q);

        let total_full: f64 = model.inertias.iter().map(|i| i.mass).sum();
        let total_reduced: f64 = reduced.inertias.iter().map(|i| i.mass).sum();
        assert_relative_eq!(total_full, total_reduced, epsilon = 1e-12);
    }

    #[test]
    fn gravity_torque_consistency() {
        // The gravity torque of the reduced model should match the full model's
        // gravity torque for the active joints when locked joints are at q_lock.
        let model = build_4link_chain();
        let q_full = vec![0.0, 0.0, 0.0, 0.0];
        let reduced = build_reduced_model(&model, &[2], &q_full);

        let q_reduced = vec![0.0; reduced.nq];
        let v_reduced = vec![0.0; reduced.nv];
        let a_reduced = vec![0.0; reduced.nv];

        let v_full = vec![0.0; model.nv];
        let a_full = vec![0.0; model.nv];

        let tau_full = rnea::rnea(&model, &q_full, &v_full, &a_full);
        let tau_reduced = rnea::rnea(&reduced, &q_reduced, &v_reduced, &a_reduced);

        // Gravity torque for j1 should match between full and reduced
        // (with locked joints contributing their weight)
        // The full model's j1 torque considers all 4 masses.
        // The reduced model's j1 torque considers j1 mass + merged j2 mass + j3+j4 masses.
        assert_relative_eq!(tau_full[0], tau_reduced[0], epsilon = 1e-10);
    }

    #[test]
    fn mass_matrix_dimensions() {
        let model = build_4link_chain();
        let q_full = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[2, 3], &q_full);

        let q_reduced = vec![0.0; reduced.nq];
        let m = crba::crba(&reduced, &q_reduced);

        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        // Mass matrix should be symmetric positive definite
        assert_relative_eq!(m[(0, 1)], m[(1, 0)], epsilon = 1e-14);
    }

    #[test]
    fn aba_runs_on_reduced_model() {
        let model = build_4link_chain();
        let q_full = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[2], &q_full);

        let q = vec![0.0; reduced.nq];
        let v = vec![0.0; reduced.nv];
        let tau = vec![0.0; reduced.nv];

        // ABA should run without panic
        let ddq = aba::aba(&reduced, &q, &v, &tau);
        assert_eq!(ddq.len(), reduced.nv);
    }

    #[test]
    fn geometry_remap_locked_joint() {
        let model = build_4link_chain();
        let mut visual = GeometryModel::new();
        visual.add(GeometryObject {
            name: "geom_j2".into(),
            parent_joint: 2,
            placement: se3::identity(),
            shape: crate::geometry::GeometryShape::Sphere { radius: 0.1 },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
            material: None,
        });
        visual.add(GeometryObject {
            name: "geom_j4".into(),
            parent_joint: 4,
            placement: se3::identity(),
            shape: crate::geometry::GeometryShape::Sphere { radius: 0.05 },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
            material: None,
        });
        let collision = GeometryModel::new();

        let q = vec![0.0; model.nq];
        let (reduced, vis, _col) =
            build_reduced_model_with_geometry(&model, &visual, &collision, &[2], &q);

        // Both geom objects should survive
        assert_eq!(vis.num_objects(), 2);

        // geom_j2 was on locked joint 2 → re-parented to j1 (new index 1)
        assert_eq!(vis.objects[0].name, "geom_j2");
        assert_eq!(vis.objects[0].parent_joint, 1);

        // geom_j4 was on unlocked joint 4 → remapped to new index 3
        assert_eq!(vis.objects[1].name, "geom_j4");
        assert_eq!(vis.objects[1].parent_joint, reduced.num_joints());
    }

    #[test]
    fn frame_model_reduction() {
        let model = build_4link_chain();
        let mut fm = FrameModel::new();
        fm.add_frame("tool_tip", 4, se3::identity());
        fm.add_frame("elbow", 2, se3::identity());

        let q = vec![0.0; model.nq];
        let reduced_fm = reduce_frame_model(&fm, &model, &[2], &q);

        assert_eq!(reduced_fm.frames.len(), 2);
        // "elbow" was on locked joint 2 → re-parented to j1 (new index 1)
        let elbow = reduced_fm.find("elbow").unwrap();
        assert_eq!(elbow.parent_joint, 1);
        // "tool_tip" was on j4 → new index 3
        let tool = reduced_fm.find("tool_tip").unwrap();
        assert_eq!(tool.parent_joint, 3);
    }

    #[test]
    fn branching_tree_reduction() {
        // Build a branching tree:
        //   universe → j1 → j2
        //             ↘ j3 → j4
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 1.0),
        );
        let inertia = LinkInertia {
            mass: 1.0,
            center_of_mass: Vector3::new(0.0, 0.0, 0.5),
            rotational_inertia: Matrix3::from_diagonal(&Vector3::new(0.01, 0.01, 0.001)),
        };
        let model = ModelBuilder::<f64>::new()
            .name("branching")
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), inertia.clone())
            .add_joint("j2", 1, joint::revolute_z(), offset, inertia.clone())
            .add_joint("j3", 1, joint::revolute_x(), offset, inertia.clone())
            .add_joint("j4", 3, joint::revolute_y(), offset, inertia.clone())
            .build();

        assert_eq!(model.num_joints(), 4);

        // Lock j3 — should merge j3 into j1, j4 becomes child of j1
        let q = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[3], &q);

        assert_eq!(reduced.num_joints(), 3); // j1, j2, j4
        assert_eq!(reduced.joints[1].name, "j1");
        assert_eq!(reduced.joints[2].name, "j2");
        assert_eq!(reduced.joints[3].name, "j4");

        // j4's parent was j3 (locked) → now j1 (index 1)
        assert_eq!(reduced.joints[3].parent, 1);
        // j2's parent unchanged: j1 (index 1)
        assert_eq!(reduced.joints[2].parent, 1);
    }

    #[test]
    fn fk_branching_consistency() {
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 1.0),
        );
        let inertia = LinkInertia {
            mass: 1.0,
            center_of_mass: Vector3::zeros(),
            rotational_inertia: Matrix3::zeros(),
        };
        let model = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), inertia.clone())
            .add_joint("j2", 1, joint::revolute_x(), offset, inertia.clone())
            .add_joint("j3", 1, joint::revolute_y(), offset, inertia.clone())
            .add_joint("j4", 3, joint::revolute_z(), offset, inertia.clone())
            .build();

        let q_full = vec![0.1, 0.2, 0.3, 0.4];
        let reduced = build_reduced_model(&model, &[3], &q_full);
        let q_reduced = vec![0.1, 0.2, 0.4]; // j1, j2, j4

        let data_full = fk::forward_kinematics(&model, &q_full);
        let data_reduced = fk::forward_kinematics(&reduced, &q_reduced);

        // j4: full index 4, reduced index 3
        assert_relative_eq!(
            se3::to_homogeneous(&data_full.oMi[4]),
            se3::to_homogeneous(&data_reduced.oMi[3]),
            epsilon = 1e-12
        );
    }

    #[test]
    fn model_name_preserved() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[2], &q);
        assert_eq!(reduced.name, "4link");
    }

    #[test]
    fn gravity_preserved() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];
        let reduced = build_reduced_model(&model, &[2], &q);
        assert_relative_eq!(reduced.gravity, model.gravity, epsilon = 1e-14);
    }

    #[test]
    #[should_panic(expected = "q_lock length")]
    fn wrong_q_lock_length_panics() {
        let model = build_4link_chain();
        build_reduced_model(&model, &[2], &[0.0, 0.0]); // too short
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn invalid_joint_index_panics() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];
        build_reduced_model(&model, &[99], &q);
    }

    #[test]
    fn universe_in_lock_list_is_ignored() {
        let model = build_4link_chain();
        let q = vec![0.0; model.nq];
        // Including 0 should be silently ignored
        let reduced = build_reduced_model(&model, &[0, 2], &q);
        assert_eq!(reduced.num_joints(), 3); // same as locking only [2]
    }

    #[test]
    fn parallel_axis_shift_identity() {
        // No shift → zero change
        let d = Vector3::new(0.0, 0.0, 0.0);
        let shift = parallel_axis_shift(1.0, &d);
        assert_relative_eq!(shift, Matrix3::zeros(), epsilon = 1e-14);
    }

    #[test]
    fn parallel_axis_shift_along_z() {
        // d = [0, 0, h] → Ixx += mh², Iyy += mh², Izz += 0
        let h = 2.0;
        let m = 3.0;
        let d = Vector3::new(0.0, 0.0, h);
        let shift = parallel_axis_shift(m, &d);
        assert_relative_eq!(shift[(0, 0)], m * h * h, epsilon = 1e-14);
        assert_relative_eq!(shift[(1, 1)], m * h * h, epsilon = 1e-14);
        assert_relative_eq!(shift[(2, 2)], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn lock_with_freefloyer() {
        // FreeFlyer base + revolute chain
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 0.5),
        );
        let inertia = LinkInertia {
            mass: 1.0,
            center_of_mass: Vector3::zeros(),
            rotational_inertia: Matrix3::from_diagonal(&Vector3::new(0.01, 0.01, 0.01)),
        };
        let model = ModelBuilder::<f64>::new()
            .add_joint("floating_base", 0, crate::joint::JointType::FreeFlyer, se3::identity(), inertia.clone())
            .add_joint("j1", 1, joint::revolute_z(), offset, inertia.clone())
            .add_joint("j2", 2, joint::revolute_z(), offset, inertia.clone())
            .build();

        assert_eq!(model.nq, 9); // 7 + 1 + 1
        assert_eq!(model.nv, 8); // 6 + 1 + 1

        // Lock the FreeFlyer base
        let mut q = model.neutral_q();
        q[7] = 0.5; // j1 = 0.5
        q[8] = 0.3; // j2 = 0.3
        let reduced = build_reduced_model(&model, &[1], &q);

        assert_eq!(reduced.nq, 2); // j1 + j2
        assert_eq!(reduced.nv, 2);
        assert_eq!(reduced.joints[1].name, "j1");
        assert_eq!(reduced.joints[2].name, "j2");
        // j1 should be child of universe after FreeFlyer is locked
        assert_eq!(reduced.joints[1].parent, 0);
    }
}
