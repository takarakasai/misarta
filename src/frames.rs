//! Operational (task-space) frames — named reference frames attached to bodies.
//!
//! An **operational frame** is a fixed offset from a joint frame, identified by
//! name. This is Pinocchio's concept of "frames" (not to be confused with joint
//! frames): things like tool-centre-points, sensor mounts, etc.
//!
//! This module provides:
//!
//! - [`Frame`] — definition of an operational frame
//! - [`FrameModel`] — collection of frames attached to a model
//! - [`compute_frame_placement`] — world placement of a frame
//! - [`compute_frame_jacobian`] — geometric Jacobian expressed at the frame
//!
//! **Pure functions**, generic over `T: RealField`.

use crate::data::Data;
use crate::fk::forward_kinematics;
use crate::model::Model;
use crate::se3::{self, SE3};
use nalgebra::{DMatrix, RealField, Vector3};

/// A named reference frame rigidly attached to a joint frame.
#[derive(Debug, Clone)]
pub struct Frame<T: RealField> {
    /// Human-readable name (e.g. `"tool0"`, `"camera_link"`).
    pub name: String,
    /// Index of the parent joint (1-based, 0 = universe).
    pub parent_joint: usize,
    /// Fixed placement of this frame relative to the parent joint frame.
    pub placement: SE3<T>,
}

/// Collection of operational frames for a robot model.
#[derive(Debug, Clone)]
pub struct FrameModel<T: RealField> {
    pub frames: Vec<Frame<T>>,
}

impl<T: RealField> FrameModel<T> {
    /// Create an empty frame model.
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Add a frame.
    pub fn add_frame(
        &mut self,
        name: impl Into<String>,
        parent_joint: usize,
        placement: SE3<T>,
    ) {
        self.frames.push(Frame {
            name: name.into(),
            parent_joint,
            placement,
        });
    }

    /// Find a frame by name.
    pub fn find(&self, name: &str) -> Option<&Frame<T>> {
        self.frames.iter().find(|f| f.name == name)
    }

    /// Find the index of a frame by name.
    pub fn find_index(&self, name: &str) -> Option<usize> {
        self.frames.iter().position(|f| f.name == name)
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether there are no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

impl<T: RealField> Default for FrameModel<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Placement computation ──────────────────────────────────────────────────

/// Compute the world placement of an operational frame.
///
/// `oMf = oMi[parent_joint] * frame.placement`
///
/// **Pure function**: runs FK internally.
pub fn compute_frame_placement<T: RealField>(
    model: &Model<T>,
    q: &[T],
    frame: &Frame<T>,
) -> SE3<T> {
    let data = forward_kinematics(model, q);
    compute_frame_placement_from_data(&data, frame)
}

/// Same as [`compute_frame_placement`] but takes pre-computed FK data.
pub fn compute_frame_placement_from_data<T: RealField>(
    data: &Data<T>,
    frame: &Frame<T>,
) -> SE3<T> {
    se3::compose(&data.oMi[frame.parent_joint], &frame.placement)
}

/// Compute world placements of all frames in a [`FrameModel`].
///
/// Returns a vector of SE3 placements, one per frame (same order as `frame_model.frames`).
pub fn compute_all_frame_placements<T: RealField>(
    model: &Model<T>,
    q: &[T],
    frame_model: &FrameModel<T>,
) -> Vec<SE3<T>> {
    let data = forward_kinematics(model, q);
    frame_model
        .frames
        .iter()
        .map(|f| compute_frame_placement_from_data(&data, f))
        .collect()
}

// ─── Jacobian computation ───────────────────────────────────────────────────

/// Compute the geometric Jacobian at an operational frame, expressed in the world frame.
///
/// This is the Jacobian that maps q̇ to the spatial velocity of the frame.
/// It differs from the joint Jacobian only in the linear part (different lever arm).
///
/// **Pure function**: runs FK internally.
pub fn compute_frame_jacobian<T: RealField>(
    model: &Model<T>,
    q: &[T],
    frame: &Frame<T>,
) -> DMatrix<T> {
    let data = forward_kinematics(model, q);
    compute_frame_jacobian_from_data(model, q, &data, frame)
}

/// Same as [`compute_frame_jacobian`] but takes pre-computed FK data.
pub fn compute_frame_jacobian_from_data<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    frame: &Frame<T>,
) -> DMatrix<T> {
    let mut jac = DMatrix::zeros(6, model.nv);

    // Target position = world position of the frame
    let o_m_f = se3::compose(&data.oMi[frame.parent_joint], &frame.placement);
    let target_pos = se3::translation(&o_m_f);

    // Walk from parent_joint to root, writing columns
    let mut current = frame.parent_joint;
    while current > 0 {
        write_joint_column(model, q, data, current, &target_pos, &mut jac);
        current = model.joints[current].parent;
    }

    jac
}

/// Write the Jacobian columns for a single joint.
fn write_joint_column<T: RealField>(
    model: &Model<T>,
    q: &[T],
    data: &Data<T>,
    joint_idx: usize,
    target_pos: &Vector3<T>,
    jac: &mut DMatrix<T>,
) {
    let joint = &model.joints[joint_idx];
    let vi = model.v_idx[joint_idx];
    let nv = joint.joint_type.nv();
    if nv == 0 {
        return;
    }

    let qi = model.q_idx[joint_idx];
    let q_slice = &q[qi..qi + joint.joint_type.nq()];
    let s_local = joint.joint_type.motion_subspace(q_slice);
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

        jac[(0, vi + col)] = w[0].clone();
        jac[(1, vi + col)] = w[1].clone();
        jac[(2, vi + col)] = w[2].clone();
        jac[(3, vi + col)] = v_at_target[0].clone();
        jac[(4, vi + col)] = v_at_target[1].clone();
        jac[(5, vi + col)] = v_at_target[2].clone();
    }
}

// ─── URDF / SDF frame extraction ───────────────────────────────────────────

/// Build a [`FrameModel`] containing one frame per link in the model.
///
/// Each frame is placed at the link's origin (= the child joint frame).
/// Frame names match the link names from `model.link_names`.
///
/// For the root link (index 0), the frame is at the world origin.
pub fn frames_from_links<T: RealField>(model: &Model<T>) -> FrameModel<T> {
    let mut fm = FrameModel::new();
    for (i, name) in model.link_names.iter().enumerate() {
        let parent_joint = if i == 0 { 0 } else { i };
        fm.add_frame(name.clone(), parent_joint, se3::identity());
    }
    fm
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fk::forward_kinematics;
    use crate::jacobian;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use approx::assert_relative_eq;
    use nalgebra::{Rotation3, Vector3};

    fn two_link_arm() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
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
            .add_joint("j2", 1, joint::revolute_z(), offset, LinkInertia::zero())
            .build()
    }

    #[test]
    fn frame_at_joint_matches_fk() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let data = forward_kinematics(&model, &q);

        // A frame with identity offset on joint 2 should match oMi[2]
        let frame = Frame {
            name: "j2_frame".into(),
            parent_joint: 2,
            placement: se3::identity(),
        };

        let placement = compute_frame_placement_from_data(&data, &frame);
        assert_relative_eq!(
            se3::to_homogeneous(&placement),
            se3::to_homogeneous(&data.oMi[2]),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn frame_with_offset() {
        let model = two_link_arm();
        let q = vec![0.0, 0.0];

        // Frame at joint 2 with 0.5m offset along X
        let offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.5, 0.0, 0.0),
        );
        let frame = Frame {
            name: "tool".into(),
            parent_joint: 2,
            placement: offset,
        };

        let placement = compute_frame_placement(&model, &q, &frame);
        // Joint 2 is at (1, 0, 0), tool is 0.5 further → (1.5, 0, 0)
        assert_relative_eq!(
            se3::translation(&placement),
            Vector3::new(1.5, 0.0, 0.0),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn frame_jacobian_at_joint_matches_joint_jacobian() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];

        // Frame at joint 2 with no offset → same as joint Jacobian
        let frame = Frame {
            name: "j2_frame".into(),
            parent_joint: 2,
            placement: se3::identity(),
        };

        let j_frame = compute_frame_jacobian(&model, &q, &frame);
        let j_joint = jacobian::compute_joint_jacobian(&model, &q, 2);

        assert_relative_eq!(j_frame, j_joint, epsilon = 1e-12);
    }

    #[test]
    fn frame_jacobian_with_offset_numerical() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];

        let tool_offset = se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.5, 0.0, 0.0),
        );
        let frame = Frame {
            name: "tool".into(),
            parent_joint: 2,
            placement: tool_offset,
        };

        let j_frame = compute_frame_jacobian(&model, &q, &frame);

        // Validate via finite difference
        let eps = 1e-8;
        for col in 0..model.nv {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;

            let p_plus = compute_frame_placement(&model, &q_plus, &frame);
            let p_minus = compute_frame_placement(&model, &q_minus, &frame);

            let t_plus = se3::translation(&p_plus);
            let t_minus = se3::translation(&p_minus);
            let v_num = (t_plus - t_minus) / (2.0 * eps);

            // Linear part (rows 3-5)
            for r in 0..3 {
                assert_relative_eq!(j_frame[(3 + r, col)], v_num[r], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn frames_from_links_count() {
        let model = two_link_arm();
        let fm = frames_from_links(&model);
        // base_link + link_1 + link_2 = 3 frames
        assert_eq!(fm.len(), 3);
    }

    #[test]
    fn frame_model_find() {
        let mut fm = FrameModel::<f64>::new();
        fm.add_frame("tool0", 2, se3::identity());
        fm.add_frame("camera", 1, se3::identity());

        assert!(fm.find("tool0").is_some());
        assert_eq!(fm.find_index("camera"), Some(1));
        assert!(fm.find("nonexistent").is_none());
    }

    #[test]
    fn compute_all_frame_placements_matches_individual() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];

        let mut fm = FrameModel::new();
        fm.add_frame("a", 1, se3::identity());
        fm.add_frame(
            "b",
            2,
            se3::from_rotation_and_translation(
                &Rotation3::identity(),
                &Vector3::new(0.5, 0.0, 0.0),
            ),
        );

        let all = compute_all_frame_placements(&model, &q, &fm);

        for (i, frame) in fm.frames.iter().enumerate() {
            let individual = compute_frame_placement(&model, &q, frame);
            assert_relative_eq!(
                se3::to_homogeneous(&all[i]),
                se3::to_homogeneous(&individual),
                epsilon = 1e-12,
            );
        }
    }
}
