//! Structural editing of a [`MisaFile`] with schema-invariant maintenance.
//!
//! The `.misa` schema cross-references entities by name: joints name their
//! parent / child links, sensors sit on links, collision pairs and loop
//! closures pair links, mimics and actuators name joints, poses and the
//! home entry key angles by joint name, and gait presets name foot links.
//! Renaming or removing an entity therefore has to update every
//! referencing table. That invariant belongs to the schema owner, so it
//! lives here rather than in each consumer (editors historically updated
//! only a subset and let the rest go stale).

use std::fmt;

use super::schema::{Joint, Link, MisaFile};

/// Why an edit was rejected. The file is left untouched on error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditError {
    UnknownLink(String),
    UnknownJoint(String),
    /// The requested new name is already taken.
    NameCollision(String),
    /// Empty / whitespace-only name.
    InvalidName(String),
    /// The root link cannot be removed.
    RootRemoval,
    /// `add_joint` would give a link a second parent (breaks the tree).
    SecondParent { child: String, existing_joint: String },
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::UnknownLink(n) => write!(f, "unknown link '{n}'"),
            EditError::UnknownJoint(n) => write!(f, "unknown joint '{n}'"),
            EditError::NameCollision(n) => write!(f, "name '{n}' is already in use"),
            EditError::InvalidName(n) => write!(f, "invalid name '{n}'"),
            EditError::RootRemoval => write!(f, "the root link cannot be removed"),
            EditError::SecondParent { child, existing_joint } => write!(
                f,
                "link '{child}' already has a parent joint ('{existing_joint}')"
            ),
        }
    }
}

impl std::error::Error for EditError {}

fn validated_new_name(new: &str) -> Result<&str, EditError> {
    let trimmed = new.trim();
    if trimmed.is_empty() {
        return Err(EditError::InvalidName(new.to_string()));
    }
    Ok(trimmed)
}

// ─── Generic core ───────────────────────────────────────────────────────────

/// Access to the name-reference tables of an editable robot description.
///
/// The `.misa` schema and any in-memory editor model that mirrors it (e.g.
/// articara's `RobotModel`) share the same cross-reference structure; this
/// trait exposes just enough of it for the edit operations to run
/// generically, so the reference-fixup invariants have exactly one
/// implementation ([`rename_link_in`], [`rename_joint_in`],
/// [`remove_link_in`]).
///
/// Contract notes:
/// - `visit_link_name_slots` must enumerate **every** slot holding a link
///   name, including each link's own `name` field and the root-link field.
/// - `visit_joint_name_slots` likewise, including each joint's own `name`.
/// - Implementations with derived indices (name → idx maps) rebuild them
///   after the edit call returns; the core only touches the slots.
pub trait EditTables {
    fn root_link(&self) -> String;
    fn has_link(&self, name: &str) -> bool;
    fn has_joint(&self, name: &str) -> bool;
    /// `(joint_name, parent_link, child_link)` for every joint.
    fn joints_topology(&self) -> Vec<(String, String, String)>;
    fn visit_link_name_slots(&mut self, f: &mut dyn FnMut(&mut String));
    fn visit_joint_name_slots(&mut self, f: &mut dyn FnMut(&mut String));
    /// Rekey joint-name-keyed maps (pose angles, home positions, ...).
    fn rekey_joint_maps(&mut self, old: &str, new: &str);
    /// Drop the named link / joint entities themselves.
    fn remove_link_entities(&mut self, names: &[String]);
    fn remove_joint_entities(&mut self, names: &[String]);
    /// Drop rows that reference a link failing `keep` (sensor mounts,
    /// collision pairs, loop closures). Gait presets are deliberately
    /// exempt — see [`remove_link_in`].
    fn retain_rows_by_link(&mut self, keep: &dyn Fn(&str) -> bool);
    /// Drop joint references failing `keep`: mimic rows, actuator joint
    /// refs (dropping actuators left empty), pose / home angle entries.
    fn retain_rows_by_joint(&mut self, keep: &dyn Fn(&str) -> bool);
}

/// Generic [`rename_link`] over any [`EditTables`] implementation.
pub fn rename_link_in(t: &mut dyn EditTables, old: &str, new: &str) -> Result<(), EditError> {
    let new = validated_new_name(new)?;
    if new == old {
        return Ok(());
    }
    if !t.has_link(old) {
        return Err(EditError::UnknownLink(old.to_string()));
    }
    if t.has_link(new) {
        return Err(EditError::NameCollision(new.to_string()));
    }
    t.visit_link_name_slots(&mut |slot| {
        if slot == old {
            *slot = new.to_string();
        }
    });
    Ok(())
}

/// Generic [`rename_joint`] over any [`EditTables`] implementation.
pub fn rename_joint_in(t: &mut dyn EditTables, old: &str, new: &str) -> Result<(), EditError> {
    let new = validated_new_name(new)?;
    if new == old {
        return Ok(());
    }
    if !t.has_joint(old) {
        return Err(EditError::UnknownJoint(old.to_string()));
    }
    if t.has_joint(new) {
        return Err(EditError::NameCollision(new.to_string()));
    }
    t.visit_joint_name_slots(&mut |slot| {
        if slot == old {
            *slot = new.to_string();
        }
    });
    t.rekey_joint_maps(old, new);
    Ok(())
}

/// Generic [`remove_link`] over any [`EditTables`] implementation.
pub fn remove_link_in(t: &mut dyn EditTables, name: &str) -> Result<Vec<String>, EditError> {
    if t.root_link() == name {
        return Err(EditError::RootRemoval);
    }
    if !t.has_link(name) {
        return Err(EditError::UnknownLink(name.to_string()));
    }

    let topo = t.joints_topology();

    // Collect the subtree: `name` plus everything reachable through joints.
    let mut removed_links: Vec<String> = vec![name.to_string()];
    let mut frontier = vec![name.to_string()];
    while let Some(parent) = frontier.pop() {
        for (_, p, c) in &topo {
            if *p == parent && !removed_links.contains(c) {
                removed_links.push(c.clone());
                frontier.push(c.clone());
            }
        }
    }
    let is_removed = |n: &str| removed_links.iter().any(|r| r == n);

    let removed_joints: Vec<String> = topo
        .iter()
        .filter(|(_, p, c)| is_removed(p) || is_removed(c))
        .map(|(n, _, _)| n.clone())
        .collect();
    let joint_removed = |n: &str| removed_joints.iter().any(|r| r == n);

    t.remove_joint_entities(&removed_joints);
    t.remove_link_entities(&removed_links);
    t.retain_rows_by_link(&|n| !is_removed(n));
    t.retain_rows_by_joint(&|n| !joint_removed(n));

    Ok(removed_links)
}

// ─── EditTables for MisaFile ────────────────────────────────────────────────

impl EditTables for MisaFile {
    fn root_link(&self) -> String {
        self.robot.root.clone()
    }

    fn has_link(&self, name: &str) -> bool {
        self.link.iter().any(|l| l.name == name)
    }

    fn has_joint(&self, name: &str) -> bool {
        self.joint.iter().any(|j| j.name == name)
    }

    fn joints_topology(&self) -> Vec<(String, String, String)> {
        self.joint
            .iter()
            .map(|j| (j.name.clone(), j.parent.clone(), j.child.clone()))
            .collect()
    }

    fn visit_link_name_slots(&mut self, f: &mut dyn FnMut(&mut String)) {
        for l in &mut self.link {
            f(&mut l.name);
        }
        f(&mut self.robot.root);
        for j in &mut self.joint {
            f(&mut j.parent);
            f(&mut j.child);
        }
        for s in &mut self.sensor {
            f(&mut s.link);
        }
        for cp in &mut self.collision_pair {
            f(&mut cp.link_a);
            f(&mut cp.link_b);
        }
        for lc in &mut self.loop_closure {
            f(&mut lc.link_a);
            f(&mut lc.link_b);
        }
        for g in &mut self.gait {
            f(&mut g.fl_foot);
            f(&mut g.fr_foot);
            f(&mut g.rl_foot);
            f(&mut g.rr_foot);
        }
    }

    fn visit_joint_name_slots(&mut self, f: &mut dyn FnMut(&mut String)) {
        for j in &mut self.joint {
            f(&mut j.name);
        }
        for m in &mut self.mimic {
            f(&mut m.joint);
            f(&mut m.source);
        }
        for a in &mut self.actuator {
            for jr in &mut a.joints {
                f(&mut jr.name);
            }
        }
    }

    fn rekey_joint_maps(&mut self, old: &str, new: &str) {
        for p in &mut self.pose {
            if let Some(v) = p.angles.remove(old) {
                p.angles.insert(new.to_string(), v);
            }
        }
        if let Some(v) = self.home.joint_positions.remove(old) {
            self.home.joint_positions.insert(new.to_string(), v);
        }
    }

    fn remove_link_entities(&mut self, names: &[String]) {
        self.link.retain(|l| !names.contains(&l.name));
    }

    fn remove_joint_entities(&mut self, names: &[String]) {
        self.joint.retain(|j| !names.contains(&j.name));
    }

    fn retain_rows_by_link(&mut self, keep: &dyn Fn(&str) -> bool) {
        self.sensor.retain(|s| keep(&s.link));
        self.collision_pair
            .retain(|cp| keep(&cp.link_a) && keep(&cp.link_b));
        self.loop_closure
            .retain(|lc| keep(&lc.link_a) && keep(&lc.link_b));
    }

    fn retain_rows_by_joint(&mut self, keep: &dyn Fn(&str) -> bool) {
        self.mimic.retain(|m| keep(&m.joint) && keep(&m.source));
        for a in &mut self.actuator {
            a.joints.retain(|jr| keep(&jr.name));
        }
        self.actuator.retain(|a| !a.joints.is_empty());
        for p in &mut self.pose {
            p.angles.retain(|k, _| keep(k));
        }
        self.home.joint_positions.retain(|k, _| keep(k));
    }
}

/// Rename a link in a [`MisaFile`], updating every reference: `robot.root`,
/// `joint.parent` / `joint.child`, `sensor.link`,
/// `collision_pair.link_a` / `link_b`, `loop_closure.link_a` / `link_b`
/// and the gait presets' four foot-link names.
///
/// Renaming to the current name is a no-op `Ok`. Thin wrapper over
/// [`rename_link_in`], which any [`EditTables`] implementation can use.
pub fn rename_link(f: &mut MisaFile, old: &str, new: &str) -> Result<(), EditError> {
    rename_link_in(f, old, new)
}

/// Rename a joint in a [`MisaFile`], updating every reference:
/// `mimic.joint` / `mimic.source`, actuator joint refs, and the
/// joint-name keys of every pose's `angles` map and of
/// `home.joint_positions`.
///
/// Renaming to the current name is a no-op `Ok`. Thin wrapper over
/// [`rename_joint_in`].
pub fn rename_joint(f: &mut MisaFile, old: &str, new: &str) -> Result<(), EditError> {
    rename_joint_in(f, old, new)
}

/// Remove a link and its entire subtree from a [`MisaFile`]. Returns the
/// removed link names (the requested link first).
///
/// Every reference to a removed link or joint is cleaned up: sensors on
/// removed links, collision pairs / loop closures touching them, mimics
/// whose follower or source joint disappeared, actuator joint refs (an
/// actuator left with no joints is dropped), and pose / home angle
/// entries of removed joints. Gait presets are left untouched — they are
/// user-authored presets whose foot fields fall back to schema defaults,
/// and silently deleting a preset would lose data the user may want to
/// re-point instead. Thin wrapper over [`remove_link_in`].
pub fn remove_link(f: &mut MisaFile, name: &str) -> Result<Vec<String>, EditError> {
    remove_link_in(f, name)
}

/// Add a link. Fails on a name collision.
pub fn add_link(f: &mut MisaFile, link: Link) -> Result<usize, EditError> {
    validated_new_name(&link.name)?;
    if f.link.iter().any(|l| l.name == link.name) {
        return Err(EditError::NameCollision(link.name));
    }
    f.link.push(link);
    Ok(f.link.len() - 1)
}

/// Add a joint. Fails on a name collision, on unknown parent / child
/// links, and when the child already has a parent joint (the kinematic
/// structure must stay a tree; closed loops are expressed via
/// `loop_closure` entries instead).
pub fn add_joint(f: &mut MisaFile, joint: Joint) -> Result<usize, EditError> {
    validated_new_name(&joint.name)?;
    if f.joint.iter().any(|j| j.name == joint.name) {
        return Err(EditError::NameCollision(joint.name));
    }
    for link in [&joint.parent, &joint.child] {
        if !f.link.iter().any(|l| &l.name == link) {
            return Err(EditError::UnknownLink(link.clone()));
        }
    }
    if let Some(existing) = f.joint.iter().find(|j| j.child == joint.child) {
        return Err(EditError::SecondParent {
            child: joint.child,
            existing_joint: existing.name.clone(),
        });
    }
    f.joint.push(joint);
    Ok(f.joint.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{parse_str, NullSource};

    /// Small robot exercising every cross-reference table.
    const FIXTURE: &str = r#"
schema = "misarta/1"

[robot]
name = "r"
root = "trunk"

[[link]]
name = "trunk"
inertial = { mass = 1.0, ixx = 0.1, iyy = 0.1, izz = 0.1 }

[[link]]
name = "thigh"
inertial = { mass = 0.5, ixx = 0.01, iyy = 0.01, izz = 0.01 }

[[link]]
name = "calf"
inertial = { mass = 0.3, ixx = 0.01, iyy = 0.01, izz = 0.01 }

[[joint]]
name = "hip"
type = "revolute"
parent = "trunk"
child = "thigh"
axis = [0, 1, 0]

[[joint]]
name = "knee"
type = "revolute"
parent = "thigh"
child = "calf"
axis = [0, 1, 0]

[[mimic]]
joint = "knee"
source = "hip"
multiplier = -1.0

[[collision_pair]]
link_a = "trunk"
link_b = "calf"
enabled = false

[[sensor]]
name = "imu"
link = "trunk"
kind = { imu = {} }

[[actuator]]
name = "hip_motor"
mode = "Position"
joints = [{ name = "hip", gear = 1.0 }]

[[actuator]]
name = "knee_motor"
mode = "Position"
joints = [{ name = "knee", gear = 1.0 }]

[[pose]]
name = "stand"
angles = { hip = 0.3, knee = -0.6 }

[[gait]]
name = "crawl"
gait_type = "Crawl"
fl_foot = "calf"
fr_foot = "calf"
rl_foot = "calf"
rr_foot = "calf"

[home]
joint_positions = { hip = 0.1, knee = -0.2 }
"#;

    fn fixture() -> MisaFile {
        parse_str(FIXTURE, &NullSource).expect("fixture parses").file
    }

    #[test]
    fn rename_link_updates_every_reference() {
        let mut f = fixture();
        rename_link(&mut f, "calf", "shin").unwrap();

        assert!(f.link.iter().any(|l| l.name == "shin"));
        assert!(f.joint.iter().any(|j| j.name == "knee" && j.child == "shin"));
        assert_eq!(f.collision_pair[0].link_b, "shin");
        assert!(f.gait[0].fl_foot == "shin" && f.gait[0].rr_foot == "shin");

        // Root rename updates robot.root and the sensor mount.
        rename_link(&mut f, "trunk", "body").unwrap();
        assert_eq!(f.robot.root, "body");
        assert_eq!(f.sensor[0].link, "body");
        assert_eq!(f.collision_pair[0].link_a, "body");
    }

    #[test]
    fn rename_link_rejects_collision_and_unknown() {
        let mut f = fixture();
        assert_eq!(
            rename_link(&mut f, "calf", "thigh"),
            Err(EditError::NameCollision("thigh".into()))
        );
        assert_eq!(
            rename_link(&mut f, "nope", "x"),
            Err(EditError::UnknownLink("nope".into()))
        );
        assert!(matches!(
            rename_link(&mut f, "calf", "  "),
            Err(EditError::InvalidName(_))
        ));
        // No-op rename succeeds.
        rename_link(&mut f, "calf", "calf").unwrap();
    }

    #[test]
    fn rename_joint_updates_mimic_actuator_pose_home() {
        let mut f = fixture();
        rename_joint(&mut f, "hip", "hip_pitch").unwrap();

        assert!(f.joint.iter().any(|j| j.name == "hip_pitch"));
        assert_eq!(f.mimic[0].source, "hip_pitch");
        assert_eq!(f.actuator[0].joints[0].name, "hip_pitch");
        assert!(f.pose[0].angles.contains_key("hip_pitch"));
        assert!(!f.pose[0].angles.contains_key("hip"));
        assert!(f.home.joint_positions.contains_key("hip_pitch"));
    }

    #[test]
    fn remove_link_removes_subtree_and_cleans_references() {
        let mut f = fixture();
        let removed = remove_link(&mut f, "thigh").unwrap();
        assert_eq!(removed, vec!["thigh".to_string(), "calf".to_string()]);

        assert_eq!(f.link.len(), 1);
        assert!(f.joint.is_empty());
        // hip + knee removed → mimic gone, both actuators gone, pose /
        // home entries emptied, collision pair with calf gone.
        assert!(f.mimic.is_empty());
        assert!(f.actuator.is_empty());
        assert!(f.pose[0].angles.is_empty());
        assert!(f.home.joint_positions.is_empty());
        assert!(f.collision_pair.is_empty());
        // Sensor on trunk survives.
        assert_eq!(f.sensor.len(), 1);
    }

    #[test]
    fn remove_link_rejects_root() {
        let mut f = fixture();
        assert_eq!(remove_link(&mut f, "trunk"), Err(EditError::RootRemoval));
    }

    #[test]
    fn add_joint_validates_tree_property() {
        let mut f = fixture();
        let mut j = f.joint[0].clone();
        j.name = "second_hip".into();
        // calf already has `knee` as parent joint.
        j.child = "calf".into();
        assert!(matches!(
            add_joint(&mut f, j),
            Err(EditError::SecondParent { .. })
        ));
    }
}
