"""Forward kinematics + Jacobian tests."""
import numpy as np
import misarta


def test_model_loaded(model):
    assert model.name == "test_robot"
    # test_robot.urdf has 2 revolute joints + 1 fixed
    assert model.nq >= 2
    assert model.nv >= 2
    assert model.njoints >= 2
    assert "joint1" in model.joint_names()
    assert "joint2" in model.joint_names()


def test_fk_identity_at_zero(model):
    q = np.zeros(model.nq)
    data = misarta.forward_kinematics(model, q)
    # Universe (index 0) is identity.
    T0 = data.oMi(0)
    np.testing.assert_allclose(T0.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T0.translation, np.zeros(3), atol=1e-12)


def test_fk_first_joint_homogeneous_is_finite(model):
    q = np.zeros(model.nq)
    data = misarta.forward_kinematics(model, q)
    T = data.oMi(1)
    H = T.homogeneous()
    assert H.shape == (4, 4)
    assert np.all(np.isfinite(H))
    # Last row of homogeneous is [0, 0, 0, 1]
    np.testing.assert_allclose(H[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_fk_changes_with_q(model):
    q0 = np.zeros(model.nq)
    q1 = np.full(model.nq, 0.5)
    d0 = misarta.forward_kinematics(model, q0)
    d1 = misarta.forward_kinematics(model, q1)
    # At least one joint placement should differ between configurations
    diff = 0.0
    for i in range(1, model.n_total):
        diff += np.linalg.norm(d0.oMi(i).homogeneous() - d1.oMi(i).homogeneous())
    assert diff > 1e-6


def test_jacobian_shape(model):
    q = np.zeros(model.nq)
    j_id = model.joint_id("joint2")
    assert j_id is not None
    J = misarta.compute_joint_jacobian(model, q, j_id, ref_frame=misarta.WORLD)
    assert J.shape == (6, model.nv)
    assert np.all(np.isfinite(J))


def test_jacobian_q_length_validation(model):
    bad_q = np.zeros(model.nq + 1)
    j_id = model.joint_id("joint2")
    try:
        misarta.compute_joint_jacobian(model, bad_q, j_id)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong-length q")
