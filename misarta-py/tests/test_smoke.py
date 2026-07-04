"""Smoke test — module import + basic attributes."""
import numpy as np
import misarta


def test_version():
    assert isinstance(misarta.__version__, str)
    assert len(misarta.__version__) > 0


def test_constants():
    assert misarta.LOCAL == 0
    assert misarta.WORLD == 1
    assert misarta.LOCAL_WORLD_ALIGNED == 2


def test_se3_identity():
    T = misarta.SE3.identity()
    R = T.rotation
    t = T.translation
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    np.testing.assert_allclose(R, np.eye(3))
    np.testing.assert_allclose(t, np.zeros(3))


def test_se3_compose_and_inverse():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    t = np.array([1.0, 2.0, 3.0])
    T = misarta.SE3(rotation=R, translation=t)
    T_inv = T.inverse()
    I = T * T_inv
    np.testing.assert_allclose(I.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(I.translation, np.zeros(3), atol=1e-12)


def test_joint_type_factory():
    rev = misarta.JointType.revolute(np.array([0.0, 0.0, 1.0]))
    assert rev.kind == "revolute"
    assert rev.nq == 1 and rev.nv == 1

    pris = misarta.JointType.prismatic(np.array([1.0, 0.0, 0.0]))
    assert pris.kind == "prismatic"

    fixed = misarta.JointType.fixed()
    assert fixed.nq == 0 and fixed.nv == 0

    free = misarta.JointType.free_flyer()
    assert free.nq == 7 and free.nv == 6
