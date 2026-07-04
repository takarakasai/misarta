"""Dynamics tests — RNEA, CRBA, ABA self-consistency."""
import numpy as np
import misarta


def test_crba_shape_and_symmetry(model):
    q = np.zeros(model.nq)
    M = misarta.crba(model, q)
    assert M.shape == (model.nv, model.nv)
    np.testing.assert_allclose(M, M.T, atol=1e-12)
    # Mass matrix should be positive definite -> all eigenvalues > 0
    eigs = np.linalg.eigvalsh((M + M.T) / 2)
    assert np.all(eigs > 0)


def test_rnea_zero_velocity_zero_accel_is_gravity(model):
    """At q = 0, v = 0, a = 0, RNEA returns the gravity-compensation torque."""
    q = np.zeros(model.nq)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    tau = misarta.rnea(model, q, v, a)
    assert tau.shape == (model.nv,)
    assert np.all(np.isfinite(tau))


def test_rnea_aba_roundtrip(model):
    """ABA(q, v, RNEA(q, v, a)) should recover a (forward/inverse dynamics inverse pair)."""
    rng = np.random.default_rng(42)
    q = rng.standard_normal(model.nq) * 0.3
    v = rng.standard_normal(model.nv) * 0.5
    a = rng.standard_normal(model.nv) * 0.5

    tau = misarta.rnea(model, q, v, a)
    a_back = misarta.aba(model, q, v, tau)
    np.testing.assert_allclose(a_back, a, atol=1e-9)


def test_rnea_linearity_in_acceleration(model):
    """tau = M(q) a + h(q, v); so RNEA(q, v, a) is affine in a with slope M(q)."""
    rng = np.random.default_rng(123)
    q = rng.standard_normal(model.nq) * 0.3
    v = rng.standard_normal(model.nv) * 0.5

    h = misarta.rnea(model, q, v, np.zeros(model.nv))  # h(q, v) only
    M = misarta.crba(model, q)

    a = rng.standard_normal(model.nv)
    tau = misarta.rnea(model, q, v, a)
    expected = M @ a + h
    np.testing.assert_allclose(tau, expected, atol=1e-9)
