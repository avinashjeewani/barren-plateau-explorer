import numpy as np
import pytest

from bpx.circuits import COSTS, gradient_first_param, make_cost, sample_params
from bpx.experiments import measure, resolve_layers


@pytest.mark.parametrize("cost", COSTS)
def test_parameter_shift_matches_finite_difference(cost):
    rng = np.random.default_rng(0)
    C = make_cost(3, cost)
    p = sample_params(rng, 6, 3, 3, "uniform")
    g = gradient_first_param(C, p)
    eps = 1e-6
    q = p.copy()
    q[:, 0, 0, 0] += eps
    assert np.allclose(g, (C(q) - C(p)) / eps, atol=1e-5)


@pytest.mark.parametrize("cost", COSTS)
def test_identity_blocks_are_identity_at_init(cost):
    """With identity-block init the circuit does nothing, so C equals its value on |+>^n."""
    rng = np.random.default_rng(1)
    n = 4
    C_id = make_cost(n, cost, identity_block=True)
    p = sample_params(rng, 5, 6, n, "identity_block")
    expected = 0.5 if cost == "local" else 1 - 2.0 ** -n
    assert np.allclose(C_id(p), expected)


def test_identity_block_gradient_is_one_over_2n():
    """Analytic check: on |+>^n the first RY changes <Z_0> at rate -1, so dC/dtheta = 1/(2n)."""
    for n in (2, 5):
        r = measure(n, 4, "local", "identity_block", n_samples=10, seed=0)
        assert np.isclose(r["grad_sq_mean"], 1 / (2 * n) ** 2)


def test_global_cost_vanishes_faster_than_local():
    small, large = 2, 7
    g = {c: [measure(n, 2, c, "uniform", 200, seed=n)["grad_sq_mean"] for n in (small, large)]
         for c in COSTS}
    drop = {c: g[c][0] / g[c][1] for c in COSTS}
    assert drop["global"] > 3 * drop["local"]


def test_resolve_layers():
    assert resolve_layers(3, 5) == 3
    assert resolve_layers("n", 5) == 5
    assert resolve_layers("2n", 5) == 10
