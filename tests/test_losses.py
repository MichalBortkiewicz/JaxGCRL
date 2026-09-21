import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgcrl.agents.crl.losses import energy_fn, pairwise_energy_fn


def _reference_cosine(x, y):
    x_n = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    y_n = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-12)
    return x_n @ y_n.T


def test_cosine_energy_is_per_sample_and_bounded():
    rng = np.random.default_rng(0)
    batch, dim = 8, 16
    # deliberately give rows very different scales so a bug that normalizes
    # over the whole batch (rather than per-row) is detectable
    sa_repr = rng.standard_normal((batch, dim)).astype(np.float32) * rng.uniform(
        0.5, 5.0, size=(batch, 1)
    ).astype(np.float32)
    g_repr = rng.standard_normal((batch, dim)).astype(np.float32) * rng.uniform(
        0.5, 5.0, size=(batch, 1)
    ).astype(np.float32)

    logits = energy_fn("cosine", jnp.asarray(sa_repr)[:, None, :], jnp.asarray(g_repr)[None, :, :])
    logits = np.asarray(logits)

    assert logits.shape == (batch, batch)
    assert np.all(logits >= -1.0 - 1e-5) and np.all(logits <= 1.0 + 1e-5)
    np.testing.assert_allclose(logits, _reference_cosine(sa_repr, g_repr), atol=1e-5)


def test_cosine_energy_matches_unbatched_case():
    rng = np.random.default_rng(1)
    batch, dim = 8, 16
    x = rng.standard_normal((batch, dim)).astype(np.float32)
    y = rng.standard_normal((batch, dim)).astype(np.float32)

    per_sample = energy_fn("cosine", jnp.asarray(x), jnp.asarray(y))
    np.testing.assert_allclose(np.asarray(per_sample), np.diag(_reference_cosine(x, y)), atol=1e-5)


@pytest.mark.parametrize("name", ["dot", "cosine", "l2"])
@pytest.mark.parametrize("batch,dim", [(4, 3), (37, 16), (256, 2048)])
def test_pairwise_energy_fn_matches_broadcast_form(name, batch, dim):
    """pairwise_energy_fn computes the (B, B) all-pairs energy matrix from
    B*D-sized matmuls/norms; it should agree with the naive broadcast form
    (energy_fn(name, x[:, None, :], y[None, :, :])), which materializes an
    O(B^2 * D) intermediate to get the same result.
    """
    key = jax.random.PRNGKey(hash((name, batch, dim)) % (2**31))
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (batch, dim))
    y = jax.random.normal(k2, (batch, dim))

    broadcast = energy_fn(name, x[:, None, :], y[None, :, :])
    fast = pairwise_energy_fn(name, x, y)

    assert fast.shape == (batch, batch)
    np.testing.assert_allclose(np.asarray(fast), np.asarray(broadcast), atol=1e-3, rtol=1e-3)


def test_pairwise_energy_fn_norm_is_unchanged():
    """"norm" (sqrt of squared distance) is deliberately left on the original
    broadcast path rather than reformulated the way "l2" is: mixing a
    matmul-computed cross term with elementwise-computed squared norms
    disagrees by ~1e-5 in float32, which sqrt's unbounded derivative near
    zero blows up into a large relative error exactly on near-identical
    x/y -- i.e. exactly on correctly matched positive pairs. So this must be
    bit-identical, not just close, to the broadcast form.
    """
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (64, 128))
    y = jax.random.normal(k2, (64, 128))

    broadcast = energy_fn("norm", x[:, None, :], y[None, :, :])
    fast = pairwise_energy_fn("norm", x, y)
    np.testing.assert_array_equal(np.asarray(fast), np.asarray(broadcast))


@pytest.mark.parametrize("name", ["dot", "cosine", "l2", "norm"])
def test_pairwise_energy_fn_no_nan_on_identical_rows(name):
    """x == y (zero distance) is the case most likely to break a pairwise
    squared-distance reformulation via floating-point cancellation; make
    sure it never produces NaN/Inf, and matches the broadcast form.
    """
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (32, 64))

    fast = pairwise_energy_fn(name, x, x)
    broadcast = energy_fn(name, x[:, None, :], x[None, :, :])

    assert np.all(np.isfinite(np.asarray(fast)))
    np.testing.assert_allclose(np.asarray(fast), np.asarray(broadcast), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("name", ["dot", "cosine", "l2"])
def test_pairwise_energy_fn_gradient_matches_broadcast_form(name):
    """Check gradients (not just values) agree, through the actual downstream
    InfoNCE loss -- a squared-sum-of-logits loss blows up float32 dynamic
    range unrealistically for l2/norm-style energies and isn't representative
    of how these logits are actually used.
    """

    def fwd_infonce(logits):
        return -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (128, 256))
    y = jax.random.normal(k2, (128, 256))

    def loss_broadcast(x, y):
        return fwd_infonce(energy_fn(name, x[:, None, :], y[None, :, :]))

    def loss_fast(x, y):
        return fwd_infonce(pairwise_energy_fn(name, x, y))

    grad_broadcast = jax.grad(loss_broadcast, argnums=(0, 1))(x, y)
    grad_fast = jax.grad(loss_fast, argnums=(0, 1))(x, y)
    for g_b, g_f in zip(grad_broadcast, grad_fast):
        np.testing.assert_allclose(np.asarray(g_f), np.asarray(g_b), atol=1e-3, rtol=1e-3)
