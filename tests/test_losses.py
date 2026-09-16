import jax.numpy as jnp
import numpy as np

from jaxgcrl.agents.crl.losses import energy_fn


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
