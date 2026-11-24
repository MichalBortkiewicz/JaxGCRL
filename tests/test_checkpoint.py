import os

import jax
import jax.numpy as jnp
import numpy as np

from jaxgcrl.agents.common.train_utils import Checkpointer


def test_roundtrip(tmp_path):
    tree = ({"a": jnp.arange(6.0).reshape(2, 3), "b": jnp.ones((1,), jnp.int32)}, jax.random.PRNGKey(3))
    ck = Checkpointer(str(tmp_path), interval_min=0.0, enabled=True)
    assert not ck.exists()
    ck.save(tree, {"epochs_done": 2, "env_steps": 10, "training_walltime": 1.5})
    assert ck.exists() and not os.path.exists(str(tmp_path / "latest.pkl.tmp"))
    template = jax.tree_util.tree_map(jnp.zeros_like, tree)
    restored, meta = ck.load(template)
    assert meta["epochs_done"] == 2
    for a, b in zip(jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    ck.finalize(tree[0], {"env_steps": 10})
    assert not ck.exists() and os.path.exists(str(tmp_path / "final_state.pkl"))


def test_disabled_does_nothing(tmp_path):
    ck = Checkpointer(str(tmp_path), enabled=False)
    ck.save({"a": jnp.ones(2)}, {})
    assert not os.path.exists(str(tmp_path / "latest.pkl"))
