import jax
import jax.numpy as jnp
import numpy as np
import pytest

from loop_transformers.model import LoopedTransformer
from loop_transformers.tasks import build_task, sample_batch


@pytest.mark.parametrize("name,order", [("parity", 2), ("s4", 24), ("a5", 60)])
def test_group_axioms(name, order):
    table, identity, got_order = build_task(name)
    table = np.asarray(table)
    assert got_order == order

    # Identity, closure, inverses, associativity.
    assert (table[identity, :] == np.arange(order)).all()
    assert (table[:, identity] == np.arange(order)).all()
    for i in range(order):
        assert identity in table[i, :]
    for _ in range(200):
        a, b, c = np.random.randint(0, order, 3)
        assert table[table[a, b], c] == table[a, table[b, c]]


def test_parity_matches_cumulative_xor():
    table, identity, _ = build_task("parity")
    x, y = sample_batch(jax.random.PRNGKey(86), table, identity, 16, 32)
    expected = np.cumsum(np.asarray(x), axis = -1) % 2
    assert (np.asarray(y) == expected).all()


@pytest.mark.parametrize("name", ["s4", "a5"])
def test_prefix_products_match_reference(name):
    table, identity, order = build_task(name)
    x, y = sample_batch(jax.random.PRNGKey(86), table, identity, 8, 16)
    x, y, table = np.asarray(x), np.asarray(y), np.asarray(table)
    for b in range(x.shape[0]):
        acc = identity
        for i in range(x.shape[1]):
            acc = table[acc, x[b, i]]
            assert y[b, i] == acc


def test_arms_agree_in_forward():
    """The arms differ only in gradient flow, so forward outputs must match."""
    model = LoopedTransformer(vocab = 24, n_classes = 24, scale = "s")
    params = model.init(jax.random.PRNGKey(86), 8)
    x = jax.random.randint(jax.random.PRNGKey(0), (4, 8), 0, 24)
    a = model.apply(params, x, 6, "bptt")
    b = model.apply(params, x, 6, "stopgrad")
    np.testing.assert_allclose(a, b, atol = 1e-4)


def test_stopgrad_equals_single_application_gradient():
    """stopgrad's gradient must equal that of one differentiated application
    placed on top of a frozen T-1 prefix."""
    model = LoopedTransformer(vocab = 24, n_classes = 24, scale = "s")
    params = model.init(jax.random.PRNGKey(86), 8)
    x = jax.random.randint(jax.random.PRNGKey(0), (4, 8), 0, 24)
    T = 5

    def loss(p, arm):
        return model.apply(p, x, T, arm).sum()

    g_stopgrad = jax.grad(loss)(params, "stopgrad")

    # Reference: build h^{T-1} outside the trace, then differentiate exactly one
    # block application plus the head and embedding.
    import flax.linen as nn

    e = model.embed.apply(params["embed"], x)
    mask = nn.make_causal_mask(jnp.ones(x.shape))
    h = jnp.zeros_like(e)
    for _ in range(T - 1):
        h = model.block.apply(params["block"], h, e, mask)
    h_frozen = jax.lax.stop_gradient(h)

    def ref_loss(p):
        e2 = model.embed.apply(p["embed"], x)
        out = model.block.apply(p["block"], h_frozen, e2, mask)
        return model.head.apply(p["head"], out).sum()

    g_ref = jax.grad(ref_loss)(params)
    for a, b in zip(jax.tree.leaves(g_stopgrad), jax.tree.leaves(g_ref)):
        np.testing.assert_allclose(a, b, atol = 1e-4)


def test_bptt_gradient_differs_from_stopgrad():
    """Sanity: the two arms must not produce the same gradient."""
    model = LoopedTransformer(vocab = 24, n_classes = 24, scale = "s")
    params = model.init(jax.random.PRNGKey(86), 8)
    x = jax.random.randint(jax.random.PRNGKey(0), (4, 8), 0, 24)

    def loss(p, arm):
        return model.apply(p, x, 5, arm).sum()

    g_a = jax.grad(loss)(params, "bptt")
    g_b = jax.grad(loss)(params, "stopgrad")
    diffs = [
        np.abs(np.asarray(a) - np.asarray(b)).max()
        for a, b in zip(jax.tree.leaves(g_a), jax.tree.leaves(g_b))
    ]
    assert max(diffs) > 1e-5
