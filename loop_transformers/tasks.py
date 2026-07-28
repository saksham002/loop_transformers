"""Prefix-product tasks: parity (Z2), S4, and A5 word problems.

Every task is per-position seq2seq: given g_1...g_n, predict p_i = g_1 o ... o g_i
at every position i. This blocks answer-only shortcuts, as in the paper.
"""

from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np


def _compose(a, b):
    """Permutation composition (a o b)(x) = a(b(x))."""
    return tuple(a[b[i]] for i in range(len(b)))


def _sign(p):
    """Sign of a permutation: +1 if even, -1 if odd."""
    n = len(p)
    seen = [False] * n
    s = 1
    for i in range(n):
        if seen[i]:
            continue
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        if length % 2 == 0:
            s = -s
    return s


def _permutation_group(degree, even_only):
    elems = [p for p in permutations(range(degree)) if not even_only or _sign(p) == 1]
    elems.sort()
    index = {p: i for i, p in enumerate(elems)}
    table = np.array(
        [[index[_compose(a, b)] for b in elems] for a in elems], dtype=np.int32
    )
    identity = index[tuple(range(degree))]
    return table, identity


def _z2_group():
    """Parity: the cyclic group Z2, where the group operation is XOR."""
    return np.array([[0, 1], [1, 0]], dtype=np.int32), 0


def _cyclic_group(order):
    """Cyclic group Z_order under addition mod order (abelian)."""
    idx = np.arange(order, dtype = np.int32)
    table = ((idx[:, None] + idx[None, :]) % order).astype(np.int32)
    return table, 0


def build_task(name):
    """Return (cayley_table, identity_index, order) for a named task."""
    if name == "parity":
        table, identity = _z2_group()
    elif name == "z60":
        table, identity = _cyclic_group(60)
    elif name == "s4":
        table, identity = _permutation_group(degree = 4, even_only = False)
    elif name == "a5":
        table, identity = _permutation_group(degree = 5, even_only = True)
    else:
        raise ValueError(f"unknown task: {name}")
    return jnp.asarray(table), identity, table.shape[0]


def sample_batch(key, table, identity, batch, n):
    """Sample uniform group words and their prefix products.

    Returns (inputs, targets), both int32 of shape (batch, n).
    """
    order = table.shape[0]
    g = jax.random.randint(key, (batch, n), 0, order, dtype = jnp.int32)

    def step(carry, g_i):
        carry = table[carry, g_i]
        return carry, carry

    init = jnp.full((batch,), identity, dtype = jnp.int32)
    _, p = jax.lax.scan(step, init, g.T)
    return g, p.T
