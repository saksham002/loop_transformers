"""Weight-tied looped transformer, following the architecture in the paper.

h^{t+1} = F_theta(h^t, e), where F_theta is a stack of pre-norm transformer
layers (4d FFN, GELU) preceded by a learned linear adapter on [h; e] (input
injection). Readout is a linear head on LN(h^T). NoPE; attention is causal,
which is what lets a NoPE model distinguish positions at all.

Params are held as an explicit dict of three sub-module param trees so that the
stopgrad arm can feed detached block params to the un-differentiated prefix
loops while the final application receives the live ones.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

# Scales s and m from the paper (Appendix D).
SCALES = {
    "s": dict(d = 128, heads = 4, layers = 2),
    "m": dict(d = 256, heads = 8, layers = 2),
}


class Block(nn.Module):
    """One application of the weight-tied loop body F_theta."""

    d: int
    heads: int
    layers: int

    @nn.compact
    def __call__(self, h, e, mask):
        x = nn.Dense(self.d, name = "adapter")(jnp.concatenate([h, e], axis = -1))
        for i in range(self.layers):
            y = nn.LayerNorm(name = f"ln_attn_{i}")(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads = self.heads, qkv_features = self.d, name = f"attn_{i}"
            )(y, y, mask = mask)
            x = x + y
            y = nn.LayerNorm(name = f"ln_mlp_{i}")(x)
            y = nn.Dense(4 * self.d, name = f"fc1_{i}")(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d, name = f"fc2_{i}")(y)
            x = x + y
        return x


class Head(nn.Module):
    """Readout: linear head on LN(h^T)."""

    n_classes: int

    @nn.compact
    def __call__(self, h):
        return nn.Dense(self.n_classes, name = "out")(nn.LayerNorm(name = "ln")(h))


class LoopedTransformer:
    """Holds the three sub-modules and drives the loop under a given arm."""

    def __init__(self, vocab, n_classes, scale):
        cfg = SCALES[scale]
        self.d = cfg["d"]
        self.embed = nn.Embed(vocab, self.d)
        self.block = Block(**cfg)
        self.head = Head(n_classes)

    def init(self, key, n):
        k_e, k_b, k_h = jax.random.split(key, 3)
        x = jnp.zeros((1, n), dtype = jnp.int32)
        p_embed = self.embed.init(k_e, x)
        e = self.embed.apply(p_embed, x)
        mask = nn.make_causal_mask(jnp.ones((1, n)))
        p_block = self.block.init(k_b, jnp.zeros_like(e), e, mask)
        h = self.block.apply(p_block, jnp.zeros_like(e), e, mask)
        p_head = self.head.init(k_h, h)
        return {"embed": p_embed, "block": p_block, "head": p_head}

    def apply(self, params, x, T, arm):
        """Run T loops and return terminal logits.

        arm="bptt": gradient flows through all T applications (paper baseline).
        arm="stopgrad": loops 1..T-1 run detached, so the compute graph is O(1)
        in T and only the final application is differentiated.
        """
        e = self.embed.apply(params["embed"], x)
        mask = nn.make_causal_mask(jnp.ones(x.shape))
        h = jnp.zeros_like(e)

        if arm == "stopgrad":
            # Detach the params and the embedding feeding the prefix loops, so
            # the entire prefix is a constant w.r.t. the differentiated params.
            block_sg = jax.lax.stop_gradient(params["block"])
            e_sg = jax.lax.stop_gradient(e)
            h = jax.lax.fori_loop(
                0, T - 1, lambda _, h: self.block.apply(block_sg, h, e_sg, mask), h
            )
            h = self.block.apply(params["block"], jax.lax.stop_gradient(h), e, mask)
        elif arm == "bptt":
            h, _ = jax.lax.scan(
                lambda h, _: (self.block.apply(params["block"], h, e, mask), None),
                h,
                length = T,
            )
        else:
            raise ValueError(f"unknown arm: {arm}")

        return self.head.apply(params["head"], h)
