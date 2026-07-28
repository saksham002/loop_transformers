# Experiments

**Question.** Can the loss be applied with a `stop_gradient` at every forward call
except the final one — supervising only the terminal output, as the paper does —
so that the loop count `T` can scale without the O(T) memory of BPTT?

```python
h = h0
for t in range(T - 1):
    h = F(h, e)                 # no gradient path
h = F(stop_grad(h), e)          # only this application is differentiated
loss = CE(head(LN(h)), y)       # terminal only, as in the paper
```

## Design

Two things are being measured, and they are kept separate.

**1. Replication of the paper's method** — Z₆₀, S₄, A₅, 4 seeds each.
Reproduces Appendix D with one intentional deviation, noted below.

**2. Our 2×2 ablation** — Z₆₀ only, 4 seeds per cell.

| Factor | Levels |
|---|---|
| Gradient | `bptt` (backprop through all T) vs `stopgrad` (detach at h^{T-1}) |
| Schedule | `fixed` (n = T = 32 throughout) vs `curr` (n = 1,2,4,8,16,32 × 10k) |

Z₆₀ carries 5 cells (4 ablation + 1 replication); S₄ and A₅ carry the
replication cell only. 7 configurations × 4 seeds = **28 runs**.

## The `paper` arm, exactly as specified

From Appendix D, verbatim: *"Length curriculum with stages n_max ∈ {4,8,16,32}
(6k–12k steps each), promotion at per-token accuracy ≥ 0.98; lengths sampled
uniformly within a stage."* Loop schedule is `prop` with multiplier m = 1, i.e.
**T = n** — the setting behind the paper's headline A₅ result.

- Stages `n_max ∈ {4, 8, 16, 32}`, **starting at 4** (not 1)
- Length **sampled uniformly** within a stage, not held fixed
- Promotion on per-token accuracy **≥ 0.98**, checked every 500 steps
- The 6k–12k range is read as the observed spread of an accuracy-gated stage, so
  a stage ends on promotion or at a **12k cap**, whichever comes first
- **Horizon curriculum** — *"used for chained A₅/S₅"* — per-position loss
  truncated to the first `h` positions, `h` ramping 4 → `n_max` within each
  stage. Applied to **A₅ only**, per the paper; Z₆₀ and S₄ do not get it.

**The one intentional deviation:** total budget is 60k steps, matching the other
four arms so the comparison is not confounded by compute. Stages promote by the
paper's rules; once the final stage is reached it holds for the remaining budget.

**Fixed settings** (Appendix D): AdamW, lr 3e-4, weight decay 0.01, grad clip
1.0, batch 256. Weight-tied block of 2 pre-norm layers, 4d FFN, GELU, input
injection via a linear adapter on `[h;e]`, readout on `LN(h^T)`, NoPE with causal
attention. Scale `s` for Z₆₀; scale `m` for S₄ and A₅.

**Reported metrics.** Final training loss (mean over the last 100 steps), and
per-token / per-sequence accuracy at n = 32 (training length), 64, and 128
(length generalisation, never trained). Token accuracy is the fraction of
positions correct; sequence accuracy requires all n positions correct at once.

**Chance floors.** Token: Z₆₀ 1/60 = 0.0167, S₄ 1/24 = 0.0417, A₅ 1/60 = 0.0167.
Sequence at n=32 is chance^32, effectively zero, so any nonzero sequence accuracy
is meaningful.

**Note on parity.** Parity (Z₂) is a *pilot/debugging* task in the paper, not part
of its selection grid, so it is excluded from these results. The task remains
implemented in `loop_transformers/tasks.py` and is still covered by the tests.

<!-- RESULTS -->
# Experiments

2x2x3 factorial: {bptt, stopgrad} x {fixed-n, curriculum} x {parity, S4, A5}, 4 seeds (86-89), 60k steps each.

Values are mean±sd over completed seeds. Chance: parity token 0.5, S4 0.0417, A5 0.0167; seq chance is ~0 at n=32.


## z60

| arm | schedule | seeds | train loss | n=32 tok | n=32 seq | n=64 tok | n=64 seq | n=128 tok | n=128 seq |
|---|---|---|---|---|---|---|---|---|---|
| bptt | fixed | 0 | — | — | — | — | — | — | — |
| bptt | curr | 3 | 0.1097±0.0593 | 0.9626±0.0379 | 0.6554±0.2725 | 0.4026±0.1206 | 0.0000±0.0000 | 0.0414±0.0147 | 0.0000±0.0000 |
| stopgrad | fixed | 4 | 0.7611±1.1793 | 0.8015±0.2876 | 0.6095±0.4118 | 0.6081±0.3965 | 0.1407±0.1237 | 0.3172±0.2024 | 0.0000±0.0000 |
| stopgrad | curr | 4 | 0.2824±0.2196 | 0.8986±0.0888 | 0.6503±0.2785 | 0.7059±0.0873 | 0.0444±0.0386 | 0.3632±0.0425 | 0.0000±0.0000 |
| bptt | **paper** | 3 | 0.5489±0.5180 | 0.6026±0.5128 | 0.2804±0.3508 | 0.0804±0.0587 | 0.0000±0.0000 | 0.0217±0.0045 | 0.0000±0.0000 |

## s4

| arm | schedule | seeds | train loss | n=32 tok | n=32 seq | n=64 tok | n=64 seq | n=128 tok | n=128 seq |
|---|---|---|---|---|---|---|---|---|---|
| bptt | **paper** | 0 | — | — | — | — | — | — | — |

## a5

| arm | schedule | seeds | train loss | n=32 tok | n=32 seq | n=64 tok | n=64 seq | n=128 tok | n=128 seq |
|---|---|---|---|---|---|---|---|---|---|
| bptt | **paper** | 2 | 0.7317±0.6248 | 0.3435±0.5657 | 0.3245±0.5621 | 0.2721±0.4412 | 0.0000±0.0000 | 0.0355±0.0326 | 0.0000±0.0000 |
