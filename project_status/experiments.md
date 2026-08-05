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

**2. Our 2×2 ablation** — Z₆₀, S₄, A₅, 4 seeds per cell.

| Factor | Levels |
|---|---|
| Gradient | `bptt` (backprop through all T) vs `stopgrad` (detach at h^{T-1}) |
| Schedule | `fixed` (n = T = 32 throughout) vs `curr` (n = 1,2,4,8,16,32 × 10k) |

Each task carries 5 cells (4 ablation + 1 replication).
15 configurations × 4 seeds = **60 runs**: the S₄/A₅ ablations were completed
in the earlier grid, and the replication cells were completed in the newer
28-run launch.

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
# Results

Each task carries the 2x2 ablation {bptt, stopgrad} x {fixed-n, curriculum} plus the paper replication. 4 seeds (86-89), 60k steps each; 60 runs total.

`paper` = Appendix D exactly: stages n_max in {4,8,16,32}, lengths sampled uniformly within a stage, promotion at per-token acc >= 0.98, T = n; horizon curriculum on A5 only.

Values are mean±sd over completed seeds. Token chance: Z60 0.0167, S4 0.0417, A5 0.0167; seq chance is ~0 at n=32.


## z60

| arm | schedule | seeds | train loss | n=32 tok | n=32 seq | n=64 tok | n=64 seq | n=128 tok | n=128 seq |
|---|---|---|---|---|---|---|---|---|---|
| bptt | fixed | 4 | 3.1718±0.9530 | 0.1303±0.1067 | 0.0000±0.0000 | 0.0375±0.0429 | 0.0000±0.0000 | 0.0274±0.0220 | 0.0000±0.0000 |
| bptt | curr | 4 | 0.1023±0.0506 | 0.9693±0.0337 | 0.7036±0.2424 | 0.4264±0.1094 | 0.0000±0.0000 | 0.0421±0.0121 | 0.0000±0.0000 |
| stopgrad | fixed | 4 | 0.7611±1.1793 | 0.8015±0.2876 | 0.6095±0.4118 | 0.6081±0.3965 | 0.1407±0.1237 | 0.3172±0.2024 | 0.0000±0.0000 |
| stopgrad | curr | 4 | 0.2824±0.2196 | 0.8986±0.0888 | 0.6503±0.2785 | 0.7059±0.0873 | 0.0444±0.0386 | 0.3632±0.0425 | 0.0000±0.0000 |
| stopgrad | currmax | 4 | 0.1470±0.0681 | 0.9022±0.0588 | 0.5829±0.2315 | 0.5549±0.1049 | 0.0004±0.0007 | 0.2843±0.0516 | 0.0000±0.0000 |
| bptt | **paper** | 4 | 0.2238±0.1363 | 0.7972±0.2141 | 0.2804±0.2864 | 0.1016±0.0242 | 0.0000±0.0000 | 0.0248±0.0010 | 0.0000±0.0000 |

## s4

| arm | schedule | seeds | train loss | n=32 tok | n=32 seq | n=64 tok | n=64 seq | n=128 tok | n=128 seq |
|---|---|---|---|---|---|---|---|---|---|
| bptt | fixed | 4 | 2.7851±0.3334 | 0.1071±0.0512 | 0.0000±0.0000 | 0.0587±0.0333 | 0.0000±0.0000 | 0.0502±0.0168 | 0.0000±0.0000 |
| bptt | curr | 4 | 2.9329±0.3924 | 0.1050±0.0738 | 0.0000±0.0000 | 0.0791±0.0440 | 0.0000±0.0000 | 0.0541±0.0152 | 0.0000±0.0000 |
| stopgrad | fixed | 4 | 3.1735±0.0029 | 0.0555±0.0066 | 0.0000±0.0000 | 0.0421±0.0002 | 0.0000±0.0000 | 0.0417±0.0001 | 0.0000±0.0000 |
| stopgrad | curr | 4 | 0.0542±0.0203 | 0.9860±0.0027 | 0.9165±0.0119 | 0.8535±0.0389 | 0.1918±0.0984 | 0.4520±0.0223 | 0.0000±0.0000 |
| stopgrad | currmax | 4 | 0.0680±0.0266 | 0.9724±0.0134 | 0.8475±0.0637 | 0.6852±0.0412 | 0.0038±0.0020 | 0.2378±0.0623 | 0.0000±0.0000 |
| bptt | **paper** | 4 | 0.0510±0.0189 | 0.9931±0.0048 | 0.9487±0.0222 | 0.9875±0.0139 | 0.8739±0.1346 | 0.7931±0.1816 | 0.2340±0.2816 |

## a5

| arm | schedule | seeds | train loss | n=32 tok | n=32 seq | n=64 tok | n=64 seq | n=128 tok | n=128 seq |
|---|---|---|---|---|---|---|---|---|---|
| bptt | fixed | 4 | 3.9081±0.0800 | 0.0621±0.0189 | 0.0000±0.0000 | 0.0168±0.0003 | 0.0000±0.0000 | 0.0170±0.0001 | 0.0000±0.0000 |
| bptt | curr | 4 | 2.4919±1.7729 | 0.4356±0.4497 | 0.2961±0.3726 | 0.2951±0.3115 | 0.0000±0.0000 | 0.0249±0.0149 | 0.0000±0.0000 |
| stopgrad | fixed | 4 | 4.0871±0.0049 | 0.0233±0.0028 | 0.0000±0.0000 | 0.0171±0.0000 | 0.0000±0.0000 | 0.0170±0.0001 | 0.0000±0.0000 |
| stopgrad | curr | 4 | 0.1449±0.0516 | 0.9598±0.0201 | 0.8134±0.0905 | 0.7837±0.0558 | 0.1495±0.0770 | 0.4030±0.0308 | 0.0000±0.0000 |
| stopgrad | currmax | 4 | 0.3097±0.2669 | 0.8044±0.1649 | 0.3879±0.3819 | 0.4492±0.1829 | 0.0000±0.0000 | 0.1796±0.1477 | 0.0000±0.0000 |
| bptt | **paper** | 4 | 0.0866±0.0521 | 0.9764±0.0159 | 0.9227±0.0384 | 0.7499±0.0887 | 0.1356±0.1601 | 0.2120±0.1843 | 0.0000±0.0000 |
