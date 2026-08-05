# Ideas

## Why fixed-n stopgrad is unstable

With `h[T - 1]` detached and only the terminal output supervised, the objective is
"given this frozen state and `e`, solve the task in **one** block application."
At init the prefix loops produce an uninformative state, making the marginal
predictor a strong attractor; this explains the pilot losses at exactly ln 2 and
ln 24. The completed Z60 grid rules out the stronger claim that fixed-n
stopgrad cannot learn: three seeds reached 0.73-0.89 sequence accuracy at n=32,
while one remained at zero. Updates learned through the final application change
the shared block used by every prefix application on the next optimiser step,
so the prefix can improve across steps even though it receives no within-step
gradient. The result is an unstable optimisation path, not an impossibility.

## Why the curriculum should improve stability

At n=1, T=1 there is no loop: the single application is fully differentiated and
the task (`p_1 = g_1`) is one-step solvable, so the block learns something real.
At n=2 the inherited `h^1` is produced by an already-trained block and is
genuinely informative, so the one-step gradient only has to learn "advance one
step." Each stage supplies the base case the next bootstraps from. Structurally
this is bootstrapped value iteration with a frozen target. On Z60, curriculum
raised stopgrad token accuracy at n=32 from 0.8015±0.2876 to 0.8986±0.0888 and
sequence accuracy from 0.6095±0.4118 to 0.6503±0.2785. The effect was much
larger on S4 and A5: fixed stopgrad had zero sequence accuracy on both, while
curriculum reached 0.9165±0.0119 and 0.8134±0.0905 respectively.

## Answers from the completed grid

- Fixed-stopgrad success is task-dependent. Three of four fixed Z60 seeds
  learned, but all fixed S4 and A5 seeds had zero sequence accuracy.
- Curriculum is decisive for stopgrad on S4 and A5, where it reaches
  0.9165±0.0119 and 0.8134±0.0905 sequence accuracy at n=32.
- Curriculum's effect on BPTT is inconsistent. It raises Z60 sequence accuracy
  from zero to 0.7036±0.2424 and A5 to 0.2961±0.3726, but S4 remains at zero.
- The paper schedule is the strongest BPTT schedule for S4 and A5, reaching
  0.9487±0.0222 and 0.9227±0.0384 sequence accuracy at n=32.
- Parity was excluded from this grid, so the pilot hypothesis about its marginal
  predictor remains untested.

## Follow-ups not yet run

**Deep supervision, O(1) memory.** Supervise every loop's output while detaching
the state between loops. Keeps memory O(1) in T but gives every `h^t` a gradient,
so it should not need a curriculum at all. The cleanest comparison against both
current arms.

**k-step truncated BPTT.** Interpolate between the arms: detach every k loops
rather than every loop. Memory O(k). Would show whether there is a minimum
gradient horizon needed for the mechanism to form.

**Push T beyond 32.** The whole point of O(1) memory. BPTT at T=128 needs ~86 GB
and OOMs a 48GB card; stopgrad stays under 1 GB. Worth running n=64/128 as
training lengths, not just eval probes.

**Length generalisation is task-dependent.** The S4 paper replication retained
0.8739 sequence accuracy at n=64 and 0.2340 at n=128. A5 fell to 0.1356 and then
zero; every Z60 cell had zero sequence accuracy at n=128. Extending the
curriculum past 32 would test whether those weaker results are a budget artifact
or a real ceiling.

**Warm-starting across loop budgets.** The paper claims mechanisms are portable
via warm-starting but not installable by input scheduling alone. A stopgrad model
trained at T=32 could be re-trained at T=128 to test portability cheaply.
