# Ideas

## Why fixed-n stopgrad should fail (and did, in the pilot)

With `h^{T-1}` detached and only the terminal output supervised, the objective is
"given this frozen state and `e`, solve the task in **one** block application."
At init the prefix loops produce an uninformative state, so that reduces to
solving 32-position prefix parity in constant depth — impossible (parity is not
in AC^0). The block settles on predicting the marginal; once it does, iterating
it still yields an uninformative state, and the prefix loops never receive
gradient to change that. It is a fixed point of the training dynamics, not slow
convergence — which is why the pilot losses sat at exactly ln 2 and ln 24.

## Why the curriculum should fix it

At n=1, T=1 there is no loop: the single application is fully differentiated and
the task (`p_1 = g_1`) is one-step solvable, so the block learns something real.
At n=2 the inherited `h^1` is produced by an already-trained block and is
genuinely informative, so the one-step gradient only has to learn "advance one
step." Each stage supplies the base case the next bootstraps from. Structurally
this is bootstrapped value iteration with a frozen target — sound only when the
target is already meaningful, which is exactly what the curriculum provides.

## Open questions the current grid will answer

- Does `curr` x `stopgrad` beat `fixed` x `stopgrad` across all three tasks and
  all four seeds, or was the pilot a seed artifact?
- Does the curriculum also help `bptt`? The pilot never ran that cell, so
  "stopgrad + curriculum works" could not be separated from "curriculum works."
  This is the main reason for the 2x2.
- Is parity genuinely pathological? Pilot hypothesis: with 2 classes a
  near-constant predictor sits at 50% and is a strong attractor, whereas guessing
  costs far more at 24/60 classes. Untested.

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

**Length generalisation is weak everywhere.** In the pilot even a model at 0.99
token accuracy on n=32 fell to chance at n=64/128 under `fixed`, and reached only
~0.87 token / ~0.20 seq under `curr`. Extending the curriculum past 32 would test
whether that is a budget artifact or a real ceiling.

**Warm-starting across loop budgets.** The paper claims mechanisms are portable
via warm-starting but not installable by input scheduling alone. A stopgrad model
trained at T=32 could be re-trained at T=128 to test portability cheaply.
