# TODO

> Check out the PR https://github.com/rel-int/optyx/pull/21 in optyx. We need to find a
> simpler way to do this, definitely I don't want to add a new module contract. Checkout
> also the PR https://github.com/discopy/discopy/issues/523 in discopy. The proposal here
> was to add a contract parameter in eval and tensor.Functor to decide which contractor to
> use between einsum, opt_einsum and quimb/cotengra. With einsum as default switching to
> opt_einsum when there are index problems. it's good to keep the quimb interface to be
> able to use the methods in that library. I would like to have minimal edits in optyx and
> handle all the contraction in DisCoPy, so that optyx remains syntactic (except for the
> PercevalBackend). What is the best solution? Propose a plan for standardising this
> between discopy and optyx

- [WIP] @f1h17k-2026-08-06 12:56 Replace `QuimbBackend._process_term` by a single call to discopy's
      `eval(contract="quimb")`, deleting the hyperoptimiser dispatch, the dtype
      promotion loop and `preprocess_quimb_tensors_safe`
- [WIP] @f1h17k-2026-08-06 12:56 Pass `**extra` through `DiscopyBackend.eval` so `contract` and `optimize`
      reach discopy; delete the dead `_get_quimb_tensor`
- [WIP] @f1h17k-2026-08-06 12:56 Repin discopy to its contraction-standardization branch with the
      `[tensor]` extra and
      declare the directly imported `quimb` and `cotengra` as dependencies
- [WIP] @f1h17k-2026-08-06 12:56 Run `pflake8 optyx` and the test suite, doctests included
