# BART vectoriZed

An implementation of Bayesian Additive Regression Trees (BART) in JAX.

If you don't know what BART is, but know XGBoost, consider BART as a sort of Bayesian XGBoost. bartz makes BART run fast enough to be used in place of XGBoost/lightgbm/etc. even on large datasets.

BART is a nonparametric Bayesian regression technique. Given training predictors $X$ and responses $y$, BART finds a function to predict $y$ given $X$. The result of the inference is a sample of possible functions, representing the uncertainty over the determination of the function.

This Python module provides an implementation of BART that runs on GPU, to process large datasets faster. It is also good on CPU. Most other implementations of BART are for R, and run on CPU only.

On CPU, bartz runs at the speed of dbarts (the fastest implementation I know of) if n > 20,000, but using 1/20 of the memory. On GPU, the speed premium depends on sample size; it is convenient over CPU only for n > 10,000. The maximum speedup is currently 200x, on an Nvidia A100 and with at least 2,000,000 observations.

[This Colab notebook](https://colab.research.google.com/github/anon-nips2026-36013/bartz/blob/main/docs/examples/basic_simdata.ipynb) runs bartz with n = 100,000 observations, p = 1000 predictors, 10,000 trees, for 2000 MCMC iterations, in 10 minutes.

BART is a very flexible method with many variants. This implementation provides only a small subset of the possible features. If you need a feature from [another BART implementation](https://anon-nips2026-36013.github.io/bartz/docs-dev/pkglist.html) or from the BART literature, please [open an issue on github](https://github.com/anon-nips2026-36013/bartz/issues).

## Links

- [Documentation (latest release)](https://anon-nips2026-36013.github.io/bartz/docs)
- [Documentation (development version)](https://anon-nips2026-36013.github.io/bartz/docs-dev)
- [Repository](https://github.com/anon-nips2026-36013/bartz)
- [Code coverage](https://anon-nips2026-36013.github.io/bartz/coverage)
- [Benchmarks](https://anon-nips2026-36013.github.io/bartz/benchmarks)
- [List of BART packages](https://anon-nips2026-36013.github.io/bartz/docs-dev/pkglist.html)

## Citing bartz

References redacted for double-blind review.
