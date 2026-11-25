"""Standard uncertainty quantification, beside UTOPIA rather than inside it.

The baselines UTOPIA is compared against: the raw point prediction, deep ensembles with a variance
head (Gaussian and negative-binomial), MC dropout, bootstrap resampling and marginal split
conformal.  Nothing in ``utopia/`` imports this package; run it against a workspace the pipeline
has already filled with ``python -m utopia.baselines config.yaml``.
"""
