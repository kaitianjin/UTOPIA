"""Prediction models.

UTOPIA implements exactly one model itself, the built-in ``default`` predictor. Every other model
(Path2Space, Loki, ...) stays in its own environment and hands its results over as files, so
nothing is registered here: UTOPIA never imports it.
"""
