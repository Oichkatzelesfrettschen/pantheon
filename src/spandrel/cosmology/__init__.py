# SPDX-License-Identifier: GPL-2.0-only
"""Cosmology sub-package: distance models and hypothesis testing.

Primary classes
---------------
SpandrelCosmology
    Tests the Spandrel (spacetime stiffness) hypothesis against Pantheon+ data.
SpandrelFitter
    MLE and MCMC fitter for SpandrelCosmology parameters.
"""

from spandrel.cosmology.spandrel_cosmology import SpandrelCosmology, SpandrelFitter

__all__ = [
    'SpandrelCosmology',
    'SpandrelFitter',
]
