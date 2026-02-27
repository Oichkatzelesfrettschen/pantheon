# SPDX-License-Identifier: GPL-2.0-only
"""Synthesis sub-package: turbulent flame theory and Phillips relation derivation."""

from spandrel.synthesis.turbulent_flame_theory import FractalFlame, TurbulentSupernovaModel
from spandrel.synthesis.unified_experiment import run_all_experiments

__all__ = [
    'TurbulentSupernovaModel',
    'FractalFlame',
    'run_all_experiments',
]
