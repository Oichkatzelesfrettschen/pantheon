# SPDX-License-Identifier: GPL-2.0-only
"""
Elevated Spandrel Project Modules

Contains research-grade simulation tools:
    - model_comparison: Bayesian evidence for cosmological models
    - alpha_chain_network: 13-isotope nuclear network
    - light_curve_synthesis: SN Ia light curve generation
    - ddt_parameter_study: Systematic DDT exploration
"""

from .alpha_chain_network import ISOTOPES, AlphaChainNetwork, Isotope
from .ddt_parameter_study import DDTParameterStudy, run_single_simulation
from .light_curve_synthesis import ArnettModel, LightCurveGenerator
from .model_comparison import (
    CPL,
    LambdaCDM,
    NestedSampler,
    RiemannResonance,
    compute_bayes_factors,
    wCDM,
)

__all__ = [
    'LambdaCDM', 'wCDM', 'CPL', 'RiemannResonance',
    'NestedSampler', 'compute_bayes_factors',
    'AlphaChainNetwork', 'Isotope', 'ISOTOPES',
    'LightCurveGenerator', 'ArnettModel',
    'DDTParameterStudy', 'run_single_simulation',
]
