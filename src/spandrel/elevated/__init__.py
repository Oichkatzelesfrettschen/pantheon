"""
Elevated Spandrel Project Modules

Contains research-grade simulation tools:
    - model_comparison: Bayesian evidence for cosmological models
    - alpha_chain_network: 13-isotope nuclear network
    - light_curve_synthesis: SN Ia light curve generation
    - ddt_parameter_study: Systematic DDT exploration
"""

from .model_comparison import (
    LambdaCDM, wCDM, CPL, RiemannResonance,
    NestedSampler, compute_bayes_factors
)
from .alpha_chain_network import AlphaChainNetwork, Isotope, ISOTOPES
from .light_curve_synthesis import LightCurveGenerator, ArnettModel
from .ddt_parameter_study import DDTParameterStudy, run_single_simulation

__all__ = [
    'LambdaCDM', 'wCDM', 'CPL', 'RiemannResonance',
    'NestedSampler', 'compute_bayes_factors',
    'AlphaChainNetwork', 'Isotope', 'ISOTOPES',
    'LightCurveGenerator', 'ArnettModel',
    'DDTParameterStudy', 'run_single_simulation',
]
