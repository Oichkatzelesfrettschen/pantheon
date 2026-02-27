"""Smoke tests for elevated simulation modules.

These tests verify that the elevated modules import and their core classes
instantiate correctly without running full (expensive) simulations.
"""

import numpy as np
import pytest


class TestAlphaChainNetwork:
    """Smoke tests for the 13-isotope alpha-chain nuclear network."""

    def test_imports(self):
        from spandrel.elevated.alpha_chain_network import ISOTOPES, AlphaChainNetwork, Isotope
        assert AlphaChainNetwork is not None
        assert Isotope is not None

    def test_instantiation(self):
        from spandrel.elevated.alpha_chain_network import AlphaChainNetwork
        network = AlphaChainNetwork()
        assert network is not None

    def test_isotopes_count(self):
        from spandrel.elevated.alpha_chain_network import ISOTOPES
        # 13-isotope network: C12 through Ni56
        assert len(ISOTOPES) == 13

    def test_burn_to_completion_returns_dict(self):
        """NSE conditions should produce a result dict."""
        from spandrel.elevated.alpha_chain_network import AlphaChainNetwork, Isotope

        network = AlphaChainNetwork()
        X_init = np.zeros(13)
        X_init[Isotope.C12] = 0.5
        X_init[Isotope.O16] = 0.5

        result = network.burn_to_completion(rho=2e7, T=6e9, X_init=X_init, t_max=0.001)
        assert isinstance(result, dict)
        assert 'X_final' in result
        assert 'e_total' in result
        assert result['e_total'] >= 0


class TestLightCurveSynthesis:
    """Smoke tests for the Arnett light curve model."""

    def test_imports(self):
        from spandrel.elevated.light_curve_synthesis import ArnettModel, LightCurveGenerator
        assert LightCurveGenerator is not None

    def test_generator_instantiation(self):
        from spandrel.elevated.light_curve_synthesis import LightCurveGenerator
        gen = LightCurveGenerator(M_Ni=0.6 * 2e33)
        assert gen is not None

    def test_generate_returns_observables(self):
        import matplotlib

        from spandrel.elevated.light_curve_synthesis import LightCurveGenerator
        matplotlib.use('Agg')
        gen = LightCurveGenerator(M_Ni=0.6 * 2e33)
        data = gen.generate()
        assert 'observables' in data
        obs = data['observables']
        assert 't_rise' in obs
        assert 'delta_m15' in obs
        assert obs['delta_m15'] > 0

    def test_peak_luminosity_physical(self):
        """Peak luminosity should be in the standard SN Ia range."""
        import matplotlib

        from spandrel.elevated.light_curve_synthesis import LightCurveGenerator
        matplotlib.use('Agg')
        gen = LightCurveGenerator(M_Ni=0.6 * 2e33)
        data = gen.generate()
        L_peak = data['observables']['L_peak']
        # Typical Type Ia: 1e42 - 1e44 erg/s
        assert 1e41 < L_peak < 1e45


class TestModelComparison:
    """Smoke tests for the Bayesian model comparison module."""

    def test_imports(self):
        from spandrel.elevated.model_comparison import (
            CPL,
            LambdaCDM,
            NestedSampler,
            RiemannResonance,
            wCDM,
        )
        assert LambdaCDM is not None

    def test_lambdacdm_E_function(self):
        """LambdaCDM E(z) should return normalized Hubble parameter."""
        from spandrel.elevated.model_comparison import LambdaCDM
        model = LambdaCDM()
        z = np.array([0.0, 0.5, 1.0])
        params = np.array([70.0, 0.3])  # H0=70, Omega_m=0.3
        E_z = model.E(z, params)
        # E(0) = 1 by definition for flat universe (Om + Ol = 1)
        assert np.isclose(E_z[0], 1.0, rtol=1e-3)
        # E(z) must be positive and increase with z for LambdaCDM
        assert np.all(E_z > 0)
        assert E_z[2] > E_z[1] > E_z[0]


class TestDDTParameterStudy:
    """Smoke tests for the DDT parameter study module."""

    def test_imports(self):
        from spandrel.elevated.ddt_parameter_study import DDTParameterStudy, ParameterPoint
        assert DDTParameterStudy is not None

    def test_instantiation(self):
        from spandrel.elevated.ddt_parameter_study import DDTParameterStudy
        study = DDTParameterStudy()
        assert study is not None
