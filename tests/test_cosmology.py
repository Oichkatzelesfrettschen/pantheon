"""
Tests for Spandrel Cosmology Framework (spandrel_cosmology.py)

Tests validate:
1. Distance calculations (comoving, luminosity, distance modulus)
2. Hubble parameter E(z)
3. Spandrel correction properties
4. Chi-squared calculation
5. Physical consistency

Run with: pytest tests/test_cosmology.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from spandrel.cosmology.spandrel_cosmology import SpandrelCosmology, SpandrelFitter
from spandrel.core.constants import C_LIGHT_KMS, H0_FIDUCIAL, OMEGA_M_FIDUCIAL


class TestHubbleParameter:
    """Test dimensionless Hubble parameter E(z)."""

    def test_E_at_z_zero(self):
        """E(z=0) should equal 1."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        assert np.isclose(cosmo.E(0), 1.0)

    def test_E_positive(self):
        """E(z) should always be positive."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        z_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        for z in z_values:
            assert cosmo.E(z) > 0

    def test_E_increases_with_z(self):
        """E(z) should increase with z (matter-dominated at high z)."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        E_0 = cosmo.E(0)
        E_1 = cosmo.E(1)
        E_2 = cosmo.E(2)
        assert E_2 > E_1 > E_0

    def test_E_flat_universe_normalization(self):
        """For flat universe, E(z)^2 = Omega_m(1+z)^3 + Omega_Lambda."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        z = 1.0
        E_squared = cosmo.E(z)**2
        expected = 0.3 * (1 + z)**3 + 0.7
        assert np.isclose(E_squared, expected)


class TestComovingDistance:
    """Test comoving distance calculation."""

    def test_zero_at_z_zero(self):
        """Comoving distance should be zero at z=0."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        assert np.isclose(cosmo.comoving_distance(0), 0.0, atol=1e-10)

    def test_positive_for_positive_z(self):
        """Comoving distance should be positive for z > 0."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        z_values = [0.1, 0.5, 1.0, 2.0]
        for z in z_values:
            assert cosmo.comoving_distance(z) > 0

    def test_increases_with_z(self):
        """Comoving distance should increase monotonically with z."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        d_prev = 0
        for z in [0.1, 0.5, 1.0, 2.0, 5.0]:
            d = cosmo.comoving_distance(z)
            assert d > d_prev
            d_prev = d

    def test_reasonable_magnitude(self):
        """Comoving distance at z=1 should be ~3000-4000 Mpc."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        d = cosmo.comoving_distance(1.0)
        assert 2500 < d < 4500  # Mpc


class TestLuminosityDistance:
    """Test luminosity distance calculation."""

    def test_equals_comoving_times_1_plus_z(self):
        """D_L = (1+z) * D_C for flat universe."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        z = 0.5
        d_L = cosmo.luminosity_distance(z)
        d_C = cosmo.comoving_distance(z)
        expected = (1 + z) * d_C
        assert np.isclose(d_L, expected)

    def test_positive(self):
        """Luminosity distance should be positive for z > 0."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        for z in [0.1, 0.5, 1.0]:
            assert cosmo.luminosity_distance(z) > 0


class TestDistanceModulus:
    """Test distance modulus calculation."""

    def test_increases_with_z(self):
        """Distance modulus should increase with z (objects farther = dimmer)."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        mu_prev = -np.inf
        for z in [0.01, 0.1, 0.5, 1.0, 2.0]:
            mu = cosmo.distance_modulus_lcdm(z)
            assert mu > mu_prev
            mu_prev = mu

    def test_typical_values(self):
        """Distance modulus at z=0.1 should be ~38-40 mag."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        mu = cosmo.distance_modulus_lcdm(0.1)
        assert 37 < mu < 42

    def test_high_z_values(self):
        """Distance modulus at z=1 should be ~44-45 mag."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        mu = cosmo.distance_modulus_lcdm(1.0)
        assert 43 < mu < 46


class TestSpandrelCorrection:
    """Test Spandrel stiffness correction."""

    def test_zero_at_z_zero(self):
        """Spandrel correction should be zero at z=0."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3, epsilon=0.1)
        assert np.isclose(cosmo.spandrel_correction(0), 0.0)

    def test_zero_when_epsilon_zero(self):
        """Spandrel correction should be zero when epsilon=0."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3, epsilon=0.0)
        for z in [0.1, 0.5, 1.0]:
            assert cosmo.spandrel_correction(z) == 0.0

    def test_proportional_to_epsilon(self):
        """Correction should scale linearly with epsilon."""
        cosmo1 = SpandrelCosmology(H0=70.0, Omega_m=0.3, epsilon=0.1)
        cosmo2 = SpandrelCosmology(H0=70.0, Omega_m=0.3, epsilon=0.2)
        z = 0.5
        corr1 = cosmo1.spandrel_correction(z)
        corr2 = cosmo2.spandrel_correction(z)
        assert np.isclose(corr2, 2 * corr1)

    def test_recovers_lcdm_when_epsilon_zero(self):
        """Spandrel model should match LambdaCDM when epsilon=0."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3, epsilon=0.0)
        z = 0.5
        mu_spandrel = cosmo.distance_modulus_spandrel(z)
        mu_lcdm = cosmo.distance_modulus_lcdm(z)
        assert np.isclose(mu_spandrel, mu_lcdm)


class TestDistanceModulusArray:
    """Test vectorized distance modulus calculation."""

    def test_array_output_shape(self):
        """Output array should match input shape."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        z_array = np.array([0.1, 0.5, 1.0, 2.0])
        mu_array = cosmo.distance_modulus_array(z_array)
        assert mu_array.shape == z_array.shape

    def test_consistent_with_scalar(self):
        """Array output should match scalar calculations."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        z_array = np.array([0.1, 0.5, 1.0])
        mu_array = cosmo.distance_modulus_array(z_array, use_spandrel=False)

        for i, z in enumerate(z_array):
            mu_scalar = cosmo.distance_modulus_lcdm(z)
            assert np.isclose(mu_array[i], mu_scalar)


class TestSpandrelFitter:
    """Test the parameter fitting framework."""

    @pytest.fixture
    def mock_data(self):
        """Create mock supernova data for testing."""
        z_obs = np.array([0.01, 0.1, 0.3, 0.5, 1.0])

        # Generate "observed" data from known cosmology
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3, epsilon=0.0)
        mu_obs = cosmo.distance_modulus_array(z_obs)
        mu_err = np.full_like(mu_obs, 0.1)  # 0.1 mag errors

        return z_obs, mu_obs, mu_err

    def test_chi_squared_at_true_params(self, mock_data):
        """Chi-squared should be small at true parameters."""
        z_obs, mu_obs, mu_err = mock_data
        fitter = SpandrelFitter(z_obs, mu_obs, mu_err)

        # True parameters
        chi2 = fitter.chi_squared((70.0, 0.3, 0.0), use_spandrel=False)

        # Should be very small since data was generated from these params
        assert chi2 < 1e-10

    def test_chi_squared_positive(self, mock_data):
        """Chi-squared should be non-negative."""
        z_obs, mu_obs, mu_err = mock_data
        fitter = SpandrelFitter(z_obs, mu_obs, mu_err)

        # Various parameter sets
        params_list = [
            (70.0, 0.3, 0.0),
            (73.0, 0.28, 0.1),
            (67.0, 0.32, -0.05),
        ]

        for params in params_list:
            chi2 = fitter.chi_squared(params)
            assert chi2 >= 0

    def test_chi_squared_sensitive_to_H0(self, mock_data):
        """Chi-squared should change significantly with H0."""
        z_obs, mu_obs, mu_err = mock_data
        fitter = SpandrelFitter(z_obs, mu_obs, mu_err)

        chi2_true = fitter.chi_squared((70.0, 0.3, 0.0))
        chi2_wrong_H0 = fitter.chi_squared((75.0, 0.3, 0.0))

        # Wrong H0 should give worse fit
        assert chi2_wrong_H0 > chi2_true

    def test_bounds_enforcement(self, mock_data):
        """Unphysical parameters should give large chi-squared."""
        z_obs, mu_obs, mu_err = mock_data
        fitter = SpandrelFitter(z_obs, mu_obs, mu_err)

        # Unphysical H0
        chi2_bad = fitter.chi_squared((150.0, 0.3, 0.0))
        assert chi2_bad > 1e9


class TestPhysicalConsistency:
    """Test physical consistency of cosmology calculations."""

    def test_hubble_constant_effect(self):
        """Higher H0 should give smaller distances."""
        cosmo_low = SpandrelCosmology(H0=67.0, Omega_m=0.3)
        cosmo_high = SpandrelCosmology(H0=73.0, Omega_m=0.3)

        z = 0.5
        d_low = cosmo_low.luminosity_distance(z)
        d_high = cosmo_high.luminosity_distance(z)

        # Higher H0 means smaller Hubble distance (c/H0)
        assert d_high < d_low

    def test_omega_m_effect(self):
        """Higher Omega_m should give smaller distances (at moderate z)."""
        cosmo_low = SpandrelCosmology(H0=70.0, Omega_m=0.2)
        cosmo_high = SpandrelCosmology(H0=70.0, Omega_m=0.4)

        z = 1.0
        d_low = cosmo_low.luminosity_distance(z)
        d_high = cosmo_high.luminosity_distance(z)

        # Higher matter density decelerates expansion more
        assert d_high < d_low

    def test_flat_universe_constraint(self):
        """Omega_Lambda should equal 1 - Omega_m for flat universe."""
        cosmo = SpandrelCosmology(H0=70.0, Omega_m=0.3)
        assert np.isclose(cosmo.Omega_Lambda, 0.7)

        cosmo2 = SpandrelCosmology(H0=70.0, Omega_m=0.25)
        assert np.isclose(cosmo2.Omega_Lambda, 0.75)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
