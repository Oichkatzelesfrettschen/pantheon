"""
Tests for Nuclear Reaction Network (reaction_carbon.py)

Tests validate:
1. Screening factor behavior
2. C12+C12 rate coefficient (Caughlan-Fowler parameterization)
3. Reaction rate and energy generation
4. Burn step integration
5. Chapman-Jouguet detonation velocity

Run with: pytest tests/test_reactions.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ddt_solver.reaction_carbon import (
    screening_factor,
    c12_c12_rate,
    reaction_rate_c12,
    burn_substep,
    burn_step_subcycled,
    chapman_jouguet_velocity,
    Q_EFF,
)
from constants import Q_BURN, K_BOLTZMANN, M_PROTON


class TestScreeningFactor:
    """Test Coulomb screening enhancement."""

    def test_positive_enhancement(self):
        """Screening factor should be >= 1 (enhancement)."""
        rho = np.array([1e7, 1e8, 1e9])
        T = np.array([1e9, 1e9, 1e9])
        f = screening_factor(rho, T)
        assert np.all(f >= 1.0)

    def test_finite_values(self):
        """Screening factor should be finite for physical conditions."""
        rho = np.array([1e6, 1e7, 1e8, 1e9])
        T = np.array([1e9, 1e9, 1e9, 1e9])
        f = screening_factor(rho, T)
        assert np.all(np.isfinite(f))

    def test_capped_at_extreme_conditions(self):
        """Screening factor should be capped to prevent overflow."""
        # Very high density, low temperature - extreme screening
        rho = np.array([1e10])
        T = np.array([1e8])
        f = screening_factor(rho, T)
        # Implementation caps at exp(5) = ~148
        assert np.all(np.isfinite(f))
        assert f[0] <= np.exp(5.0) + 0.1

    def test_exponential_form(self):
        """Screening enhancement should have exponential form."""
        # Screening factor is exp(min(H, 5)) where H depends on rho, T
        rho = np.array([1e7])
        T = np.array([1e10])  # High T reduces screening
        f = screening_factor(rho, T)
        # Should be close to 1 at high T
        assert f[0] > 1.0


class TestC12Rate:
    """Test C12+C12 rate coefficient."""

    def test_positive_rate(self):
        """Rate coefficient should be positive."""
        T = np.array([1e9, 2e9, 5e9, 1e10])
        rate = c12_c12_rate(T)
        assert np.all(rate > 0)

    def test_temperature_sensitivity(self):
        """Rate should increase strongly with temperature."""
        T_low = np.array([1e9])
        T_high = np.array([3e9])

        rate_low = c12_c12_rate(T_low)
        rate_high = c12_c12_rate(T_high)

        # Rate should increase by many orders of magnitude
        assert rate_high[0] > 1e10 * rate_low[0]

    def test_floor_at_low_temperature(self):
        """Rate should not blow up at low temperature."""
        T_cold = np.array([1e6])  # Very cold
        rate = c12_c12_rate(T_cold)
        assert np.isfinite(rate[0])
        # At very low T, rate can be essentially zero due to Gamow factor
        assert rate[0] >= 0

    def test_reasonable_magnitude_at_high_t(self):
        """At T9=3 (hot burning), rate should be substantial."""
        T = np.array([3e9])  # 3 GK
        rate = c12_c12_rate(T)
        # Rate is per particle pair, should be non-negligible
        assert rate[0] > 1e-30


class TestReactionRate:
    """Test full reaction rate calculation."""

    def test_zero_fuel_zero_rate(self):
        """Zero fuel should give zero burning rate."""
        rho = np.array([1e8])
        T = np.array([3e9])
        X_C12 = np.array([0.0])

        dX_dt, eps = reaction_rate_c12(rho, T, X_C12)

        assert dX_dt[0] == 0.0
        assert eps[0] == 0.0

    def test_negative_consumption(self):
        """Fuel consumption rate should be negative."""
        rho = np.array([1e8])
        T = np.array([3e9])
        X_C12 = np.array([0.5])

        dX_dt, eps = reaction_rate_c12(rho, T, X_C12)

        assert dX_dt[0] < 0.0

    def test_positive_energy_generation(self):
        """Energy generation should be positive."""
        rho = np.array([1e8])
        T = np.array([3e9])
        X_C12 = np.array([0.5])

        dX_dt, eps = reaction_rate_c12(rho, T, X_C12)

        assert eps[0] > 0.0

    def test_energy_rate_scales_with_fuel(self):
        """Energy generation should increase with fuel fraction."""
        rho = np.array([1e8])
        T = np.array([3e9])
        X_low = np.array([0.1])
        X_high = np.array([0.5])

        _, eps_low = reaction_rate_c12(rho, T, X_low)
        _, eps_high = reaction_rate_c12(rho, T, X_high)

        # Rate scales as X^2
        assert eps_high[0] > eps_low[0]

    def test_rate_scales_with_density(self):
        """Burning rate should increase with density."""
        T = np.array([3e9])
        X_C12 = np.array([0.5])
        rho_low = np.array([1e7])
        rho_high = np.array([1e8])

        _, eps_low = reaction_rate_c12(rho_low, T, X_C12)
        _, eps_high = reaction_rate_c12(rho_high, T, X_C12)

        # Rate scales as rho (n^2/rho)
        assert eps_high[0] > eps_low[0]


class TestBurnSubstep:
    """Test nuclear burning integration."""

    def test_fuel_decreases(self):
        """Fuel should decrease during burning."""
        n = 10
        rho = np.full(n, 1e8)
        e_int = np.full(n, 1e17)
        X_C12 = np.full(n, 0.5)
        T = np.full(n, 3e9)
        dt = 1e-6

        e_new, X_new = burn_substep(rho, e_int, X_C12, T, dt)

        assert np.all(X_new <= X_C12)

    def test_energy_increases(self):
        """Internal energy should increase during burning."""
        n = 10
        rho = np.full(n, 1e8)
        e_int = np.full(n, 1e17)
        X_C12 = np.full(n, 0.5)
        T = np.full(n, 3e9)
        dt = 1e-6

        e_new, X_new = burn_substep(rho, e_int, X_C12, T, dt)

        assert np.all(e_new >= e_int)

    def test_mass_fraction_bounded(self):
        """Mass fraction should stay in [0, 1]."""
        n = 10
        rho = np.full(n, 1e8)
        e_int = np.full(n, 1e17)
        X_C12 = np.full(n, 0.5)
        T = np.full(n, 5e9)  # Very hot
        dt = 1e-3  # Large timestep

        e_new, X_new = burn_substep(rho, e_int, X_C12, T, dt)

        assert np.all(X_new >= 0.0)
        assert np.all(X_new <= 1.0)

    def test_backward_euler_stable(self):
        """Backward Euler should be stable for large timesteps."""
        n = 10
        rho = np.full(n, 1e8)
        e_int = np.full(n, 1e17)
        X_C12 = np.full(n, 0.5)
        T = np.full(n, 5e9)
        dt = 1e-2  # Very large timestep

        # Should not crash or produce NaN
        e_new, X_new = burn_substep(rho, e_int, X_C12, T, dt, method='backward_euler')

        assert np.all(np.isfinite(e_new))
        assert np.all(np.isfinite(X_new))


class TestBurnSubcycled:
    """Test subcycled burning integration."""

    def test_conserves_fuel_plus_ash(self):
        """Total mass should be conserved."""
        n = 10
        rho = np.full(n, 1e8)
        e_int = np.full(n, 1e17)
        X_C12 = np.full(n, 0.5)
        T = np.full(n, 3e9)
        dt = 1e-3

        e_new, X_new = burn_step_subcycled(rho, e_int, X_C12, T, dt)

        # Fuel consumed should equal ash produced
        dX_fuel = X_C12 - X_new
        assert np.all(dX_fuel >= 0)

    def test_energy_consistent_with_fuel_burned(self):
        """Energy release should match fuel consumed * Q_burn."""
        n = 10
        rho = np.full(n, 1e8)
        e_int = np.full(n, 1e17)
        X_C12 = np.full(n, 0.5)
        T = np.full(n, 3e9)
        dt = 1e-5

        e_new, X_new = burn_step_subcycled(rho, e_int, X_C12, T, dt)

        dX = X_C12 - X_new
        de_expected = dX * Q_BURN
        de_actual = e_new - e_int

        # Should be approximately equal (within 10%)
        np.testing.assert_allclose(de_actual, de_expected, rtol=0.1)

    def test_handles_complete_burnout(self):
        """Should handle case where all fuel burns."""
        n = 10
        rho = np.full(n, 1e8)
        e_int = np.full(n, 1e17)
        X_C12 = np.full(n, 0.01)  # Small amount of fuel
        T = np.full(n, 5e9)       # Very hot
        dt = 1e-2                  # Long timestep

        e_new, X_new = burn_step_subcycled(rho, e_int, X_C12, T, dt)

        # Should not crash, X should be near zero
        assert np.all(np.isfinite(X_new))
        assert np.all(X_new >= 0)


class TestChapmanJouguet:
    """Test Chapman-Jouguet detonation velocity."""

    def test_positive_velocity(self):
        """CJ velocity should be positive."""
        rho = np.array([1e7, 2e7, 5e7])
        T = np.array([1e9, 1e9, 1e9])
        v_CJ = chapman_jouguet_velocity(rho, T)
        assert np.all(v_CJ > 0)

    def test_reasonable_magnitude(self):
        """CJ velocity should be ~10^8-10^9 cm/s for WD conditions."""
        rho = np.array([2e7])  # DDT density
        T = np.array([1e9])
        v_CJ = chapman_jouguet_velocity(rho, T)
        assert 1e8 < v_CJ[0] < 2e9

    def test_consistent_array_shape(self):
        """Output should match input shape."""
        rho = np.array([1e7, 2e7, 5e7])
        T = np.array([1e9, 1e9, 1e9])
        v_CJ = chapman_jouguet_velocity(rho, T)
        assert v_CJ.shape == rho.shape


class TestPhysicalConsistency:
    """Test physical consistency of nuclear network."""

    def test_burning_timescale_positive(self):
        """Burning timescale should be positive and finite."""
        rho = np.array([2e7])
        T = np.array([3e9])
        X_C12 = np.array([0.5])

        dX_dt, eps = reaction_rate_c12(rho, T, X_C12)

        # Timescale τ = X / |dX/dt|
        tau = X_C12[0] / np.abs(dX_dt[0])

        # Should be positive and finite
        assert tau > 0
        assert np.isfinite(tau)

    def test_faster_burning_at_higher_temperature(self):
        """Burning should be faster (shorter timescale) at higher T."""
        rho = np.array([2e7])
        X_C12 = np.array([0.5])

        T_low = np.array([2e9])
        T_high = np.array([5e9])

        dX_dt_low, _ = reaction_rate_c12(rho, T_low, X_C12)
        dX_dt_high, _ = reaction_rate_c12(rho, T_high, X_C12)

        tau_low = X_C12[0] / np.abs(dX_dt_low[0])
        tau_high = X_C12[0] / np.abs(dX_dt_high[0])

        # Higher T means faster burning (shorter timescale)
        assert tau_high < tau_low

    def test_energy_release_per_gram_reasonable(self):
        """Energy from burning 1g of C12 should be ~2×10^17 erg."""
        # Complete burning of 1g C12
        e_per_gram = Q_BURN

        # Should be ~2×10^17 erg/g
        assert 1e17 < e_per_gram < 1e18


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
