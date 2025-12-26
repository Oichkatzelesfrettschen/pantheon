"""
Tests for White Dwarf Equation of State (eos_white_dwarf.py)

Tests validate:
1. Degenerate pressure limits (non-relativistic and ultra-relativistic)
2. EOS inversion consistency (rho,T -> e -> T roundtrip)
3. Sound speed causality (c_s < c)
4. Effective gamma bounds (4/3 to 5/3)
5. Physical value ranges

Run with: pytest tests/test_eos.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from spandrel.ddt.eos_white_dwarf import (
    electron_density,
    fermi_momentum,
    relativity_parameter,
    pressure_degenerate,
    pressure_ions,
    pressure_radiation,
    energy_degenerate,
    sound_speed,
    effective_gamma,
    eos_from_rho_T,
    temperature_from_rho_e,
    EOSState,
)
from spandrel.core.constants import C_LIGHT_CGS, K_BOLTZMANN, M_ELECTRON, M_PROTON


class TestElectronDensity:
    """Test electron density calculation."""

    def test_positive_density(self):
        """Electron density should be positive for positive mass density."""
        rho = np.array([1e6, 1e7, 1e8, 1e9])
        n_e = electron_density(rho)
        assert np.all(n_e > 0)

    def test_linear_scaling(self):
        """Electron density should scale linearly with mass density."""
        rho1, rho2 = 1e7, 2e7
        n_e1 = electron_density(np.array([rho1]))
        n_e2 = electron_density(np.array([rho2]))
        assert np.isclose(n_e2 / n_e1, 2.0, rtol=1e-10)

    def test_typical_wd_density(self):
        """Check electron density at typical WD central density."""
        rho = np.array([2e9])  # g/cm^3
        n_e = electron_density(rho)
        # n_e ~ Y_e * rho / m_p ~ 0.5 * 2e9 / 1.67e-24 ~ 6e32
        assert 1e32 < n_e[0] < 1e33


class TestFermiMomentum:
    """Test Fermi momentum calculation."""

    def test_positive_momentum(self):
        """Fermi momentum should be positive."""
        n_e = np.array([1e30, 1e31, 1e32])
        p_F = fermi_momentum(n_e)
        assert np.all(p_F > 0)

    def test_scaling_with_density(self):
        """p_F should scale as n_e^(1/3)."""
        n_e1, n_e2 = 1e30, 8e30  # factor of 8
        p_F1 = fermi_momentum(np.array([n_e1]))
        p_F2 = fermi_momentum(np.array([n_e2]))
        # p_F2/p_F1 should be 8^(1/3) = 2
        assert np.isclose(p_F2 / p_F1, 2.0, rtol=1e-10)


class TestRelativityParameter:
    """Test dimensionless relativity parameter x = p_F/(m_e*c)."""

    def test_non_relativistic_limit(self):
        """Low density should give x << 1 (non-relativistic)."""
        rho = np.array([1e5])  # Low density
        n_e = electron_density(rho)
        p_F = fermi_momentum(n_e)
        x = relativity_parameter(p_F)
        assert x[0] < 1.0

    def test_ultra_relativistic_limit(self):
        """High density should give x >> 1 (ultra-relativistic)."""
        rho = np.array([1e10])  # Very high density
        n_e = electron_density(rho)
        p_F = fermi_momentum(n_e)
        x = relativity_parameter(p_F)
        assert x[0] > 10.0


class TestDegeneratePressure:
    """Test Chandrasekhar degenerate electron pressure."""

    def test_positive_pressure(self):
        """Pressure should always be positive."""
        rho = np.array([1e6, 1e7, 1e8, 1e9, 1e10])
        P = pressure_degenerate(rho)
        assert np.all(P > 0)

    def test_non_relativistic_scaling(self):
        """At low density, P should scale as rho^(5/3)."""
        # Use low densities where x << 1
        rho1, rho2 = 1e5, 2e5
        P1 = pressure_degenerate(np.array([rho1]))
        P2 = pressure_degenerate(np.array([rho2]))
        # P ∝ rho^(5/3) -> P2/P1 = 2^(5/3) ~ 3.17
        expected_ratio = 2.0 ** (5.0 / 3.0)
        actual_ratio = P2[0] / P1[0]
        assert np.isclose(actual_ratio, expected_ratio, rtol=0.1)

    def test_ultra_relativistic_scaling(self):
        """At high density, P should scale as rho^(4/3)."""
        # Use high densities where x >> 1
        rho1, rho2 = 1e10, 2e10
        P1 = pressure_degenerate(np.array([rho1]))
        P2 = pressure_degenerate(np.array([rho2]))
        # P ∝ rho^(4/3) -> P2/P1 = 2^(4/3) ~ 2.52
        expected_ratio = 2.0 ** (4.0 / 3.0)
        actual_ratio = P2[0] / P1[0]
        assert np.isclose(actual_ratio, expected_ratio, rtol=0.1)

    def test_monotonic_increase(self):
        """Pressure should increase monotonically with density."""
        rho = np.logspace(6, 10, 20)
        P = pressure_degenerate(rho)
        assert np.all(np.diff(P) > 0)


class TestIonPressure:
    """Test ideal gas ion pressure."""

    def test_positive_pressure(self):
        """Ion pressure should be positive for T > 0."""
        rho = np.array([1e7])
        T = np.array([1e9])
        P_ion = pressure_ions(rho, T)
        assert P_ion[0] > 0

    def test_linear_temperature_scaling(self):
        """P_ion should scale linearly with T."""
        rho = np.array([1e7])
        T1, T2 = np.array([1e9]), np.array([2e9])
        P1 = pressure_ions(rho, T1)
        P2 = pressure_ions(rho, T2)
        assert np.isclose(P2 / P1, 2.0, rtol=1e-10)


class TestRadiationPressure:
    """Test radiation pressure P_rad = aT^4/3."""

    def test_positive_pressure(self):
        """Radiation pressure should be positive."""
        T = np.array([1e9, 5e9, 1e10])
        P_rad = pressure_radiation(T)
        assert np.all(P_rad > 0)

    def test_t4_scaling(self):
        """P_rad should scale as T^4."""
        T1, T2 = np.array([1e9]), np.array([2e9])
        P1 = pressure_radiation(T1)
        P2 = pressure_radiation(T2)
        assert np.isclose(P2 / P1, 16.0, rtol=1e-10)

    def test_negligible_at_wd_temps(self):
        """Radiation pressure should be << degenerate at WD conditions."""
        rho = np.array([2e9])
        T = np.array([5e8])  # 0.5 GK
        P_deg = pressure_degenerate(rho)
        P_rad = pressure_radiation(T)
        assert P_rad[0] < 0.01 * P_deg[0]  # Less than 1%


class TestSoundSpeed:
    """Test adiabatic sound speed."""

    def test_causality(self):
        """Sound speed must be less than speed of light."""
        rho = np.logspace(6, 10, 20)
        T = np.full_like(rho, 1e9)
        state = eos_from_rho_T(rho, T)
        assert np.all(state.cs < C_LIGHT_CGS)

    def test_positive(self):
        """Sound speed should be positive."""
        rho = np.array([1e7, 1e8, 1e9])
        T = np.array([1e9, 1e9, 1e9])
        state = eos_from_rho_T(rho, T)
        assert np.all(state.cs > 0)

    def test_typical_values(self):
        """Sound speed at DDT conditions should be ~10^8-10^9 cm/s."""
        rho = np.array([2e7])  # DDT density
        T = np.array([3e9])    # Hot spot
        state = eos_from_rho_T(rho, T)
        assert 1e8 < state.cs[0] < 1e10


class TestEffectiveGamma:
    """Test effective adiabatic index."""

    def test_bounds(self):
        """gamma_eff should be between 4/3 (ultra-rel) and 5/3 (non-rel)."""
        rho = np.logspace(6, 10, 20)
        T = np.full_like(rho, 1e9)
        state = eos_from_rho_T(rho, T)
        assert np.all(state.gamma_eff >= 4.0 / 3.0 - 0.01)
        assert np.all(state.gamma_eff <= 5.0 / 3.0 + 0.01)

    def test_approaches_5_3_at_low_density(self):
        """At low density (non-rel), gamma -> 5/3."""
        rho = np.array([1e5])
        T = np.array([1e8])
        state = eos_from_rho_T(rho, T)
        assert np.isclose(state.gamma_eff[0], 5.0 / 3.0, rtol=0.1)

    def test_approaches_4_3_at_high_density(self):
        """At high density (ultra-rel), gamma -> 4/3."""
        rho = np.array([1e10])
        T = np.array([1e9])
        state = eos_from_rho_T(rho, T)
        assert np.isclose(state.gamma_eff[0], 4.0 / 3.0, rtol=0.1)


class TestEOSInversion:
    """Test EOS inversion (rho, e -> T)."""

    def test_roundtrip_consistency(self):
        """(rho, T) -> e -> T should recover original T."""
        rho = np.array([1e7, 2e7, 5e7, 1e8])
        T_original = np.array([1e9, 2e9, 3e9, 5e9])

        state = eos_from_rho_T(rho, T_original)
        T_recovered = temperature_from_rho_e(rho, state.e_int)

        np.testing.assert_allclose(T_recovered, T_original, rtol=1e-3)

    def test_convergence_at_extremes(self):
        """Should converge at extreme (but physical) conditions."""
        # Low T
        rho = np.array([1e8])
        T_low = np.array([1e8])
        state = eos_from_rho_T(rho, T_low)
        T_rec = temperature_from_rho_e(rho, state.e_int)
        assert np.isclose(T_rec[0], T_low[0], rtol=0.01)

        # High T
        T_high = np.array([1e10])
        state = eos_from_rho_T(rho, T_high)
        T_rec = temperature_from_rho_e(rho, state.e_int)
        assert np.isclose(T_rec[0], T_high[0], rtol=0.01)


class TestEOSState:
    """Test complete EOS state computation."""

    def test_all_fields_populated(self):
        """EOSState should have all fields populated."""
        rho = np.array([1e7])
        T = np.array([1e9])
        state = eos_from_rho_T(rho, T)

        assert state.rho is not None
        assert state.T is not None
        assert state.P is not None
        assert state.e_int is not None
        assert state.cs is not None
        assert state.gamma_eff is not None

    def test_pressure_components(self):
        """Total pressure should be sum of components."""
        rho = np.array([2e9])
        T = np.array([5e9])
        state = eos_from_rho_T(rho, T)

        P_deg = pressure_degenerate(rho)
        P_ion = pressure_ions(rho, T)
        P_rad = pressure_radiation(T)
        P_total = P_deg + P_ion + P_rad

        np.testing.assert_allclose(state.P, P_total, rtol=1e-10)


class TestPhysicalConsistency:
    """Test physical consistency of EOS."""

    def test_thermodynamic_stability(self):
        """dP/drho should be positive (mechanical stability)."""
        rho = np.logspace(6, 10, 100)
        T = np.full_like(rho, 1e9)
        state = eos_from_rho_T(rho, T)

        dP_drho = np.gradient(state.P, rho)
        assert np.all(dP_drho > 0)

    def test_energy_positive(self):
        """Internal energy should be positive."""
        rho = np.array([1e7, 1e8, 1e9])
        T = np.array([1e9, 1e9, 1e9])
        state = eos_from_rho_T(rho, T)
        assert np.all(state.e_int > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
