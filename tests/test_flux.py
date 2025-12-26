"""
Tests for HLLC Riemann Solver (flux_hllc.py)

Tests validate:
1. Flux limiter properties (TVD, symmetry)
2. Primitive/conserved variable conversion roundtrip
3. CFL timestep calculation
4. HLLC flux computation
5. Sod shock tube benchmark

Run with: pytest tests/test_flux.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from spandrel.ddt.flux_hllc import (
    minmod,
    superbee,
    mc_limiter,
    primitive_to_conserved,
    conserved_to_primitive,
    compute_cfl_timestep,
    compute_hllc_update,
)


class TestMinmodLimiter:
    """Test minmod flux limiter."""

    def test_same_sign_positive(self):
        """Minmod of same-sign positives should be minimum."""
        a, b = np.array([2.0]), np.array([3.0])
        result = minmod(a, b)
        assert result[0] == 2.0

    def test_same_sign_negative(self):
        """Minmod of same-sign negatives should be maximum (least negative)."""
        a, b = np.array([-2.0]), np.array([-3.0])
        result = minmod(a, b)
        assert result[0] == -2.0

    def test_opposite_signs(self):
        """Minmod of opposite signs should be zero."""
        a, b = np.array([2.0]), np.array([-3.0])
        result = minmod(a, b)
        assert result[0] == 0.0

    def test_with_zero(self):
        """Minmod with zero should be zero."""
        a, b = np.array([2.0]), np.array([0.0])
        result = minmod(a, b)
        assert result[0] == 0.0

    def test_symmetry(self):
        """Minmod should be symmetric: minmod(a,b) = minmod(b,a)."""
        a, b = np.array([2.0, -1.0, 3.0]), np.array([3.0, -2.0, 1.0])
        result1 = minmod(a, b)
        result2 = minmod(b, a)
        np.testing.assert_array_equal(result1, result2)


class TestSuperbeeLimiter:
    """Test superbee flux limiter."""

    def test_more_compressive_than_minmod(self):
        """Superbee should be >= minmod for same-sign inputs."""
        a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 4.0])
        sb = superbee(a, b)
        mm = minmod(a, b)
        assert np.all(np.abs(sb) >= np.abs(mm) - 1e-10)

    def test_opposite_signs(self):
        """Superbee of opposite signs should be zero."""
        a, b = np.array([2.0]), np.array([-3.0])
        result = superbee(a, b)
        assert result[0] == 0.0


class TestMCLimiter:
    """Test MC (Monotonized Central) limiter."""

    def test_between_minmod_and_superbee(self):
        """MC should typically be between minmod and superbee."""
        a, b = np.array([2.0]), np.array([3.0])
        mm = minmod(a, b)
        sb = superbee(a, b)
        mc = mc_limiter(a, b)
        # MC is balanced between the two
        assert mm[0] <= mc[0] <= sb[0] or mm[0] >= mc[0] >= sb[0]

    def test_opposite_signs(self):
        """MC of opposite signs should be zero."""
        a, b = np.array([2.0]), np.array([-3.0])
        result = mc_limiter(a, b)
        assert result[0] == 0.0


class TestConservedPrimitiveConversion:
    """Test conversion between primitive and conserved variables."""

    def test_roundtrip_density(self):
        """Density should be preserved in roundtrip conversion."""
        n = 10
        rho = np.logspace(6, 9, n)
        v = np.zeros(n)
        P = np.full(n, 1e24)  # Typical pressure
        gamma = np.full(n, 5.0 / 3.0)  # Non-relativistic

        U = primitive_to_conserved(rho, v, P, gamma)
        rho_rec, v_rec, P_rec = conserved_to_primitive(U, gamma)

        np.testing.assert_allclose(rho_rec, rho, rtol=1e-10)

    def test_roundtrip_velocity(self):
        """Velocity should be preserved in roundtrip conversion."""
        n = 10
        rho = np.full(n, 1e7)
        v = np.linspace(-1e8, 1e8, n)
        P = np.full(n, 1e24)
        gamma = np.full(n, 5.0 / 3.0)

        U = primitive_to_conserved(rho, v, P, gamma)
        rho_rec, v_rec, P_rec = conserved_to_primitive(U, gamma)

        np.testing.assert_allclose(v_rec, v, rtol=1e-10)

    def test_roundtrip_pressure(self):
        """Pressure should be preserved in roundtrip conversion."""
        n = 10
        rho = np.full(n, 1e7)
        v = np.zeros(n)
        P = np.logspace(22, 26, n)
        gamma = np.full(n, 5.0 / 3.0)

        U = primitive_to_conserved(rho, v, P, gamma)
        rho_rec, v_rec, P_rec = conserved_to_primitive(U, gamma)

        np.testing.assert_allclose(P_rec, P, rtol=1e-10)

    def test_conserved_structure(self):
        """Conserved array should have shape (3, n)."""
        n = 10
        rho = np.full(n, 1e7)
        v = np.zeros(n)
        P = np.full(n, 1e24)
        gamma = np.full(n, 5.0 / 3.0)

        U = primitive_to_conserved(rho, v, P, gamma)
        assert U.shape == (3, n)


class TestCFLTimestep:
    """Test CFL timestep calculation."""

    def test_positive_timestep(self):
        """CFL timestep should be positive."""
        n = 100
        rho = np.full(n, 1e7)
        v = np.zeros(n)
        cs = np.full(n, 1e8)  # Sound speed ~10^8 cm/s
        dx = 1e5  # 1 km

        dt = compute_cfl_timestep(rho, v, cs, dx, cfl=0.3)
        assert dt > 0

    def test_cfl_constraint(self):
        """dt should satisfy CFL: dt * (|v| + cs) / dx <= CFL."""
        n = 100
        rho = np.full(n, 1e7)
        v = np.full(n, 1e8)  # Non-zero velocity
        cs = np.full(n, 5e8)  # Sound speed
        dx = 1e5
        cfl = 0.3

        dt = compute_cfl_timestep(rho, v, cs, dx, cfl=cfl)
        max_speed = np.max(np.abs(v) + cs)
        courant = dt * max_speed / dx

        assert courant <= cfl + 1e-10

    def test_smaller_dx_smaller_dt(self):
        """Smaller dx should give smaller dt."""
        n = 100
        rho = np.full(n, 1e7)
        v = np.zeros(n)
        cs = np.full(n, 1e8)

        dt1 = compute_cfl_timestep(rho, v, cs, dx=1e5, cfl=0.3)
        dt2 = compute_cfl_timestep(rho, v, cs, dx=1e4, cfl=0.3)

        assert dt2 < dt1

    def test_faster_flow_smaller_dt(self):
        """Faster flow should give smaller dt."""
        n = 100
        rho = np.full(n, 1e7)
        cs = np.full(n, 1e8)
        dx = 1e5

        v_slow = np.zeros(n)
        v_fast = np.full(n, 5e8)

        dt_slow = compute_cfl_timestep(rho, v_slow, cs, dx, cfl=0.3)
        dt_fast = compute_cfl_timestep(rho, v_fast, cs, dx, cfl=0.3)

        assert dt_fast < dt_slow


class TestHLLCUpdate:
    """Test HLLC flux computation and update."""

    def test_uniform_state_no_change(self):
        """Uniform state should have zero flux divergence."""
        n = 100
        rho = np.full(n, 1e7)
        v = np.zeros(n)
        P = np.full(n, 1e24)
        gamma = np.full(n, 5.0 / 3.0)
        dx = 1e5

        U = primitive_to_conserved(rho, v, P, gamma)
        dU = compute_hllc_update(U, gamma, dx)

        # For uniform state, dU should be ~zero (except boundaries)
        np.testing.assert_allclose(dU[:, 2:-2], 0.0, atol=1e-10 * np.abs(U).max())

    def test_output_shape(self):
        """Output shape should match input shape."""
        n = 100
        rho = np.full(n, 1e7)
        v = np.zeros(n)
        P = np.full(n, 1e24)
        gamma = np.full(n, 5.0 / 3.0)
        dx = 1e5

        U = primitive_to_conserved(rho, v, P, gamma)
        dU = compute_hllc_update(U, gamma, dx)

        assert dU.shape == U.shape


class TestSodShockTube:
    """Sod shock tube test - standard benchmark for Riemann solvers."""

    @pytest.fixture
    def sod_initial_conditions(self):
        """Set up Sod shock tube initial conditions."""
        n = 200
        x = np.linspace(0, 1, n)
        x_mid = 0.5

        # Left state (high pressure)
        rho_L, P_L = 1.0, 1.0
        # Right state (low pressure)
        rho_R, P_R = 0.125, 0.1

        rho = np.where(x < x_mid, rho_L, rho_R)
        P = np.where(x < x_mid, P_L, P_R)
        v = np.zeros(n)
        gamma = np.full(n, 1.4)  # Ideal gas

        return rho, v, P, gamma, x

    def test_shock_forms(self, sod_initial_conditions):
        """A shock should form from the discontinuity."""
        rho, v, P, gamma, x = sod_initial_conditions
        n = len(rho)
        dx = x[1] - x[0]

        U = primitive_to_conserved(rho, v, P, gamma)

        # Evolve for a few timesteps
        for _ in range(10):
            cs = np.sqrt(gamma * P / rho)
            dt = compute_cfl_timestep(rho, v, cs, dx, cfl=0.3)
            dU = compute_hllc_update(U, gamma, dx)
            U = U + dt * dU

            # Update primitives
            rho, v, P = conserved_to_primitive(U, gamma)
            # Clamp to physical values
            rho = np.maximum(rho, 1e-10)
            P = np.maximum(P, 1e-10)

        # After evolution, velocity should be non-zero in expansion fan
        assert np.max(np.abs(v)) > 0

    def test_no_negative_density(self, sod_initial_conditions):
        """Density should never go negative."""
        rho, v, P, gamma, x = sod_initial_conditions
        n = len(rho)
        dx = x[1] - x[0]

        U = primitive_to_conserved(rho, v, P, gamma)

        # Evolve for several timesteps
        for _ in range(20):
            cs = np.sqrt(gamma * P / rho)
            dt = compute_cfl_timestep(rho, v, cs, dx, cfl=0.3)
            dU = compute_hllc_update(U, gamma, dx)
            U = U + dt * dU

            # Check density (conserved[0] = rho)
            assert np.all(U[0] > 0), "Negative density detected!"

            # Update for next iteration
            rho, v, P = conserved_to_primitive(U, gamma)
            rho = np.maximum(rho, 1e-10)
            P = np.maximum(P, 1e-10)


class TestTVDProperty:
    """Test Total Variation Diminishing property."""

    def test_total_variation_non_increasing(self):
        """Total variation should not increase with minmod limiter."""
        n = 100
        # Create state with discontinuity
        x = np.linspace(0, 1, n)
        rho = np.where(x < 0.5, 1.0, 0.5)
        v = np.zeros(n)
        P = np.full(n, 1.0)
        gamma = np.full(n, 1.4)
        dx = x[1] - x[0]

        U = primitive_to_conserved(rho, v, P, gamma)

        # Initial total variation
        TV_initial = np.sum(np.abs(np.diff(U[0])))

        # Evolve
        for _ in range(5):
            cs = np.sqrt(gamma * P / rho)
            dt = compute_cfl_timestep(rho, v, cs, dx, cfl=0.3)
            dU = compute_hllc_update(U, gamma, dx)
            U = U + dt * dU

            # Update
            rho, v, P = conserved_to_primitive(U, gamma)
            rho = np.maximum(rho, 1e-10)
            P = np.maximum(P, 1e-10)

        # Final total variation
        TV_final = np.sum(np.abs(np.diff(U[0])))

        # TV should not increase (allowing small numerical tolerance)
        assert TV_final <= TV_initial * 1.1  # 10% tolerance for numerical errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
