"""
Tests for the Pantheon+SH0ES data interface.

Run with: pytest tests/
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from spandrel.core.data_interface import PantheonData, load_pantheon, DataStats


class TestPantheonData:
    """Test suite for PantheonData class."""

    @pytest.fixture
    def data(self):
        """Load data fixture."""
        return PantheonData()

    def test_loads_successfully(self, data):
        """Data should load without errors."""
        assert data is not None
        assert len(data) > 0

    def test_expected_count(self, data):
        """Should have approximately 1700 supernovae."""
        assert 1600 < len(data) < 1800

    def test_stats_populated(self, data):
        """Stats should be computed."""
        assert isinstance(data.stats, DataStats)
        assert data.stats.total_valid > 0
        assert data.stats.z_min > 0
        assert data.stats.z_max > data.stats.z_min

    def test_redshift_range(self, data):
        """Redshift should be within expected bounds."""
        z, _, _ = data.get_cosmology_data()
        assert z.min() >= 0.001
        assert z.max() <= 2.5
        assert np.all(z > 0)

    def test_distance_modulus_positive(self, data):
        """Distance modulus should be positive."""
        _, mu, _ = data.get_cosmology_data()
        assert np.all(mu > 0)

    def test_errors_positive(self, data):
        """Measurement errors should be positive."""
        _, _, mu_err = data.get_cosmology_data()
        assert np.all(mu_err > 0)

    def test_no_nan_values(self, data):
        """No NaN values in cosmology data."""
        z, mu, mu_err = data.get_cosmology_data()
        assert not np.isnan(z).any()
        assert not np.isnan(mu).any()
        assert not np.isnan(mu_err).any()

    def test_sorted_by_redshift(self, data):
        """Data should be sorted by redshift."""
        z, _, _ = data.get_cosmology_data()
        assert np.all(np.diff(z) >= 0)

    def test_calibrator_subset(self, data):
        """Calibrator subset should exist."""
        calibrators = data.get_calibrator_subset()
        assert len(calibrators) > 0
        assert len(calibrators) < len(data)
        assert (calibrators['IS_CALIBRATOR'] == 1).all()

    def test_hubble_flow_subset(self, data):
        """Hubble flow subset should exclude local universe."""
        hf = data.get_hubble_flow_subset(z_cut=0.01)
        assert len(hf) > 0
        assert len(hf) < len(data)
        assert (hf['zHD'] > 0.01).all()

    def test_validation_report(self, data):
        """Validation should return complete report."""
        report = data.validate()
        assert 'total_entries' in report
        assert 'rejected_entries' in report
        assert 'has_nan_z' in report
        assert report['has_nan_z'] == False
        assert report['has_nan_mu'] == False


class TestLoadFunction:
    """Test convenience load function."""

    def test_returns_tuple(self):
        """Should return tuple of three arrays."""
        result = load_pantheon()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_arrays_same_length(self):
        """All arrays should have same length."""
        z, mu, mu_err = load_pantheon()
        assert len(z) == len(mu) == len(mu_err)

    def test_respects_z_limits(self):
        """Should respect z_min and z_max parameters."""
        z, _, _ = load_pantheon(z_min=0.1, z_max=1.0)
        assert z.min() >= 0.1
        assert z.max() <= 1.0


class TestConstants:
    """Test constants module."""

    def test_imports_cleanly(self):
        """Constants should import without error."""
        from spandrel.core.constants import C_LIGHT, M_SUN, GAMMA_1
        assert C_LIGHT > 0
        assert M_SUN > 0
        assert GAMMA_1 > 14 and GAMMA_1 < 15


class TestDDTSolver:
    """Basic tests for DDT solver modules."""

    def test_eos_imports(self):
        """EOS module should import."""
        from spandrel.ddt.eos_white_dwarf import eos_from_rho_T
        assert callable(eos_from_rho_T)

    def test_flux_imports(self):
        """Flux module should import."""
        from spandrel.ddt.flux_hllc import compute_hllc_update
        assert callable(compute_hllc_update)

    def test_reaction_imports(self):
        """Reaction module should import."""
        from spandrel.ddt.reaction_carbon import c12_c12_rate
        assert callable(c12_c12_rate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
