"""
Clean data interface for Pantheon+SH0ES supernova dataset.

This module provides a unified, validated interface to the Pantheon+SH0ES
Type Ia supernova compilation used for cosmological analysis.

Usage:
    from data_interface import PantheonData

    data = PantheonData()
    z, mu, mu_err = data.get_cosmology_data()

    # Or access full DataFrame
    df = data.dataframe
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd

# Default data file path
DATA_FILE = Path(__file__).parent / "Pantheon+SH0ES.dat"


@dataclass
class DataStats:
    """Statistics about loaded data."""
    total_raw: int
    total_valid: int
    z_min: float
    z_max: float
    mu_min: float
    mu_max: float
    n_surveys: int
    n_calibrators: int


class PantheonData:
    """
    Clean interface to Pantheon+SH0ES supernova dataset.

    Attributes:
        dataframe: Full pandas DataFrame with all columns
        stats: DataStats object with summary statistics

    Methods:
        get_cosmology_data(): Returns (z, mu, mu_err) arrays for fitting
        get_calibrator_subset(): Returns calibrator SNe only
        get_hubble_flow_subset(): Returns z > 0.01 subset
        validate(): Run validation checks on loaded data
    """

    def __init__(
        self,
        filepath: Optional[Path] = None,
        z_min: float = 0.001,
        z_max: float = 2.5
    ):
        """
        Load and validate Pantheon+SH0ES dataset.

        Args:
            filepath: Path to data file. Default: Pantheon+SH0ES.dat
            z_min: Minimum redshift cutoff
            z_max: Maximum redshift cutoff
        """
        self.filepath = Path(filepath) if filepath else DATA_FILE
        self.z_min = z_min
        self.z_max = z_max

        self._load_data()
        self._validate_data()
        self._compute_stats()

    def _load_data(self) -> None:
        """Load raw data from file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

        # Load with whitespace delimiter
        self._raw_df = pd.read_csv(
            self.filepath,
            sep=r'\s+',
            comment='#'
        )
        self._total_raw = len(self._raw_df)

    def _validate_data(self) -> None:
        """Apply quality cuts and validation."""
        df = self._raw_df.copy()

        # Extract key columns
        z = df['zHD'].values
        mu = df['MU_SH0ES'].values
        mu_err = df['MU_SH0ES_ERR_DIAG'].values

        # Quality mask
        mask = (
            (z > self.z_min) &
            (z < self.z_max) &
            (mu > 0) &
            np.isfinite(mu) &
            np.isfinite(mu_err) &
            (mu_err > 0)
        )

        # Apply mask
        self.dataframe = df[mask].copy()
        self.dataframe = self.dataframe.sort_values('zHD').reset_index(drop=True)

        self._validation_mask = mask
        self._n_rejected = (~mask).sum()

    def _compute_stats(self) -> None:
        """Compute summary statistics."""
        df = self.dataframe
        self.stats = DataStats(
            total_raw=self._total_raw,
            total_valid=len(df),
            z_min=df['zHD'].min(),
            z_max=df['zHD'].max(),
            mu_min=df['MU_SH0ES'].min(),
            mu_max=df['MU_SH0ES'].max(),
            n_surveys=df['IDSURVEY'].nunique(),
            n_calibrators=(df['IS_CALIBRATOR'] == 1).sum()
        )

    def get_cosmology_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get arrays for cosmological fitting.

        Returns:
            Tuple of (redshift, distance_modulus, distance_modulus_error)
        """
        df = self.dataframe
        return (
            df['zHD'].values.astype(np.float64),
            df['MU_SH0ES'].values.astype(np.float64),
            df['MU_SH0ES_ERR_DIAG'].values.astype(np.float64)
        )

    def get_calibrator_subset(self) -> pd.DataFrame:
        """Get Cepheid-calibrated supernova subset."""
        return self.dataframe[self.dataframe['IS_CALIBRATOR'] == 1].copy()

    def get_hubble_flow_subset(self, z_cut: float = 0.01) -> pd.DataFrame:
        """Get Hubble flow subset (excludes local universe)."""
        return self.dataframe[self.dataframe['zHD'] > z_cut].copy()

    def get_light_curve_params(self) -> pd.DataFrame:
        """Get SALT2 light curve parameters."""
        cols = ['CID', 'zHD', 'x1', 'x1ERR', 'c', 'cERR', 'mB', 'mBERR']
        return self.dataframe[cols].copy()

    def get_host_properties(self) -> pd.DataFrame:
        """Get host galaxy properties."""
        cols = ['CID', 'zHD', 'HOST_RA', 'HOST_DEC', 'HOST_LOGMASS', 'HOST_LOGMASS_ERR']
        return self.dataframe[cols].copy()

    def validate(self) -> dict:
        """
        Run validation checks and return report.

        Returns:
            Dictionary with validation results
        """
        df = self.dataframe
        z, mu, mu_err = self.get_cosmology_data()

        checks = {
            'total_entries': len(df),
            'rejected_entries': self._n_rejected,
            'redshift_range': (z.min(), z.max()),
            'mu_range': (mu.min(), mu.max()),
            'has_nan_z': np.isnan(z).any(),
            'has_nan_mu': np.isnan(mu).any(),
            'has_negative_errors': (mu_err <= 0).any(),
            'surveys_present': df['IDSURVEY'].unique().tolist(),
            'calibrator_count': (df['IS_CALIBRATOR'] == 1).sum(),
            'median_z': np.median(z),
            'median_mu_err': np.median(mu_err),
        }

        return checks

    def __repr__(self) -> str:
        return (
            f"PantheonData(n={self.stats.total_valid}, "
            f"z=[{self.stats.z_min:.4f}, {self.stats.z_max:.4f}], "
            f"surveys={self.stats.n_surveys})"
        )

    def __len__(self) -> int:
        return len(self.dataframe)


def load_pantheon(
    filepath: Optional[Path] = None,
    z_min: float = 0.001,
    z_max: float = 2.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load Pantheon data arrays directly.

    Returns:
        Tuple of (z, mu, mu_err) numpy arrays
    """
    data = PantheonData(filepath=filepath, z_min=z_min, z_max=z_max)
    return data.get_cosmology_data()


if __name__ == "__main__":
    # Quick test
    print("Loading Pantheon+SH0ES data...")
    data = PantheonData()
    print(data)
    print(f"\nValidation report:")
    for key, val in data.validate().items():
        print(f"  {key}: {val}")
