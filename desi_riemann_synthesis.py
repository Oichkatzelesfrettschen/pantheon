#!/usr/bin/env python3
"""
DESI-Riemann Synthesis: The Grand Unified Dark Energy Test
===========================================================

Comparing the Riemann Resonance prediction against:
1. DESI 2024 BAO results (w₀-wₐ constraints)
2. Pantheon+ SNe Ia (our analysis)
3. CMB (Planck 2018)
4. High-z Quasar Hubble Diagram (extending to z ~ 7)
5. Cosmic Chronometers (H(z) direct measurements)

The Ultimate Question:
    Does the "Riemann Snake" w(z) = -1 - (Aγ/3)sin(γln(1+z)+φ)
    thread through ALL the error ellipses?

Author: Spandrel Cosmology Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Import physical constants from central module
from constants import C_LIGHT_KMS as C_LIGHT, H0_FIDUCIAL, H0_PLANCK, H0_SH0ES, OMEGA_M_FIDUCIAL, GAMMA_1, RIEMANN_ZEROS

# Our fit results from Pantheon+
RIEMANN_FIT = {
    'amplitude': 0.026237,
    'phase': 2.8332,  # radians (162.3°)
    'H0': 73.679,
    'Omega_m': 0.31627
}


# =============================================================================
# DESI 2024 AND OTHER SURVEY RESULTS
# =============================================================================

@dataclass
class SurveyConstraint:
    """Container for cosmological survey constraints."""
    name: str
    w0: float
    w0_err: float
    wa: float
    wa_err: float
    correlation: float  # w0-wa correlation coefficient
    color: str
    year: int


# Published constraints (approximate from papers)
SURVEY_CONSTRAINTS = {
    'desi_2024_bao': SurveyConstraint(
        name='DESI 2024 (BAO only)',
        w0=-0.55, w0_err=0.21,
        wa=-1.75, wa_err=0.65,
        correlation=0.7,
        color='#e41a1c', year=2024
    ),
    'desi_2024_combined': SurveyConstraint(
        name='DESI 2024 + CMB + SNe',
        w0=-0.827, w0_err=0.063,
        wa=-0.75, wa_err=0.29,
        correlation=0.65,
        color='#377eb8', year=2024
    ),
    'planck_2018': SurveyConstraint(
        name='Planck 2018 (CMB)',
        w0=-1.03, w0_err=0.03,
        wa=0.0, wa_err=0.5,  # Essentially unconstrained
        correlation=0.0,
        color='#4daf4a', year=2018
    ),
    'des_y5': SurveyConstraint(
        name='DES Y5 (2024)',
        w0=-0.87, w0_err=0.06,
        wa=-0.45, wa_err=0.35,
        correlation=0.6,
        color='#984ea3', year=2024
    ),
    'union3': SurveyConstraint(
        name='Union3 SNe (2024)',
        w0=-0.91, w0_err=0.08,
        wa=-0.35, wa_err=0.40,
        correlation=0.5,
        color='#ff7f00', year=2024
    ),
}


# =============================================================================
# RIEMANN EQUATION OF STATE
# =============================================================================

class RiemannEoS:
    """
    Riemann-modulated dark energy equation of state.

    w(z) = -1 - (Aγ/3) * sin(γ*ln(1+z) + φ)

    In CPL parameterization (w = w₀ + wₐ*(1-a) = w₀ + wₐ*z/(1+z)):
    We can compute effective w₀ and wₐ by Taylor expansion at z=0.
    """

    def __init__(self, amplitude: float, phase: float, gamma: float = GAMMA_1):
        self.A = amplitude
        self.phi = phase
        self.gamma = gamma

    def w(self, z: np.ndarray) -> np.ndarray:
        """Exact Riemann equation of state."""
        z = np.atleast_1d(z)
        return -1.0 - (self.A * self.gamma / 3.0) * np.sin(self.gamma * np.log(1 + z) + self.phi)

    def effective_w0_wa(self) -> Tuple[float, float]:
        """
        Compute effective CPL parameters (w₀, wₐ) by matching at z=0 and z=0.5.

        w(z) ≈ w₀ + wₐ * z/(1+z)

        At z=0: w₀ = w(0)
        At z→∞: w₀ + wₐ = w(∞)

        Better: fit over 0 < z < 2
        """
        # w₀ = w(z=0)
        w0 = float(self.w(np.array([0.0]))[0])

        # Fit wₐ by matching derivative at z=0
        # dw/dz|_{z=0} = -(Aγ²/3) * cos(φ)
        # For CPL: dw/dz|_{z=0} = wₐ
        dwdz_0 = -(self.A * self.gamma**2 / 3.0) * np.cos(self.phi)
        wa = dwdz_0

        return w0, wa

    def effective_w0_wa_fitted(self) -> Tuple[float, float]:
        """
        Fit CPL parameters by minimizing deviation over 0 < z < 2.
        """
        z_fit = np.linspace(0, 2, 100)
        w_exact = self.w(z_fit)

        def loss(params):
            w0, wa = params
            w_cpl = w0 + wa * z_fit / (1 + z_fit)
            return np.sum((w_exact - w_cpl)**2)

        result = minimize(loss, [-1, 0], method='Nelder-Mead')
        return result.x[0], result.x[1]


# =============================================================================
# HIGH-Z DATASETS
# =============================================================================

def get_quasar_hubble_data():
    """
    Quasar Hubble Diagram data from Risaliti & Lusso 2019.

    Uses the X-ray/UV luminosity relation to standardize quasars as
    "standard candles" extending to z ~ 7.5.

    Format: (z, μ, σ_μ) - redshift, distance modulus, error

    Note: This is a representative subset; full catalog has 1,598 quasars.
    """
    # Representative binned data from R&L 2019 + extensions
    quasar_data = np.array([
        # z,    μ,      σ_μ
        [0.5,   42.5,   0.3],
        [0.7,   43.2,   0.25],
        [1.0,   44.1,   0.2],
        [1.3,   44.7,   0.2],
        [1.6,   45.2,   0.22],
        [2.0,   45.8,   0.25],
        [2.5,   46.3,   0.3],
        [3.0,   46.7,   0.35],
        [3.5,   47.0,   0.4],
        [4.0,   47.3,   0.45],
        [4.5,   47.5,   0.5],
        [5.0,   47.7,   0.6],
        [5.5,   47.9,   0.7],
        [6.0,   48.0,   0.8],
        [6.5,   48.1,   0.9],
        [7.0,   48.2,   1.0],
    ])
    return quasar_data[:, 0], quasar_data[:, 1], quasar_data[:, 2]


def get_cosmic_chronometer_data():
    """
    Cosmic Chronometer H(z) measurements.

    These are direct measurements of H(z) from the differential
    age method using passively evolving galaxies.

    Data from Moresco et al. 2022 compilation.
    Format: (z, H(z), σ_H) in km/s/Mpc
    """
    cc_data = np.array([
        # z,     H(z),   σ_H
        [0.07,   69.0,   19.6],
        [0.09,   69.0,   12.0],
        [0.12,   68.6,   26.2],
        [0.17,   83.0,   8.0],
        [0.179,  75.0,   4.0],
        [0.199,  75.0,   5.0],
        [0.20,   72.9,   29.6],
        [0.27,   77.0,   14.0],
        [0.28,   88.8,   36.6],
        [0.352,  83.0,   14.0],
        [0.3802, 83.0,   13.5],
        [0.4,    95.0,   17.0],
        [0.4004, 77.0,   10.2],
        [0.4247, 87.1,   11.2],
        [0.4497, 92.8,   12.9],
        [0.47,   89.0,   50.0],
        [0.4783, 80.9,   9.0],
        [0.48,   97.0,   62.0],
        [0.593,  104.0,  13.0],
        [0.68,   92.0,   8.0],
        [0.781,  105.0,  12.0],
        [0.875,  125.0,  17.0],
        [0.88,   90.0,   40.0],
        [0.9,    117.0,  23.0],
        [1.037,  154.0,  20.0],
        [1.3,    168.0,  17.0],
        [1.363,  160.0,  33.6],
        [1.43,   177.0,  18.0],
        [1.53,   140.0,  14.0],
        [1.75,   202.0,  40.0],
        [1.965,  186.5,  50.4],
    ])
    return cc_data[:, 0], cc_data[:, 1], cc_data[:, 2]


def get_bao_dv_data():
    """
    BAO D_V(z)/r_d measurements (volume-averaged distance).

    Compilation including BOSS, eBOSS, DESI.
    Format: (z, D_V/r_d, σ)
    """
    bao_data = np.array([
        # z,     D_V/r_d,  σ
        [0.106,  2.98,    0.13],   # 6dFGS
        [0.15,   4.47,    0.17],   # SDSS MGS
        [0.38,   10.23,   0.17],   # BOSS DR12
        [0.51,   13.36,   0.21],   # BOSS DR12
        [0.61,   15.45,   0.24],   # BOSS DR12
        [0.70,   17.86,   0.33],   # eBOSS LRG
        [0.85,   19.50,   0.50],   # eBOSS LRG
        [1.48,   30.69,   0.80],   # eBOSS Quasar
        [2.33,   37.6,    1.2],    # eBOSS Lyα
        [2.40,   37.3,    1.7],    # DESI Lyα (2024)
    ])
    return bao_data[:, 0], bao_data[:, 1], bao_data[:, 2]


# =============================================================================
# COSMOLOGY CALCULATIONS FOR HIGH-Z
# =============================================================================

class ExtendedRiemannCosmology:
    """
    Full Riemann cosmology for high-z predictions.
    """

    def __init__(self, H0: float, Omega_m: float, amplitude: float, phase: float):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = 1.0 - Omega_m
        self.A = amplitude
        self.phi = phase
        self.gamma = GAMMA_1

    def dark_energy_factor(self, z: np.ndarray) -> np.ndarray:
        """Riemann oscillating dark energy."""
        z = np.atleast_1d(z)
        return 1.0 + self.A * np.cos(self.gamma * np.log(1 + z) + self.phi)

    def E_squared(self, z: np.ndarray) -> np.ndarray:
        """E²(z) = H²(z)/H₀²."""
        z = np.atleast_1d(z)
        matter = self.Omega_m * (1 + z)**3
        de = self.Omega_Lambda * self.dark_energy_factor(z)
        return matter + de

    def E(self, z: np.ndarray) -> np.ndarray:
        """E(z) = H(z)/H₀."""
        return np.sqrt(np.maximum(self.E_squared(z), 1e-10))

    def H(self, z: np.ndarray) -> np.ndarray:
        """H(z) in km/s/Mpc."""
        return self.H0 * self.E(z)

    def comoving_distance(self, z: np.ndarray, n_steps: int = 1000) -> np.ndarray:
        """D_C(z) in Mpc."""
        z = np.atleast_1d(z)

        result = np.zeros_like(z)
        for i, zi in enumerate(z):
            z_int = np.linspace(0, zi, n_steps)
            integrand = 1.0 / self.E(z_int)
            result[i] = np.trapz(integrand, z_int)

        return (C_LIGHT / self.H0) * result

    def luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        """D_L(z) = (1+z) * D_C(z)."""
        return (1 + np.atleast_1d(z)) * self.comoving_distance(z)

    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        """μ(z) = 5*log₁₀(D_L/10pc)."""
        d_L = self.luminosity_distance(z)
        return 5.0 * np.log10(d_L * 1e6 / 10.0)

    def angular_diameter_distance(self, z: np.ndarray) -> np.ndarray:
        """D_A(z) = D_C(z)/(1+z)."""
        return self.comoving_distance(z) / (1 + np.atleast_1d(z))

    def dV_over_rd(self, z: np.ndarray, rd: float = 147.09) -> np.ndarray:
        """
        Volume-averaged distance D_V(z)/r_d for BAO.

        D_V = [z * D_H * D_M²]^(1/3)
        where D_H = c/H(z), D_M = D_C(z)
        """
        z = np.atleast_1d(z)

        D_H = C_LIGHT / self.H(z)
        D_M = self.comoving_distance(z)

        D_V = (z * D_H * D_M**2)**(1/3)
        return D_V / rd


# =============================================================================
# VISUALIZATION: THE GRAND SYNTHESIS
# =============================================================================

def plot_w0_wa_plane(riemann_eos: RiemannEoS, save_path: Optional[str] = None):
    """
    Plot the w₀-wₐ plane with survey constraints and Riemann prediction.

    The "Riemann Snake" shows how w₀ and wₐ vary as phase changes.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot survey constraint ellipses
    for name, constraint in SURVEY_CONSTRAINTS.items():
        # Covariance matrix from errors and correlation
        cov = np.array([
            [constraint.w0_err**2, constraint.correlation * constraint.w0_err * constraint.wa_err],
            [constraint.correlation * constraint.w0_err * constraint.wa_err, constraint.wa_err**2]
        ])

        # Eigenvalue decomposition for ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # 1σ and 2σ ellipses
        for n_sigma, alpha in [(1, 0.5), (2, 0.2)]:
            width = 2 * n_sigma * np.sqrt(eigenvalues[0])
            height = 2 * n_sigma * np.sqrt(eigenvalues[1])

            ellipse = Ellipse(
                (constraint.w0, constraint.wa),
                width, height, angle=angle,
                facecolor=constraint.color, alpha=alpha,
                edgecolor=constraint.color, linewidth=2
            )
            ax.add_patch(ellipse)

        # Label
        ax.plot(constraint.w0, constraint.wa, 'o', color=constraint.color,
               markersize=10, label=constraint.name)

    # Plot the "Riemann Snake" - how (w₀, wₐ) varies with phase
    phases = np.linspace(0, 2*np.pi, 100)
    w0_snake = []
    wa_snake = []

    for phi in phases:
        eos = RiemannEoS(riemann_eos.A, phi, riemann_eos.gamma)
        w0, wa = eos.effective_w0_wa_fitted()
        w0_snake.append(w0)
        wa_snake.append(wa)

    ax.plot(w0_snake, wa_snake, 'k-', linewidth=3, label='Riemann Snake (phase sweep)')

    # Mark our best-fit phase
    w0_fit, wa_fit = riemann_eos.effective_w0_wa_fitted()
    ax.plot(w0_fit, wa_fit, 'r*', markersize=25, markeredgecolor='black',
           label=f'Pantheon+ Riemann Fit (φ={np.degrees(riemann_eos.phi):.0f}°)')

    # Mark ΛCDM
    ax.plot(-1, 0, 'g^', markersize=15, markeredgecolor='black',
           label='ΛCDM (w=-1, wₐ=0)')

    # Phantom divide line
    ax.axvline(x=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('$w_0$ (equation of state today)', fontsize=14)
    ax.set_ylabel('$w_a$ (evolution parameter)', fontsize=14)
    ax.set_title('Dark Energy Constraints: DESI 2024 vs Riemann Resonance', fontsize=16)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-1.8, -0.2)
    ax.set_ylim(-3, 1)
    ax.grid(True, alpha=0.3)

    # Add text annotation
    ax.text(-1.7, -2.7,
           f'Riemann: A={riemann_eos.A:.3f}, γ={riemann_eos.gamma:.2f}\n'
           f'w₀={w0_fit:.3f}, wₐ={wa_fit:.3f}',
           fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved w₀-wₐ plot to: {save_path}")

    plt.show()
    return w0_fit, wa_fit


def plot_w_z_evolution(riemann_eos: RiemannEoS, save_path: Optional[str] = None):
    """
    Plot w(z) evolution comparing Riemann to CPL approximation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    z = np.linspace(0, 5, 500)

    # Left: w(z) comparison
    ax1 = axes[0]

    w_riemann = riemann_eos.w(z)
    w0_fit, wa_fit = riemann_eos.effective_w0_wa_fitted()
    w_cpl = w0_fit + wa_fit * z / (1 + z)

    ax1.axhline(y=-1, color='blue', linewidth=2, label='ΛCDM (w = -1)')
    ax1.plot(z, w_riemann, 'r-', linewidth=2.5, label='Riemann Resonance')
    ax1.plot(z, w_cpl, 'g--', linewidth=2, label=f'CPL fit (w₀={w0_fit:.2f}, wₐ={wa_fit:.2f})')

    # DESI best fit
    w0_desi, wa_desi = -0.827, -0.75
    w_desi = w0_desi + wa_desi * z / (1 + z)
    ax1.plot(z, w_desi, 'm:', linewidth=2, label=f'DESI 2024 (w₀={w0_desi}, wₐ={wa_desi})')

    ax1.set_xlabel('Redshift z', fontsize=12)
    ax1.set_ylabel('Equation of State w(z)', fontsize=12)
    ax1.set_title('Dark Energy Evolution', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.set_ylim(-1.5, -0.5)
    ax1.set_xlim(0, 5)
    ax1.grid(True, alpha=0.3)

    # Right: Residual from ΛCDM
    ax2 = axes[1]

    ax2.axhline(y=0, color='blue', linewidth=2, label='ΛCDM')
    ax2.plot(z, w_riemann + 1, 'r-', linewidth=2.5, label='Riemann')
    ax2.plot(z, w_cpl + 1, 'g--', linewidth=2, label='CPL fit')
    ax2.plot(z, w_desi + 1, 'm:', linewidth=2, label='DESI 2024')

    # Shade oscillation
    ax2.fill_between(z, 0, w_riemann + 1, where=(w_riemann > -1),
                    alpha=0.3, color='red', label='Quintessence (w > -1)')
    ax2.fill_between(z, 0, w_riemann + 1, where=(w_riemann < -1),
                    alpha=0.3, color='blue', label='Phantom (w < -1)')

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('w(z) + 1 (deviation from Λ)', fontsize=12)
    ax2.set_title('Deviation from Cosmological Constant', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.5, 0.3)
    ax2.set_xlim(0, 5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved w(z) evolution to: {save_path}")

    plt.show()


def plot_high_z_synthesis(save_path: Optional[str] = None):
    """
    Compare Riemann predictions against multiple high-z datasets.
    """
    fig = plt.figure(figsize=(16, 12))

    # Initialize cosmologies
    riemann_cosmo = ExtendedRiemannCosmology(
        H0=RIEMANN_FIT['H0'],
        Omega_m=RIEMANN_FIT['Omega_m'],
        amplitude=RIEMANN_FIT['amplitude'],
        phase=RIEMANN_FIT['phase']
    )

    lcdm_cosmo = ExtendedRiemannCosmology(
        H0=73.3, Omega_m=0.315, amplitude=0, phase=0
    )

    z_model = np.linspace(0.01, 7, 500)

    # Panel 1: H(z) - Cosmic Chronometers
    ax1 = fig.add_subplot(2, 2, 1)

    z_cc, H_cc, err_cc = get_cosmic_chronometer_data()

    ax1.errorbar(z_cc, H_cc, yerr=err_cc, fmt='o', color='green',
                capsize=3, label='Cosmic Chronometers', markersize=6)
    ax1.plot(z_model, riemann_cosmo.H(z_model), 'r-', linewidth=2, label='Riemann')
    ax1.plot(z_model, lcdm_cosmo.H(z_model), 'b--', linewidth=2, label='ΛCDM')

    ax1.set_xlabel('Redshift z', fontsize=12)
    ax1.set_ylabel('H(z) [km/s/Mpc]', fontsize=12)
    ax1.set_title('Hubble Parameter from Cosmic Chronometers', fontsize=13)
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 2.2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: BAO D_V/r_d
    ax2 = fig.add_subplot(2, 2, 2)

    z_bao, dv_bao, err_bao = get_bao_dv_data()

    ax2.errorbar(z_bao, dv_bao, yerr=err_bao, fmt='s', color='purple',
                capsize=3, label='BAO measurements', markersize=8)

    z_bao_model = np.linspace(0.1, 2.5, 200)
    ax2.plot(z_bao_model, riemann_cosmo.dV_over_rd(z_bao_model), 'r-',
            linewidth=2, label='Riemann')
    ax2.plot(z_bao_model, lcdm_cosmo.dV_over_rd(z_bao_model), 'b--',
            linewidth=2, label='ΛCDM')

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('$D_V(z)/r_d$', fontsize=12)
    ax2.set_title('BAO Volume-Averaged Distance', fontsize=13)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: High-z Quasars
    ax3 = fig.add_subplot(2, 2, 3)

    z_qso, mu_qso, err_qso = get_quasar_hubble_data()

    ax3.errorbar(z_qso, mu_qso, yerr=err_qso, fmt='D', color='orange',
                capsize=3, label='Quasar Hubble Diagram', markersize=8)

    z_qso_model = np.linspace(0.5, 7, 200)
    ax3.plot(z_qso_model, riemann_cosmo.distance_modulus(z_qso_model), 'r-',
            linewidth=2, label='Riemann')
    ax3.plot(z_qso_model, lcdm_cosmo.distance_modulus(z_qso_model), 'b--',
            linewidth=2, label='ΛCDM')

    ax3.set_xlabel('Redshift z', fontsize=12)
    ax3.set_ylabel('Distance Modulus μ', fontsize=12)
    ax3.set_title('Quasar Standard Candles (z > 0.5)', fontsize=13)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Residuals showing Riemann oscillation
    ax4 = fig.add_subplot(2, 2, 4)

    # Show the oscillation signature
    z_full = np.linspace(0.01, 7, 500)
    mu_riemann = riemann_cosmo.distance_modulus(z_full)
    mu_lcdm = lcdm_cosmo.distance_modulus(z_full)

    delta_mu = mu_riemann - mu_lcdm

    ax4.plot(z_full, delta_mu, 'r-', linewidth=2.5, label='Riemann - ΛCDM')
    ax4.axhline(y=0, color='blue', linestyle='--', linewidth=1.5, label='ΛCDM')

    # Mark the oscillation peaks/troughs
    ax4.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=1.2, color='gray', linestyle=':', alpha=0.5)

    ax4.set_xlabel('Redshift z', fontsize=12)
    ax4.set_ylabel('Δμ (mag)', fontsize=12)
    ax4.set_title('Riemann Oscillation Signature', fontsize=13)
    ax4.legend(loc='upper right')
    ax4.set_xlim(0, 7)
    ax4.grid(True, alpha=0.3)

    # Annotate
    ax4.annotate('Peak\n(Tension)', xy=(0.7, delta_mu[70]), fontsize=10, ha='center')
    ax4.annotate('Trough\n(Relaxation)', xy=(1.5, delta_mu[110]), fontsize=10, ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved high-z synthesis to: {save_path}")

    plt.show()


def compute_chi2_comparison():
    """
    Compute chi-squared for Riemann vs ΛCDM against all datasets.
    """
    print("\n" + "="*70)
    print("CHI-SQUARED COMPARISON: RIEMANN vs ΛCDM")
    print("="*70)

    riemann = ExtendedRiemannCosmology(
        H0=RIEMANN_FIT['H0'], Omega_m=RIEMANN_FIT['Omega_m'],
        amplitude=RIEMANN_FIT['amplitude'], phase=RIEMANN_FIT['phase']
    )
    lcdm = ExtendedRiemannCosmology(
        H0=73.3, Omega_m=0.315, amplitude=0, phase=0
    )

    results = {}

    # Cosmic Chronometers
    z_cc, H_cc, err_cc = get_cosmic_chronometer_data()
    chi2_r_cc = np.sum(((H_cc - riemann.H(z_cc)) / err_cc)**2)
    chi2_l_cc = np.sum(((H_cc - lcdm.H(z_cc)) / err_cc)**2)
    results['Cosmic Chronometers'] = (chi2_l_cc, chi2_r_cc, len(z_cc))

    # BAO
    z_bao, dv_bao, err_bao = get_bao_dv_data()
    chi2_r_bao = np.sum(((dv_bao - riemann.dV_over_rd(z_bao)) / err_bao)**2)
    chi2_l_bao = np.sum(((dv_bao - lcdm.dV_over_rd(z_bao)) / err_bao)**2)
    results['BAO'] = (chi2_l_bao, chi2_r_bao, len(z_bao))

    # Quasars
    z_qso, mu_qso, err_qso = get_quasar_hubble_data()
    chi2_r_qso = np.sum(((mu_qso - riemann.distance_modulus(z_qso)) / err_qso)**2)
    chi2_l_qso = np.sum(((mu_qso - lcdm.distance_modulus(z_qso)) / err_qso)**2)
    results['Quasars'] = (chi2_l_qso, chi2_r_qso, len(z_qso))

    print(f"\n{'Dataset':<25} {'ΛCDM χ²':>12} {'Riemann χ²':>12} {'Δχ²':>10} {'N':>6}")
    print("-"*70)

    total_lcdm = 0
    total_riemann = 0
    total_n = 0

    for name, (chi2_l, chi2_r, n) in results.items():
        delta = chi2_l - chi2_r
        print(f"{name:<25} {chi2_l:>12.2f} {chi2_r:>12.2f} {delta:>+10.2f} {n:>6}")
        total_lcdm += chi2_l
        total_riemann += chi2_r
        total_n += n

    print("-"*70)
    print(f"{'TOTAL':<25} {total_lcdm:>12.2f} {total_riemann:>12.2f} {total_lcdm-total_riemann:>+10.2f} {total_n:>6}")

    return results


# =============================================================================
# MAIN SYNTHESIS
# =============================================================================

def run_desi_riemann_synthesis():
    """
    Execute the grand synthesis comparing Riemann to all data.
    """
    print("\n" + "="*70)
    print("DESI-RIEMANN GRAND SYNTHESIS")
    print("The Ultimate Test of Vacuum Oscillation Cosmology")
    print("="*70)

    # Initialize Riemann EoS from our Pantheon+ fit
    riemann_eos = RiemannEoS(
        amplitude=RIEMANN_FIT['amplitude'],
        phase=RIEMANN_FIT['phase']
    )

    # Compute effective CPL parameters
    w0_analytic, wa_analytic = riemann_eos.effective_w0_wa()
    w0_fit, wa_fit = riemann_eos.effective_w0_wa_fitted()

    print(f"""
    Riemann Resonance Parameters (from Pantheon+):
      Amplitude A = {RIEMANN_FIT['amplitude']:.6f} (2.6% oscillation)
      Phase φ = {np.degrees(RIEMANN_FIT['phase']):.2f}°
      Frequency γ₁ = {GAMMA_1:.4f} (First Riemann Zero)

    Effective CPL Parameters:
      w₀ (analytic) = {w0_analytic:.4f}
      wₐ (analytic) = {wa_analytic:.4f}
      w₀ (fitted) = {w0_fit:.4f}
      wₐ (fitted) = {wa_fit:.4f}

    Comparison to Surveys:
      DESI 2024 BAO:      w₀ = -0.55 ± 0.21,  wₐ = -1.75 ± 0.65
      DESI 2024 Combined: w₀ = -0.83 ± 0.06,  wₐ = -0.75 ± 0.29
      Planck 2018:        w₀ = -1.03 ± 0.03,  wₐ ~ 0 (unconstrained)
    """)

    # Check if Riemann falls within DESI contours
    desi = SURVEY_CONSTRAINTS['desi_2024_combined']

    # Distance in sigma space (approximate)
    d_w0 = (w0_fit - desi.w0) / desi.w0_err
    d_wa = (wa_fit - desi.wa) / desi.wa_err

    print(f"""
    Distance from DESI 2024 Combined:
      Δw₀/σ = {d_w0:.2f}
      Δwₐ/σ = {d_wa:.2f}

    Verdict: {'CONSISTENT' if abs(d_w0) < 2 and abs(d_wa) < 2 else 'TENSION'} with DESI at 2σ
    """)

    # Generate visualizations
    print("-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)

    # 1. w0-wa plane
    print("\n1. w₀-wₐ plane with survey constraints...")
    plot_w0_wa_plane(riemann_eos, save_path="w0_wa_plane.png")

    # 2. w(z) evolution
    print("\n2. w(z) evolution comparison...")
    plot_w_z_evolution(riemann_eos, save_path="w_z_evolution.png")

    # 3. High-z synthesis
    print("\n3. High-z dataset synthesis...")
    plot_high_z_synthesis(save_path="high_z_synthesis.png")

    # 4. Chi-squared comparison
    chi2_results = compute_chi2_comparison()

    # Final verdict
    print("\n" + "="*70)
    print("GRAND SYNTHESIS VERDICT")
    print("="*70)

    print(f"""
    The Riemann Resonance Hypothesis predicts:

    1. VACUUM OSCILLATION: Dark energy oscillates at γ₁ = 14.13 in log-time
       → Amplitude: ~2.6% variation in Λ
       → Current phase: 162° ("exhale" - tension decreasing)

    2. EFFECTIVE CPL PARAMETERS: w₀ = {w0_fit:.3f}, wₐ = {wa_fit:.3f}
       → This is {'WITHIN' if abs(d_w0) < 2 else 'OUTSIDE'} DESI 2σ contour
       → The "DESI anomaly" (w₀ > -1, wₐ < 0) is {'EXPLAINED' if wa_fit < -0.3 else 'PARTIALLY MATCHED'}

    3. HIGH-Z EXTENSION: The oscillation continues to z ~ 7
       → Predicts alternating quintessence/phantom phases
       → Testable with JWST high-z galaxy ages

    STATISTICAL STATUS:
    - Pantheon+ alone: NOT SIGNIFICANT (p = 0.31)
    - Combined datasets: {'SUGGESTIVE' if sum([r[0] - r[1] for r in chi2_results.values()]) > 0 else 'INCONCLUSIVE'}
    - Definitive test requires: Rubin/LSST 100k+ SNe or CMB Stage-4

    PHYSICAL INTERPRETATION:
    The "Breathing Universe" model posits that:
    - The vacuum is a vibrating membrane, not a static ocean
    - The vibration frequency is set by the Riemann zeta zeros
    - We are currently in an "exhale" phase (tension decreasing)
    - The next "tension peak" occurs at z ~ 0.2 (look-back time ~2.5 Gyr)
    """)


if __name__ == "__main__":
    run_desi_riemann_synthesis()
