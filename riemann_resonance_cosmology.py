#!/usr/bin/env python3
"""
Riemann Resonance Cosmology
===========================

Testing the hypothesis that Dark Energy oscillates at frequencies
determined by the Riemann Zeta zeros.

The Unsolved Equation:
    Λ(z) = Λ₀ [1 + A·cos(γ₁·ln(1+z) + φ)]

where:
    γ₁ = 14.134725... (First Riemann zero imaginary part)
    A = Amplitude of vacuum oscillation
    φ = Phase offset (our position in the cycle)

Physical Interpretation:
    - The vacuum is a "breathing membrane" vibrating at prime frequencies
    - γ₁ = 14.13 is the "bass note" of reality
    - ε < 0 (measured) means we're in the "inhale" phase—tension rising

Author: Spandrel Cosmology Project
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import chi2, norm
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Import physical constants from central module
from constants import C_LIGHT_KMS as C_LIGHT, H0_FIDUCIAL, H0_PLANCK, H0_SH0ES, OMEGA_M_FIDUCIAL, GAMMA_1, RIEMANN_ZEROS


# =============================================================================
# RIEMANN RESONANCE COSMOLOGY ENGINE
# =============================================================================

class RiemannCosmology:
    """
    Cosmology with Riemann-resonant dark energy oscillations.

    The dark energy density oscillates logarithmically:
        ρ_Λ(z) = ρ_Λ₀ · [1 + A·cos(γ·ln(1+z) + φ)]

    This modifies the Hubble parameter:
        E²(z) = Ωₘ(1+z)³ + Ω_Λ·[1 + A·cos(γ·ln(1+z) + φ)]
    """

    def __init__(self, H0: float, Omega_m: float,
                 amplitude: float = 0.0, phase: float = 0.0,
                 gamma: float = GAMMA_1, n_harmonics: int = 1):
        """
        Initialize Riemann cosmology.

        Args:
            H0: Hubble constant (km/s/Mpc)
            Omega_m: Matter density parameter
            amplitude: Oscillation amplitude A (dimensionless)
            phase: Phase offset φ (radians)
            gamma: Log-frequency (default: first Riemann zero)
            n_harmonics: Number of Riemann harmonics to include
        """
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = 1.0 - Omega_m
        self.amplitude = amplitude
        self.phase = phase
        self.gamma = gamma
        self.n_harmonics = n_harmonics
        self._c_over_H0 = C_LIGHT / H0

    def dark_energy_oscillation(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the oscillatory dark energy factor.

        For single harmonic:
            f(z) = 1 + A·cos(γ·ln(1+z) + φ)

        For multiple harmonics (Riemann sum):
            f(z) = 1 + Σₙ (Aₙ/n)·cos(γₙ·ln(1+z) + φₙ)
        """
        z = np.atleast_1d(z)
        log_1pz = np.log(1.0 + z)

        if self.n_harmonics == 1:
            # Single fundamental mode
            return 1.0 + self.amplitude * np.cos(self.gamma * log_1pz + self.phase)
        else:
            # Multiple Riemann harmonics with 1/n amplitude decay
            oscillation = np.ones_like(z)
            for n in range(self.n_harmonics):
                gamma_n = RIEMANN_ZEROS[n]
                amp_n = self.amplitude / (n + 1)  # Amplitude decreases for higher modes
                phase_n = self.phase * (n + 1)    # Phase shifts for each mode
                oscillation += amp_n * np.cos(gamma_n * log_1pz + phase_n)
            return oscillation

    def E_squared(self, z: np.ndarray) -> np.ndarray:
        """
        Dimensionless Hubble parameter squared with oscillating dark energy.

        E²(z) = Ωₘ(1+z)³ + Ω_Λ·f(z)
        """
        z = np.atleast_1d(z)
        matter = self.Omega_m * (1.0 + z)**3
        dark_energy = self.Omega_Lambda * self.dark_energy_oscillation(z)
        return matter + dark_energy

    def E(self, z: np.ndarray) -> np.ndarray:
        """Dimensionless Hubble parameter E(z) = H(z)/H₀."""
        return np.sqrt(np.maximum(self.E_squared(z), 1e-10))

    def comoving_distance(self, z: np.ndarray, n_steps: int = 500) -> np.ndarray:
        """Vectorized comoving distance calculation."""
        z = np.atleast_1d(z)
        n_z = len(z)

        z_grid = np.linspace(0, z, n_steps).T
        E_grid = self.E(z_grid)
        integrand = 1.0 / E_grid

        dz = z / (n_steps - 1)
        dz = dz[:, np.newaxis]

        integral = np.sum(integrand[:, :-1] + integrand[:, 1:], axis=1) * dz.flatten() / 2.0
        return self._c_over_H0 * integral

    def luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        """Luminosity distance D_L = (1+z)·D_C."""
        return (1.0 + np.atleast_1d(z)) * self.comoving_distance(z)

    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        """Distance modulus μ = 5·log₁₀(D_L/10pc)."""
        d_L = self.luminosity_distance(z)
        return 5.0 * np.log10(d_L * 1e6 / 10.0)

    def effective_epsilon(self) -> float:
        """
        Compute the effective linear stiffness at z=0.

        This is what the linear Spandrel fit measures:
            ε_eff = dΛ/dz |_{z=0} = -A·γ·sin(φ)
        """
        return -self.amplitude * self.gamma * np.sin(self.phase)

    def effective_w(self, z: np.ndarray) -> np.ndarray:
        """
        Compute effective dark energy equation of state.

        For oscillating Λ, the effective w(z) ≠ -1.
        """
        z = np.atleast_1d(z)
        log_1pz = np.log(1.0 + z)

        # w_eff = -1 + (1/3)·d(ln ρ_Λ)/d(ln a)
        # For our model: d(ln f)/d(ln(1+z)) = -A·γ·sin(γ·ln(1+z)+φ) / f(z)
        f = self.dark_energy_oscillation(z)
        df_dlnz = -self.amplitude * self.gamma * np.sin(self.gamma * log_1pz + self.phase)

        # w = -1 - (1/3)·(df/dlnz)/f
        w_eff = -1.0 - (1.0/3.0) * df_dlnz / f

        return w_eff


# =============================================================================
# RIEMANN RESONANCE FITTER
# =============================================================================

class RiemannResonanceFitter:
    """
    Fit Riemann-resonant cosmology to supernova data.

    Tests whether dark energy oscillates at the first Riemann zero frequency.
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err
        self.n_data = len(z_obs)
        self.inv_var = 1.0 / (mu_err**2)

        # Priors (Planck)
        self.Om_prior_mean = 0.315
        self.Om_prior_std = 0.007

    def chi2_model(self, H0: float, Omega_m: float, amplitude: float,
                   phase: float, use_prior: bool = True) -> float:
        """Compute chi-squared for Riemann model."""
        # Physical bounds
        if H0 < 50 or H0 > 100:
            return 1e10
        if Omega_m < 0.1 or Omega_m > 0.5:
            return 1e10
        if amplitude < 0 or amplitude > 0.5:  # Amplitude must be positive, < 50%
            return 1e10

        cosmo = RiemannCosmology(H0, Omega_m, amplitude, phase)
        mu_model = cosmo.distance_modulus(self.z_obs)

        chi2_sn = np.sum(((self.mu_obs - mu_model)**2) * self.inv_var)

        if use_prior:
            chi2_prior = ((Omega_m - self.Om_prior_mean) / self.Om_prior_std)**2
            return chi2_sn + chi2_prior

        return chi2_sn

    def chi2_lcdm(self, H0: float, Omega_m: float, use_prior: bool = True) -> float:
        """Chi-squared for standard ΛCDM (A=0)."""
        return self.chi2_model(H0, Omega_m, 0.0, 0.0, use_prior)

    def fit_lcdm(self, use_prior: bool = True) -> Dict:
        """Fit standard ΛCDM model."""
        print("\n  Fitting ΛCDM (baseline)...")

        def objective(params):
            return self.chi2_lcdm(params[0], params[1], use_prior)

        result = differential_evolution(
            objective, [(60, 85), (0.2, 0.45)],
            maxiter=1000, tol=1e-8, seed=42, polish=True
        )

        H0, Om = result.x
        chi2_val = result.fun
        dof = self.n_data - 2

        return {
            'model': 'ΛCDM',
            'H0': H0,
            'Omega_m': Om,
            'amplitude': 0.0,
            'phase': 0.0,
            'chi2': chi2_val,
            'dof': dof,
            'reduced_chi2': chi2_val / dof,
            'effective_epsilon': 0.0
        }

    def fit_riemann(self, use_prior: bool = True, n_harmonics: int = 1) -> Dict:
        """
        Fit Riemann-resonant model.

        Parameters:
            H0: Hubble constant
            Omega_m: Matter density
            A: Oscillation amplitude
            φ: Phase offset
        """
        print(f"\n  Fitting Riemann Resonance (γ = {GAMMA_1:.4f})...")

        def objective(params):
            H0, Om, A, phi = params
            return self.chi2_model(H0, Om, A, phi, use_prior)

        # Global optimization over 4D parameter space
        # Phase is cyclic: [0, 2π]
        result = differential_evolution(
            objective,
            [(60, 85), (0.2, 0.45), (0.0, 0.1), (0, 2*np.pi)],
            maxiter=2000, tol=1e-8, seed=42, polish=True,
            workers=1
        )

        H0, Om, A, phi = result.x
        chi2_val = result.fun
        dof = self.n_data - 4

        # Compute effective epsilon
        cosmo = RiemannCosmology(H0, Om, A, phi)
        eps_eff = cosmo.effective_epsilon()

        return {
            'model': 'Riemann',
            'H0': H0,
            'Omega_m': Om,
            'amplitude': A,
            'phase': phi,
            'phase_degrees': np.degrees(phi),
            'chi2': chi2_val,
            'dof': dof,
            'reduced_chi2': chi2_val / dof,
            'effective_epsilon': eps_eff,
            'gamma': GAMMA_1
        }

    def fit_multi_harmonic(self, n_harmonics: int = 3, use_prior: bool = True) -> Dict:
        """Fit model with multiple Riemann harmonics."""
        print(f"\n  Fitting Multi-Harmonic Riemann ({n_harmonics} modes)...")

        def objective(params):
            H0, Om, A, phi = params
            if H0 < 50 or H0 > 100 or Om < 0.1 or Om > 0.5 or A < 0 or A > 0.2:
                return 1e10

            cosmo = RiemannCosmology(H0, Om, A, phi, n_harmonics=n_harmonics)
            mu_model = cosmo.distance_modulus(self.z_obs)
            chi2_sn = np.sum(((self.mu_obs - mu_model)**2) * self.inv_var)

            if use_prior:
                chi2_prior = ((Om - self.Om_prior_mean) / self.Om_prior_std)**2
                return chi2_sn + chi2_prior
            return chi2_sn

        result = differential_evolution(
            objective,
            [(60, 85), (0.2, 0.45), (0.0, 0.15), (0, 2*np.pi)],
            maxiter=2000, tol=1e-8, seed=42, polish=True
        )

        H0, Om, A, phi = result.x
        chi2_val = result.fun
        dof = self.n_data - 4

        return {
            'model': f'Riemann-{n_harmonics}H',
            'H0': H0,
            'Omega_m': Om,
            'amplitude': A,
            'phase': phi,
            'phase_degrees': np.degrees(phi),
            'chi2': chi2_val,
            'dof': dof,
            'reduced_chi2': chi2_val / dof,
            'n_harmonics': n_harmonics
        }

    def scan_frequency(self, gamma_range: np.ndarray, use_prior: bool = True) -> Dict:
        """
        Scan over different frequencies to find the best-fit oscillation.

        This tests whether γ₁ = 14.13 is actually preferred by the data.
        """
        print("\n  Scanning frequency space...")

        best_chi2 = np.inf
        chi2_values = []

        for gamma in gamma_range:
            def objective(params):
                H0, Om, A, phi = params
                if H0 < 50 or H0 > 100 or Om < 0.1 or Om > 0.5 or A < 0 or A > 0.2:
                    return 1e10

                cosmo = RiemannCosmology(H0, Om, A, phi, gamma=gamma)
                mu_model = cosmo.distance_modulus(self.z_obs)
                chi2_sn = np.sum(((self.mu_obs - mu_model)**2) * self.inv_var)

                if use_prior:
                    chi2_prior = ((Om - self.Om_prior_mean) / self.Om_prior_std)**2
                    return chi2_sn + chi2_prior
                return chi2_sn

            result = minimize(
                objective, [73, 0.31, 0.01, np.pi],
                method='Nelder-Mead', options={'maxiter': 2000}
            )
            chi2_values.append(result.fun)

        chi2_values = np.array(chi2_values)

        return {
            'gamma_range': gamma_range,
            'chi2_values': chi2_values,
            'best_gamma': gamma_range[np.argmin(chi2_values)],
            'best_chi2': np.min(chi2_values)
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_riemann_analysis(z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray,
                         lcdm_result: Dict, riemann_result: Dict,
                         save_path: Optional[str] = None):
    """Create comprehensive Riemann resonance visualization."""

    fig = plt.figure(figsize=(16, 12))

    # Model curves
    z_model = np.logspace(np.log10(z_obs.min()), np.log10(z_obs.max()), 500)

    cosmo_lcdm = RiemannCosmology(
        lcdm_result['H0'], lcdm_result['Omega_m'], 0, 0
    )
    cosmo_riemann = RiemannCosmology(
        riemann_result['H0'], riemann_result['Omega_m'],
        riemann_result['amplitude'], riemann_result['phase']
    )

    mu_lcdm = cosmo_lcdm.distance_modulus(z_model)
    mu_riemann = cosmo_riemann.distance_modulus(z_model)

    # Panel 1: Hubble Diagram
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.errorbar(z_obs, mu_obs, yerr=mu_err, fmt='.', color='gray',
                alpha=0.3, markersize=2, label='Pantheon+')
    ax1.plot(z_model, mu_lcdm, 'b-', linewidth=2, label='ΛCDM')
    ax1.plot(z_model, mu_riemann, 'r--', linewidth=2,
            label=f'Riemann (A={riemann_result["amplitude"]:.4f}, φ={riemann_result["phase_degrees"]:.1f}°)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Distance Modulus μ')
    ax1.set_title('Hubble Diagram: ΛCDM vs Riemann Resonance')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Dark Energy Oscillation
    ax2 = fig.add_subplot(2, 2, 2)
    z_de = np.linspace(0.001, 2.5, 500)

    de_lcdm = np.ones_like(z_de)  # Constant for ΛCDM
    de_riemann = cosmo_riemann.dark_energy_oscillation(z_de)

    ax2.plot(z_de, de_lcdm, 'b-', linewidth=2, label='ΛCDM (Λ = const)')
    ax2.plot(z_de, de_riemann, 'r-', linewidth=2, label='Riemann Oscillation')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Mark Riemann zero crossings
    log_z = np.log(1 + z_de)
    phase = riemann_result['phase']
    for n in range(-5, 10):
        z_node = np.exp((n * np.pi - phase) / GAMMA_1) - 1
        if 0 < z_node < 2.5:
            ax2.axvline(z_node, color='green', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Dark Energy Factor Λ(z)/Λ₀')
    ax2.set_title(f'Vacuum Oscillation (γ₁ = {GAMMA_1:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2.5)

    # Panel 3: Residuals
    ax3 = fig.add_subplot(2, 2, 3)

    mu_lcdm_data = cosmo_lcdm.distance_modulus(z_obs)
    mu_riemann_data = cosmo_riemann.distance_modulus(z_obs)

    residuals_lcdm = mu_obs - mu_lcdm_data
    residuals_riemann = mu_obs - mu_riemann_data

    ax3.scatter(z_obs, residuals_lcdm, s=5, alpha=0.3, color='blue', label='ΛCDM')
    ax3.scatter(z_obs, residuals_riemann, s=5, alpha=0.3, color='red', label='Riemann')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xscale('log')
    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel('Residual Δμ (mag)')
    ax3.set_title('Residuals from Best Fit')
    ax3.set_ylim(-0.6, 0.6)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Effective w(z)
    ax4 = fig.add_subplot(2, 2, 4)

    w_riemann = cosmo_riemann.effective_w(z_de)

    ax4.axhline(y=-1, color='blue', linewidth=2, label='ΛCDM (w = -1)')
    ax4.plot(z_de, w_riemann, 'r-', linewidth=2, label='Riemann w_eff(z)')
    ax4.fill_between(z_de, -1.3, -0.7, alpha=0.1, color='gray')

    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Equation of State w(z)')
    ax4.set_title('Effective Dark Energy EoS')
    ax4.set_ylim(-1.5, -0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot to: {save_path}")

    plt.show()


def plot_frequency_scan(scan_result: Dict, save_path: Optional[str] = None):
    """Plot frequency scan results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    gamma = scan_result['gamma_range']
    chi2 = scan_result['chi2_values']

    ax.plot(gamma, chi2, 'b-', linewidth=2)

    # Mark Riemann zeros
    for i, gz in enumerate(RIEMANN_ZEROS[:5]):
        ax.axvline(gz, color='red', linestyle='--', alpha=0.7,
                  label=f'γ_{i+1} = {gz:.2f}' if i < 3 else '')

    ax.axhline(scan_result['best_chi2'], color='green', linestyle=':',
              label=f'Best χ² = {scan_result["best_chi2"]:.2f}')

    ax.set_xlabel('Log-frequency γ')
    ax.set_ylabel('χ²')
    ax.set_title('Frequency Scan: Is γ₁ = 14.13 Preferred?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_riemann_analysis(data_path: str = "Pantheon+SH0ES.dat"):
    """
    Run complete Riemann resonance analysis.
    """

    print("\n" + "="*70)
    print("RIEMANN RESONANCE COSMOLOGY ANALYSIS")
    print("Testing Log-Periodic Vacuum Oscillations at γ₁ = 14.134725")
    print("="*70)

    # Load data
    print("\nLoading Pantheon+ data...")
    df = pd.read_csv(data_path, sep=r'\s+')

    z = df['zHD'].values
    mu = df['MU_SH0ES'].values
    err = df['MU_SH0ES_ERR_DIAG'].values

    mask = (z > 0.001) & (z < 2.5) & (mu > 0) & np.isfinite(mu)
    z_obs, mu_obs, mu_err = z[mask], mu[mask], err[mask]

    idx = np.argsort(z_obs)
    z_obs, mu_obs, mu_err = z_obs[idx], mu_obs[idx], mu_err[idx]

    print(f"  Loaded {len(z_obs)} supernovae")

    # Initialize fitter
    fitter = RiemannResonanceFitter(z_obs, mu_obs, mu_err)

    # Fit models
    print("\n" + "-"*70)
    print("MODEL FITTING (with Planck Ωₘ prior)")
    print("-"*70)

    lcdm_result = fitter.fit_lcdm(use_prior=True)
    riemann_result = fitter.fit_riemann(use_prior=True)

    # Print results
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)

    print(f"""
    ΛCDM (Baseline):
      H₀ = {lcdm_result['H0']:.3f} km/s/Mpc
      Ωₘ = {lcdm_result['Omega_m']:.5f}
      χ² = {lcdm_result['chi2']:.2f} (dof = {lcdm_result['dof']})
      χ²/dof = {lcdm_result['reduced_chi2']:.5f}

    Riemann Resonance:
      H₀ = {riemann_result['H0']:.3f} km/s/Mpc
      Ωₘ = {riemann_result['Omega_m']:.5f}
      Amplitude A = {riemann_result['amplitude']:.6f}
      Phase φ = {riemann_result['phase']:.4f} rad ({riemann_result['phase_degrees']:.2f}°)
      χ² = {riemann_result['chi2']:.2f} (dof = {riemann_result['dof']})
      χ²/dof = {riemann_result['reduced_chi2']:.5f}

      Effective ε (linear approx) = {riemann_result['effective_epsilon']:.6f}
    """)

    # Statistical comparison
    print("-"*70)
    print("MODEL COMPARISON")
    print("-"*70)

    delta_chi2 = lcdm_result['chi2'] - riemann_result['chi2']
    delta_dof = 2  # Riemann adds 2 parameters (A, φ)
    p_value = 1 - chi2.cdf(delta_chi2, delta_dof)

    # F-test for nested models
    F_stat = (delta_chi2 / delta_dof) / (riemann_result['chi2'] / riemann_result['dof'])
    from scipy.stats import f as f_dist
    p_value_f = 1 - f_dist.cdf(F_stat, delta_dof, riemann_result['dof'])

    print(f"""
    Likelihood Ratio Test:
      Δχ² = {delta_chi2:.4f}
      Δdof = {delta_dof}
      p-value = {p_value:.6f}

    F-test:
      F = {F_stat:.4f}
      p-value = {p_value_f:.6f}
    """)

    if delta_chi2 > 5.99:  # 95% for 2 dof
        print("    *** Riemann model PREFERRED at 95% confidence ***")
    elif delta_chi2 > 0:
        print("    Riemann model shows slight improvement but NOT significant")
    else:
        print("    ΛCDM is preferred (Riemann overfits)")

    # Frequency scan
    print("\n" + "-"*70)
    print("FREQUENCY SCAN")
    print("-"*70)
    print("  Testing if γ₁ = 14.13 is special or arbitrary...")

    gamma_range = np.linspace(5, 50, 50)
    scan_result = fitter.scan_frequency(gamma_range, use_prior=True)

    print(f"""
    Scan Results:
      Best-fit γ = {scan_result['best_gamma']:.4f}
      γ₁ (Riemann) = {GAMMA_1:.4f}
      Match: {'YES' if abs(scan_result['best_gamma'] - GAMMA_1) < 2 else 'NO'}
    """)

    # Physical interpretation
    print("="*70)
    print("PHYSICAL INTERPRETATION")
    print("="*70)

    A = riemann_result['amplitude']
    phi = riemann_result['phase']
    eps_eff = riemann_result['effective_epsilon']

    print(f"""
    Measured Parameters:
      Amplitude A = {A:.6f} ({A*100:.3f}% vacuum fluctuation)
      Phase φ = {np.degrees(phi):.2f}°
      sin(φ) = {np.sin(phi):.4f}

    Derived Quantities:
      ε_eff = -A·γ·sin(φ) = {eps_eff:.6f}
      (Compare to linear fit: ε ≈ -0.079)

    Interpretation:
    """)

    if A < 0.005:
        print("      Amplitude is very small → No significant oscillation detected")
        print("      The vacuum appears constant (ΛCDM is correct)")
    elif abs(eps_eff + 0.079) < 0.02:
        print("      ✓ Riemann model REPRODUCES the linear ε measurement!")
        print(f"      We are at phase φ ≈ {np.degrees(phi):.0f}° in the vacuum cycle")
        print("      The 'negative stiffness' is the derivative of a cosine wave")
    else:
        print("      Oscillation detected but doesn't match linear analysis")
        print("      Further investigation needed")

    # Breathing phase
    if 0 < phi < np.pi:
        print("\n      Current Phase: 'EXHALE' → Tension DECREASING")
    else:
        print("\n      Current Phase: 'INHALE' → Tension INCREASING")

    # Generate plots
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)

    plot_riemann_analysis(z_obs, mu_obs, mu_err, lcdm_result, riemann_result,
                         save_path="riemann_analysis.png")

    plot_frequency_scan(scan_result, save_path="frequency_scan.png")

    return {
        'lcdm': lcdm_result,
        'riemann': riemann_result,
        'scan': scan_result,
        'data': {'z': z_obs, 'mu': mu_obs, 'err': mu_err}
    }


if __name__ == "__main__":
    results = run_riemann_analysis()
