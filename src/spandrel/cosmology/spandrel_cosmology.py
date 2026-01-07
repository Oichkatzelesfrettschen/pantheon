#!/usr/bin/env python3
"""
Spandrel Cosmology Analysis Framework
=====================================

Tests the "Spandrel Hypothesis" against the Pantheon+ Type Ia Supernova dataset.

The Spandrel Hypothesis proposes that spacetime exhibits "stiffness" (epsilon) that
modifies the standard LambdaCDM distance-redshift relationship. This stiffness
parameter could explain the Hubble Tension by bridging early-universe (high
stiffness) and late-universe (relaxed stiffness) expansion rates.

Dataset: Pantheon+SH0ES (1,701 Type Ia Supernovae)
Reference: Scolnic et al. 2022, ApJ 938 113

Author: Spandrel Cosmology Project
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Import physical constants from central module
from spandrel.core.constants import C_LIGHT_KMS as C_LIGHT, H0_FIDUCIAL, H0_PLANCK, H0_SH0ES, OMEGA_M_FIDUCIAL, GAMMA_1, RIEMANN_ZEROS
from spandrel.core.data_interface import PantheonData
try:
    from spandrel_core.cosmology import distance_modulus_lcdm as distance_modulus_lcdm_core
    from spandrel_core.likelihood import chi2_diagonal as chi2_diagonal_core
except ImportError as exc:
    raise ImportError(
        "spandrel-core is required for distance modulus calculations; install it "
        "or ensure the sibling `spandrel-core/src` is on PYTHONPATH."
    ) from exc

class SpandrelCosmology:
    """
    Implements the Spandrel Cosmology model with stiffness parameter.

    The Spandrel modification to LambdaCDM introduces a "stiffness" parameter epsilon
    that modifies the distance-redshift relationship:

    mu_spandrel(z) = mu_LambdaCDM(z) + epsilon * f(z)

    where f(z) captures the scale-dependent stiffness effect.

    Parameters:
    - H0: Hubble constant (km/s/Mpc)
    - Omega_m: Matter density parameter
    - Omega_Lambda: Dark energy density (= 1 - Omega_m for flat universe)
    - epsilon: Stiffness parameter (epsilon = 0 recovers standard LambdaCDM)
    """

    def __init__(self, H0: float = 70.0, Omega_m: float = 0.3, epsilon: float = 0.0):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = 1.0 - Omega_m  # Flat universe
        self.epsilon = epsilon

    def E(self, z: float) -> float:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.

        For flat LambdaCDM: E(z) = sqrt(Omega_m(1+z)^3 + Omega_Lambda)
        """
        return np.sqrt(
            self.Omega_m * (1 + z)**3 + self.Omega_Lambda
        )

    def comoving_distance(self, z: float) -> float:
        """
        Comoving distance in Mpc (for flat universe, equals luminosity distance / (1+z)).

        D_C(z) = (c/H0) * integral_0ᶻ dz'/E(z')
        """
        integrand = lambda zp: 1.0 / self.E(zp)
        result, _ = quad(integrand, 0, z)
        return (C_LIGHT / self.H0) * result

    def luminosity_distance(self, z: float) -> float:
        """
        Luminosity distance D_L(z) = (1+z) * D_C(z) for flat universe.
        """
        return (1 + z) * self.comoving_distance(z)

    def distance_modulus_lcdm(self, z: float) -> float:
        """
        Standard LambdaCDM distance modulus: mu = 5*log10(D_L/10pc)
        """
        return float(distance_modulus_lcdm_core(z, Om0=self.Omega_m, H0=self.H0))

    def distance_modulus(self, z: float) -> float:
        """Alias for distance_modulus_lcdm."""
        return self.distance_modulus_lcdm(z)

    def spandrel_correction(self, z: float) -> float:
        """
        Spandrel stiffness correction term.

        The correction models "surface tension" in expanding spacetime:
        Δmu = epsilon * ln(1 + z) * (1 - 1/(1+z)^2)

        Physical interpretation:
        - At z=0: correction = 0 (local measurements unaffected)
        - At high z: correction grows logarithmically
        - The (1 - 1/(1+z)^2) term represents relaxation from early stiffness
        """
        return self.epsilon * np.log(1 + z) * (1 - 1/(1 + z)**2)

    def distance_modulus_spandrel(self, z: float) -> float:
        """
        Spandrel-modified distance modulus: mu_spandrel = mu_LambdaCDM + correction
        """
        return self.distance_modulus_lcdm(z) + self.spandrel_correction(z)

    def distance_modulus_array(self, z_array: np.ndarray, use_spandrel: bool = True) -> np.ndarray:
        """Vectorized distance modulus calculation."""
        if use_spandrel:
            return np.array([self.distance_modulus_spandrel(z) for z in z_array])
        else:
            return np.array([self.distance_modulus_lcdm(z) for z in z_array])


class SpandrelFitter:
    """
    Maximum likelihood fitter for Spandrel cosmology parameters.

    Fits H0, Omega_m, and epsilon to minimize chi-squared residuals
    against observed supernova distance moduli.
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err
        self.n_data = len(z_obs)

        # Fit results
        self.best_fit = None
        self.covariance = None
        self.chi2_min = None
        self.dof = None

    def chi_squared(self, params: Tuple[float, float, float], use_spandrel: bool = True) -> float:
        """
        Compute chi-squared statistic for given parameters.

        chi^2 = Σ [(mu_obs - mu_model)^2 / sigma^2]
        """
        H0, Omega_m, epsilon = params

        # Physical bounds check
        if H0 < 50 or H0 > 100 or Omega_m < 0.1 or Omega_m > 0.5:
            return 1e10

        cosmo = SpandrelCosmology(H0=H0, Omega_m=Omega_m, epsilon=epsilon)
        mu_model = cosmo.distance_modulus_array(self.z_obs, use_spandrel=use_spandrel)

        residuals = (self.mu_obs - mu_model) / self.mu_err
        return chi2_diagonal_core(self.mu_obs - mu_model, self.mu_err)

    def fit_lcdm(self, initial_guess: Tuple[float, float] = (70.0, 0.3)) -> Dict[str, Any]:
        """
        Fit standard LambdaCDM (epsilon = 0).

        Returns dict with H0, Omega_m, chi2, reduced_chi2, p_value
        """
        print("\n" + "="*60)
        print("Fitting Standard LambdaCDM Model (epsilon = 0)")
        print("="*60)

        def objective(params):
            return self.chi_squared((params[0], params[1], 0.0), use_spandrel=False)

        result = minimize(
            objective,
            initial_guess,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
        )

        H0_fit, Om_fit = result.x
        chi2_val = result.fun
        dof = self.n_data - 2
        reduced_chi2 = chi2_val / dof
        p_value = 1 - chi2.cdf(chi2_val, dof)

        lcdm_result = {
            'H0': H0_fit,
            'Omega_m': Om_fit,
            'epsilon': 0.0,
            'chi2': chi2_val,
            'dof': dof,
            'reduced_chi2': reduced_chi2,
            'p_value': p_value,
            'model': 'LambdaCDM'
        }

        print(f"  H_0 = {H0_fit:.2f} km/s/Mpc")
        print(f"  Omegaₘ = {Om_fit:.4f}")
        print(f"  chi^2 = {chi2_val:.2f} (dof = {dof})")
        print(f"  chi^2/dof = {reduced_chi2:.4f}")
        print(f"  p-value = {p_value:.4f}")

        return lcdm_result

    def fit_spandrel(self, initial_guess: Tuple[float, float, float] = (70.0, 0.3, 0.0),
                     use_global: bool = True) -> Dict[str, Any]:
        """
        Fit Spandrel cosmology with stiffness parameter.

        Returns dict with H0, Omega_m, epsilon, chi2, reduced_chi2, p_value
        """
        print("\n" + "="*60)
        print("Fitting Spandrel Cosmology Model (epsilon free)")
        print("="*60)

        def objective(params):
            return self.chi_squared(params, use_spandrel=True)

        if use_global:
            # Global optimization to avoid local minima
            bounds = [(60, 85), (0.15, 0.45), (-0.5, 0.5)]
            result = differential_evolution(
                objective,
                bounds,
                maxiter=1000,
                tol=1e-7,
                seed=42,
                workers=1,  # Single worker to avoid pickling issues
                updating='immediate'
            )
        else:
            result = minimize(
                objective,
                initial_guess,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
            )

        H0_fit, Om_fit, eps_fit = result.x
        chi2_val = result.fun
        dof = self.n_data - 3
        reduced_chi2 = chi2_val / dof
        p_value = 1 - chi2.cdf(chi2_val, dof)

        self.best_fit = result.x
        self.chi2_min = chi2_val
        self.dof = dof

        spandrel_result = {
            'H0': H0_fit,
            'Omega_m': Om_fit,
            'epsilon': eps_fit,
            'chi2': chi2_val,
            'dof': dof,
            'reduced_chi2': reduced_chi2,
            'p_value': p_value,
            'model': 'Spandrel'
        }

        print(f"  H_0 = {H0_fit:.2f} km/s/Mpc")
        print(f"  Omegaₘ = {Om_fit:.4f}")
        print(f"  epsilon (stiffness) = {eps_fit:.6f}")
        print(f"  chi^2 = {chi2_val:.2f} (dof = {dof})")
        print(f"  chi^2/dof = {reduced_chi2:.4f}")
        print(f"  p-value = {p_value:.4f}")

        return spandrel_result

    def compute_parameter_errors(self, best_params: Tuple[float, float, float],
                                  delta_chi2: float = 1.0) -> Dict[str, float]:
        """
        Estimate parameter uncertainties using delta-chi-squared method.

        1sigma errors correspond to Δchi^2 = 1 for one parameter of interest.
        """
        print("\nEstimating parameter uncertainties (Δchi^2 = 1)...")

        H0_best, Om_best, eps_best = best_params
        chi2_best = self.chi_squared(best_params)

        errors = {}
        param_names = ['H0', 'Omega_m', 'epsilon']
        step_sizes = [0.1, 0.001, 0.001]

        for i, (name, step) in enumerate(zip(param_names, step_sizes)):
            # Search upward
            params_up = list(best_params)
            while True:
                params_up[i] += step
                chi2_up = self.chi_squared(tuple(params_up))
                if chi2_up - chi2_best > delta_chi2:
                    break
                if params_up[i] > best_params[i] * 2:  # Safety limit
                    break

            # Search downward
            params_down = list(best_params)
            while True:
                params_down[i] -= step
                chi2_down = self.chi_squared(tuple(params_down))
                if chi2_down - chi2_best > delta_chi2:
                    break
                if params_down[i] < best_params[i] * 0.5:  # Safety limit
                    break

            errors[name] = (params_up[i] - params_down[i]) / 2

        return errors

    def likelihood_ratio_test(self, lcdm_result: Dict, spandrel_result: Dict) -> Dict[str, float]:
        """
        Perform likelihood ratio test to assess if Spandrel model is preferred.

        Tests H0: epsilon = 0 (LambdaCDM) vs H1: epsilon ≠ 0 (Spandrel)

        The test statistic -2Δln(L) = Δchi^2 follows chi^2(1) under the null.
        """
        delta_chi2 = lcdm_result['chi2'] - spandrel_result['chi2']
        delta_dof = lcdm_result['dof'] - spandrel_result['dof']  # Should be 1

        # p-value for the improvement
        p_value = 1 - chi2.cdf(delta_chi2, abs(delta_dof))

        # Significance in sigma
        from scipy.stats import norm
        sigma = norm.ppf(1 - p_value/2) if p_value > 0 else float('inf')

        return {
            'delta_chi2': delta_chi2,
            'delta_dof': delta_dof,
            'p_value': p_value,
            'sigma': sigma,
            'prefers_spandrel': delta_chi2 > 3.84  # 95% confidence threshold
        }


class SpandrelVisualizer:
    """
    Visualization tools for Spandrel cosmology analysis.
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err

    def plot_hubble_diagram(self, lcdm_result: Dict, spandrel_result: Dict,
                            save_path: Optional[str] = None):
        """
        Create the Hubble diagram with both model fits.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                                  gridspec_kw={'height_ratios': [3, 1]})

        # Create model curves
        z_model = np.logspace(np.log10(self.z_obs.min()), np.log10(self.z_obs.max()), 500)

        cosmo_lcdm = SpandrelCosmology(H0=lcdm_result['H0'], Omega_m=lcdm_result['Omega_m'], epsilon=0)
        cosmo_spandrel = SpandrelCosmology(
            H0=spandrel_result['H0'],
            Omega_m=spandrel_result['Omega_m'],
            epsilon=spandrel_result['epsilon']
        )

        mu_lcdm = cosmo_lcdm.distance_modulus_array(z_model, use_spandrel=False)
        mu_spandrel = cosmo_spandrel.distance_modulus_array(z_model, use_spandrel=True)

        # Top panel: Hubble diagram
        ax1 = axes[0]
        ax1.errorbar(self.z_obs, self.mu_obs, yerr=self.mu_err, fmt='.',
                    color='gray', alpha=0.3, markersize=3, label='Pantheon+ SNe Ia')

        ax1.plot(z_model, mu_lcdm, 'b-', linewidth=2,
                label=f'LambdaCDM: H_0={lcdm_result["H0"]:.1f}, Omegaₘ={lcdm_result["Omega_m"]:.3f}')
        ax1.plot(z_model, mu_spandrel, 'r--', linewidth=2,
                label=f'Spandrel: H_0={spandrel_result["H0"]:.1f}, Omegaₘ={spandrel_result["Omega_m"]:.3f}, epsilon={spandrel_result["epsilon"]:.4f}')

        ax1.set_xscale('log')
        ax1.set_xlabel('Redshift z', fontsize=12)
        ax1.set_ylabel('Distance Modulus mu (mag)', fontsize=12)
        ax1.set_title('Pantheon+ Hubble Diagram: LambdaCDM vs Spandrel Cosmology', fontsize=14)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Bottom panel: Residuals
        ax2 = axes[1]

        mu_lcdm_at_data = cosmo_lcdm.distance_modulus_array(self.z_obs, use_spandrel=False)
        mu_spandrel_at_data = cosmo_spandrel.distance_modulus_array(self.z_obs, use_spandrel=True)

        residuals_lcdm = self.mu_obs - mu_lcdm_at_data
        residuals_spandrel = self.mu_obs - mu_spandrel_at_data

        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.scatter(self.z_obs, residuals_lcdm, c='blue', s=5, alpha=0.3, label='LambdaCDM residuals')
        ax2.scatter(self.z_obs, residuals_spandrel, c='red', s=5, alpha=0.3, label='Spandrel residuals')

        ax2.set_xscale('log')
        ax2.set_xlabel('Redshift z', fontsize=12)
        ax2.set_ylabel('Residual Δmu (mag)', fontsize=12)
        ax2.set_ylim(-1, 1)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved Hubble diagram to: {save_path}")

        plt.show()

    def plot_stiffness_effect(self, spandrel_result: Dict, save_path: Optional[str] = None):
        """
        Visualize the Spandrel stiffness correction as a function of redshift.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        z_range = np.linspace(0.001, 2.5, 500)
        epsilon = spandrel_result['epsilon']

        cosmo = SpandrelCosmology(epsilon=epsilon)
        corrections = np.array([cosmo.spandrel_correction(z) for z in z_range])

        ax.plot(z_range, corrections, 'r-', linewidth=2, label=f'Spandrel correction (epsilon = {epsilon:.4f})')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # Mark key redshift regimes
        ax.axvspan(0, 0.01, alpha=0.2, color='green', label='Local (z < 0.01)')
        ax.axvspan(1.0, 2.5, alpha=0.2, color='blue', label='Early Universe (z > 1)')

        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Distance Modulus Correction Δmu (mag)', fontsize=12)
        ax.set_title('Spandrel Stiffness Effect on Distance Measurements', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved stiffness plot to: {save_path}")

        plt.show()

    def plot_chi2_contours(self, fitter: SpandrelFitter, best_params: Tuple[float, float, float],
                           save_path: Optional[str] = None):
        """
        Plot chi-squared contours in H0-epsilon parameter space.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        H0_best, Om_best, eps_best = best_params

        # Create grid
        H0_range = np.linspace(H0_best - 5, H0_best + 5, 50)
        eps_range = np.linspace(eps_best - 0.1, eps_best + 0.1, 50)

        chi2_grid = np.zeros((len(eps_range), len(H0_range)))

        for i, eps in enumerate(eps_range):
            for j, H0 in enumerate(H0_range):
                chi2_grid[i, j] = fitter.chi_squared((H0, Om_best, eps))

        chi2_min = fitter.chi2_min

        # Contour levels for 1sigma, 2sigma, 3sigma (Δchi^2 = 2.30, 6.17, 11.8 for 2 params)
        levels = chi2_min + np.array([2.30, 6.17, 11.8])

        contour = ax.contour(H0_range, eps_range, chi2_grid, levels=levels,
                            colors=['green', 'blue', 'red'])
        ax.clabel(contour, fmt={levels[0]: '1sigma', levels[1]: '2sigma', levels[2]: '3sigma'})

        ax.plot(H0_best, eps_best, 'k*', markersize=15, label='Best fit')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='LambdaCDM (epsilon=0)')

        ax.set_xlabel('H_0 (km/s/Mpc)', fontsize=12)
        ax.set_ylabel('Stiffness epsilon', fontsize=12)
        ax.set_title('chi^2 Confidence Contours: H_0 vs Stiffness', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved contour plot to: {save_path}")

        plt.show()


def run_full_analysis(data_path: str = "Pantheon+SH0ES.dat"):
    """
    Execute complete Spandrel Hypothesis analysis on Pantheon+ data.
    """
    print("\n" + "="*70)
    print("SPANDREL COSMOLOGY ANALYSIS")
    print("Testing the Stiffness Hypothesis against Pantheon+ SNe Ia")
    print("="*70)

    # 1. Load data
    data = PantheonData(filepath=Path(data_path))
    z_obs, mu_obs, mu_err = data.get_cosmology_data()
    meta = data.validate()

    # 2. Fit models
    fitter = SpandrelFitter(z_obs, mu_obs, mu_err)

    lcdm_result = fitter.fit_lcdm()
    spandrel_result = fitter.fit_spandrel()

    # 3. Compute uncertainties
    errors = fitter.compute_parameter_errors(
        (spandrel_result['H0'], spandrel_result['Omega_m'], spandrel_result['epsilon'])
    )

    # 4. Statistical comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON: Likelihood Ratio Test")
    print("="*60)

    lr_test = fitter.likelihood_ratio_test(lcdm_result, spandrel_result)

    print(f"  Δchi^2 (LambdaCDM - Spandrel) = {lr_test['delta_chi2']:.4f}")
    print(f"  p-value = {lr_test['p_value']:.6f}")
    print(f"  Significance = {lr_test['sigma']:.2f}sigma")

    if lr_test['prefers_spandrel']:
        print("\n  *** Spandrel model PREFERRED over LambdaCDM at 95% confidence ***")
    else:
        print("\n  LambdaCDM is statistically sufficient (no evidence for stiffness)")

    # 5. Interpretation
    print("\n" + "="*60)
    print("PHYSICAL INTERPRETATION")
    print("="*60)

    eps = spandrel_result['epsilon']
    eps_err = errors.get('epsilon', 0)

    print(f"\n  Stiffness Parameter: epsilon = {eps:.6f} +/- {eps_err:.6f}")

    if abs(eps) < 2 * eps_err:
        print("\n  RESULT: epsilon is consistent with zero within 2sigma")
        print("  INTERPRETATION: The Universe is purely Associative.")
        print("  Standard LambdaCDM is correct. The 'Surface Tension' is effectively zero.")
        print("  The Hubble Tension may be a systematic measurement error.")
    elif eps > 0:
        print("\n  RESULT: epsilon > 0 detected!")
        print("  INTERPRETATION: CONFIRMATION of Spandrel Cosmology")
        print("  The universe was 'stiffer' in the past.")
        print("  Dark Energy density may be decaying as the bubble expands (1/R).")
        print("  This could mathematically resolve the Hubble Tension.")
    else:
        print("\n  RESULT: epsilon < 0 detected")
        print("  INTERPRETATION: Universe is 'softening' over time (unexpected)")
        print("  This requires further theoretical investigation.")

    # 6. Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    viz = SpandrelVisualizer(z_obs, mu_obs, mu_err)

    viz.plot_hubble_diagram(lcdm_result, spandrel_result, save_path="hubble_diagram.png")
    viz.plot_stiffness_effect(spandrel_result, save_path="stiffness_effect.png")
    viz.plot_chi2_contours(fitter, fitter.best_fit, save_path="chi2_contours.png")

    # 7. Summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"""
    Dataset: Pantheon+ ({meta['total_entries']} Type Ia Supernovae)
    Redshift Range: z = {meta['redshift_range'][0]:.4f} to {meta['redshift_range'][1]:.4f}

    LambdaCDM Best Fit:
      H_0 = {lcdm_result['H0']:.2f} km/s/Mpc
      Omegaₘ = {lcdm_result['Omega_m']:.4f}
      chi^2/dof = {lcdm_result['reduced_chi2']:.4f}

    Spandrel Best Fit:
      H_0 = {spandrel_result['H0']:.2f} +/- {errors.get('H0', 0):.2f} km/s/Mpc
      Omegaₘ = {spandrel_result['Omega_m']:.4f} +/- {errors.get('Omega_m', 0):.4f}
      epsilon  = {spandrel_result['epsilon']:.6f} +/- {errors.get('epsilon', 0):.6f}
      chi^2/dof = {spandrel_result['reduced_chi2']:.4f}

    Model Comparison:
      Δchi^2 = {lr_test['delta_chi2']:.4f}
      Significance: {lr_test['sigma']:.2f}sigma
      Spandrel preferred: {lr_test['prefers_spandrel']}
    """)

    return {
        'lcdm': lcdm_result,
        'spandrel': spandrel_result,
        'errors': errors,
        'likelihood_ratio': lr_test,
        'data': {'z': z_obs, 'mu': mu_obs, 'err': mu_err}
    }


if __name__ == "__main__":
    results = run_full_analysis()
