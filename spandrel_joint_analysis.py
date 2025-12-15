#!/usr/bin/env python3
"""
Spandrel Cosmology Joint Analysis
=================================

Breaking the Ωₘ-ε degeneracy using external cosmological priors.

The Problem:
  SNe Ia measure relative brightness (expansion history) but cannot
  independently constrain matter density. This creates a degeneracy:
  - High Ωₘ + Low ε ≈ Low Ωₘ + High ε

The Solution:
  Apply Gaussian priors from BAO/CMB measurements to "pin" Ωₘ,
  forcing the stiffness parameter ε to reveal itself.

Priors Used:
  - Planck 2018 (CMB): Ωₘ = 0.315 ± 0.007
  - DESI BAO 2024: Ωₘ = 0.295 ± 0.015 (optional, shows tension)
  - Combined: Weighted average

Author: Spandrel Cosmology Project
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2, norm
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import warnings

warnings.filterwarnings('ignore')

# Import physical constants from central module
from constants import C_LIGHT_KMS as C_LIGHT, H0_FIDUCIAL, H0_PLANCK, H0_SH0ES, OMEGA_M_FIDUCIAL, GAMMA_1, RIEMANN_ZEROS

# Number of CPU cores
NUM_CORES = mp.cpu_count()


# =============================================================================
# EXTERNAL PRIORS (The "Cosmic Vise")
# =============================================================================

@dataclass
class CosmologicalPrior:
    """Container for external cosmological constraints."""
    name: str
    param: str
    mean: float
    std: float

    def chi2(self, value: float) -> float:
        """Compute chi-squared penalty for deviation from prior."""
        return ((value - self.mean) / self.std) ** 2

    def log_prior(self, value: float) -> float:
        """Compute log of Gaussian prior probability."""
        return -0.5 * self.chi2(value)


# Standard cosmological priors
PRIORS = {
    'planck2018': CosmologicalPrior(
        name='Planck 2018 (CMB)',
        param='Omega_m',
        mean=0.315,
        std=0.007
    ),
    'planck2018_h0': CosmologicalPrior(
        name='Planck 2018 H0',
        param='H0',
        mean=67.4,
        std=0.5
    ),
    'desi2024': CosmologicalPrior(
        name='DESI BAO 2024',
        param='Omega_m',
        mean=0.295,
        std=0.015
    ),
    'sh0es2022': CosmologicalPrior(
        name='SH0ES 2022',
        param='H0',
        mean=73.04,
        std=1.04
    ),
    'bao_consensus': CosmologicalPrior(
        name='BAO Consensus',
        param='Omega_m',
        mean=0.310,
        std=0.010
    ),
}


# =============================================================================
# COSMOLOGY ENGINE (Vectorized)
# =============================================================================

class SpandrelCosmologyEngine:
    """Fast vectorized cosmology calculations."""

    def __init__(self, H0: float, Omega_m: float, epsilon: float = 0.0):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = 1.0 - Omega_m
        self.epsilon = epsilon
        self._c_over_H0 = C_LIGHT / H0

    def E(self, z: np.ndarray) -> np.ndarray:
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        z = np.atleast_1d(z)
        return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)

    def comoving_distance(self, z: np.ndarray, n_steps: int = 500) -> np.ndarray:
        """Vectorized comoving distance calculation."""
        z = np.atleast_1d(z)
        n_z = len(z)

        # Integration grid
        z_grid = np.linspace(0, z, n_steps).T
        E_grid = self.E(z_grid)
        integrand = 1.0 / E_grid

        dz = z / (n_steps - 1)
        dz = dz[:, np.newaxis]

        integral = np.sum(integrand[:, :-1] + integrand[:, 1:], axis=1) * dz.flatten() / 2.0
        return self._c_over_H0 * integral

    def luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        """Luminosity distance D_L = (1+z) * D_C."""
        return (1.0 + np.atleast_1d(z)) * self.comoving_distance(z)

    def distance_modulus_lcdm(self, z: np.ndarray) -> np.ndarray:
        """Standard ΛCDM distance modulus."""
        d_L = self.luminosity_distance(z)
        return 5.0 * np.log10(d_L * 1e6 / 10.0)

    def spandrel_correction(self, z: np.ndarray) -> np.ndarray:
        """
        Spandrel stiffness correction.

        Δμ = ε * ln(1+z) * (1 - 1/(1+z)²)
        """
        z = np.atleast_1d(z)
        one_plus_z = 1.0 + z
        return self.epsilon * np.log(one_plus_z) * (1.0 - 1.0/one_plus_z**2)

    def distance_modulus_spandrel(self, z: np.ndarray) -> np.ndarray:
        """Spandrel-modified distance modulus."""
        return self.distance_modulus_lcdm(z) + self.spandrel_correction(z)


# =============================================================================
# JOINT LIKELIHOOD FITTER
# =============================================================================

class JointAnalysisFitter:
    """
    Joint analysis fitter combining SNe Ia with external priors.

    The total chi-squared is:
        χ²_total = χ²_SN + Σ χ²_prior

    This "vise" clamps down on parameters, breaking degeneracies.
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err
        self.n_data = len(z_obs)

        # Precompute inverse variance
        self.inv_var = 1.0 / (mu_err ** 2)

        # Active priors
        self.active_priors: List[CosmologicalPrior] = []

    def add_prior(self, prior: CosmologicalPrior):
        """Add an external prior to the analysis."""
        self.active_priors.append(prior)
        print(f"  Added prior: {prior.name} ({prior.param} = {prior.mean} ± {prior.std})")

    def clear_priors(self):
        """Remove all priors."""
        self.active_priors = []

    def chi2_sn(self, H0: float, Omega_m: float, epsilon: float) -> float:
        """Compute SNe Ia chi-squared."""
        cosmo = SpandrelCosmologyEngine(H0, Omega_m, epsilon)
        mu_model = cosmo.distance_modulus_spandrel(self.z_obs)
        return np.sum(((self.mu_obs - mu_model) ** 2) * self.inv_var)

    def chi2_priors(self, H0: float, Omega_m: float, epsilon: float) -> float:
        """Compute chi-squared contribution from all priors."""
        chi2_prior = 0.0
        for prior in self.active_priors:
            if prior.param == 'Omega_m':
                chi2_prior += prior.chi2(Omega_m)
            elif prior.param == 'H0':
                chi2_prior += prior.chi2(H0)
            elif prior.param == 'epsilon':
                chi2_prior += prior.chi2(epsilon)
        return chi2_prior

    def chi2_total(self, params: np.ndarray, use_spandrel: bool = True) -> float:
        """Total chi-squared = SNe + Priors."""
        if use_spandrel:
            H0, Omega_m, epsilon = params
        else:
            H0, Omega_m = params
            epsilon = 0.0

        # Physical bounds
        if H0 < 50 or H0 > 100 or Omega_m < 0.1 or Omega_m > 0.5:
            return 1e10
        if use_spandrel and (epsilon < -1.0 or epsilon > 1.0):
            return 1e10

        chi2_sn = self.chi2_sn(H0, Omega_m, epsilon)
        chi2_prior = self.chi2_priors(H0, Omega_m, epsilon)

        return chi2_sn + chi2_prior

    def fit(self, use_spandrel: bool = True,
            use_global: bool = True) -> Dict:
        """
        Perform joint fit with active priors.
        """
        model_name = "Spandrel" if use_spandrel else "ΛCDM"
        n_priors = len(self.active_priors)

        print(f"\n  Fitting {model_name} with {n_priors} external prior(s)...")

        def objective(params):
            return self.chi2_total(params, use_spandrel)

        if use_spandrel:
            bounds = [(60, 85), (0.15, 0.45), (-0.5, 0.5)]
            x0 = [70, 0.3, 0.0]
            n_params = 3
        else:
            bounds = [(60, 85), (0.15, 0.45)]
            x0 = [70, 0.3]
            n_params = 2

        if use_global:
            result = differential_evolution(
                objective, bounds, maxiter=2000, tol=1e-8,
                seed=42, workers=1, polish=True
            )
        else:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 10000})

        if use_spandrel:
            H0_fit, Om_fit, eps_fit = result.x
        else:
            H0_fit, Om_fit = result.x
            eps_fit = 0.0

        chi2_val = result.fun
        chi2_sn_val = self.chi2_sn(H0_fit, Om_fit, eps_fit)
        chi2_prior_val = self.chi2_priors(H0_fit, Om_fit, eps_fit)

        # Degrees of freedom accounting for priors
        dof = self.n_data - n_params
        reduced_chi2 = chi2_sn_val / dof  # Use SNe chi2 for goodness of fit
        p_value = 1 - chi2.cdf(chi2_sn_val, dof)

        return {
            'model': model_name,
            'H0': H0_fit,
            'Omega_m': Om_fit,
            'epsilon': eps_fit,
            'chi2_total': chi2_val,
            'chi2_sn': chi2_sn_val,
            'chi2_prior': chi2_prior_val,
            'dof': dof,
            'reduced_chi2': reduced_chi2,
            'p_value': p_value,
            'n_priors': n_priors
        }


# =============================================================================
# MCMC WITH PRIORS
# =============================================================================

class JointMCMC:
    """MCMC sampler with external priors."""

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray,
                 priors: List[CosmologicalPrior] = None):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err
        self.inv_var = 1.0 / (mu_err ** 2)
        self.priors = priors or []

    def log_likelihood(self, H0: float, Omega_m: float, epsilon: float) -> float:
        """Log-likelihood from SNe Ia."""
        cosmo = SpandrelCosmologyEngine(H0, Omega_m, epsilon)
        mu_model = cosmo.distance_modulus_spandrel(self.z_obs)
        chi2 = np.sum(((self.mu_obs - mu_model) ** 2) * self.inv_var)
        return -0.5 * chi2

    def log_prior(self, H0: float, Omega_m: float, epsilon: float) -> float:
        """Log-prior including external constraints."""
        # Uniform bounds
        if not (50 < H0 < 100):
            return -np.inf
        if not (0.1 < Omega_m < 0.5):
            return -np.inf
        if not (-1.0 < epsilon < 1.0):
            return -np.inf

        # External Gaussian priors
        log_p = 0.0
        for prior in self.priors:
            if prior.param == 'Omega_m':
                log_p += prior.log_prior(Omega_m)
            elif prior.param == 'H0':
                log_p += prior.log_prior(H0)

        return log_p

    def log_posterior(self, params: np.ndarray) -> float:
        """Log-posterior = log-prior + log-likelihood."""
        H0, Omega_m, epsilon = params
        lp = self.log_prior(H0, Omega_m, epsilon)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(H0, Omega_m, epsilon)

    def run_chain(self, n_samples: int, initial: np.ndarray,
                  proposal_sigma: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, float]:
        """Run single MCMC chain."""
        np.random.seed(seed)

        chain = np.zeros((n_samples, 3))
        current = initial.copy()
        current_lp = self.log_posterior(current)

        accepted = 0

        for i in range(n_samples):
            proposal = current + proposal_sigma * np.random.randn(3)
            proposal_lp = self.log_posterior(proposal)

            if np.log(np.random.random()) < proposal_lp - current_lp:
                current = proposal
                current_lp = proposal_lp
                accepted += 1

            chain[i] = current

        acceptance_rate = accepted / n_samples
        return chain, acceptance_rate

    def run_parallel(self, n_samples: int = 5000, n_burn: int = 1000,
                    n_chains: int = 4) -> Dict:
        """Run parallel MCMC chains."""
        print(f"\n  Running {n_chains} MCMC chains ({n_samples} samples each)...")

        initial = np.array([72.0, 0.31, 0.0])
        proposal_sigma = np.array([0.3, 0.005, 0.015])

        chains = []

        for i in range(n_chains):
            init_perturbed = initial + 0.1 * np.random.randn(3)
            chain, acc_rate = self.run_chain(
                n_samples + n_burn, init_perturbed, proposal_sigma, seed=42+i
            )
            chains.append(chain[n_burn:])
            print(f"    Chain {i}: acceptance = {acc_rate:.3f}")

        combined = np.vstack(chains)

        # Compute statistics
        stats = {}
        param_names = ['H0', 'Omega_m', 'epsilon']
        for i, name in enumerate(param_names):
            samples = combined[:, i]
            stats[name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                'q16': np.percentile(samples, 16),
                'q84': np.percentile(samples, 84),
                'q2.5': np.percentile(samples, 2.5),
                'q97.5': np.percentile(samples, 97.5)
            }

        return {'chains': combined, 'stats': stats}


# =============================================================================
# OSCILLATORY SPANDREL MODEL (Resonance Node Hypothesis)
# =============================================================================

class OscillatorySpandrelEngine(SpandrelCosmologyEngine):
    """
    Extended Spandrel model with oscillatory decay.

    The "Relaxed Node" hypothesis suggests that stiffness oscillates:

    Δμ = ε * sin(ω * ln(1+z)) * exp(-γ * z) * f(z)

    where:
    - ω: oscillation frequency in log-redshift
    - γ: decay rate
    - f(z): the standard Spandrel shape function
    """

    def __init__(self, H0: float, Omega_m: float, epsilon: float = 0.0,
                 omega: float = 2.0, gamma: float = 0.5):
        super().__init__(H0, Omega_m, epsilon)
        self.omega = omega
        self.gamma = gamma

    def spandrel_correction(self, z: np.ndarray) -> np.ndarray:
        """Oscillatory Spandrel correction."""
        z = np.atleast_1d(z)
        one_plus_z = 1.0 + z

        # Base shape function
        shape = np.log(one_plus_z) * (1.0 - 1.0/one_plus_z**2)

        # Oscillatory modulation
        oscillation = np.sin(self.omega * np.log(one_plus_z))

        # Exponential decay
        decay = np.exp(-self.gamma * z)

        return self.epsilon * shape * oscillation * decay


# =============================================================================
# MAIN JOINT ANALYSIS
# =============================================================================

def load_pantheon_data(filepath: str = "Pantheon+SH0ES.dat") -> Tuple[np.ndarray, ...]:
    """Load Pantheon+ data."""
    print(f"\nLoading data from {filepath}...")
    df = pd.read_csv(filepath, sep=r'\s+')

    z = df['zHD'].values
    mu = df['MU_SH0ES'].values
    err = df['MU_SH0ES_ERR_DIAG'].values

    mask = (z > 0.001) & (z < 2.5) & (mu > 0) & np.isfinite(mu) & np.isfinite(err)

    z_obs = z[mask]
    mu_obs = mu[mask]
    mu_err = err[mask]

    # Sort by redshift
    idx = np.argsort(z_obs)

    print(f"  Loaded {len(z_obs)} supernovae (z = {z_obs.min():.4f} to {z_obs.max():.4f})")

    return z_obs[idx], mu_obs[idx], mu_err[idx]


def run_degeneracy_analysis():
    """
    Run the full degeneracy-breaking analysis.

    Compares:
    1. Free fit (no priors) - shows degeneracy
    2. Planck prior on Ωₘ - breaks degeneracy
    3. SH0ES prior on H₀ - alternative anchor
    4. Combined priors - tightest constraints
    """

    print("\n" + "="*70)
    print("SPANDREL JOINT ANALYSIS: Breaking the Ωₘ-ε Degeneracy")
    print("="*70)

    # Load data
    z_obs, mu_obs, mu_err = load_pantheon_data()

    # Initialize fitter
    fitter = JointAnalysisFitter(z_obs, mu_obs, mu_err)

    results = {}

    # =========================================================================
    # Analysis 1: Free fit (baseline showing degeneracy)
    # =========================================================================
    print("\n" + "-"*70)
    print("ANALYSIS 1: Free Fit (No Priors) - Demonstrating Degeneracy")
    print("-"*70)

    fitter.clear_priors()

    results['lcdm_free'] = fitter.fit(use_spandrel=False)
    results['spandrel_free'] = fitter.fit(use_spandrel=True)

    print(f"\n  ΛCDM Free:     H₀ = {results['lcdm_free']['H0']:.3f}, "
          f"Ωₘ = {results['lcdm_free']['Omega_m']:.4f}")
    print(f"  Spandrel Free: H₀ = {results['spandrel_free']['H0']:.3f}, "
          f"Ωₘ = {results['spandrel_free']['Omega_m']:.4f}, "
          f"ε = {results['spandrel_free']['epsilon']:.6f}")

    delta_Om = results['spandrel_free']['Omega_m'] - results['lcdm_free']['Omega_m']
    print(f"\n  ⚠️  Matter density shift: ΔΩₘ = {delta_Om:+.4f}")
    print(f"      This {abs(delta_Om)/results['lcdm_free']['Omega_m']*100:.1f}% shift absorbs the stiffness!")

    # =========================================================================
    # Analysis 2: Planck prior on Ωₘ (The "Cosmic Vise")
    # =========================================================================
    print("\n" + "-"*70)
    print("ANALYSIS 2: Planck CMB Prior (Ωₘ = 0.315 ± 0.007)")
    print("-"*70)
    print("  Applying the 'Cosmic Vise' to pin down matter density...")

    fitter.clear_priors()
    fitter.add_prior(PRIORS['planck2018'])

    results['lcdm_planck'] = fitter.fit(use_spandrel=False)
    results['spandrel_planck'] = fitter.fit(use_spandrel=True)

    print(f"\n  ΛCDM + Planck:     H₀ = {results['lcdm_planck']['H0']:.3f}, "
          f"Ωₘ = {results['lcdm_planck']['Omega_m']:.4f}")
    print(f"  Spandrel + Planck: H₀ = {results['spandrel_planck']['H0']:.3f}, "
          f"Ωₘ = {results['spandrel_planck']['Omega_m']:.4f}, "
          f"ε = {results['spandrel_planck']['epsilon']:.6f}")

    print(f"\n  χ²_prior contribution: {results['spandrel_planck']['chi2_prior']:.2f}")

    # =========================================================================
    # Analysis 3: DESI BAO prior (newer, slightly different)
    # =========================================================================
    print("\n" + "-"*70)
    print("ANALYSIS 3: DESI BAO 2024 Prior (Ωₘ = 0.295 ± 0.015)")
    print("-"*70)

    fitter.clear_priors()
    fitter.add_prior(PRIORS['desi2024'])

    results['spandrel_desi'] = fitter.fit(use_spandrel=True)

    print(f"\n  Spandrel + DESI: H₀ = {results['spandrel_desi']['H0']:.3f}, "
          f"Ωₘ = {results['spandrel_desi']['Omega_m']:.4f}, "
          f"ε = {results['spandrel_desi']['epsilon']:.6f}")

    # =========================================================================
    # Analysis 4: Combined Priors (Maximum constraint)
    # =========================================================================
    print("\n" + "-"*70)
    print("ANALYSIS 4: Combined Priors (Planck Ωₘ + SH0ES H₀)")
    print("-"*70)
    print("  Testing for internal consistency with local H₀...")

    fitter.clear_priors()
    fitter.add_prior(PRIORS['planck2018'])
    fitter.add_prior(PRIORS['sh0es2022'])

    results['spandrel_combined'] = fitter.fit(use_spandrel=True)

    print(f"\n  Spandrel Combined: H₀ = {results['spandrel_combined']['H0']:.3f}, "
          f"Ωₘ = {results['spandrel_combined']['Omega_m']:.4f}, "
          f"ε = {results['spandrel_combined']['epsilon']:.6f}")
    print(f"  χ²_prior = {results['spandrel_combined']['chi2_prior']:.2f} "
          f"(tension between priors!)")

    # =========================================================================
    # MCMC with Planck Prior for proper uncertainties
    # =========================================================================
    print("\n" + "-"*70)
    print("ANALYSIS 5: MCMC Posterior with Planck Prior")
    print("-"*70)

    mcmc = JointMCMC(z_obs, mu_obs, mu_err, priors=[PRIORS['planck2018']])
    mcmc_result = mcmc.run_parallel(n_samples=3000, n_burn=500, n_chains=4)

    results['mcmc_planck'] = mcmc_result

    print("\n  Posterior estimates (with Planck prior):")
    for param, stats in mcmc_result['stats'].items():
        print(f"    {param}: {stats['mean']:.6f} ± {stats['std']:.6f} "
              f"(95% CI: [{stats['q2.5']:.6f}, {stats['q97.5']:.6f}])")

    # =========================================================================
    # Summary and Interpretation
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: Stiffness Parameter Under Different Priors")
    print("="*70)

    print(f"""
    Analysis                      ε (stiffness)      Ωₘ (matter)
    ─────────────────────────────────────────────────────────────
    Free fit (degeneracy)         {results['spandrel_free']['epsilon']:+.6f}       {results['spandrel_free']['Omega_m']:.4f}
    + Planck Ωₘ prior             {results['spandrel_planck']['epsilon']:+.6f}       {results['spandrel_planck']['Omega_m']:.4f}
    + DESI BAO prior              {results['spandrel_desi']['epsilon']:+.6f}       {results['spandrel_desi']['Omega_m']:.4f}
    + Combined (Planck+SH0ES)     {results['spandrel_combined']['epsilon']:+.6f}       {results['spandrel_combined']['Omega_m']:.4f}

    MCMC with Planck prior:
      ε = {mcmc_result['stats']['epsilon']['mean']:.6f} ± {mcmc_result['stats']['epsilon']['std']:.6f}
      95% CI: [{mcmc_result['stats']['epsilon']['q2.5']:.6f}, {mcmc_result['stats']['epsilon']['q97.5']:.6f}]
    """)

    # Statistical test
    eps_mean = mcmc_result['stats']['epsilon']['mean']
    eps_std = mcmc_result['stats']['epsilon']['std']
    significance = abs(eps_mean) / eps_std

    print("="*70)
    print("INTERPRETATION")
    print("="*70)

    if significance < 2:
        print(f"""
    Detection Significance: {significance:.2f}σ (NOT SIGNIFICANT)

    Result: ε is STILL consistent with zero even after breaking the
            Ωₘ-ε degeneracy with Planck priors.

    Conclusion: The "Relaxed Node" hypothesis may be correct.
                The current epoch (z ≈ 0) appears to be at a
                resonance node where stiffness is minimal.

    Next Steps:
      1. Test oscillatory Spandrel models
      2. Examine higher-z data (CMB, high-z quasars)
      3. Look for stiffness in the early universe (z > 2)
        """)
    elif eps_mean > 0:
        print(f"""
    Detection Significance: {significance:.2f}σ (POTENTIALLY SIGNIFICANT!)

    Result: ε > 0 detected after degeneracy breaking!

    Interpretation: With Ωₘ pinned to Planck values, the data
                    REQUIRES positive stiffness to fit.

    Physical Meaning:
      - The universe was "stiffer" at higher redshifts
      - Dark energy may not be a cosmological constant
      - This could partially explain the Hubble tension
        """)
    else:
        print(f"""
    Detection Significance: {significance:.2f}σ

    Result: ε < 0 detected (universe "softening")

    This unexpected result requires further investigation.
        """)

    # Likelihood ratio test with priors
    print("\n" + "-"*70)
    print("LIKELIHOOD RATIO TEST (with Planck prior)")
    print("-"*70)

    delta_chi2 = results['lcdm_planck']['chi2_total'] - results['spandrel_planck']['chi2_total']
    p_value = 1 - chi2.cdf(delta_chi2, 1)
    sigma = norm.ppf(1 - p_value/2) if p_value > 0 else float('inf')

    print(f"  Δχ² = {delta_chi2:.4f}")
    print(f"  p-value = {p_value:.6f}")
    print(f"  Significance = {sigma:.2f}σ")

    if delta_chi2 > 3.84:
        print("\n  *** Spandrel model PREFERRED at 95% confidence ***")
    else:
        print("\n  ΛCDM remains statistically sufficient")

    return results


if __name__ == "__main__":
    results = run_degeneracy_analysis()
