#!/usr/bin/env python3
"""
Spandrel Cosmology HPC Analysis Framework
==========================================

High-Performance Computing implementation for testing the Spandrel Hypothesis
against the Pantheon+ Type Ia Supernova dataset.

Optimizations:
- Vectorized NumPy operations for CPU cache efficiency
- Multiprocessing MCMC with parallel chains
- MLX Metal GPU acceleration (Apple Silicon native)
- Nested sampling for Bayesian evidence computation
- Memory-mapped data for large datasets

Author: Spandrel Cosmology Project
Hardware Target: Apple Silicon (M1/M2/M3) with Metal GPU
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad_vec, solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2, norm
from scipy.special import logsumexp
from typing import Tuple, Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Import physical constants from central module
from constants import C_LIGHT_KMS as C_LIGHT, H0_FIDUCIAL, H0_PLANCK, H0_SH0ES, OMEGA_M_FIDUCIAL, GAMMA_1, RIEMANN_ZEROS

# CPU core count for parallel operations
NUM_CORES = mp.cpu_count()
print(f"Detected {NUM_CORES} CPU cores for parallel computation")

# Attempt to import MLX for Metal GPU acceleration
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
    print("MLX Metal GPU acceleration: AVAILABLE")
except ImportError:
    HAS_MLX = False
    print("MLX Metal GPU acceleration: NOT AVAILABLE (install with: pip install mlx)")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CosmologyParams:
    """Container for cosmological parameters."""
    H0: float = 70.0
    Omega_m: float = 0.3
    Omega_Lambda: float = 0.7
    epsilon: float = 0.0  # Spandrel stiffness
    w0: float = -1.0      # Dark energy equation of state (w0)
    wa: float = 0.0       # Dark energy evolution (wa in w(a) = w0 + wa*(1-a))

    def __post_init__(self):
        if self.Omega_Lambda == 0.7:  # Default flat universe
            self.Omega_Lambda = 1.0 - self.Omega_m


@dataclass
class FitResult:
    """Container for fit results."""
    params: CosmologyParams
    chi2: float
    dof: int
    reduced_chi2: float
    p_value: float
    model_name: str
    errors: Dict[str, float] = field(default_factory=dict)
    covariance: Optional[np.ndarray] = None
    chain: Optional[np.ndarray] = None  # MCMC chain


@dataclass
class BayesianEvidence:
    """Container for Bayesian model comparison."""
    log_evidence: float
    log_evidence_err: float
    information: float  # Kullback-Leibler divergence
    n_live: int
    n_iterations: int


# =============================================================================
# VECTORIZED COSMOLOGY ENGINE (CPU-OPTIMIZED)
# =============================================================================

class VectorizedCosmology:
    """
    High-performance vectorized cosmology calculations.

    Uses NumPy's SIMD operations and cache-friendly memory access patterns
    for maximum CPU throughput.
    """

    def __init__(self, params: CosmologyParams):
        self.params = params
        # Pre-compute constants for efficiency
        self._c_over_H0 = C_LIGHT / params.H0

    def E_squared(self, z: np.ndarray) -> np.ndarray:
        """
        Vectorized E²(z) = H²(z)/H₀² for flat wCDM cosmology.

        E²(z) = Ωₘ(1+z)³ + Ω_Λ * f(z)

        where f(z) accounts for dark energy equation of state.
        """
        z = np.atleast_1d(z)
        one_plus_z = 1.0 + z

        # Matter contribution
        matter = self.params.Omega_m * one_plus_z**3

        # Dark energy with w(a) = w0 + wa*(1-a) = w0 + wa*z/(1+z)
        if self.params.w0 == -1.0 and self.params.wa == 0.0:
            # Cosmological constant (fast path)
            dark_energy = self.params.Omega_Lambda
        else:
            # General w(z) integration
            # For w(a) = w0 + wa*(1-a), the density evolution is:
            # ρ_DE ∝ (1+z)^(3(1+w0+wa)) * exp(-3*wa*z/(1+z))
            w0, wa = self.params.w0, self.params.wa
            exponent = 3.0 * (1.0 + w0 + wa)
            dark_energy = self.params.Omega_Lambda * (
                one_plus_z**exponent * np.exp(-3.0 * wa * z / one_plus_z)
            )

        return matter + dark_energy

    def E(self, z: np.ndarray) -> np.ndarray:
        """Vectorized E(z) = H(z)/H₀."""
        return np.sqrt(self.E_squared(z))

    def comoving_distance_vectorized(self, z: np.ndarray, n_steps: int = 1000) -> np.ndarray:
        """
        Vectorized comoving distance using trapezoidal integration.

        D_C(z) = (c/H₀) * ∫₀ᶻ dz'/E(z')

        This is much faster than scipy.integrate.quad for arrays.
        """
        z = np.atleast_1d(z)
        n_z = len(z)

        # Create integration grid for all redshifts simultaneously
        # Shape: (n_steps, n_z)
        z_grid = np.linspace(0, z, n_steps).T  # Shape: (n_z, n_steps)

        # Compute E(z) for entire grid
        E_grid = self.E(z_grid)  # Shape: (n_z, n_steps)

        # Trapezoidal integration along axis 1
        integrand = 1.0 / E_grid
        dz = z / (n_steps - 1)
        dz = dz[:, np.newaxis]  # Shape: (n_z, 1)

        # Trapezoidal rule
        integral = np.sum(integrand[:, :-1] + integrand[:, 1:], axis=1) * dz.flatten() / 2.0

        return self._c_over_H0 * integral

    def luminosity_distance_vectorized(self, z: np.ndarray, n_steps: int = 1000) -> np.ndarray:
        """
        Vectorized luminosity distance D_L(z) = (1+z) * D_C(z).
        """
        z = np.atleast_1d(z)
        d_c = self.comoving_distance_vectorized(z, n_steps)
        return (1.0 + z) * d_c

    def distance_modulus_vectorized(self, z: np.ndarray, n_steps: int = 1000) -> np.ndarray:
        """
        Vectorized distance modulus μ = 5*log₁₀(D_L/10pc).
        """
        d_L = self.luminosity_distance_vectorized(z, n_steps)
        return 5.0 * np.log10(d_L * 1e6 / 10.0)

    def spandrel_correction_vectorized(self, z: np.ndarray) -> np.ndarray:
        """
        Vectorized Spandrel stiffness correction.

        Δμ = ε * ln(1+z) * (1 - 1/(1+z)²)
        """
        z = np.atleast_1d(z)
        one_plus_z = 1.0 + z
        return self.params.epsilon * np.log(one_plus_z) * (1.0 - 1.0/one_plus_z**2)

    def distance_modulus_spandrel_vectorized(self, z: np.ndarray, n_steps: int = 1000) -> np.ndarray:
        """
        Vectorized Spandrel-modified distance modulus.

        μ_spandrel = μ_ΛCDM + Δμ_stiffness
        """
        mu_lcdm = self.distance_modulus_vectorized(z, n_steps)
        correction = self.spandrel_correction_vectorized(z)
        return mu_lcdm + correction


# =============================================================================
# MLX METAL GPU ACCELERATION (APPLE SILICON)
# =============================================================================

if HAS_MLX:
    class MLXCosmology:
        """
        Metal GPU-accelerated cosmology using Apple's MLX framework.

        This runs directly on the Apple Silicon GPU with unified memory,
        providing massive parallelism for likelihood evaluations.
        """

        def __init__(self, params: CosmologyParams):
            self.params = params
            self._c_over_H0 = mx.array(C_LIGHT / params.H0)
            self.Omega_m = mx.array(params.Omega_m)
            self.Omega_Lambda = mx.array(params.Omega_Lambda)
            self.epsilon = mx.array(params.epsilon)
            self.w0 = mx.array(params.w0)
            self.wa = mx.array(params.wa)

        def E_squared_mlx(self, z: mx.array) -> mx.array:
            """GPU-accelerated E²(z)."""
            one_plus_z = 1.0 + z
            matter = self.Omega_m * mx.power(one_plus_z, 3)

            if float(self.w0) == -1.0 and float(self.wa) == 0.0:
                dark_energy = self.Omega_Lambda
            else:
                exponent = 3.0 * (1.0 + self.w0 + self.wa)
                dark_energy = self.Omega_Lambda * (
                    mx.power(one_plus_z, exponent) *
                    mx.exp(-3.0 * self.wa * z / one_plus_z)
                )

            return matter + dark_energy

        def E_mlx(self, z: mx.array) -> mx.array:
            """GPU-accelerated E(z)."""
            return mx.sqrt(self.E_squared_mlx(z))

        def comoving_distance_mlx(self, z: mx.array, n_steps: int = 1000) -> mx.array:
            """
            GPU-accelerated comoving distance via trapezoidal integration.
            """
            # Create integration grid on GPU
            z_max = mx.max(z)
            z_grid = mx.linspace(0, float(z_max), n_steps)

            # Broadcast for all target redshifts
            # z shape: (n_z,), z_grid shape: (n_steps,)
            # We need z_grid scaled to each z
            scale = z / z_max  # (n_z,)
            z_integration = mx.outer(scale, z_grid)  # (n_z, n_steps)

            # Compute E(z) on GPU
            E_values = self.E_mlx(z_integration)
            integrand = 1.0 / E_values

            # Trapezoidal integration
            dz = z / (n_steps - 1)
            dz = mx.reshape(dz, (-1, 1))

            integral = mx.sum(integrand[:, :-1] + integrand[:, 1:], axis=1) * mx.squeeze(dz) / 2.0

            return self._c_over_H0 * integral

        def luminosity_distance_mlx(self, z: mx.array, n_steps: int = 1000) -> mx.array:
            """GPU-accelerated luminosity distance."""
            d_c = self.comoving_distance_mlx(z, n_steps)
            return (1.0 + z) * d_c

        def distance_modulus_mlx(self, z: mx.array, n_steps: int = 1000) -> mx.array:
            """GPU-accelerated distance modulus."""
            d_L = self.luminosity_distance_mlx(z, n_steps)
            return 5.0 * mx.log10(d_L * 1e6 / 10.0)

        def spandrel_correction_mlx(self, z: mx.array) -> mx.array:
            """GPU-accelerated Spandrel correction."""
            one_plus_z = 1.0 + z
            return self.epsilon * mx.log(one_plus_z) * (1.0 - 1.0/mx.power(one_plus_z, 2))

        def distance_modulus_spandrel_mlx(self, z: mx.array, n_steps: int = 1000) -> mx.array:
            """GPU-accelerated Spandrel distance modulus."""
            mu_lcdm = self.distance_modulus_mlx(z, n_steps)
            correction = self.spandrel_correction_mlx(z)
            return mu_lcdm + correction

        def chi_squared_mlx(self, z_obs: mx.array, mu_obs: mx.array,
                           mu_err: mx.array, use_spandrel: bool = True) -> float:
            """
            GPU-accelerated chi-squared computation.
            """
            if use_spandrel:
                mu_model = self.distance_modulus_spandrel_mlx(z_obs)
            else:
                mu_model = self.distance_modulus_mlx(z_obs)

            residuals = (mu_obs - mu_model) / mu_err
            chi2_val = mx.sum(mx.power(residuals, 2))

            # Synchronize and return as Python float
            mx.eval(chi2_val)
            return float(chi2_val)


# =============================================================================
# PARALLEL MCMC ENGINE
# =============================================================================

class ParallelMCMC:
    """
    Parallel Markov Chain Monte Carlo sampler with multiple chains.

    Uses multiprocessing for embarrassingly parallel chain execution
    on all available CPU cores.
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray,
                 n_chains: int = None, use_gpu: bool = True):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err
        self.n_chains = n_chains or NUM_CORES
        self.use_gpu = use_gpu and HAS_MLX

        # Precompute inverse variance for efficiency
        self.inv_var = 1.0 / (mu_err**2)

        # Convert to MLX arrays if using GPU
        if self.use_gpu:
            self.z_obs_mlx = mx.array(z_obs)
            self.mu_obs_mlx = mx.array(mu_obs)
            self.mu_err_mlx = mx.array(mu_err)

    def log_likelihood(self, params: np.ndarray, use_spandrel: bool = True) -> float:
        """
        Compute log-likelihood for given parameters.

        params: [H0, Omega_m, epsilon] or [H0, Omega_m] for ΛCDM
        """
        if len(params) == 3:
            H0, Omega_m, epsilon = params
        else:
            H0, Omega_m = params
            epsilon = 0.0

        # Physical bounds (uniform prior)
        if H0 < 50 or H0 > 100 or Omega_m < 0.05 or Omega_m > 0.7:
            return -np.inf
        if use_spandrel and (epsilon < -1.0 or epsilon > 1.0):
            return -np.inf

        cosmo_params = CosmologyParams(H0=H0, Omega_m=Omega_m, epsilon=epsilon)

        if self.use_gpu:
            cosmo = MLXCosmology(cosmo_params)
            chi2_val = cosmo.chi_squared_mlx(
                self.z_obs_mlx, self.mu_obs_mlx, self.mu_err_mlx, use_spandrel
            )
        else:
            cosmo = VectorizedCosmology(cosmo_params)
            if use_spandrel:
                mu_model = cosmo.distance_modulus_spandrel_vectorized(self.z_obs)
            else:
                mu_model = cosmo.distance_modulus_vectorized(self.z_obs)
            chi2_val = np.sum(((self.mu_obs - mu_model) / self.mu_err)**2)

        return -0.5 * chi2_val

    def log_prior(self, params: np.ndarray, use_spandrel: bool = True) -> float:
        """
        Compute log-prior for given parameters.

        Using broad uniform priors:
        - H0: [50, 100] km/s/Mpc
        - Omega_m: [0.05, 0.7]
        - epsilon: [-1, 1] (if Spandrel)
        """
        if len(params) == 3:
            H0, Omega_m, epsilon = params
        else:
            H0, Omega_m = params
            epsilon = 0.0

        # Uniform priors
        if not (50 < H0 < 100):
            return -np.inf
        if not (0.05 < Omega_m < 0.7):
            return -np.inf
        if use_spandrel and not (-1.0 < epsilon < 1.0):
            return -np.inf

        return 0.0  # Uniform prior

    def log_posterior(self, params: np.ndarray, use_spandrel: bool = True) -> float:
        """Log posterior = log prior + log likelihood."""
        lp = self.log_prior(params, use_spandrel)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params, use_spandrel)

    def run_single_chain(self, chain_id: int, n_samples: int,
                         initial_params: np.ndarray, proposal_sigma: np.ndarray,
                         use_spandrel: bool, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single MCMC chain using Metropolis-Hastings.

        Returns: (chain samples, log_prob for each sample)
        """
        np.random.seed(seed + chain_id)

        n_params = len(initial_params)
        chain = np.zeros((n_samples, n_params))
        log_probs = np.zeros(n_samples)

        current_params = initial_params.copy()
        current_log_prob = self.log_posterior(current_params, use_spandrel)

        accepted = 0

        for i in range(n_samples):
            # Propose new parameters
            proposal = current_params + proposal_sigma * np.random.randn(n_params)
            proposal_log_prob = self.log_posterior(proposal, use_spandrel)

            # Metropolis-Hastings acceptance
            log_alpha = proposal_log_prob - current_log_prob

            if np.log(np.random.random()) < log_alpha:
                current_params = proposal
                current_log_prob = proposal_log_prob
                accepted += 1

            chain[i] = current_params
            log_probs[i] = current_log_prob

        acceptance_rate = accepted / n_samples
        print(f"  Chain {chain_id}: acceptance rate = {acceptance_rate:.3f}")

        return chain, log_probs

    def run_parallel_chains(self, n_samples: int = 10000, n_burn: int = 2000,
                           use_spandrel: bool = True,
                           initial_params: np.ndarray = None) -> Dict[str, Any]:
        """
        Run multiple MCMC chains in parallel.

        Returns dict with:
        - chains: combined chains (n_chains * n_samples, n_params)
        - log_probs: log probabilities
        - diagnostics: convergence diagnostics
        """
        print(f"\nRunning {self.n_chains} parallel MCMC chains...")
        print(f"  Samples per chain: {n_samples + n_burn} (burn-in: {n_burn})")

        n_params = 3 if use_spandrel else 2

        if initial_params is None:
            initial_params = np.array([70.0, 0.3, 0.0][:n_params])

        # Proposal widths (tuned for ~25% acceptance)
        proposal_sigma = np.array([0.5, 0.01, 0.02][:n_params])

        start_time = time.time()

        # Run chains in parallel
        chains = []
        log_probs_all = []

        # Use ProcessPoolExecutor for true parallelism
        # Note: On Apple Silicon, this uses all performance and efficiency cores
        with ProcessPoolExecutor(max_workers=self.n_chains) as executor:
            futures = []
            for chain_id in range(self.n_chains):
                # Slightly perturb initial conditions for each chain
                init = initial_params + 0.1 * np.random.randn(n_params)
                future = executor.submit(
                    self.run_single_chain, chain_id, n_samples + n_burn,
                    init, proposal_sigma, use_spandrel, 42
                )
                futures.append(future)

            for future in futures:
                chain, log_probs = future.result()
                # Remove burn-in
                chains.append(chain[n_burn:])
                log_probs_all.append(log_probs[n_burn:])

        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f}s ({n_samples * self.n_chains / elapsed:.0f} samples/s)")

        # Combine chains
        combined_chain = np.vstack(chains)
        combined_log_probs = np.concatenate(log_probs_all)

        # Compute diagnostics
        diagnostics = self._compute_diagnostics(chains)

        return {
            'chains': combined_chain,
            'log_probs': combined_log_probs,
            'individual_chains': chains,
            'diagnostics': diagnostics,
            'elapsed_time': elapsed
        }

    def _compute_diagnostics(self, chains: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute MCMC convergence diagnostics.

        Includes:
        - Gelman-Rubin R-hat statistic
        - Effective sample size
        - Autocorrelation time
        """
        n_chains = len(chains)
        n_samples = chains[0].shape[0]
        n_params = chains[0].shape[1]

        # Gelman-Rubin R-hat
        chain_means = np.array([np.mean(c, axis=0) for c in chains])
        chain_vars = np.array([np.var(c, axis=0, ddof=1) for c in chains])

        overall_mean = np.mean(chain_means, axis=0)

        B = n_samples * np.var(chain_means, axis=0, ddof=1)  # Between-chain variance
        W = np.mean(chain_vars, axis=0)  # Within-chain variance

        var_estimate = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        R_hat = np.sqrt(var_estimate / W)

        # Effective sample size (simple estimate)
        combined = np.vstack(chains)
        n_eff = np.zeros(n_params)

        for p in range(n_params):
            acf = np.correlate(combined[:, p] - np.mean(combined[:, p]),
                              combined[:, p] - np.mean(combined[:, p]), mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]

            # Find where ACF drops below 0.05
            tau = 1 + 2 * np.sum(acf[1:min(100, len(acf))])
            n_eff[p] = len(combined) / tau

        return {
            'R_hat': R_hat,
            'n_eff': n_eff,
            'converged': np.all(R_hat < 1.1)
        }

    def compute_statistics(self, chain: np.ndarray,
                          param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute posterior statistics from MCMC chain.
        """
        stats = {}
        for i, name in enumerate(param_names):
            samples = chain[:, i]
            stats[name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'median': np.median(samples),
                'q16': np.percentile(samples, 16),
                'q84': np.percentile(samples, 84),
                'q2.5': np.percentile(samples, 2.5),
                'q97.5': np.percentile(samples, 97.5)
            }
        return stats


# =============================================================================
# NESTED SAMPLING FOR BAYESIAN EVIDENCE
# =============================================================================

class NestedSampler:
    """
    Nested sampling implementation for computing Bayesian evidence.

    This allows proper model comparison between ΛCDM and Spandrel cosmology
    using the Bayes factor.
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray,
                 n_live: int = 500):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err
        self.n_live = n_live

        # Pre-compute for efficiency
        self.inv_var = 1.0 / (mu_err**2)

    def log_likelihood_fast(self, params: np.ndarray, use_spandrel: bool) -> float:
        """Fast log-likelihood computation."""
        if len(params) == 3:
            H0, Omega_m, epsilon = params
        else:
            H0, Omega_m = params
            epsilon = 0.0

        cosmo = VectorizedCosmology(CosmologyParams(H0=H0, Omega_m=Omega_m, epsilon=epsilon))

        if use_spandrel:
            mu_model = cosmo.distance_modulus_spandrel_vectorized(self.z_obs)
        else:
            mu_model = cosmo.distance_modulus_vectorized(self.z_obs)

        chi2_val = np.sum(((self.mu_obs - mu_model)**2) * self.inv_var)
        return -0.5 * chi2_val

    def sample_prior(self, n_samples: int, use_spandrel: bool) -> np.ndarray:
        """Sample from the prior distribution."""
        n_params = 3 if use_spandrel else 2
        samples = np.zeros((n_samples, n_params))

        # H0 ~ Uniform(50, 100)
        samples[:, 0] = np.random.uniform(50, 100, n_samples)
        # Omega_m ~ Uniform(0.05, 0.7)
        samples[:, 1] = np.random.uniform(0.05, 0.7, n_samples)

        if use_spandrel:
            # epsilon ~ Uniform(-1, 1)
            samples[:, 2] = np.random.uniform(-1, 1, n_samples)

        return samples

    def run(self, use_spandrel: bool = True,
            tol: float = 0.1, max_iter: int = 50000) -> BayesianEvidence:
        """
        Run nested sampling to compute Bayesian evidence.

        Returns BayesianEvidence dataclass with log(Z) and uncertainty.
        """
        print(f"\nRunning Nested Sampling ({'Spandrel' if use_spandrel else 'ΛCDM'})...")
        print(f"  Live points: {self.n_live}")

        n_params = 3 if use_spandrel else 2

        # Initialize live points from prior
        live_points = self.sample_prior(self.n_live, use_spandrel)
        live_log_L = np.array([
            self.log_likelihood_fast(p, use_spandrel) for p in live_points
        ])

        # Nested sampling loop
        log_Z = -np.inf  # Log evidence
        log_X = 0.0      # Log prior volume
        H = 0.0          # Information

        dead_points = []
        dead_log_L = []

        start_time = time.time()

        for iteration in range(max_iter):
            # Find worst live point
            worst_idx = np.argmin(live_log_L)
            worst_log_L = live_log_L[worst_idx]
            worst_point = live_points[worst_idx].copy()

            # Save dead point
            dead_points.append(worst_point)
            dead_log_L.append(worst_log_L)

            # Update evidence
            log_X_new = log_X - 1.0 / self.n_live
            log_weight = np.log(np.exp(log_X) - np.exp(log_X_new)) + worst_log_L
            log_Z = np.logaddexp(log_Z, log_weight)

            # Update information
            if np.isfinite(log_Z):
                H = np.exp(log_weight - log_Z) * worst_log_L + \
                    np.exp(log_Z - log_weight) * (H - log_Z) + log_Z

            log_X = log_X_new

            # Check convergence
            remaining_evidence = np.max(live_log_L) + log_X
            if remaining_evidence - log_Z < np.log(tol):
                print(f"  Converged at iteration {iteration}")
                break

            # Replace worst point with new point from constrained prior
            # Using simple MCMC within likelihood constraint
            new_point = self._sample_constrained(
                live_points, worst_log_L, use_spandrel, n_steps=20
            )
            new_log_L = self.log_likelihood_fast(new_point, use_spandrel)

            live_points[worst_idx] = new_point
            live_log_L[worst_idx] = new_log_L

            if iteration % 1000 == 0:
                print(f"  Iteration {iteration}: log(Z) = {log_Z:.2f}")

        # Add remaining live points
        for i in range(self.n_live):
            log_weight = log_X - np.log(self.n_live) + live_log_L[i]
            log_Z = np.logaddexp(log_Z, log_weight)

        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f}s")
        print(f"  Final log(Z) = {log_Z:.4f}")

        # Estimate uncertainty (simplified)
        log_Z_err = np.sqrt(H / self.n_live)

        return BayesianEvidence(
            log_evidence=log_Z,
            log_evidence_err=log_Z_err,
            information=H,
            n_live=self.n_live,
            n_iterations=iteration
        )

    def _sample_constrained(self, live_points: np.ndarray, min_log_L: float,
                           use_spandrel: bool, n_steps: int) -> np.ndarray:
        """Sample new point with likelihood > min_log_L using MCMC."""
        # Start from random live point
        idx = np.random.randint(len(live_points))
        current = live_points[idx].copy()

        n_params = len(current)
        step_size = np.array([1.0, 0.02, 0.05][:n_params])

        for _ in range(n_steps):
            # Propose
            proposal = current + step_size * np.random.randn(n_params)

            # Check bounds
            if proposal[0] < 50 or proposal[0] > 100:
                continue
            if proposal[1] < 0.05 or proposal[1] > 0.7:
                continue
            if use_spandrel and (proposal[2] < -1 or proposal[2] > 1):
                continue

            # Check likelihood constraint
            prop_log_L = self.log_likelihood_fast(proposal, use_spandrel)
            if prop_log_L > min_log_L:
                current = proposal

        return current


# =============================================================================
# EXTENDED DATA LOADER
# =============================================================================

class PantheonPlusLoaderHPC:
    """
    High-performance loader for Pantheon+ data with memory mapping support.
    """

    def __init__(self, filepath: str = "Pantheon+SH0ES.dat"):
        self.filepath = filepath
        self.dataframe = None
        self.z_obs = None
        self.mu_obs = None
        self.mu_err = None
        self.metadata = {}

    def load(self, z_min: float = 0.001, z_max: float = 2.5,
             use_sh0es_calibration: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load Pantheon+ data with optional SH0ES calibration.
        """
        print(f"Loading Pantheon+ dataset from: {self.filepath}")

        # Load with optimized dtypes
        dtype_spec = {
            'zHD': np.float64,
            'MU_SH0ES': np.float64,
            'MU_SH0ES_ERR_DIAG': np.float64,
            'IS_CALIBRATOR': np.int8
        }

        self.dataframe = pd.read_csv(
            self.filepath,
            sep=r'\s+',
            usecols=['zHD', 'MU_SH0ES', 'MU_SH0ES_ERR_DIAG', 'IS_CALIBRATOR', 'IDSURVEY'],
            dtype=dtype_spec
        )

        # Extract and filter
        z_raw = self.dataframe['zHD'].values
        mu_raw = self.dataframe['MU_SH0ES'].values
        err_raw = self.dataframe['MU_SH0ES_ERR_DIAG'].values

        mask = (
            (z_raw > z_min) &
            (z_raw < z_max) &
            (mu_raw > 0) &
            np.isfinite(mu_raw) &
            np.isfinite(err_raw)
        )

        self.z_obs = np.ascontiguousarray(z_raw[mask])
        self.mu_obs = np.ascontiguousarray(mu_raw[mask])
        self.mu_err = np.ascontiguousarray(err_raw[mask])

        # Sort by redshift for cache efficiency
        sort_idx = np.argsort(self.z_obs)
        self.z_obs = self.z_obs[sort_idx]
        self.mu_obs = self.mu_obs[sort_idx]
        self.mu_err = self.mu_err[sort_idx]

        # Metadata
        self.metadata = {
            'total_raw': len(z_raw),
            'total_valid': len(self.z_obs),
            'z_range': (float(self.z_obs.min()), float(self.z_obs.max())),
            'mu_range': (float(self.mu_obs.min()), float(self.mu_obs.max())),
            'n_calibrators': int(self.dataframe['IS_CALIBRATOR'].sum()),
            'n_surveys': int(self.dataframe['IDSURVEY'].nunique())
        }

        print(f"  Total supernovae: {self.metadata['total_raw']}")
        print(f"  Valid after cuts: {self.metadata['total_valid']}")
        print(f"  Redshift range: z = {self.metadata['z_range'][0]:.4f} to {self.metadata['z_range'][1]:.4f}")
        print(f"  Calibrator SNe (Cepheid hosts): {self.metadata['n_calibrators']}")
        print(f"  Number of surveys: {self.metadata['n_surveys']}")

        return self.z_obs, self.mu_obs, self.mu_err

    def get_redshift_bins(self, n_bins: int = 10) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Split data into redshift bins for systematic tests."""
        if self.z_obs is None:
            self.load()

        z_edges = np.percentile(self.z_obs, np.linspace(0, 100, n_bins + 1))
        bins = []

        for i in range(n_bins):
            mask = (self.z_obs >= z_edges[i]) & (self.z_obs < z_edges[i + 1])
            bins.append((
                self.z_obs[mask],
                self.mu_obs[mask],
                self.mu_err[mask]
            ))

        return bins


# =============================================================================
# HIGH-LEVEL ANALYSIS PIPELINE
# =============================================================================

class SpandrelAnalysisPipeline:
    """
    Complete analysis pipeline for Spandrel cosmology hypothesis testing.

    Combines:
    - Maximum likelihood fitting
    - MCMC posterior sampling
    - Bayesian evidence computation
    - Model comparison statistics
    """

    def __init__(self, data_path: str = "Pantheon+SH0ES.dat"):
        self.data_path = data_path
        self.loader = PantheonPlusLoaderHPC(data_path)
        self.z_obs = None
        self.mu_obs = None
        self.mu_err = None
        self.results = {}

    def load_data(self, z_min: float = 0.001, z_max: float = 2.5):
        """Load and prepare supernova data."""
        self.z_obs, self.mu_obs, self.mu_err = self.loader.load(z_min, z_max)
        return self

    def fit_mle(self, model: str = 'both') -> Dict[str, FitResult]:
        """
        Maximum Likelihood Estimation for ΛCDM and/or Spandrel.
        """
        print("\n" + "="*70)
        print("MAXIMUM LIKELIHOOD ESTIMATION")
        print("="*70)

        results = {}

        if model in ['lcdm', 'both']:
            results['lcdm'] = self._fit_mle_model(use_spandrel=False)

        if model in ['spandrel', 'both']:
            results['spandrel'] = self._fit_mle_model(use_spandrel=True)

        self.results['mle'] = results
        return results

    def _fit_mle_model(self, use_spandrel: bool) -> FitResult:
        """Internal MLE fitting."""
        model_name = "Spandrel" if use_spandrel else "ΛCDM"
        print(f"\nFitting {model_name} model...")

        def objective(params):
            if use_spandrel:
                H0, Om, eps = params
            else:
                H0, Om = params
                eps = 0.0

            if H0 < 50 or H0 > 100 or Om < 0.05 or Om > 0.7:
                return 1e10

            cosmo = VectorizedCosmology(CosmologyParams(H0=H0, Omega_m=Om, epsilon=eps))

            if use_spandrel:
                mu_model = cosmo.distance_modulus_spandrel_vectorized(self.z_obs)
            else:
                mu_model = cosmo.distance_modulus_vectorized(self.z_obs)

            return np.sum(((self.mu_obs - mu_model) / self.mu_err)**2)

        # Global optimization
        if use_spandrel:
            bounds = [(60, 85), (0.15, 0.45), (-0.3, 0.3)]
            x0 = [70, 0.3, 0.0]
        else:
            bounds = [(60, 85), (0.15, 0.45)]
            x0 = [70, 0.3]

        result = differential_evolution(
            objective, bounds, maxiter=2000, tol=1e-8, seed=42,
            workers=1, updating='immediate', polish=True
        )

        if use_spandrel:
            H0_fit, Om_fit, eps_fit = result.x
            n_params = 3
        else:
            H0_fit, Om_fit = result.x
            eps_fit = 0.0
            n_params = 2

        chi2_val = result.fun
        dof = len(self.z_obs) - n_params
        reduced_chi2 = chi2_val / dof
        p_value = 1 - chi2.cdf(chi2_val, dof)

        print(f"  H₀ = {H0_fit:.3f} km/s/Mpc")
        print(f"  Ωₘ = {Om_fit:.5f}")
        if use_spandrel:
            print(f"  ε = {eps_fit:.6f}")
        print(f"  χ² = {chi2_val:.2f} (dof = {dof})")
        print(f"  χ²/dof = {reduced_chi2:.5f}")

        return FitResult(
            params=CosmologyParams(H0=H0_fit, Omega_m=Om_fit, epsilon=eps_fit),
            chi2=chi2_val,
            dof=dof,
            reduced_chi2=reduced_chi2,
            p_value=p_value,
            model_name=model_name
        )

    def run_mcmc(self, model: str = 'both', n_samples: int = 10000,
                 n_burn: int = 2000, n_chains: int = None) -> Dict[str, Dict]:
        """
        Run parallel MCMC sampling for posterior estimation.
        """
        print("\n" + "="*70)
        print("PARALLEL MCMC SAMPLING")
        print("="*70)

        mcmc = ParallelMCMC(self.z_obs, self.mu_obs, self.mu_err, n_chains=n_chains)
        results = {}

        if model in ['lcdm', 'both']:
            print("\n--- ΛCDM Model ---")
            results['lcdm'] = mcmc.run_parallel_chains(
                n_samples=n_samples, n_burn=n_burn, use_spandrel=False,
                initial_params=np.array([70.0, 0.3])
            )
            results['lcdm']['stats'] = mcmc.compute_statistics(
                results['lcdm']['chains'], ['H0', 'Omega_m']
            )

        if model in ['spandrel', 'both']:
            print("\n--- Spandrel Model ---")
            results['spandrel'] = mcmc.run_parallel_chains(
                n_samples=n_samples, n_burn=n_burn, use_spandrel=True,
                initial_params=np.array([70.0, 0.3, 0.0])
            )
            results['spandrel']['stats'] = mcmc.compute_statistics(
                results['spandrel']['chains'], ['H0', 'Omega_m', 'epsilon']
            )

        self.results['mcmc'] = results
        return results

    def compute_evidence(self, model: str = 'both',
                        n_live: int = 500) -> Dict[str, BayesianEvidence]:
        """
        Compute Bayesian evidence using nested sampling.
        """
        print("\n" + "="*70)
        print("BAYESIAN EVIDENCE COMPUTATION")
        print("="*70)

        sampler = NestedSampler(self.z_obs, self.mu_obs, self.mu_err, n_live=n_live)
        results = {}

        if model in ['lcdm', 'both']:
            results['lcdm'] = sampler.run(use_spandrel=False)

        if model in ['spandrel', 'both']:
            results['spandrel'] = sampler.run(use_spandrel=True)

        # Compute Bayes factor if both models
        if 'lcdm' in results and 'spandrel' in results:
            log_bayes = results['spandrel'].log_evidence - results['lcdm'].log_evidence
            print(f"\n  log(Bayes Factor) = {log_bayes:.4f}")
            print(f"  Bayes Factor = {np.exp(log_bayes):.4f}")

            if log_bayes > 1:
                print("  → Strong evidence for Spandrel over ΛCDM")
            elif log_bayes > 0:
                print("  → Weak evidence for Spandrel over ΛCDM")
            elif log_bayes > -1:
                print("  → Weak evidence for ΛCDM over Spandrel")
            else:
                print("  → Strong evidence for ΛCDM over Spandrel")

        self.results['evidence'] = results
        return results

    def likelihood_ratio_test(self) -> Dict[str, float]:
        """
        Perform likelihood ratio test between ΛCDM and Spandrel.
        """
        if 'mle' not in self.results:
            self.fit_mle()

        lcdm = self.results['mle']['lcdm']
        spandrel = self.results['mle']['spandrel']

        delta_chi2 = lcdm.chi2 - spandrel.chi2
        delta_dof = 1  # Spandrel has one extra parameter

        p_value = 1 - chi2.cdf(delta_chi2, delta_dof)
        sigma = norm.ppf(1 - p_value/2) if p_value > 0 else float('inf')

        print("\n" + "="*70)
        print("LIKELIHOOD RATIO TEST: ΛCDM vs Spandrel")
        print("="*70)
        print(f"  Δχ² = {delta_chi2:.4f}")
        print(f"  p-value = {p_value:.6f}")
        print(f"  Significance = {sigma:.2f}σ")

        if delta_chi2 > 3.84:  # 95% confidence
            print("\n  *** Spandrel model PREFERRED at 95% confidence ***")
        else:
            print("\n  ΛCDM is statistically sufficient")

        return {
            'delta_chi2': delta_chi2,
            'p_value': p_value,
            'sigma': sigma,
            'prefers_spandrel': delta_chi2 > 3.84
        }

    def summary_report(self):
        """
        Print comprehensive analysis summary.
        """
        print("\n" + "="*70)
        print("SPANDREL COSMOLOGY ANALYSIS SUMMARY")
        print("="*70)

        print(f"\nDataset: Pantheon+ ({self.loader.metadata.get('total_valid', 'N/A')} SNe Ia)")
        print(f"Redshift range: z = {self.loader.metadata.get('z_range', ('N/A', 'N/A'))[0]:.4f} to "
              f"{self.loader.metadata.get('z_range', ('N/A', 'N/A'))[1]:.4f}")

        if 'mle' in self.results:
            print("\n--- Maximum Likelihood Results ---")
            for name, result in self.results['mle'].items():
                print(f"\n{result.model_name}:")
                print(f"  H₀ = {result.params.H0:.3f} km/s/Mpc")
                print(f"  Ωₘ = {result.params.Omega_m:.5f}")
                if result.params.epsilon != 0:
                    print(f"  ε = {result.params.epsilon:.6f}")
                print(f"  χ²/dof = {result.reduced_chi2:.5f}")

        if 'mcmc' in self.results:
            print("\n--- MCMC Posterior Estimates ---")
            for name, result in self.results['mcmc'].items():
                print(f"\n{name.upper()}:")
                for param, stats in result['stats'].items():
                    print(f"  {param} = {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"(95% CI: [{stats['q2.5']:.4f}, {stats['q97.5']:.4f}])")
                if result['diagnostics']['converged']:
                    print("  ✓ Chains converged (R̂ < 1.1)")
                else:
                    print("  ⚠ Chains may not have converged")

        if 'evidence' in self.results:
            print("\n--- Bayesian Evidence ---")
            for name, evidence in self.results['evidence'].items():
                print(f"  {name.upper()}: log(Z) = {evidence.log_evidence:.4f} ± {evidence.log_evidence_err:.4f}")

            if 'lcdm' in self.results['evidence'] and 'spandrel' in self.results['evidence']:
                log_bf = (self.results['evidence']['spandrel'].log_evidence -
                         self.results['evidence']['lcdm'].log_evidence)
                print(f"\n  log(Bayes Factor Spandrel/ΛCDM) = {log_bf:.4f}")

        # Physical interpretation
        print("\n--- Physical Interpretation ---")

        if 'mcmc' in self.results and 'spandrel' in self.results['mcmc']:
            eps_stats = self.results['mcmc']['spandrel']['stats'].get('epsilon', {})
            eps_mean = eps_stats.get('mean', 0)
            eps_std = eps_stats.get('std', 0)

            if abs(eps_mean) < 2 * eps_std:
                print(f"\n  ε = {eps_mean:.6f} ± {eps_std:.6f}")
                print("  Result: ε is consistent with zero within 2σ")
                print("  Interpretation: The Universe is purely Associative.")
                print("  Standard ΛCDM appears sufficient.")
            elif eps_mean > 0:
                print(f"\n  ε = {eps_mean:.6f} ± {eps_std:.6f}")
                print("  Result: ε > 0 detected!")
                print("  Interpretation: Evidence for Spandrel stiffness")
                print("  The universe may have been 'stiffer' in the past.")
            else:
                print(f"\n  ε = {eps_mean:.6f} ± {eps_std:.6f}")
                print("  Result: ε < 0 detected")
                print("  Interpretation: Universe appears to be 'softening'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_full_hpc_analysis(data_path: str = "Pantheon+SH0ES.dat",
                          run_mcmc: bool = True,
                          run_evidence: bool = True,
                          n_mcmc_samples: int = 5000,
                          n_live: int = 300) -> SpandrelAnalysisPipeline:
    """
    Run the complete high-performance Spandrel analysis pipeline.
    """
    print("\n" + "="*70)
    print("SPANDREL COSMOLOGY HPC ANALYSIS")
    print("Testing the Stiffness Hypothesis with Maximum Parallelism")
    print("="*70)
    print(f"\nHardware Configuration:")
    print(f"  CPU Cores: {NUM_CORES}")
    print(f"  Metal GPU: {'Available (MLX)' if HAS_MLX else 'Not Available'}")
    print(f"  NumPy BLAS: {np.__config__.show() if hasattr(np.__config__, 'show') else 'Unknown'}")

    # Initialize pipeline
    pipeline = SpandrelAnalysisPipeline(data_path)

    # Load data
    pipeline.load_data()

    # Maximum likelihood estimation
    pipeline.fit_mle()

    # Likelihood ratio test
    pipeline.likelihood_ratio_test()

    # MCMC sampling
    if run_mcmc:
        pipeline.run_mcmc(n_samples=n_mcmc_samples, n_burn=1000)

    # Bayesian evidence
    if run_evidence:
        pipeline.compute_evidence(n_live=n_live)

    # Summary report
    pipeline.summary_report()

    return pipeline


if __name__ == "__main__":
    # Run full analysis with all features
    pipeline = run_full_hpc_analysis(
        run_mcmc=True,
        run_evidence=True,
        n_mcmc_samples=5000,
        n_live=300
    )
