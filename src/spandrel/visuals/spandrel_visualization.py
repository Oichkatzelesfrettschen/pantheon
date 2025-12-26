#!/usr/bin/env python3
"""
Spandrel Cosmology Visualization Suite
======================================

Publication-quality visualization tools for Spandrel cosmology analysis.

Features:
- Corner plots for MCMC posteriors
- Hubble diagram with residuals
- Model comparison plots
- Redshift-binned analysis
- Evidence visualization

Author: Spandrel Cosmology Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Import physical constants from central module
from spandrel.core.constants import C_LIGHT_KMS as C_LIGHT, H0_FIDUCIAL, H0_PLANCK, H0_SH0ES, OMEGA_M_FIDUCIAL, GAMMA_1, RIEMANN_ZEROS

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color schemes
COLORS = {
    'lcdm': '#2166ac',       # Blue
    'spandrel': '#b2182b',   # Red
    'data': '#636363',       # Gray
    'planck': '#1b9e77',     # Teal (Planck H0)
    'sh0es': '#d95f02',      # Orange (SH0ES H0)
}


class CornerPlot:
    """
    Generate corner plots for MCMC posterior visualization.
    """

    def __init__(self, samples: np.ndarray, labels: List[str],
                 truths: Optional[List[float]] = None,
                 color: str = COLORS['spandrel']):
        self.samples = samples
        self.labels = labels
        self.truths = truths
        self.color = color
        self.n_params = samples.shape[1]

    def plot(self, figsize: Tuple[int, int] = None,
             quantiles: List[float] = [0.16, 0.5, 0.84],
             show_titles: bool = True,
             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create corner plot showing 1D and 2D marginalized posteriors.
        """
        if figsize is None:
            figsize = (3 * self.n_params, 3 * self.n_params)

        fig, axes = plt.subplots(self.n_params, self.n_params, figsize=figsize)

        for i in range(self.n_params):
            for j in range(self.n_params):
                ax = axes[i, j]

                if j > i:
                    # Upper triangle: empty
                    ax.axis('off')
                elif i == j:
                    # Diagonal: 1D histogram
                    self._plot_1d_hist(ax, i, quantiles, show_titles)
                else:
                    # Lower triangle: 2D contour
                    self._plot_2d_contour(ax, j, i)

                # Labels
                if i == self.n_params - 1:
                    ax.set_xlabel(self.labels[j])
                if j == 0 and i > 0:
                    ax.set_ylabel(self.labels[i])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved corner plot to: {save_path}")

        return fig

    def _plot_1d_hist(self, ax: plt.Axes, param_idx: int,
                      quantiles: List[float], show_titles: bool):
        """Plot 1D marginalized posterior."""
        data = self.samples[:, param_idx]

        # Histogram
        n, bins, patches = ax.hist(data, bins=50, density=True,
                                   color=self.color, alpha=0.7,
                                   edgecolor='white', linewidth=0.5)

        # KDE overlay
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), color='black', linewidth=1.5)

        # Quantile lines
        q_values = np.percentile(data, np.array(quantiles) * 100)
        for q in q_values:
            ax.axvline(q, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

        # Truth value
        if self.truths is not None and self.truths[param_idx] is not None:
            ax.axvline(self.truths[param_idx], color='green', linewidth=2)

        # Title with statistics
        if show_titles:
            median = q_values[1]
            lower = median - q_values[0]
            upper = q_values[2] - median
            ax.set_title(f"${median:.4f}_{{-{lower:.4f}}}^{{+{upper:.4f}}}$",
                        fontsize=10)

        ax.set_yticks([])

    def _plot_2d_contour(self, ax: plt.Axes, x_idx: int, y_idx: int):
        """Plot 2D marginalized posterior with contours."""
        x = self.samples[:, x_idx]
        y = self.samples[:, y_idx]

        # 2D histogram for density estimation
        h, xedges, yedges = np.histogram2d(x, y, bins=50)
        h = h.T  # Transpose for correct orientation

        # Compute contour levels for 1sigma, 2sigma, 3sigma
        h_sorted = np.sort(h.flatten())[::-1]
        h_cumsum = np.cumsum(h_sorted) / np.sum(h_sorted)

        levels = []
        for sigma_frac in [0.393, 0.865, 0.989]:  # 1sigma, 2sigma, 3sigma in 2D
            idx = np.searchsorted(h_cumsum, sigma_frac)
            if idx < len(h_sorted):
                levels.append(h_sorted[idx])

        levels = sorted(levels)

        # Plot filled contours
        xc = (xedges[:-1] + xedges[1:]) / 2
        yc = (yedges[:-1] + yedges[1:]) / 2

        ax.contourf(xc, yc, h, levels=[levels[0], levels[1], levels[2], h.max()],
                   colors=[self.color], alpha=[0.2, 0.4, 0.6])
        ax.contour(xc, yc, h, levels=levels, colors=[self.color],
                  linewidths=[1, 1, 1])

        # Truth values
        if self.truths is not None:
            if self.truths[x_idx] is not None:
                ax.axvline(self.truths[x_idx], color='green', linewidth=1, alpha=0.7)
            if self.truths[y_idx] is not None:
                ax.axhline(self.truths[y_idx], color='green', linewidth=1, alpha=0.7)


class HubbleDiagramPlot:
    """
    Create Hubble diagram visualizations with model comparisons.
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err

    def plot_comparison(self, models: Dict[str, Dict[str, Any]],
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Hubble diagram comparing multiple models.

        models: dict with format {name: {'z': array, 'mu': array, 'params': {...}}}
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)

        # Main Hubble diagram
        ax1 = fig.add_subplot(gs[0])

        # Data points
        ax1.errorbar(self.z_obs, self.mu_obs, yerr=self.mu_err,
                    fmt='.', color=COLORS['data'], alpha=0.3, markersize=3,
                    label=f'Pantheon+ ({len(self.z_obs)} SNe Ia)', zorder=1)

        # Model curves
        z_model = np.logspace(np.log10(self.z_obs.min()), np.log10(self.z_obs.max()), 500)

        for name, model in models.items():
            color = COLORS.get(name.lower(), 'purple')
            params = model.get('params', {})
            label = f"{name}: H_0={params.get('H0', '?'):.1f}, Omegaₘ={params.get('Omega_m', '?'):.3f}"
            if 'epsilon' in params and params['epsilon'] != 0:
                label += f", epsilon={params['epsilon']:.4f}"

            ax1.plot(model['z'], model['mu'], '-', color=color, linewidth=2,
                    label=label, zorder=2)

        ax1.set_xscale('log')
        ax1.set_ylabel('Distance Modulus mu (mag)')
        ax1.set_title('Pantheon+ Hubble Diagram: LambdaCDM vs Spandrel Cosmology', fontsize=14)
        ax1.legend(loc='lower right', framealpha=0.9)
        ax1.set_xlim(self.z_obs.min() * 0.9, self.z_obs.max() * 1.1)

        # Reference H0 values
        ax1.axhline(y=0, alpha=0)  # Hidden, just for legend spacing
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylim(ax1.get_ylim())
        ax1_twin.set_ylabel('Luminosity Distance (Mpc)', alpha=0.5)
        ax1_twin.set_yticks([])

        # Residual panels
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        model_names = list(models.keys())
        if len(model_names) >= 2:
            self._plot_residuals(ax2, models[model_names[0]], model_names[0])
            self._plot_residuals(ax3, models[model_names[1]], model_names[1])
        elif len(model_names) == 1:
            self._plot_residuals(ax2, models[model_names[0]], model_names[0])
            ax3.axis('off')

        ax2.set_ylabel('Δmu (mag)')
        ax3.set_ylabel('Δmu (mag)')
        ax3.set_xlabel('Redshift z')

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        if save_path:
            plt.savefig(save_path)
            print(f"Saved Hubble diagram to: {save_path}")

        return fig

    def _plot_residuals(self, ax: plt.Axes, model: Dict, name: str):
        """Plot residuals for a single model."""
        # Interpolate model to data redshifts
        mu_model_at_data = np.interp(self.z_obs, model['z'], model['mu'])
        residuals = self.mu_obs - mu_model_at_data

        color = COLORS.get(name.lower(), 'purple')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.scatter(self.z_obs, residuals, s=5, alpha=0.3, color=color)

        # Binned residuals
        z_bins = np.logspace(np.log10(self.z_obs.min()), np.log10(self.z_obs.max()), 15)
        bin_centers = []
        bin_means = []
        bin_stds = []

        for i in range(len(z_bins) - 1):
            mask = (self.z_obs >= z_bins[i]) & (self.z_obs < z_bins[i + 1])
            if np.sum(mask) > 2:
                bin_centers.append(np.sqrt(z_bins[i] * z_bins[i + 1]))
                bin_means.append(np.mean(residuals[mask]))
                bin_stds.append(np.std(residuals[mask]) / np.sqrt(np.sum(mask)))

        ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                   fmt='o', color=color, markersize=6, capsize=3,
                   label=f'{name} binned')

        ax.set_ylim(-0.5, 0.5)
        ax.legend(loc='upper right', fontsize=9)


class StiffnessEvolutionPlot:
    """
    Visualize the Spandrel stiffness effect across redshift.
    """

    def __init__(self, epsilon: float, epsilon_err: float = 0.0):
        self.epsilon = epsilon
        self.epsilon_err = epsilon_err

    def plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create stiffness evolution visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        z_range = np.linspace(0.001, 2.5, 500)

        # Left panel: Correction magnitude
        ax1 = axes[0]

        # Central value
        correction = self.epsilon * np.log(1 + z_range) * (1 - 1/(1 + z_range)**2)
        ax1.plot(z_range, correction, 'r-', linewidth=2, label=f'epsilon = {self.epsilon:.4f}')

        # Error band
        if self.epsilon_err > 0:
            corr_high = (self.epsilon + self.epsilon_err) * np.log(1 + z_range) * (1 - 1/(1 + z_range)**2)
            corr_low = (self.epsilon - self.epsilon_err) * np.log(1 + z_range) * (1 - 1/(1 + z_range)**2)
            ax1.fill_between(z_range, corr_low, corr_high, color='red', alpha=0.2)

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        # Mark cosmological epochs
        ax1.axvspan(0, 0.01, alpha=0.15, color='green', label='Local (z < 0.01)')
        ax1.axvspan(0.5, 1.0, alpha=0.15, color='blue', label='Cosmic noon (0.5 < z < 1)')
        ax1.axvspan(1.0, 2.5, alpha=0.15, color='purple', label='Early universe (z > 1)')

        ax1.set_xlabel('Redshift z')
        ax1.set_ylabel('Distance Modulus Correction Δmu (mag)')
        ax1.set_title('Spandrel Stiffness Effect on Distance Measurements')
        ax1.legend(loc='best', fontsize=9)

        # Right panel: Effective equation of state modification
        ax2 = axes[1]

        # The stiffness can be interpreted as modifying w_eff
        w_eff_modification = self.epsilon * (1 - 1/(1 + z_range)**2) / (1 + z_range)
        ax2.plot(z_range, -1 + w_eff_modification, 'b-', linewidth=2)
        ax2.axhline(y=-1, color='black', linestyle='--', linewidth=1,
                   label='Cosmological constant (w = -1)')

        ax2.set_xlabel('Redshift z')
        ax2.set_ylabel('Effective w(z)')
        ax2.set_title('Effective Dark Energy Equation of State')
        ax2.legend(loc='best')
        ax2.set_ylim(-1.3, -0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved stiffness plot to: {save_path}")

        return fig


class HubbleTensionPlot:
    """
    Visualize the Hubble tension and how Spandrel might resolve it.
    """

    def __init__(self, mcmc_results: Dict[str, Dict]):
        self.mcmc_results = mcmc_results

    def plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create Hubble tension visualization."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Reference values
        H0_planck = 67.4
        H0_planck_err = 0.5
        H0_sh0es = 73.04
        H0_sh0es_err = 1.04

        # Plot reference bands
        y_planck = 1
        y_sh0es = 2

        # Planck
        ax.errorbar(H0_planck, y_planck, xerr=H0_planck_err * 2, fmt='o',
                   color=COLORS['planck'], markersize=12, capsize=5,
                   label=f'Planck 2018: {H0_planck:.1f} +/- {H0_planck_err:.1f}')
        ax.axvspan(H0_planck - H0_planck_err, H0_planck + H0_planck_err,
                  alpha=0.2, color=COLORS['planck'])

        # SH0ES
        ax.errorbar(H0_sh0es, y_sh0es, xerr=H0_sh0es_err * 2, fmt='s',
                   color=COLORS['sh0es'], markersize=12, capsize=5,
                   label=f'SH0ES 2022: {H0_sh0es:.1f} +/- {H0_sh0es_err:.1f}')
        ax.axvspan(H0_sh0es - H0_sh0es_err, H0_sh0es + H0_sh0es_err,
                  alpha=0.2, color=COLORS['sh0es'])

        # Our results
        y_offset = 3
        for name, result in self.mcmc_results.items():
            stats = result.get('stats', {})
            H0_stats = stats.get('H0', {})
            H0_mean = H0_stats.get('mean', 70)
            H0_std = H0_stats.get('std', 1)

            color = COLORS.get(name.lower(), 'gray')
            marker = 'D' if 'spandrel' in name.lower() else '^'

            ax.errorbar(H0_mean, y_offset, xerr=H0_std * 2, fmt=marker,
                       color=color, markersize=12, capsize=5,
                       label=f'{name}: {H0_mean:.2f} +/- {H0_std:.2f}')
            y_offset += 1

        # Tension indicator
        tension = (H0_sh0es - H0_planck) / np.sqrt(H0_planck_err**2 + H0_sh0es_err**2)
        ax.annotate(f'Tension: {tension:.1f}sigma', xy=(70.2, 1.5),
                   fontsize=12, ha='center', color='red')
        ax.annotate('', xy=(H0_planck + H0_planck_err, 1.5),
                   xytext=(H0_sh0es - H0_sh0es_err, 1.5),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))

        ax.set_xlabel('$H_0$ (km/s/Mpc)', fontsize=12)
        ax.set_yticks([])
        ax.set_xlim(63, 78)
        ax.set_title('The Hubble Tension: Comparison of $H_0$ Measurements', fontsize=14)
        ax.legend(loc='upper right', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved Hubble tension plot to: {save_path}")

        return fig


class ModelComparisonPlot:
    """
    Visualize model comparison statistics.
    """

    def __init__(self, mle_results: Dict, evidence_results: Optional[Dict] = None):
        self.mle_results = mle_results
        self.evidence_results = evidence_results

    def plot_chi2_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create chi-squared comparison plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        models = list(self.mle_results.keys())
        chi2_values = [self.mle_results[m].chi2 for m in models]
        reduced_chi2 = [self.mle_results[m].reduced_chi2 for m in models]

        colors = [COLORS.get(m.lower(), 'gray') for m in models]

        # Chi-squared values
        ax1 = axes[0]
        bars1 = ax1.bar(models, chi2_values, color=colors, edgecolor='black')
        ax1.set_ylabel('chi^2')
        ax1.set_title('Total Chi-squared')

        for bar, val in zip(bars1, chi2_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)

        # Reduced chi-squared
        ax2 = axes[1]
        bars2 = ax2.bar(models, reduced_chi2, color=colors, edgecolor='black')
        ax2.set_ylabel('chi^2/dof')
        ax2.set_title('Reduced Chi-squared')
        ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Ideal fit')

        for bar, val in zip(bars2, reduced_chi2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved chi2 comparison to: {save_path}")

        return fig

    def plot_evidence_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create Bayesian evidence comparison plot."""
        if self.evidence_results is None:
            print("No evidence results available")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(self.evidence_results.keys())
        log_Z = [self.evidence_results[m].log_evidence for m in models]
        log_Z_err = [self.evidence_results[m].log_evidence_err for m in models]

        colors = [COLORS.get(m.lower(), 'gray') for m in models]

        bars = ax.bar(models, log_Z, yerr=log_Z_err, color=colors,
                     edgecolor='black', capsize=5)

        ax.set_ylabel('log(Z) (Bayesian Evidence)')
        ax.set_title('Bayesian Model Comparison')

        # Add Bayes factor annotation
        if len(models) >= 2:
            log_bf = log_Z[1] - log_Z[0]  # Assuming second is Spandrel
            interpretation = ""
            if log_bf > 5:
                interpretation = "Decisive for Spandrel"
            elif log_bf > 2.5:
                interpretation = "Strong for Spandrel"
            elif log_bf > 1:
                interpretation = "Substantial for Spandrel"
            elif log_bf > 0:
                interpretation = "Weak for Spandrel"
            elif log_bf > -1:
                interpretation = "Weak for LambdaCDM"
            elif log_bf > -2.5:
                interpretation = "Substantial for LambdaCDM"
            else:
                interpretation = "Strong for LambdaCDM"

            ax.text(0.5, 0.95, f'log(Bayes Factor) = {log_bf:.2f}\n{interpretation}',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        for bar, val, err in zip(bars, log_Z, log_Z_err):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.5,
                   f'{val:.2f}+/-{err:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved evidence comparison to: {save_path}")

        return fig


class Chi2ContourPlot:
    """
    Create chi-squared contour plots for parameter constraints.
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, mu_err: np.ndarray):
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err

    def compute_chi2_grid(self, H0_range: np.ndarray, param2_range: np.ndarray,
                          param2_name: str = 'epsilon',
                          fixed_params: Dict = None) -> np.ndarray:
        """Compute chi-squared on a parameter grid."""
        from spandrel.cosmology.spandrel_cosmology_hpc import VectorizedCosmology, CosmologyParams

        chi2_grid = np.zeros((len(param2_range), len(H0_range)))

        fixed = fixed_params or {}

        for i, p2 in enumerate(param2_range):
            for j, H0 in enumerate(H0_range):
                params = CosmologyParams(
                    H0=H0,
                    Omega_m=fixed.get('Omega_m', 0.3),
                    epsilon=p2 if param2_name == 'epsilon' else fixed.get('epsilon', 0.0)
                )

                if param2_name == 'Omega_m':
                    params.Omega_m = p2

                cosmo = VectorizedCosmology(params)
                mu_model = cosmo.distance_modulus_spandrel_vectorized(self.z_obs)
                chi2_grid[i, j] = np.sum(((self.mu_obs - mu_model) / self.mu_err)**2)

        return chi2_grid

    def plot(self, best_fit: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Create chi-squared contour plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        H0_best = best_fit['H0']
        Om_best = best_fit['Omega_m']
        eps_best = best_fit.get('epsilon', 0)

        # H0 vs epsilon
        ax1 = axes[0]
        H0_range = np.linspace(H0_best - 5, H0_best + 5, 60)
        eps_range = np.linspace(eps_best - 0.15, eps_best + 0.15, 60)

        chi2_grid1 = self.compute_chi2_grid(
            H0_range, eps_range, 'epsilon',
            {'Omega_m': Om_best}
        )

        chi2_min = np.min(chi2_grid1)
        levels = chi2_min + np.array([2.30, 6.17, 11.8])  # 1sigma, 2sigma, 3sigma for 2 params

        contour1 = ax1.contour(H0_range, eps_range, chi2_grid1, levels=levels,
                               colors=['green', 'blue', 'red'])
        ax1.clabel(contour1, fmt={levels[0]: '1sigma', levels[1]: '2sigma', levels[2]: '3sigma'})

        ax1.contourf(H0_range, eps_range, chi2_grid1,
                    levels=[chi2_min, levels[0], levels[1], levels[2], chi2_grid1.max()],
                    colors=[COLORS['spandrel']], alpha=[0.4, 0.3, 0.2, 0.1])

        ax1.plot(H0_best, eps_best, 'k*', markersize=15, label='Best fit')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, label='LambdaCDM (epsilon=0)')

        ax1.set_xlabel('H_0 (km/s/Mpc)')
        ax1.set_ylabel('Stiffness epsilon')
        ax1.set_title('chi^2 Contours: H_0 vs Stiffness')
        ax1.legend(loc='upper right')

        # H0 vs Omega_m
        ax2 = axes[1]
        Om_range = np.linspace(Om_best - 0.1, Om_best + 0.1, 60)

        chi2_grid2 = self.compute_chi2_grid(
            H0_range, Om_range, 'Omega_m',
            {'epsilon': eps_best}
        )

        chi2_min2 = np.min(chi2_grid2)
        levels2 = chi2_min2 + np.array([2.30, 6.17, 11.8])

        contour2 = ax2.contour(H0_range, Om_range, chi2_grid2, levels=levels2,
                               colors=['green', 'blue', 'red'])
        ax2.clabel(contour2, fmt={levels2[0]: '1sigma', levels2[1]: '2sigma', levels2[2]: '3sigma'})

        ax2.contourf(H0_range, Om_range, chi2_grid2,
                    levels=[chi2_min2, levels2[0], levels2[1], levels2[2], chi2_grid2.max()],
                    colors=[COLORS['lcdm']], alpha=[0.4, 0.3, 0.2, 0.1])

        ax2.plot(H0_best, Om_best, 'k*', markersize=15, label='Best fit')

        ax2.set_xlabel('H_0 (km/s/Mpc)')
        ax2.set_ylabel('Omegaₘ')
        ax2.set_title('chi^2 Contours: H_0 vs Omegaₘ')
        ax2.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved chi2 contours to: {save_path}")

        return fig


def create_publication_figures(pipeline, output_dir: str = "."):
    """
    Generate all publication-quality figures from analysis pipeline.
    """
    import os

    print("\n" + "="*70)
    print("GENERATING PUBLICATION FIGURES")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Hubble diagram with model comparison
    if 'mle' in pipeline.results:
        print("\n1. Creating Hubble diagram...")

        hubble_plot = HubbleDiagramPlot(pipeline.z_obs, pipeline.mu_obs, pipeline.mu_err)

        # Generate model curves
        from spandrel.cosmology.spandrel_cosmology_hpc import VectorizedCosmology, CosmologyParams

        z_model = np.logspace(np.log10(pipeline.z_obs.min()),
                              np.log10(pipeline.z_obs.max()), 500)

        models = {}
        for name, result in pipeline.results['mle'].items():
            cosmo = VectorizedCosmology(result.params)
            if result.params.epsilon != 0:
                mu_model = cosmo.distance_modulus_spandrel_vectorized(z_model)
            else:
                mu_model = cosmo.distance_modulus_vectorized(z_model)

            models[name] = {
                'z': z_model,
                'mu': mu_model,
                'params': {
                    'H0': result.params.H0,
                    'Omega_m': result.params.Omega_m,
                    'epsilon': result.params.epsilon
                }
            }

        hubble_plot.plot_comparison(models, save_path=f"{output_dir}/hubble_diagram_hpc.png")

    # 2. Corner plot for MCMC posteriors
    if 'mcmc' in pipeline.results and 'spandrel' in pipeline.results['mcmc']:
        print("\n2. Creating corner plot...")

        mcmc_spandrel = pipeline.results['mcmc']['spandrel']
        corner = CornerPlot(
            mcmc_spandrel['chains'],
            labels=['$H_0$', '$\\Omega_m$', '$\\epsilon$'],
            truths=[None, None, 0.0],
            color=COLORS['spandrel']
        )
        corner.plot(save_path=f"{output_dir}/corner_spandrel.png")

    # 3. Stiffness evolution plot
    if 'mcmc' in pipeline.results and 'spandrel' in pipeline.results['mcmc']:
        print("\n3. Creating stiffness evolution plot...")

        eps_stats = pipeline.results['mcmc']['spandrel']['stats']['epsilon']
        stiff_plot = StiffnessEvolutionPlot(eps_stats['mean'], eps_stats['std'])
        stiff_plot.plot(save_path=f"{output_dir}/stiffness_evolution.png")

    # 4. Hubble tension plot
    if 'mcmc' in pipeline.results:
        print("\n4. Creating Hubble tension plot...")

        tension_plot = HubbleTensionPlot(pipeline.results['mcmc'])
        tension_plot.plot(save_path=f"{output_dir}/hubble_tension.png")

    # 5. Model comparison plots
    if 'mle' in pipeline.results:
        print("\n5. Creating model comparison plots...")

        comparison = ModelComparisonPlot(
            pipeline.results['mle'],
            pipeline.results.get('evidence')
        )
        comparison.plot_chi2_comparison(save_path=f"{output_dir}/chi2_comparison.png")

        if pipeline.results.get('evidence'):
            comparison.plot_evidence_comparison(save_path=f"{output_dir}/evidence_comparison.png")

    # 6. Chi-squared contours
    if 'mle' in pipeline.results and 'spandrel' in pipeline.results['mle']:
        print("\n6. Creating chi-squared contour plots...")

        chi2_plot = Chi2ContourPlot(pipeline.z_obs, pipeline.mu_obs, pipeline.mu_err)
        best_fit = {
            'H0': pipeline.results['mle']['spandrel'].params.H0,
            'Omega_m': pipeline.results['mle']['spandrel'].params.Omega_m,
            'epsilon': pipeline.results['mle']['spandrel'].params.epsilon
        }
        chi2_plot.plot(best_fit, save_path=f"{output_dir}/chi2_contours.png")

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    # Test with mock data
    print("Testing visualization suite...")

    # Generate mock posterior samples
    np.random.seed(42)
    n_samples = 5000

    mock_samples = np.column_stack([
        np.random.normal(72, 1, n_samples),     # H0
        np.random.normal(0.28, 0.02, n_samples), # Omega_m
        np.random.normal(0.01, 0.02, n_samples)  # epsilon
    ])

    # Test corner plot
    corner = CornerPlot(mock_samples, ['$H_0$', '$\\Omega_m$', '$\\epsilon$'])
    fig = corner.plot(save_path="test_corner.png")
    plt.close()

    print("Visualization tests complete!")
