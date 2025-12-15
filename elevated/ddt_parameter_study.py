#!/usr/bin/env python3
"""
DDT Parameter Study: Systematic Exploration of Detonation Conditions

Investigates how key parameters affect DDT success and Ni-56 yield:
    1. Gradient width (Zel'dovich critical length)
    2. Initial density (DDT threshold)
    3. Hot spot temperature
    4. Hot spot size

This produces the "DDT Phase Diagram" - a map of parameter space showing
where detonation occurs vs. where the flame fizzles.

Scientific questions:
    - What is the minimum gradient width for DDT?
    - How does density affect the critical gradient?
    - What determines the transition from subluminous to overluminous SNe?
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import M_SUN
from ddt_solver.main_zeldovich import SimulationConfig, ZeldovichDDTSolver


# =============================================================================
# PARAMETER SPACE
# =============================================================================
@dataclass
class ParameterPoint:
    """Single point in parameter space."""
    gradient_width: float    # cm
    rho_ambient: float       # g/cm³
    T_hotspot: float         # K
    hotspot_width: float     # cm


@dataclass
class SimulationResult:
    """Result from a single simulation."""
    params: ParameterPoint
    detonation: bool
    shock_velocity: float    # cm/s
    max_temperature: float   # K
    Ni56_fraction: float     # Mass fraction that reached NSE
    burn_fraction: float     # Fraction of domain that burned
    runtime: float           # seconds


# =============================================================================
# SINGLE SIMULATION RUNNER
# =============================================================================
def run_single_simulation(params: ParameterPoint, t_end: float = 0.01,
                          verbose: bool = False) -> SimulationResult:
    """
    Run a single DDT simulation with given parameters.

    Returns summary result without full field data.
    """
    import time
    start_time = time.time()

    # Configure simulation
    config = SimulationConfig(
        n_cells=512,  # Reduced for parameter study
        domain_size=1e7,
        rho_ambient=params.rho_ambient,
        T_ambient=5e8,
        T_hotspot=params.T_hotspot,
        hotspot_width=params.hotspot_width,
        gradient_width=params.gradient_width,
        X_C12_initial=0.5,
        cfl=0.3,
        t_end=t_end,
        max_steps=20000,
        plot_interval=10000,  # No plotting
        verbose=verbose
    )

    # Run simulation
    solver = ZeldovichDDTSolver(config)

    # Suppress plotting
    try:
        solver.run(show_plots=False)
    except Exception as e:
        # Handle numerical failures gracefully
        return SimulationResult(
            params=params,
            detonation=False,
            shock_velocity=0.0,
            max_temperature=0.0,
            Ni56_fraction=0.0,
            burn_fraction=0.0,
            runtime=time.time() - start_time
        )

    # Extract results
    from ddt_solver.flux_hllc import conserved_to_primitive
    rho, v, _ = conserved_to_primitive(solver.U, solver.gamma_eff)

    # Temperature analysis
    T_max = np.max(solver.T)
    T_NSE = 5e9  # NSE threshold

    # Ni-56 proxy: fraction above NSE temperature
    Ni56_fraction = np.mean(solver.T > T_NSE)

    # Burn fraction: where carbon depleted
    burn_fraction = np.mean(solver.X_C12 < 0.4)

    runtime = time.time() - start_time

    return SimulationResult(
        params=params,
        detonation=solver.detonation_detected,
        shock_velocity=solver.shock_velocity,
        max_temperature=T_max,
        Ni56_fraction=Ni56_fraction,
        burn_fraction=burn_fraction,
        runtime=runtime
    )


# =============================================================================
# PARAMETER SCAN
# =============================================================================
class DDTParameterStudy:
    """
    Systematic parameter study for DDT conditions.
    """

    def __init__(self):
        self.results: List[SimulationResult] = []

    def scan_gradient_width(self, widths: np.ndarray,
                            rho: float = 2e7, T_hot: float = 3e9,
                            parallel: bool = True) -> List[SimulationResult]:
        """
        Scan gradient width at fixed density and temperature.

        This finds the critical gradient length for DDT.
        """
        print(f"Scanning gradient width: {len(widths)} points")
        print(f"Fixed: ρ = {rho:.1e} g/cm³, T_hot = {T_hot:.1e} K")

        params_list = [
            ParameterPoint(
                gradient_width=w,
                rho_ambient=rho,
                T_hotspot=T_hot,
                hotspot_width=5e5
            )
            for w in widths
        ]

        results = self._run_batch(params_list, parallel)
        self.results.extend(results)

        return results

    def scan_2d_grid(self, gradient_widths: np.ndarray, densities: np.ndarray,
                     T_hot: float = 3e9, parallel: bool = True) -> np.ndarray:
        """
        2D scan of gradient width vs density.

        Returns 2D array of detonation outcomes.
        """
        print(f"2D scan: {len(gradient_widths)} × {len(densities)} = {len(gradient_widths)*len(densities)} points")

        params_list = []
        for rho in densities:
            for w in gradient_widths:
                params_list.append(ParameterPoint(
                    gradient_width=w,
                    rho_ambient=rho,
                    T_hotspot=T_hot,
                    hotspot_width=5e5
                ))

        results = self._run_batch(params_list, parallel)
        self.results.extend(results)

        # Reshape to 2D grid
        n_w = len(gradient_widths)
        n_rho = len(densities)

        detonation_grid = np.zeros((n_rho, n_w))
        Ni56_grid = np.zeros((n_rho, n_w))

        for i, result in enumerate(results):
            i_rho = i // n_w
            i_w = i % n_w
            detonation_grid[i_rho, i_w] = 1 if result.detonation else 0
            Ni56_grid[i_rho, i_w] = result.Ni56_fraction

        return detonation_grid, Ni56_grid

    def _run_batch(self, params_list: List[ParameterPoint],
                   parallel: bool = True) -> List[SimulationResult]:
        """Run a batch of simulations."""
        results = []

        if parallel:
            # Parallel execution
            n_workers = min(8, len(params_list))
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(run_single_simulation, p): p for p in params_list}

                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        results.append(result)
                        status = "DDT" if result.detonation else "No DDT"
                        print(f"  [{i+1}/{len(params_list)}] λ={result.params.gradient_width/1e5:.0f}km, "
                              f"ρ={result.params.rho_ambient:.1e}: {status}")
                    except Exception as e:
                        print(f"  [{i+1}/{len(params_list)}] Failed: {e}")
        else:
            # Serial execution
            for i, params in enumerate(params_list):
                result = run_single_simulation(params)
                results.append(result)
                status = "DDT" if result.detonation else "No DDT"
                print(f"  [{i+1}/{len(params_list)}] λ={params.gradient_width/1e5:.0f}km: {status}")

        return results


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_gradient_scan(results: List[SimulationResult], save_path: str = None):
    """Plot results from gradient width scan."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    widths = np.array([r.params.gradient_width for r in results]) / 1e5  # km
    velocities = np.array([r.shock_velocity for r in results]) / 1e8
    T_max = np.array([r.max_temperature for r in results]) / 1e9
    Ni56 = np.array([r.Ni56_fraction for r in results])
    detonations = np.array([r.detonation for r in results])

    # Sound speed for reference
    cs_ref = 4.5  # 10^8 cm/s

    # Panel 1: Shock velocity vs gradient width
    ax = axes[0, 0]
    colors = ['#3fb950' if d else '#f85149' for d in detonations]
    ax.scatter(widths, velocities, c=colors, s=80, alpha=0.8)
    ax.axhline(cs_ref, color='white', linestyle='--', alpha=0.5, label=f'Sound speed ({cs_ref}×10⁸ cm/s)')
    ax.set_xlabel('Gradient Width (km)')
    ax.set_ylabel('Shock Velocity (10⁸ cm/s)')
    ax.set_title('Shock Velocity vs Gradient Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Peak temperature
    ax = axes[0, 1]
    ax.scatter(widths, T_max, c=colors, s=80, alpha=0.8)
    ax.axhline(5.0, color='purple', linestyle='--', alpha=0.7, label='NSE threshold (5 GK)')
    ax.set_xlabel('Gradient Width (km)')
    ax.set_ylabel('Peak Temperature (10⁹ K)')
    ax.set_title('Peak Temperature vs Gradient Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Ni-56 fraction
    ax = axes[1, 0]
    ax.scatter(widths, Ni56 * 100, c=colors, s=80, alpha=0.8)
    ax.set_xlabel('Gradient Width (km)')
    ax.set_ylabel('NSE Fraction (%)')
    ax.set_title('Ni-56 Yield Proxy vs Gradient Length')
    ax.grid(True, alpha=0.3)

    # Panel 4: DDT success summary
    ax = axes[1, 1]

    # Find critical gradient
    ddt_widths = widths[detonations]
    no_ddt_widths = widths[~detonations]

    if len(ddt_widths) > 0 and len(no_ddt_widths) > 0:
        lambda_crit = (np.min(ddt_widths) + np.max(no_ddt_widths)) / 2
    elif len(ddt_widths) > 0:
        lambda_crit = np.min(ddt_widths) * 0.9
    else:
        lambda_crit = np.max(widths) * 1.1

    ax.bar(['Detonation', 'No Detonation'], [np.sum(detonations), np.sum(~detonations)],
          color=['#3fb950', '#f85149'])
    ax.set_ylabel('Count')
    ax.set_title(f'DDT Success Rate (λ_crit ≈ {lambda_crit:.0f} km)')

    # Add text summary
    summary = f"""
    Critical Gradient: {lambda_crit:.0f} km

    Detonation: {np.sum(detonations)} / {len(detonations)}
    Max v_shock: {np.max(velocities):.2f} × 10⁸ cm/s
    Max Ni-56:   {np.max(Ni56)*100:.0f}%
    """
    ax.text(0.5, 0.6, summary, transform=ax.transAxes, fontsize=10,
           ha='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#1a1a2e'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='#0d1117')
        print(f"Saved: {save_path}")

    plt.show()


def plot_2d_phase_diagram(gradient_widths: np.ndarray, densities: np.ndarray,
                          detonation_grid: np.ndarray, Ni56_grid: np.ndarray,
                          save_path: str = None):
    """Plot 2D DDT phase diagram."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    widths_km = gradient_widths / 1e5
    densities_7 = densities / 1e7

    # Panel 1: Detonation success
    ax = axes[0]
    im = ax.imshow(detonation_grid, origin='lower', aspect='auto',
                  extent=[widths_km.min(), widths_km.max(),
                         densities_7.min(), densities_7.max()],
                  cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xlabel('Gradient Width (km)')
    ax.set_ylabel('Density (10⁷ g/cm³)')
    ax.set_title('DDT Phase Diagram')

    # Add contour at transition
    ax.contour(widths_km, densities_7, detonation_grid, levels=[0.5],
              colors='white', linewidths=2)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Detonation Success')

    # Panel 2: Ni-56 yield
    ax = axes[1]
    im = ax.imshow(Ni56_grid * 100, origin='lower', aspect='auto',
                  extent=[widths_km.min(), widths_km.max(),
                         densities_7.min(), densities_7.max()],
                  cmap='inferno', vmin=0, vmax=100)

    ax.set_xlabel('Gradient Width (km)')
    ax.set_ylabel('Density (10⁷ g/cm³)')
    ax.set_title('Ni-56 Yield (NSE Fraction)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NSE Fraction (%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='#0d1117')
        print(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def run_parameter_study():
    """Execute full parameter study."""
    print("=" * 70)
    print("DDT PARAMETER STUDY")
    print("Systematic exploration of detonation conditions")
    print("=" * 70)

    study = DDTParameterStudy()

    # 1. Gradient width scan at fixed density
    print("\n" + "─" * 50)
    print("SCAN 1: Gradient Width (fixed ρ = 2×10⁷ g/cm³)")
    print("─" * 50)

    gradient_widths = np.linspace(5e5, 5e6, 12)  # 5-50 km
    results_1d = study.scan_gradient_width(gradient_widths, rho=2e7, parallel=False)

    plot_gradient_scan(results_1d, save_path=Path(__file__).parent / 'gradient_scan.png')

    # Summary
    print("\n" + "=" * 70)
    print("PARAMETER STUDY COMPLETE")
    print("=" * 70)

    # Critical gradient analysis
    ddt_results = [r for r in results_1d if r.detonation]
    no_ddt_results = [r for r in results_1d if not r.detonation]

    if ddt_results and no_ddt_results:
        lambda_crit = (min(r.params.gradient_width for r in ddt_results) +
                      max(r.params.gradient_width for r in no_ddt_results)) / 2
        print(f"\nCritical gradient length: λ_crit ≈ {lambda_crit/1e5:.0f} km")
        print(f"  (Below this: no DDT)")
        print(f"  (Above this: successful detonation)")

    # Best case
    if ddt_results:
        best = max(ddt_results, key=lambda r: r.Ni56_fraction)
        print(f"\nOptimal configuration:")
        print(f"  Gradient width: {best.params.gradient_width/1e5:.0f} km")
        print(f"  Shock velocity: {best.shock_velocity:.2e} cm/s")
        print(f"  Ni-56 fraction: {best.Ni56_fraction*100:.0f}%")

    return study


if __name__ == "__main__":
    study = run_parameter_study()
