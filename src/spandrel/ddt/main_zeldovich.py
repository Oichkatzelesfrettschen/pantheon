#!/usr/bin/env python3
"""
Zel'dovich Gradient Mechanism DDT Simulation

This code simulates the spontaneous deflagration-to-detonation transition
in a Type Ia supernova using the Zel'dovich gradient mechanism.

The key physics:
    1. A temperature gradient creates a "spontaneous wave" with velocity
       u_sp = |dT/dx|^(-1) * |dT/dt|_burn
    2. When u_sp ~ c_s (sound speed), the burning front can couple
       with the pressure wave -> detonation

The simulation uses:
    - Degenerate electron EOS (Chandrasekhar formula)
    - HLLC Riemann solver with MUSCL reconstruction
    - C12+C12 nuclear burning network
    - Strang splitting for hydro/reaction coupling

Reference:
    Zel'dovich et al. (1970), Astronaut. Acta 15, 313
    Khokhlov (1991), A&A 245, 114

Usage:
    python main_zeldovich.py

Output:
    - Real-time plots of density, velocity, temperature, composition
    - Detection of detonation (shock velocity > sound speed)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spandrel.core.constants import K_BOLTZMANN, M_PROTON, Q_BURN
from spandrel.ddt.eos_white_dwarf import (
    eos_from_rho_T, eos_from_rho_e, temperature_from_rho_e
)
from spandrel.ddt.flux_hllc import (
    compute_hllc_update, compute_cfl_timestep,
    conserved_to_primitive, primitive_to_conserved
)
from spandrel.ddt.reaction_carbon import (
    burn_step_subcycled, chapman_jouguet_velocity
)


@dataclass
class SimulationConfig:
    """Configuration for DDT simulation."""
    # Domain
    n_cells: int = 1024
    domain_size: float = 1e7       # cm (100 km)

    # Initial conditions
    rho_ambient: float = 2e7       # g/cm^3 (DDT-favorable density)
    T_ambient: float = 5e8         # K (below ignition)
    T_hotspot: float = 3e9         # K (burning temperature)
    hotspot_width: float = 5e5     # cm (5 km)
    gradient_width: float = 2e6    # cm (20 km) - CRITICAL for DDT

    # Composition
    X_C12_initial: float = 0.5     # 50% carbon (rest is oxygen, inert for simplicity)

    # Time integration
    cfl: float = 0.3
    t_end: float = 0.1             # seconds
    max_steps: int = 100000

    # Output
    plot_interval: int = 200
    verbose: bool = True


class ZeldovichDDTSolver:
    """
    1D Reactive Euler Solver for DDT via Zel'dovich Gradient Mechanism.

    Implements Strang splitting:
        1. Half-step reaction
        2. Full-step hydro
        3. Half-step reaction
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        # Grid
        self.dx = config.domain_size / config.n_cells
        self.x = np.linspace(0.5*self.dx, config.domain_size - 0.5*self.dx, config.n_cells)

        # Conserved variables: U = [rho, rhov, E]
        self.U = np.zeros((3, config.n_cells))

        # Auxiliary: mass fractions
        self.X_C12 = np.zeros(config.n_cells)

        # Cached thermodynamic quantities
        self.T = np.zeros(config.n_cells)
        self.P = np.zeros(config.n_cells)
        self.gamma_eff = np.zeros(config.n_cells)
        self.cs = np.zeros(config.n_cells)

        # Diagnostics
        self.time = 0.0
        self.step = 0
        self.detonation_detected = False
        self.shock_velocity = 0.0

        # Initialize
        self._setup_initial_conditions()

    def _setup_initial_conditions(self):
        """
        Set up the Zel'dovich gradient initial conditions.

        We create a temperature gradient from T_hotspot at x=0 to T_ambient.
        The key is the gradient LENGTH - this determines u_sp.
        """
        cfg = self.config

        # Uniform density (approximately; EOS will adjust)
        rho = np.full(cfg.n_cells, cfg.rho_ambient)

        # Temperature profile: linear gradient from hotspot
        # T(x) = T_hotspot - (T_hotspot - T_ambient) * x / gradient_width for x < gradient_width
        T = np.full(cfg.n_cells, cfg.T_ambient)

        # Hotspot at left boundary
        hotspot_cells = int(cfg.hotspot_width / self.dx)
        T[:hotspot_cells] = cfg.T_hotspot

        # Gradient region
        gradient_cells = int(cfg.gradient_width / self.dx)
        gradient_end = hotspot_cells + gradient_cells

        if gradient_end < cfg.n_cells:
            x_gradient = np.linspace(0, 1, gradient_cells)
            T[hotspot_cells:gradient_end] = cfg.T_hotspot - (cfg.T_hotspot - cfg.T_ambient) * x_gradient

        # Initial composition
        self.X_C12 = np.full(cfg.n_cells, cfg.X_C12_initial)

        # Compute EOS
        state = eos_from_rho_T(rho, T)

        # Store thermodynamic state
        self.T = T
        self.P = state.P
        self.gamma_eff = state.gamma_eff
        self.cs = state.cs

        # Initialize conserved variables
        v = np.zeros(cfg.n_cells)  # Initially at rest
        self.U = primitive_to_conserved(rho, v, state.P, state.gamma_eff)

        if cfg.verbose:
            print("Initial Conditions:")
            print(f"  Domain: {cfg.domain_size/1e5:.0f} km = {cfg.n_cells} cells")
            print(f"  rho = {cfg.rho_ambient:.2e} g/cm^3")
            print(f"  T_hot = {cfg.T_hotspot:.2e} K, T_cold = {cfg.T_ambient:.2e} K")
            print(f"  Gradient length: {cfg.gradient_width/1e5:.1f} km")
            print(f"  Sound speed: {state.cs[0]:.2e} cm/s")
            print(f"  Chapman-Jouguet velocity: {chapman_jouguet_velocity(rho[:1], T[:1])[0]:.2e} cm/s")

    def _update_thermodynamics(self):
        """Update T, P, gamma from conserved variables U and composition X_C12."""
        rho, v, _ = conserved_to_primitive(self.U, self.gamma_eff)

        # Specific internal energy
        E_kinetic = 0.5 * rho * v**2
        e_int = (self.U[2] - E_kinetic) / rho
        e_int = np.maximum(e_int, 1e14)  # Floor

        # EOS inversion to get T
        state = eos_from_rho_e(rho, e_int, self.T)

        self.T = state.T
        self.P = state.P
        self.gamma_eff = state.gamma_eff
        self.cs = state.cs

    def _hydro_step(self, dt: float):
        """
        Advance hydrodynamics by dt using HLLC.

        dU/dt = -dF/dx (no source terms in hydro step)
        """
        # HLLC flux divergence
        dU_dt = compute_hllc_update(self.U, self.gamma_eff, self.dx, limiter='mc')

        # Second-order Runge-Kutta (Heun's method)
        U_star = self.U + dt * dU_dt

        # Recompute flux at predictor state
        rho_star, v_star, P_star = conserved_to_primitive(U_star, self.gamma_eff)
        gamma_star = self.gamma_eff  # Approximate

        dU_dt_star = compute_hllc_update(U_star, gamma_star, self.dx, limiter='mc')

        # Corrector
        self.U = self.U + 0.5 * dt * (dU_dt + dU_dt_star)

        # Apply boundary conditions (transmissive/outflow)
        self._apply_boundary_conditions()

    def _reaction_step(self, dt: float):
        """
        Advance nuclear burning by dt.

        Updates internal energy and composition.
        """
        rho, v, _ = conserved_to_primitive(self.U, self.gamma_eff)

        # Current internal energy
        E_kinetic = 0.5 * rho * v**2
        e_int = (self.U[2] - E_kinetic) / rho

        # Burn with subcycling for stiff reactions
        e_int_new, X_C12_new = burn_step_subcycled(
            rho, e_int, self.X_C12, self.T, dt,
            max_dX=0.05, max_subcycles=1000
        )

        # Update conserved energy
        self.U[2] = rho * e_int_new + E_kinetic

        # Update composition
        self.X_C12 = X_C12_new

    def _apply_boundary_conditions(self):
        """Transmissive (outflow) boundary conditions."""
        # Left boundary: zero-gradient
        self.U[:, 0] = self.U[:, 1]

        # Right boundary: zero-gradient
        self.U[:, -1] = self.U[:, -2]

    def _detect_detonation(self) -> bool:
        """
        Check if a detonation has formed.

        Criteria:
            1. Shock velocity > local sound speed (supersonic)
            2. Sharp density/pressure jump
        """
        rho, v, P = conserved_to_primitive(self.U, self.gamma_eff)

        # Find maximum velocity (shock front)
        i_max_v = np.argmax(np.abs(v))
        self.shock_velocity = np.abs(v[i_max_v])

        # Local sound speed at shock
        cs_shock = self.cs[i_max_v]

        # Check Mach number
        mach = self.shock_velocity / cs_shock if cs_shock > 0 else 0

        # Detonation criterion: Mach > 1, significant velocity, and burning
        if mach > 1.0 and self.shock_velocity > 3e8 and np.max(self.T) > 3e9:
            if not self.detonation_detected:
                self.detonation_detected = True
                if self.config.verbose:
                    print(f"\n*** DETONATION DETECTED at t = {self.time:.4e} s ***")
                    print(f"    Shock velocity: {self.shock_velocity:.3e} cm/s")
                    print(f"    Mach number: {mach:.2f}")
                    print(f"    Max Temperature: {np.max(self.T):.2e} K")
                    print(f"    Position: x = {self.x[i_max_v]/1e5:.1f} km")

        return self.detonation_detected

    def _compute_timestep(self) -> float:
        """Compute CFL-limited timestep."""
        rho, v, _ = conserved_to_primitive(self.U, self.gamma_eff)

        dt_hydro = compute_cfl_timestep(rho, v, self.cs, self.dx, self.config.cfl)

        # Also limit by burning timescale in hot regions
        # (Handled by subcycling, but we cap dt for safety)
        dt_max = 1e-5  # 10 µs max

        return min(dt_hydro, dt_max)

    def run(self, show_plots: bool = True):
        """
        Main simulation loop with Strang splitting.

        Strang splitting achieves second-order accuracy:
            1. Reaction half-step (dt/2)
            2. Hydro full-step (dt)
            3. Reaction half-step (dt/2)
        """
        cfg = self.config

        if show_plots:
            plt.ion()
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        if cfg.verbose:
            print(f"\nStarting simulation (t_end = {cfg.t_end:.2e} s)")
            print("-" * 60)

        while self.time < cfg.t_end and self.step < cfg.max_steps:
            # Compute timestep
            dt = self._compute_timestep()
            if self.time + dt > cfg.t_end:
                dt = cfg.t_end - self.time

            # Strang splitting
            # Step 1: Half reaction
            self._reaction_step(0.5 * dt)
            self._update_thermodynamics()

            # Step 2: Full hydro
            self._hydro_step(dt)
            self._update_thermodynamics()

            # Step 3: Half reaction
            self._reaction_step(0.5 * dt)
            self._update_thermodynamics()

            # Advance time
            self.time += dt
            self.step += 1

            # Check for detonation
            self._detect_detonation()

            # Plotting
            if show_plots and self.step % cfg.plot_interval == 0:
                self._plot_state(fig, axes)

            # Progress
            if cfg.verbose and self.step % 1000 == 0:
                rho, v, _ = conserved_to_primitive(self.U, self.gamma_eff)
                print(f"Step {self.step:6d}: t = {self.time:.4e} s, "
                      f"max|v| = {np.max(np.abs(v)):.2e} cm/s, "
                      f"max T = {np.max(self.T):.2e} K")

        if show_plots:
            plt.ioff()
            self._plot_state(fig, axes)
            plt.savefig(Path(__file__).parent / 'ddt_result.png', dpi=150)
            plt.show()

        # Final summary
        if cfg.verbose:
            print("\n" + "=" * 60)
            print("Simulation Complete")
            print(f"  Final time: {self.time:.4e} s")
            print(f"  Total steps: {self.step}")
            print(f"  Detonation: {'YES' if self.detonation_detected else 'NO'}")
            if self.detonation_detected:
                D_CJ = chapman_jouguet_velocity(
                    np.array([cfg.rho_ambient]),
                    np.array([cfg.T_hotspot])
                )[0]
                print(f"  Shock velocity: {self.shock_velocity:.3e} cm/s")
                print(f"  Chapman-Jouguet: {D_CJ:.3e} cm/s")
                print(f"  Ratio v_shock/D_CJ: {self.shock_velocity/D_CJ:.2f}")

    def _plot_state(self, fig, axes):
        """Plot current simulation state."""
        rho, v, P = conserved_to_primitive(self.U, self.gamma_eff)

        x_km = self.x / 1e5  # Convert to km

        for ax in axes.flat:
            ax.clear()

        # Density
        axes[0, 0].plot(x_km, rho / 1e7, 'b-', linewidth=1)
        axes[0, 0].set_ylabel('rho (10⁷ g/cm^3)')
        axes[0, 0].set_title('Density')
        axes[0, 0].grid(True, alpha=0.3)

        # Velocity
        axes[0, 1].plot(x_km, v / 1e8, 'r-', linewidth=1)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_ylabel('v (10⁸ cm/s)')
        axes[0, 1].set_title(f'Velocity (max = {np.max(np.abs(v)):.2e} cm/s)')
        axes[0, 1].grid(True, alpha=0.3)

        # Temperature
        axes[1, 0].plot(x_km, self.T / 1e9, 'orange', linewidth=1)
        axes[1, 0].set_ylabel('T (10⁹ K)')
        axes[1, 0].set_xlabel('x (km)')
        axes[1, 0].set_title('Temperature')
        axes[1, 0].grid(True, alpha=0.3)

        # Composition
        axes[1, 1].plot(x_km, self.X_C12, 'g-', linewidth=1)
        axes[1, 1].set_ylabel('X(C12)')
        axes[1, 1].set_xlabel('x (km)')
        axes[1, 1].set_title('Carbon Mass Fraction')
        axes[1, 1].set_ylim(-0.05, 0.55)
        axes[1, 1].grid(True, alpha=0.3)

        status = "DETONATION!" if self.detonation_detected else "Deflagration"
        fig.suptitle(f'Zel\'dovich DDT Simulation: t = {self.time:.4e} s, step = {self.step} [{status}]',
                     fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.01)


def run_ddt_simulation():
    """Run the DDT simulation with default parameters."""
    # Configure for DDT-favorable conditions
    # The gradient width is CRITICAL - too short = shock outruns flame
    #                                  too long = flame dies out
    config = SimulationConfig(
        n_cells=1024,
        domain_size=1e7,            # 100 km
        rho_ambient=2e7,            # g/cm^3
        T_ambient=5e8,              # 500 MK
        T_hotspot=3e9,              # 3 GK
        hotspot_width=5e5,          # 5 km
        gradient_width=2e6,         # 20 km - Zel'dovich critical length
        X_C12_initial=0.5,
        cfl=0.3,
        t_end=0.02,                 # 20 ms
        max_steps=50000,
        plot_interval=100,
        verbose=True
    )

    solver = ZeldovichDDTSolver(config)
    solver.run(show_plots=True)

    return solver


if __name__ == "__main__":
    print("=" * 60)
    print("Zel'dovich Gradient Mechanism DDT Simulation")
    print("Realistic Physics (No Numerology)")
    print("=" * 60)
    print()

    solver = run_ddt_simulation()
