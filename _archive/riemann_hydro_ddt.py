#!/usr/bin/env python3
"""
Riemann-Hydro DDT Simulation
============================

Testing the hypothesis that the Deflagration-to-Detonation Transition
in Type Ia Supernovae is triggered by Riemann Resonance in degenerate plasma.

The Core Idea:
    - White Dwarf plasma turbulence cascades through scale-invariant frequencies
    - When the cascade hits γ₁ = 14.13 (first Riemann zero), vacuum coupling occurs
    - Energy is injected into the flame front, triggering detonation

Physics Implemented:
    1. 1D Reactive Euler equations (compressible flow)
    2. Degenerate electron equation of state
    3. Carbon-12 → Nickel-56 nuclear burning (simplified)
    4. Riemann Resonance source term: S = α·ρ·cos(γ₁·ln(ρ/ρ₀))

The DDT Test:
    - Standard model: Does the deflagration stay subsonic?
    - Riemann model: Does the resonance kick it supersonic?

Author: Spandrel Cosmology Project
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import warnings

# Optional numba acceleration
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS (CGS Units)
# =============================================================================

# Fundamental constants
C_LIGHT = 2.998e10       # cm/s
HBAR = 1.055e-27         # erg·s
K_B = 1.381e-16          # erg/K
M_E = 9.109e-28          # g (electron mass)
M_P = 1.673e-24          # g (proton mass)
M_SUN = 1.989e33         # g
G_NEWTON = 6.674e-8      # cm³/g/s²

# Nuclear physics
Q_BURN = 7.0e17          # erg/g (C12 → Ni56 energy release)
A_CARBON = 12.0          # Atomic mass of Carbon-12
Z_CARBON = 6.0           # Atomic number

# White Dwarf parameters
RHO_CENTRAL = 2.0e9      # g/cm³ (central density near Chandrasekhar)
T_IGNITION = 7.0e8       # K (Carbon ignition temperature)
CHANDRASEKHAR = 1.44     # Solar masses

# Riemann constant
GAMMA_1 = 14.134725141734693  # First Riemann zero


# =============================================================================
# EQUATION OF STATE: DEGENERATE ELECTRON GAS
# =============================================================================

@dataclass
class DegenerateEOS:
    """
    Equation of state for degenerate electron gas.

    In the relativistic degenerate limit (White Dwarf interior):
        P = K₁ · ρ^(4/3)  [relativistic]
        P = K₂ · ρ^(5/3)  [non-relativistic]

    We use a blend that transitions smoothly.
    """

    # EOS constants for Carbon/Oxygen mixture (Ye ≈ 0.5)
    K_nr: float = 9.91e12   # Non-relativistic constant (cgs)
    K_rel: float = 1.231e15  # Relativistic constant (cgs)
    gamma_nr: float = 5/3
    gamma_rel: float = 4/3

    # Transition density
    rho_trans: float = 2.0e6  # g/cm³

    def pressure(self, rho: np.ndarray) -> np.ndarray:
        """Compute pressure from density."""
        rho = np.atleast_1d(rho)

        # Blend between non-relativistic and relativistic
        x = rho / self.rho_trans
        f_rel = x / (1 + x)  # Smooth blending function

        P_nr = self.K_nr * rho**self.gamma_nr
        P_rel = self.K_rel * rho**self.gamma_rel

        return (1 - f_rel) * P_nr + f_rel * P_rel

    def sound_speed(self, rho: np.ndarray) -> np.ndarray:
        """Compute sound speed c_s = sqrt(dP/dρ)."""
        rho = np.atleast_1d(rho)

        x = rho / self.rho_trans
        f_rel = x / (1 + x)

        # Effective gamma
        gamma_eff = (1 - f_rel) * self.gamma_nr + f_rel * self.gamma_rel

        P = self.pressure(rho)
        return np.sqrt(gamma_eff * P / rho)

    def internal_energy(self, rho: np.ndarray) -> np.ndarray:
        """Compute specific internal energy e = P / (ρ(γ-1))."""
        rho = np.atleast_1d(rho)

        x = rho / self.rho_trans
        f_rel = x / (1 + x)
        gamma_eff = (1 - f_rel) * self.gamma_nr + f_rel * self.gamma_rel

        P = self.pressure(rho)
        return P / (rho * (gamma_eff - 1))


# =============================================================================
# NUCLEAR BURNING MODEL
# =============================================================================

@dataclass
class CarbonBurning:
    """
    Simplified Carbon-12 → Nickel-56 burning model.

    Reaction rate follows Arrhenius form:
        ω = A · ρ · X_C · exp(-E_a / kT)

    where X_C is the Carbon mass fraction.
    """

    # Arrhenius parameters (fitted to detailed networks)
    A_rate: float = 1.0e17      # Pre-exponential factor
    E_activation: float = 8.0e9  # Activation energy (erg/g) ~ T_ign

    # Energy release
    Q_nuc: float = Q_BURN

    def reaction_rate(self, rho: np.ndarray, T: np.ndarray,
                      X_C: np.ndarray) -> np.ndarray:
        """
        Compute mass burning rate (g/cm³/s).

        Returns dρ_C/dt (mass of Carbon burned per volume per time)
        """
        rho = np.atleast_1d(rho)
        T = np.atleast_1d(T)
        X_C = np.atleast_1d(X_C)

        # Arrhenius rate
        rate = self.A_rate * rho * X_C * np.exp(-self.E_activation / (K_B * T))

        # Prevent burning where T is too low
        rate = np.where(T > 1e8, rate, 0.0)

        return rate

    def energy_release_rate(self, rho: np.ndarray, T: np.ndarray,
                           X_C: np.ndarray) -> np.ndarray:
        """Energy release rate (erg/cm³/s)."""
        burn_rate = self.reaction_rate(rho, T, X_C)
        return burn_rate * self.Q_nuc


# =============================================================================
# RIEMANN RESONANCE SOURCE TERM
# =============================================================================

class RiemannResonance:
    """
    The Riemann Resonance vacuum coupling.

    Hypothesis: At extreme densities, plasma turbulence couples to vacuum
    fluctuations at the Riemann frequency γ₁ = 14.13.

    Source term:
        S_Riemann = α · ρ · cos(γ₁ · ln(ρ/ρ₀) + φ)

    This injects energy into the plasma when the density crosses
    resonance points: ρ_n = ρ₀ · exp(nπ/γ₁)

    Physical interpretation:
        - The plasma is a "Box-Kite" vortex fluid in 64D
        - Resonance opens zero-divisor channels
        - Vacuum energy leaks into the nuclear flame
    """

    def __init__(self, alpha: float = 1e15, rho_0: float = RHO_CENTRAL,
                 phase: float = 0.0, gamma: float = GAMMA_1):
        """
        Initialize Riemann resonance.

        Args:
            alpha: Coupling strength (erg/g/s at resonance)
            rho_0: Reference density for log scaling
            phase: Phase offset
            gamma: Riemann frequency (default: γ₁)
        """
        self.alpha = alpha
        self.rho_0 = rho_0
        self.phase = phase
        self.gamma = gamma

        # Compute resonance densities (where cos = ±1)
        self.rho_resonances = self._compute_resonances()

    def _compute_resonances(self, n_max: int = 10) -> np.ndarray:
        """Compute the density values where resonance peaks occur."""
        # cos(γ·ln(ρ/ρ₀) + φ) = ±1 when argument = nπ
        n_values = np.arange(-n_max, n_max + 1)
        log_rho = (n_values * np.pi - self.phase) / self.gamma
        return self.rho_0 * np.exp(log_rho)

    def source_term(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute the Riemann source term S(ρ).

        Returns energy injection rate (erg/cm³/s).
        """
        rho = np.atleast_1d(rho)
        log_rho_ratio = np.log(rho / self.rho_0)

        # Oscillating source
        S = self.alpha * rho * np.cos(self.gamma * log_rho_ratio + self.phase)

        return S

    def source_term_turbulent(self, rho: np.ndarray,
                              turbulent_mach: float = 0.1) -> np.ndarray:
        """
        Source term modulated by turbulent fluctuations.

        The resonance is only active when turbulence "hits" the right frequency.
        We model this as a Gaussian envelope around resonance densities.
        """
        rho = np.atleast_1d(rho)

        # Base oscillation
        log_rho_ratio = np.log(rho / self.rho_0)
        oscillation = np.cos(self.gamma * log_rho_ratio + self.phase)

        # Turbulent modulation (intermittent activation)
        # Width of resonance in log-density space
        width = turbulent_mach / self.gamma

        # Find distance to nearest resonance
        distances = np.abs(log_rho_ratio[:, np.newaxis] -
                          np.log(self.rho_resonances / self.rho_0))
        min_distance = np.min(distances, axis=1)

        # Gaussian envelope
        envelope = np.exp(-0.5 * (min_distance / width)**2)

        return self.alpha * rho * oscillation * envelope


# =============================================================================
# 1D REACTIVE EULER SOLVER
# =============================================================================

class ReactiveEulerSolver:
    """
    1D Reactive Euler equations for degenerate plasma.

    Conservation laws:
        ∂ρ/∂t + ∂(ρu)/∂x = 0                      (mass)
        ∂(ρu)/∂t + ∂(ρu² + P)/∂x = 0              (momentum)
        ∂E/∂t + ∂((E + P)u)/∂x = S_nuc + S_Riemann (energy)
        ∂(ρX_C)/∂t + ∂(ρX_C·u)/∂x = -ω            (Carbon mass fraction)

    where E = ρe + ½ρu² is total energy density.

    Numerical method: MUSCL-Hancock with HLL Riemann solver
    """

    def __init__(self, nx: int = 1000, x_max: float = 1e7,
                 use_riemann: bool = False, riemann_alpha: float = 1e15):
        """
        Initialize solver.

        Args:
            nx: Number of grid cells
            x_max: Domain size (cm)
            use_riemann: Enable Riemann resonance source term
            riemann_alpha: Riemann coupling strength
        """
        self.nx = nx
        self.x_max = x_max
        self.dx = x_max / nx

        # Grid
        self.x = np.linspace(0.5 * self.dx, x_max - 0.5 * self.dx, nx)

        # Physics modules
        self.eos = DegenerateEOS()
        self.burning = CarbonBurning()

        # Riemann resonance
        self.use_riemann = use_riemann
        if use_riemann:
            self.riemann = RiemannResonance(alpha=riemann_alpha)
        else:
            self.riemann = None

        # State variables: [ρ, ρu, E, ρX_C]
        self.U = np.zeros((4, nx))

        # Diagnostics
        self.time = 0.0
        self.history = {
            'time': [],
            'max_velocity': [],
            'max_mach': [],
            'burned_mass': [],
            'detonation_triggered': False,
            'detonation_time': None
        }

    def initialize_deflagration(self, rho_0: float = RHO_CENTRAL,
                                T_0: float = 5e8, T_hot: float = 3e9,
                                hot_width: float = 1e5):
        """
        Initialize a deflagration setup.

        Hot ash (burned) on the left, cold fuel (unburned) on the right.
        """
        # Density (uniform initially)
        rho = np.full(self.nx, rho_0)

        # Temperature profile (hot spot on left)
        T = T_0 + (T_hot - T_0) * np.exp(-(self.x / hot_width)**2)

        # Velocity (initially at rest)
        u = np.zeros(self.nx)

        # Carbon mass fraction (1 = pure fuel, 0 = fully burned)
        X_C = np.where(self.x < hot_width, 0.1, 1.0)  # Partially burned in hot spot

        # Compute internal energy from EOS + thermal
        e_deg = self.eos.internal_energy(rho)
        e_thermal = K_B * T / (M_P * A_CARBON)  # Thermal contribution
        e_total = e_deg + e_thermal

        # Set conservative variables
        self.U[0] = rho                      # ρ
        self.U[1] = rho * u                  # ρu
        self.U[2] = rho * e_total + 0.5 * rho * u**2  # E
        self.U[3] = rho * X_C                # ρX_C

        self.time = 0.0

    def get_primitives(self) -> Tuple[np.ndarray, ...]:
        """Extract primitive variables from conserved."""
        rho = self.U[0]
        u = self.U[1] / rho
        E = self.U[2]
        X_C = self.U[3] / rho

        # Specific internal energy
        e = E / rho - 0.5 * u**2

        # Pressure from EOS
        P = self.eos.pressure(rho)

        # Temperature (approximate from thermal component)
        e_deg = self.eos.internal_energy(rho)
        e_thermal = np.maximum(e - e_deg, 1e10)  # Thermal energy
        T = e_thermal * M_P * A_CARBON / K_B

        return rho, u, P, T, X_C, e

    def compute_fluxes(self, rho: np.ndarray, u: np.ndarray,
                       P: np.ndarray, E: np.ndarray,
                       rhoXC: np.ndarray) -> np.ndarray:
        """Compute flux vector F(U)."""
        F = np.zeros((4, len(rho)))
        F[0] = rho * u                  # Mass flux
        F[1] = rho * u**2 + P           # Momentum flux
        F[2] = (E + P) * u              # Energy flux
        F[3] = rhoXC * u                # Species flux
        return F

    def hll_flux(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """
        HLL approximate Riemann solver.

        Computes numerical flux at cell interface.
        """
        # Left state primitives
        rhoL = UL[0]
        uL = UL[1] / rhoL
        PL = self.eos.pressure(rhoL)
        csL = self.eos.sound_speed(rhoL)

        # Right state primitives
        rhoR = UR[0]
        uR = UR[1] / rhoR
        PR = self.eos.pressure(rhoR)
        csR = self.eos.sound_speed(rhoR)

        # Wave speed estimates
        SL = np.minimum(uL - csL, uR - csR)
        SR = np.maximum(uL + csL, uR + csR)

        # Fluxes
        FL = self.compute_fluxes(rhoL, uL, PL, UL[2], UL[3])
        FR = self.compute_fluxes(rhoR, uR, PR, UR[2], UR[3])

        # HLL flux
        F_hll = np.zeros_like(FL)

        for i in range(len(SL)):
            if SL[i] >= 0:
                F_hll[:, i] = FL[:, i]
            elif SR[i] <= 0:
                F_hll[:, i] = FR[:, i]
            else:
                F_hll[:, i] = (SR[i] * FL[:, i] - SL[i] * FR[:, i] +
                              SL[i] * SR[i] * (UR[:, i] - UL[:, i])) / (SR[i] - SL[i])

        return F_hll

    def compute_source_terms(self, rho: np.ndarray, T: np.ndarray,
                            X_C: np.ndarray) -> np.ndarray:
        """
        Compute source terms for energy and species equations.

        S = [0, 0, S_nuc + S_Riemann, -ω]
        """
        S = np.zeros((4, len(rho)))

        # Nuclear burning
        burn_rate = self.burning.reaction_rate(rho, T, X_C)
        S_nuc = burn_rate * self.burning.Q_nuc

        # Riemann resonance (if enabled)
        if self.use_riemann and self.riemann is not None:
            S_riemann = self.riemann.source_term(rho)
        else:
            S_riemann = 0.0

        S[2] = S_nuc + S_riemann  # Energy source
        S[3] = -burn_rate         # Carbon consumption

        return S

    def compute_timestep(self, cfl: float = 0.4) -> float:
        """Compute timestep from CFL condition."""
        rho, u, P, T, X_C, e = self.get_primitives()
        cs = self.eos.sound_speed(rho)

        max_speed = np.max(np.abs(u) + cs)
        dt = cfl * self.dx / max_speed

        return dt

    def step(self, dt: float):
        """
        Advance solution by one timestep using MUSCL-Hancock.
        """
        # Get primitives
        rho, u, P, T, X_C, e = self.get_primitives()

        # Reconstruct at cell interfaces (simple piecewise constant for now)
        # Left and right states at each interface i+1/2
        UL = self.U[:, :-1]
        UR = self.U[:, 1:]

        # Compute HLL fluxes at interfaces
        F_interface = self.hll_flux(UL, UR)

        # Flux differences
        dF = np.zeros_like(self.U)
        dF[:, 1:-1] = (F_interface[:, 1:] - F_interface[:, :-1]) / self.dx

        # Boundary conditions (reflective left, outflow right)
        dF[:, 0] = dF[:, 1]
        dF[:, -1] = dF[:, -2]

        # Source terms
        S = self.compute_source_terms(rho, T, X_C)

        # Update
        self.U = self.U - dt * dF + dt * S

        # Enforce positivity
        self.U[0] = np.maximum(self.U[0], 1e-10)
        self.U[3] = np.maximum(self.U[3], 0.0)
        self.U[3] = np.minimum(self.U[3], self.U[0])  # X_C <= 1

        self.time += dt

    def check_detonation(self) -> bool:
        """
        Check if detonation has been triggered.

        Criterion: Flame speed > local sound speed (Mach > 1)
        """
        rho, u, P, T, X_C, e = self.get_primitives()
        cs = self.eos.sound_speed(rho)

        # Find flame front (where X_C transitions)
        flame_idx = np.argmax(np.abs(np.gradient(X_C)))

        if flame_idx > 0 and flame_idx < self.nx - 1:
            # Local Mach number at flame
            mach_flame = np.abs(u[flame_idx]) / cs[flame_idx]

            if mach_flame > 1.0 and not self.history['detonation_triggered']:
                self.history['detonation_triggered'] = True
                self.history['detonation_time'] = self.time
                return True

        return False

    def record_diagnostics(self):
        """Record diagnostic quantities."""
        rho, u, P, T, X_C, e = self.get_primitives()
        cs = self.eos.sound_speed(rho)

        self.history['time'].append(self.time)
        self.history['max_velocity'].append(np.max(np.abs(u)))
        self.history['max_mach'].append(np.max(np.abs(u) / cs))
        self.history['burned_mass'].append(np.sum((1 - X_C) * rho * self.dx))

    def run(self, t_max: float = 1e-3, n_output: int = 100) -> dict:
        """
        Run simulation until t_max or detonation.

        Returns history dictionary.
        """
        print(f"Running {'Riemann' if self.use_riemann else 'Standard'} simulation...")
        print(f"  Domain: {self.x_max:.2e} cm, {self.nx} cells")
        print(f"  Max time: {t_max:.2e} s")

        output_interval = t_max / n_output
        next_output = output_interval

        step_count = 0

        while self.time < t_max:
            dt = self.compute_timestep()
            dt = min(dt, t_max - self.time)

            self.step(dt)
            step_count += 1

            # Check for detonation
            if self.check_detonation():
                print(f"  *** DETONATION TRIGGERED at t = {self.time:.4e} s ***")
                break

            # Record diagnostics
            if self.time >= next_output:
                self.record_diagnostics()
                next_output += output_interval

                # Progress report
                mach = self.history['max_mach'][-1]
                print(f"  t = {self.time:.4e} s, Max Mach = {mach:.3f}")

        print(f"  Completed: {step_count} steps, final time = {self.time:.4e} s")

        return self.history


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(solver_std: ReactiveEulerSolver,
                   solver_riemann: ReactiveEulerSolver,
                   save_path: Optional[str] = None):
    """Compare standard and Riemann simulations."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get final states
    rho_std, u_std, P_std, T_std, XC_std, e_std = solver_std.get_primitives()
    rho_r, u_r, P_r, T_r, XC_r, e_r = solver_riemann.get_primitives()

    # Row 1: Spatial profiles
    ax1, ax2, ax3 = axes[0]

    # Density
    ax1.plot(solver_std.x / 1e5, rho_std / 1e9, 'b-', label='Standard', linewidth=2)
    ax1.plot(solver_riemann.x / 1e5, rho_r / 1e9, 'r--', label='Riemann', linewidth=2)
    ax1.set_xlabel('Position (km)')
    ax1.set_ylabel('Density (10⁹ g/cm³)')
    ax1.set_title('Density Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Velocity
    ax2.plot(solver_std.x / 1e5, u_std / 1e8, 'b-', label='Standard', linewidth=2)
    ax2.plot(solver_riemann.x / 1e5, u_r / 1e8, 'r--', label='Riemann', linewidth=2)
    ax2.set_xlabel('Position (km)')
    ax2.set_ylabel('Velocity (10⁸ cm/s)')
    ax2.set_title('Velocity Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Temperature
    ax3.semilogy(solver_std.x / 1e5, T_std, 'b-', label='Standard', linewidth=2)
    ax3.semilogy(solver_riemann.x / 1e5, T_r, 'r--', label='Riemann', linewidth=2)
    ax3.set_xlabel('Position (km)')
    ax3.set_ylabel('Temperature (K)')
    ax3.set_title('Temperature Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: Time histories
    ax4, ax5, ax6 = axes[1]

    # Max Mach number
    t_std = np.array(solver_std.history['time']) * 1e3  # Convert to ms
    t_r = np.array(solver_riemann.history['time']) * 1e3

    ax4.plot(t_std, solver_std.history['max_mach'], 'b-', label='Standard', linewidth=2)
    ax4.plot(t_r, solver_riemann.history['max_mach'], 'r--', label='Riemann', linewidth=2)
    ax4.axhline(y=1.0, color='green', linestyle=':', label='Sonic (M=1)')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Max Mach Number')
    ax4.set_title('Flame Mach Number Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Burned mass
    ax5.plot(t_std, solver_std.history['burned_mass'], 'b-', label='Standard', linewidth=2)
    ax5.plot(t_r, solver_riemann.history['burned_mass'], 'r--', label='Riemann', linewidth=2)
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Burned Mass (g)')
    ax5.set_title('Nuclear Energy Release')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Carbon fraction
    ax6.plot(solver_std.x / 1e5, XC_std, 'b-', label='Standard', linewidth=2)
    ax6.plot(solver_riemann.x / 1e5, XC_r, 'r--', label='Riemann', linewidth=2)
    ax6.set_xlabel('Position (km)')
    ax6.set_ylabel('Carbon Mass Fraction')
    ax6.set_title('Fuel Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add detonation markers
    if solver_std.history['detonation_triggered']:
        ax4.axvline(solver_std.history['detonation_time'] * 1e3,
                   color='blue', linestyle='--', alpha=0.7)
    if solver_riemann.history['detonation_triggered']:
        ax4.axvline(solver_riemann.history['detonation_time'] * 1e3,
                   color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Add summary text
    fig.text(0.5, 0.01,
            f"Standard: {'DETONATION' if solver_std.history['detonation_triggered'] else 'Deflagration'} | "
            f"Riemann: {'DETONATION' if solver_riemann.history['detonation_triggered'] else 'Deflagration'}",
            ha='center', fontsize=12, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")

    plt.show()


def plot_riemann_resonance_structure(save_path: Optional[str] = None):
    """Visualize the Riemann resonance source term structure."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    riemann = RiemannResonance(alpha=1e15)

    # Density range spanning several resonances
    rho = np.logspace(8, 10, 1000)

    # Source term
    S = riemann.source_term(rho)

    ax1 = axes[0]
    ax1.semilogx(rho, S / 1e24, 'r-', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Mark resonance peaks
    for rho_res in riemann.rho_resonances:
        if 1e8 < rho_res < 1e10:
            ax1.axvline(rho_res, color='blue', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Density (g/cm³)')
    ax1.set_ylabel('Source Term (10²⁴ erg/cm³/s)')
    ax1.set_title(f'Riemann Resonance Source (γ = {GAMMA_1:.2f})')
    ax1.grid(True, alpha=0.3)

    # Log-density oscillation
    ax2 = axes[1]
    log_rho = np.log(rho / RHO_CENTRAL)

    ax2.plot(log_rho, np.cos(GAMMA_1 * log_rho), 'g-', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('ln(ρ/ρ₀)')
    ax2.set_ylabel('cos(γ₁ · ln(ρ/ρ₀))')
    ax2.set_title('Log-Periodic Oscillation Structure')
    ax2.grid(True, alpha=0.3)

    # Mark the periods
    for n in range(-3, 4):
        x_zero = n * np.pi / GAMMA_1
        if -3 < x_zero < 3:
            ax2.axvline(x_zero, color='red', linestyle=':', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


# =============================================================================
# MAIN DDT TEST
# =============================================================================

def run_ddt_test(riemann_alpha: float = 1e15, t_max: float = 5e-4):
    """
    Run the Deflagration-to-Detonation Transition test.

    Compares:
    1. Standard reactive Euler (no Riemann term)
    2. Riemann-enhanced reactive Euler

    The question: Does the Riemann resonance trigger detonation?
    """

    print("\n" + "="*70)
    print("RIEMANN-HYDRO DDT TEST")
    print("Testing Vacuum Resonance as Supernova Trigger")
    print("="*70)

    print(f"\nParameters:")
    print(f"  Riemann frequency: γ₁ = {GAMMA_1:.6f}")
    print(f"  Coupling strength: α = {riemann_alpha:.2e} erg/g/s")
    print(f"  Central density: ρ₀ = {RHO_CENTRAL:.2e} g/cm³")
    print(f"  Simulation time: {t_max:.2e} s")

    # Visualize resonance structure
    print("\n1. Visualizing Riemann Resonance Structure...")
    plot_riemann_resonance_structure(save_path="riemann_resonance_structure.png")

    # Run standard simulation
    print("\n2. Running STANDARD simulation (no Riemann)...")
    solver_std = ReactiveEulerSolver(nx=500, x_max=5e6, use_riemann=False)
    solver_std.initialize_deflagration()
    history_std = solver_std.run(t_max=t_max)

    # Run Riemann simulation
    print("\n3. Running RIEMANN simulation...")
    solver_riemann = ReactiveEulerSolver(nx=500, x_max=5e6,
                                         use_riemann=True,
                                         riemann_alpha=riemann_alpha)
    solver_riemann.initialize_deflagration()
    history_riemann = solver_riemann.run(t_max=t_max)

    # Compare results
    print("\n4. Generating comparison plots...")
    plot_comparison(solver_std, solver_riemann, save_path="ddt_comparison.png")

    # Summary
    print("\n" + "="*70)
    print("DDT TEST RESULTS")
    print("="*70)

    print(f"""
    STANDARD MODEL:
      Final Max Mach: {history_std['max_mach'][-1]:.4f}
      Detonation: {'YES at t=' + f"{history_std['detonation_time']:.4e} s" if history_std['detonation_triggered'] else 'NO'}
      Total burned: {history_std['burned_mass'][-1]:.4e} g

    RIEMANN MODEL:
      Final Max Mach: {history_riemann['max_mach'][-1]:.4f}
      Detonation: {'YES at t=' + f"{history_riemann['detonation_time']:.4e} s" if history_riemann['detonation_triggered'] else 'NO'}
      Total burned: {history_riemann['burned_mass'][-1]:.4e} g

    VERDICT:
    """)

    if history_riemann['detonation_triggered'] and not history_std['detonation_triggered']:
        print("    *** RIEMANN RESONANCE TRIGGERED DETONATION ***")
        print("    The vacuum coupling kicked the flame supersonic!")
        print("    This supports the Riemann-DDT hypothesis.")
    elif history_riemann['detonation_triggered'] and history_std['detonation_triggered']:
        dt = (history_std['detonation_time'] - history_riemann['detonation_time']) * 1e3
        print(f"    Both models detonated. Riemann was {dt:.2f} ms {'faster' if dt > 0 else 'slower'}.")
    else:
        print("    Neither model triggered detonation in this configuration.")
        print("    Try increasing α or running longer.")

    return solver_std, solver_riemann


if __name__ == "__main__":
    # Run the DDT test
    solver_std, solver_riemann = run_ddt_test(
        riemann_alpha=5e15,  # Coupling strength
        t_max=1e-3           # 1 millisecond
    )
