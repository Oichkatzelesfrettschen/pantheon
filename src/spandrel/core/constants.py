"""
Physical constants for the Spandrel Project.

This module re-exports constants from spandrel_core for backward compatibility.
The canonical source is now spandrel_core.constants.

Import via:
    from spandrel.core.constants import *                    # All constants
    from spandrel.core.constants import C_LIGHT_CGS, M_SUN   # Specific constants

Or prefer the new canonical import:
    from spandrel_core.constants import C_LIGHT_CGS, M_SUN
"""

# Re-export all constants from spandrel_core
from spandrel_core.constants import (
    # Speed of light (dual units)
    C_LIGHT,
    C_LIGHT_CGS,
    C_LIGHT_KMS,
    # Fundamental
    HBAR,
    H_PLANCK,
    K_BOLTZMANN,
    G_NEWTON,
    SIGMA_SB,
    A_RAD,
    # Particles
    M_ELECTRON,
    M_PROTON,
    M_NEUTRON,
    M_AMU,
    # Astrophysical
    M_SUN,
    R_SUN,
    L_SUN,
    AU,
    PC,
    MPC,
    # Cosmological
    H0_FIDUCIAL,
    H0_PLANCK,
    H0_SH0ES,
    OMEGA_M_FIDUCIAL,
    OMEGA_LAMBDA_FIDUCIAL,
    # Riemann
    RIEMANN_ZEROS,
    GAMMA_1,
    # Nuclear
    Q_C12_C12,
    Q_BURN,
    Q_NI56,
    E_NI56,
    E_CO56,
    # White dwarf
    RHO_CENTRAL_WD,
    RHO_DDT,
    M_CHANDRASEKHAR,
    R_WD,
    # Decay timescales
    DAY,
    TAU_NI56,
    TAU_CO56,
    HALF_LIFE_NI56,
    HALF_LIFE_CO56,
    # Composition
    A_BAR,
    Z_BAR,
    Y_E,
    # Conversions
    KM_TO_CM,
    CM_TO_KM,
    MPC_TO_KM,
    KM_TO_MPC,
    ERG_TO_MEV,
    MEV_TO_ERG,
)

__all__ = [
    # Speed of light (dual units)
    'C_LIGHT', 'C_LIGHT_CGS', 'C_LIGHT_KMS',
    # Fundamental
    'HBAR', 'H_PLANCK', 'K_BOLTZMANN', 'G_NEWTON', 'SIGMA_SB', 'A_RAD',
    # Particles
    'M_ELECTRON', 'M_PROTON', 'M_NEUTRON', 'M_AMU',
    # Astrophysical
    'M_SUN', 'R_SUN', 'L_SUN', 'AU', 'PC', 'MPC',
    # Cosmological
    'H0_FIDUCIAL', 'H0_PLANCK', 'H0_SH0ES',
    'OMEGA_M_FIDUCIAL', 'OMEGA_LAMBDA_FIDUCIAL',
    # Riemann
    'RIEMANN_ZEROS', 'GAMMA_1',
    # Nuclear
    'Q_C12_C12', 'Q_BURN', 'Q_NI56', 'E_NI56', 'E_CO56',
    # White dwarf
    'RHO_CENTRAL_WD', 'RHO_DDT', 'M_CHANDRASEKHAR', 'R_WD',
    # Decay timescales
    'DAY', 'TAU_NI56', 'TAU_CO56', 'HALF_LIFE_NI56', 'HALF_LIFE_CO56',
    # Composition
    'A_BAR', 'Z_BAR', 'Y_E',
    # Conversions
    'KM_TO_CM', 'CM_TO_KM', 'MPC_TO_KM', 'KM_TO_MPC', 'ERG_TO_MEV', 'MEV_TO_ERG',
]
