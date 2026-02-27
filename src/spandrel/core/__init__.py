# SPDX-License-Identifier: GPL-2.0-only
"""Core utilities and shared constants for Spandrel.

Exports the most-used constants and data-interface symbols so that callers
can write ``from spandrel.core import C_LIGHT_CGS`` instead of reaching
into sub-modules directly.
"""

from spandrel.core.constants import (
    A_RAD,
    C_LIGHT_CGS,
    C_LIGHT_KMS,
    DAY,
    GAMMA_1,
    H0_FIDUCIAL,
    H0_PLANCK,
    H0_SH0ES,
    HBAR,
    K_BOLTZMANN,
    L_SUN,
    M_AMU,
    M_ELECTRON,
    M_PROTON,
    M_SUN,
    MEV_TO_ERG,
    OMEGA_M_FIDUCIAL,
    Q_BURN,
    RIEMANN_ZEROS,
    SIGMA_SB,
    TAU_CO56,
    TAU_NI56,
)
from spandrel.core.data_interface import DataStats, PantheonData, load_pantheon

__all__ = [
    'C_LIGHT_CGS', 'C_LIGHT_KMS',
    'H0_FIDUCIAL', 'H0_PLANCK', 'H0_SH0ES',
    'OMEGA_M_FIDUCIAL',
    'K_BOLTZMANN', 'M_PROTON', 'M_ELECTRON',
    'M_SUN', 'L_SUN', 'DAY', 'SIGMA_SB', 'A_RAD', 'HBAR',
    'Q_BURN', 'TAU_NI56', 'TAU_CO56', 'M_AMU', 'MEV_TO_ERG',
    'GAMMA_1', 'RIEMANN_ZEROS',
    'PantheonData', 'load_pantheon', 'DataStats',
]
