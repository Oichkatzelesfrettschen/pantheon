# SPDX-License-Identifier: GPL-2.0-only
"""
DDT Solver: 1D Reactive Euler Code for Type Ia Supernovae

Implements the Zel'dovich gradient mechanism for spontaneous
deflagration-to-detonation transition.
"""

from .eos_white_dwarf import EOSState, eos_from_rho_e, eos_from_rho_T
from .flux_hllc import compute_cfl_timestep, compute_hllc_update
from .reaction_carbon import burn_step_subcycled, chapman_jouguet_velocity, reaction_rate_c12

__all__ = [
    'eos_from_rho_T',
    'eos_from_rho_e',
    'EOSState',
    'compute_hllc_update',
    'compute_cfl_timestep',
    'reaction_rate_c12',
    'burn_step_subcycled',
    'chapman_jouguet_velocity',
]
