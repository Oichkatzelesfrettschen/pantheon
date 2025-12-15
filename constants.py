"""
Physical constants for the Spandrel Project.

This module provides centralized constants to avoid duplication across modules.
Constants are provided in multiple unit systems where needed:
- CGS (cm, g, s) for astrophysics/DDT solver
- Cosmological units (km/s, Mpc) for distance calculations

Import via:
    from constants import *                    # All constants
    from constants import C_LIGHT_CGS, M_SUN   # Specific constants
"""

# =============================================================================
# SPEED OF LIGHT (dual units for different domains)
# =============================================================================
C_LIGHT_CGS = 2.99792458e10     # Speed of light [cm/s] - for astrophysics
C_LIGHT_KMS = 299792.458        # Speed of light [km/s] - for cosmology
C_LIGHT = C_LIGHT_CGS           # Default: CGS (backward compatible)

# =============================================================================
# FUNDAMENTAL CONSTANTS (CGS)
# =============================================================================
HBAR = 1.0545718e-27            # Reduced Planck constant [erg·s]
H_PLANCK = 6.62607015e-27       # Planck constant [erg·s]
K_BOLTZMANN = 1.380649e-16      # Boltzmann constant [erg/K]
G_NEWTON = 6.674e-8             # Gravitational constant [cm³/g/s²]
SIGMA_SB = 5.670374e-5          # Stefan-Boltzmann constant [erg/cm²/s/K⁴]
A_RAD = 7.5657e-15              # Radiation constant [erg/cm³/K⁴]

# =============================================================================
# PARTICLE MASSES (CGS)
# =============================================================================
M_ELECTRON = 9.1093837e-28      # Electron mass [g]
M_PROTON = 1.6726219e-24        # Proton mass [g]
M_NEUTRON = 1.6749275e-24       # Neutron mass [g]
M_AMU = 1.6605390e-24           # Atomic mass unit [g]

# =============================================================================
# ASTROPHYSICAL CONSTANTS
# =============================================================================
M_SUN = 1.989e33                # Solar mass [g] (IAU nominal)
R_SUN = 6.96e10                 # Solar radius [cm]
L_SUN = 3.828e33                # Solar luminosity [erg/s]
AU = 1.496e13                   # Astronomical unit [cm]
PC = 3.086e18                   # Parsec [cm]
MPC = 3.086e24                  # Megaparsec [cm]

# =============================================================================
# COSMOLOGICAL CONSTANTS
# =============================================================================
H0_FIDUCIAL = 70.0              # Fiducial Hubble constant [km/s/Mpc]
H0_PLANCK = 67.4                # Planck 2018 H0 [km/s/Mpc]
H0_SH0ES = 73.0                 # SH0ES 2022 H0 [km/s/Mpc]
OMEGA_M_FIDUCIAL = 0.3          # Fiducial matter density parameter
OMEGA_LAMBDA_FIDUCIAL = 0.7     # Fiducial dark energy density parameter

# =============================================================================
# RIEMANN ZETA ZEROS
# =============================================================================
RIEMANN_ZEROS = [
    14.134725141734693790,      # γ₁
    21.022039638771554993,      # γ₂
    25.010857580145688763,      # γ₃
    30.424876125859513210,      # γ₄
    32.935061587739189691,      # γ₅
]
GAMMA_1 = RIEMANN_ZEROS[0]      # First Riemann zero

# =============================================================================
# NUCLEAR PHYSICS CONSTANTS
# =============================================================================
Q_C12_C12 = 1.7e18              # C12+C12 energy release [erg/g] (average)
Q_BURN = 2.0e17                 # Complete C→NSE burning [erg/g]
Q_NI56 = 1.75e17                # Ni56 decay energy [erg/g]
E_NI56 = 1.75e17                # Ni56 specific energy [erg/g] (alias)
E_CO56 = 3.73e16                # Co56 specific energy [erg/g]

# =============================================================================
# WHITE DWARF PARAMETERS
# =============================================================================
RHO_CENTRAL_WD = 2e9            # Central density of Chandrasekhar WD [g/cm³]
RHO_DDT = 2e7                   # DDT-favorable density [g/cm³]
M_CHANDRASEKHAR = 1.44 * M_SUN  # Chandrasekhar mass limit [g]
R_WD = 2e8                      # Typical WD radius [cm] (~2000 km)

# =============================================================================
# NUCLEAR DECAY TIMESCALES
# =============================================================================
DAY = 86400.0                   # Seconds per day [s]
TAU_NI56 = 8.8 * DAY            # Ni56 decay timescale [s] (8.8 days)
TAU_CO56 = 111.3 * DAY          # Co56 decay timescale [s] (111.3 days)
HALF_LIFE_NI56 = 6.1 * DAY      # Ni56 half-life [s] (6.1 days)
HALF_LIFE_CO56 = 77.0 * DAY     # Co56 half-life [s] (77 days)

# =============================================================================
# COMPOSITION PARAMETERS (for C/O white dwarf)
# =============================================================================
A_BAR = 12.0                    # Mean atomic mass (pure C12)
Z_BAR = 6.0                     # Mean atomic number (pure C12)
Y_E = Z_BAR / A_BAR             # Electron fraction = 0.5

# =============================================================================
# CONVERSION FACTORS
# =============================================================================
KM_TO_CM = 1e5                  # km to cm
CM_TO_KM = 1e-5                 # cm to km
MPC_TO_KM = 3.086e19            # Mpc to km
KM_TO_MPC = 1.0 / MPC_TO_KM     # km to Mpc
ERG_TO_MEV = 6.242e5            # erg to MeV
MEV_TO_ERG = 1.602e-6           # MeV to erg

# =============================================================================
# EXPORTS
# =============================================================================
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
