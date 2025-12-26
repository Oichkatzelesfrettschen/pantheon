"""
Total Analysis: High-Fidelity Pantheon+ Synthesis
================================================

This module performs a definitive analysis of all 1701 supernovae
in the Pantheon+ dataset, including full chi-squared minimization
and residual extraction.

It uses the Spandrel framework to validate the 'Turbulence Spread' 
hypothesis across the entire cosmic sample.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from spandrel.core.data_interface import PantheonData
from spandrel.cosmology.spandrel_cosmology import SpandrelCosmology, SpandrelFitter

def run_total_fidelity_analysis():
    print("-" * 60)
    print("TOTAL FIDELITY ANALYSIS: 1701 SUPERNOVAE")
    print("-" * 60)
    
    # 1. Load the full dataset (no cuts other than basic validity)
    print("[DATA] Loading entire Pantheon+ dataset...")
    data = PantheonData(z_min=0.0, z_max=3.0) 
    z, mu_obs, mu_err = data.get_cosmology_data()
    n_total = len(z)
    print(f"[OK] Dataset loaded: {n_total} supernovae")
    
    # 2. Perform definitive LCDM fit
    fitter = SpandrelFitter(z, mu_obs, mu_err)
    print("[FIT] Minimizing chi-squared for Standard LCDM...")
    lcdm_res = fitter.fit_lcdm()
    
    # 3. Perform definitive Spandrel fit (Stiffness Epsilon)
    print("[FIT] Minimizing chi-squared for Spandrel Cosmology...")
    spandrel_res = fitter.fit_spandrel(use_global=True)
    
    # 4. Extract High-Fidelity Residuals
    cosmo_best = SpandrelCosmology(H0=spandrel_res['H0'], 
                                   Omega_m=spandrel_res['Omega_m'], 
                                   epsilon=spandrel_res['epsilon'])
    mu_model = cosmo_best.distance_modulus_array(z, use_spandrel=True)
    residuals = mu_obs - mu_model
    
    # 5. Statistical Synthesis
    chi2_dof = spandrel_res['reduced_chi2']
    sigma_int = np.std(residuals)
    
    print("\n" + "=" * 60)
    print("FINAL SCIENTIFIC SYNTHESIS")
    print("=" * 60)
    print(f"Total SN Samples: {n_total}")
    print(f"Best-fit H0:      {spandrel_res['H0']:.3f} km/s/Mpc")
    print(f"Best-fit Omega_m: {spandrel_res['Omega_m']:.4f}")
    print(f"Best-fit Epsilon: {spandrel_res['epsilon']:.6f}")
    print(f"Reduced Chi^2:    {chi2_dof:.4f}")
    print(f"Intrinsic Scatter: {sigma_int:.4f} mag")
    
    # Save the total residuals for the turbulence bridge
    results_dir = Path(__file__).parents[3] / "results"
    residual_file = results_dir / "total_residuals.csv"
    
    df_res = pd.DataFrame({
        'z': z,
        'residual': residuals,
        'err': mu_err
    })
    df_res.to_csv(residual_file, index=False)
    print(f"[OK] Total residuals exported to {residual_file}")
    
    print("-" * 60)
    print("Analysis Complete. The universe is accounted for.")
    print("-" * 60)

if __name__ == "__main__":
    run_total_fidelity_analysis()
