#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Spandrel Cosmology Analysis Runner
==================================

Main entry point for running the complete Spandrel cosmology analysis
with maximal parallelization on Apple Silicon.

Usage:
    python run_analysis.py                    # Full analysis (MCMC + Evidence)
    python run_analysis.py --quick            # Quick MLE-only analysis
    python run_analysis.py --mcmc-only        # Skip evidence computation
    python run_analysis.py --cores 4          # Limit CPU cores

Author: Spandrel Cosmology Project
"""

import argparse
import os
import sys
import time


def print_banner():
    """Print analysis banner."""
    banner = """
+======================================================================+
|                                                                      |
|     #######+######+  #####+ ###+   ##+######+ ######+ #######+##+    |
|     ##+====+##+==##+##+==##+####+  ##|##+==##+##+==##+##+====+##|    |
|     #######+######++#######|##+##+ ##|##|  ##|######++#####+  ##|    |
|     +====##|##+===+ ##+==##|##|+##+##|##|  ##|##+==##+##+==+  ##|    |
|     #######|##|     ##|  ##|##| +####|######++##|  ##|#######+#######|
|     +======++=+     +=+  +=++=+  +===++=====+ +=+  +=++======++======+
|                                                                      |
|              COSMOLOGY HYPOTHESIS TESTING FRAMEWORK                  |
|                                                                      |
|           Testing Spacetime Stiffness with Pantheon+ SNe Ia          |
|                                                                      |
+======================================================================+
    """
    print(banner)


def check_dependencies():
    """Check and report available dependencies."""
    print("\n[DEPS] Checking dependencies...\n")

    deps = {}

    # Core dependencies
    try:
        import numpy as np
        deps['numpy'] = f"[OK] NumPy {np.__version__}"
    except ImportError:
        deps['numpy'] = "[X] NumPy NOT FOUND (required)"
        return False, deps

    try:
        import pandas as pd
        deps['pandas'] = f"[OK] Pandas {pd.__version__}"
    except ImportError:
        deps['pandas'] = "[X] Pandas NOT FOUND (required)"
        return False, deps

    try:
        import scipy
        deps['scipy'] = f"[OK] SciPy {scipy.__version__}"
    except ImportError:
        deps['scipy'] = "[X] SciPy NOT FOUND (required)"
        return False, deps

    try:
        import matplotlib
        deps['matplotlib'] = f"[OK] Matplotlib {matplotlib.__version__}"
    except ImportError:
        deps['matplotlib'] = "[X] Matplotlib NOT FOUND (required)"
        return False, deps

    # Optional GPU acceleration
    try:
        import mlx.core as mx  # noqa: F401
        deps['mlx'] = "[OK] MLX (Metal GPU) available"
    except ImportError:
        deps['mlx'] = "( ) MLX not installed (optional, pip install mlx)"

    for _name, status in deps.items():
        print(f"  {status}")

    return True, deps


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Spandrel Cosmology Analysis Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                     Full analysis
  python run_analysis.py --quick             Quick MLE-only
  python run_analysis.py --mcmc-samples 20000  High-precision MCMC
  python run_analysis.py --no-plots          Skip visualization
        """
    )

    parser.add_argument('--data', type=str, default='Pantheon+SH0ES.dat',
                       help='Path to Pantheon+ data file')

    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis (MLE only, no MCMC/evidence)')

    parser.add_argument('--mcmc-only', action='store_true',
                       help='Run MCMC but skip nested sampling evidence')

    parser.add_argument('--mcmc-samples', type=int, default=5000,
                       help='Number of MCMC samples per chain (default: 5000)')

    parser.add_argument('--mcmc-chains', type=int, default=None,
                       help='Number of MCMC chains (default: CPU cores)')

    parser.add_argument('--evidence-live', type=int, default=300,
                       help='Number of nested sampling live points (default: 300)')

    parser.add_argument('--cores', type=int, default=None,
                       help='Number of CPU cores to use (default: all)')

    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')

    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results and plots')

    parser.add_argument('--z-min', type=float, default=0.001,
                       help='Minimum redshift cutoff')

    parser.add_argument('--z-max', type=float, default=2.5,
                       help='Maximum redshift cutoff')

    return parser.parse_args()


def run_analysis(
    *,
    quick_mode: bool = False,
    data_path: str = 'Pantheon+SH0ES.dat',
    output_dir: str = 'results',
    mcmc_only: bool = False,
    mcmc_samples: int = 5000,
    mcmc_chains: int = None,
    evidence_live: int = 300,
    no_plots: bool = False,
    z_min: float = 0.001,
    z_max: float = 2.5,
):
    """Run the main analysis pipeline.

    All parameters are keyword-only so that callers (CLI, scripts, notebooks)
    can pass exactly what they need without constructing an argparse Namespace.
    """
    from spandrel.cosmology.spandrel_cosmology_hpc import (
        NUM_CORES,
        SpandrelAnalysisPipeline,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pipeline
    print("\n[OBS] Initializing Spandrel Analysis Pipeline...")
    print(f"   Data file: {data_path}")
    print(f"   Output directory: {output_dir}")

    pipeline = SpandrelAnalysisPipeline(data_path)

    # Load data
    print("\n[DATA] Loading Pantheon+ supernova data...")
    pipeline.load_data(z_min=z_min, z_max=z_max)

    # Maximum likelihood estimation
    print("\n[FIT] Running Maximum Likelihood Estimation...")
    start_mle = time.time()
    pipeline.fit_mle()
    mle_time = time.time() - start_mle
    print(f"   MLE completed in {mle_time:.2f}s")

    # Likelihood ratio test
    print("\n[CALC] Performing Likelihood Ratio Test...")
    lr_result = pipeline.likelihood_ratio_test()

    # MCMC sampling
    if not quick_mode:
        print("\n[LINK] Running Parallel MCMC Sampling...")
        print(f"   Samples per chain: {mcmc_samples}")
        print(f"   Number of chains: {mcmc_chains or NUM_CORES}")

        start_mcmc = time.time()
        mcmc_results = pipeline.run_mcmc(
            n_samples=mcmc_samples,
            n_burn=min(2000, mcmc_samples // 2),
            n_chains=mcmc_chains
        )
        mcmc_time = time.time() - start_mcmc
        print(f"   MCMC completed in {mcmc_time:.2f}s")

        # Save MCMC chains
        import numpy as np
        for name, result in mcmc_results.items():
            chain_file = os.path.join(output_dir, f'mcmc_chain_{name}.npy')
            np.save(chain_file, result['chains'])
            print(f"   Saved chain to: {chain_file}")

    # Nested sampling for evidence
    if not quick_mode and not mcmc_only:
        print("\n[STAT] Computing Bayesian Evidence (Nested Sampling)...")
        print(f"   Live points: {evidence_live}")

        start_evidence = time.time()
        pipeline.compute_evidence(n_live=evidence_live)
        evidence_time = time.time() - start_evidence
        print(f"   Evidence computation completed in {evidence_time:.2f}s")

    # Summary report
    pipeline.summary_report()

    # Generate visualizations
    if not no_plots:
        print("\n[PLOT] Generating publication figures...")
        try:
            from spandrel_visualization import create_publication_figures
            create_publication_figures(pipeline, output_dir)
        except ImportError as e:
            print(f"   Warning: Could not generate plots: {e}")

    # Save summary to file
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("SPANDREL COSMOLOGY ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")

        f.write(f"Dataset: Pantheon+ ({pipeline.loader.metadata.get('total_valid', 'N/A')} SNe Ia)\n")
        f.write(f"Redshift range: z = {z_min:.4f} to {z_max:.4f}\n\n")

        f.write("MLE RESULTS:\n")
        f.write("-"*30 + "\n")
        for name, result in pipeline.results['mle'].items():
            f.write(f"\n{name.upper()}:\n")
            f.write(f"  H0 = {result.params.H0:.4f} km/s/Mpc\n")
            f.write(f"  Omega_m = {result.params.Omega_m:.6f}\n")
            if result.params.epsilon != 0:
                f.write(f"  epsilon = {result.params.epsilon:.8f}\n")
            f.write(f"  chi2/dof = {result.reduced_chi2:.6f}\n")

        f.write("\nLIKELIHOOD RATIO TEST:\n")
        f.write(f"  Delta chi2 = {lr_result['delta_chi2']:.4f}\n")
        f.write(f"  p-value = {lr_result['p_value']:.6f}\n")
        f.write(f"  Significance = {lr_result['sigma']:.2f} sigma\n")

        if 'mcmc' in pipeline.results:
            f.write("\nMCMC POSTERIOR ESTIMATES:\n")
            f.write("-"*30 + "\n")
            for name, result in pipeline.results['mcmc'].items():
                f.write(f"\n{name.upper()}:\n")
                for param, stats in result['stats'].items():
                    f.write(f"  {param} = {stats['mean']:.6f} +/- {stats['std']:.6f}\n")

        if 'evidence' in pipeline.results:
            f.write("\nBAYESIAN EVIDENCE:\n")
            f.write("-"*30 + "\n")
            for name, evidence in pipeline.results['evidence'].items():
                f.write(f"  {name}: log(Z) = {evidence.log_evidence:.4f} +/- {evidence.log_evidence_err:.4f}\n")

    print(f"\n[LOG] Summary saved to: {summary_file}")

    return pipeline


def main():
    """Main entry point."""
    # Set thread counts before importing NumPy/SciPy so they take effect.
    # Done here rather than at import time to avoid side effects when the
    # module is imported by other code (e.g. the CLI or tests).
    n_cpu = str(os.cpu_count())
    os.environ.setdefault('OMP_NUM_THREADS', n_cpu)
    os.environ.setdefault('MKL_NUM_THREADS', n_cpu)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', n_cpu)

    print_banner()

    # Check dependencies
    ok, deps = check_dependencies()
    if not ok:
        print("\n[X] Missing required dependencies. Please install them first.")
        sys.exit(1)

    # Parse arguments
    args = parse_args()

    # Set core limit if specified
    if args.cores:
        # This affects the HPC module's NUM_CORES
        os.environ['OMP_NUM_THREADS'] = str(args.cores)
        print(f"\n[CONFIG]  Using {args.cores} CPU cores")

    # Check for data file
    if not os.path.exists(args.data):
        print(f"\n[X] Data file not found: {args.data}")
        print("   Download from: https://github.com/PantheonPlusSH0ES/DataRelease")
        sys.exit(1)

    # Run analysis
    start_total = time.time()

    try:
        pipeline = run_analysis(
            quick_mode=args.quick,
            data_path=args.data,
            output_dir=args.output_dir,
            mcmc_only=args.mcmc_only,
            mcmc_samples=args.mcmc_samples,
            mcmc_chains=args.mcmc_chains,
            evidence_live=args.evidence_live,
            no_plots=args.no_plots,
            z_min=args.z_min,
            z_max=args.z_max,
        )
    except KeyboardInterrupt:
        print("\n\n[WARN]  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Analysis failed with error: {e}")
        raise

    total_time = time.time() - start_total

    print("\n[OK] Analysis complete!")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"   Results saved to: {args.output_dir}/")

    # Final physical interpretation
    if 'mcmc' in pipeline.results and 'spandrel' in pipeline.results['mcmc']:
        eps_stats = pipeline.results['mcmc']['spandrel']['stats']['epsilon']
        eps = eps_stats['mean']
        eps_err = eps_stats['std']

        print("\n" + "="*70)
        print("PHYSICAL INTERPRETATION")
        print("="*70)

        if abs(eps) < 2 * eps_err:
            print(f"""
    Stiffness Parameter: epsilon = {eps:.6f} +/- {eps_err:.6f}

    RESULT: epsilon is consistent with zero within 2sigma

    INTERPRETATION: The Universe appears purely Associative.
    Standard LambdaCDM is statistically sufficient to describe the data.
    The "Surface Tension" effect, if present, is below detection threshold.

    The Hubble Tension, if real, likely has a different explanation:
    - Systematic errors in distance ladder calibration
    - New physics in the early universe (not late-time)
    - Sample variance effects
            """)
        elif eps > 0:
            print(f"""
    Stiffness Parameter: epsilon = {eps:.6f} +/- {eps_err:.6f}

    RESULT: epsilon > 0 detected at {abs(eps/eps_err):.1f}sigma significance!

    INTERPRETATION: Evidence for Spandrel Cosmology

    The universe exhibits positive stiffness, suggesting:
    - Spacetime was "stiffer" in the past
    - Dark energy density may be evolving (not constant)
    - The effective equation of state w_eff(z) deviates from -1

    Implications for Hubble Tension:
    - This could bridge early and late universe H0 measurements
    - High-z (CMB) sees a "stiffer" universe -> lower apparent H0
    - Low-z (Cepheids) sees a "relaxed" universe -> higher apparent H0
            """)
        else:
            print(f"""
    Stiffness Parameter: epsilon = {eps:.6f} +/- {eps_err:.6f}

    RESULT: epsilon < 0 detected

    INTERPRETATION: Universe appears to be "softening" over time

    This unexpected result could indicate:
    - Phantom dark energy (w < -1)
    - Growing dark energy density
    - Systematic effects in the data requiring investigation
            """)


if __name__ == "__main__":
    main()
