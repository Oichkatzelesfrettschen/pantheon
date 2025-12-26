import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Spandrel: Cosmological Hypothesis Testing & Supernova Synthesis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available modules")

    # Subcommand: synthesis (The Unified Experiment)
    parser_synth = subparsers.add_parser("synthesis", help="Run the Unified Turbulence-Phillips Experiment")
    
    # Subcommand: ddt (The Simulation)
    parser_ddt = subparsers.add_parser("ddt", help="Run Zeldovich DDT Simulation")
    parser_ddt.add_argument("--quick", action="store_true", help="Run in low-res mode for verification")

    # Subcommand: cosmology (The Analysis)
    parser_cosmo = subparsers.add_parser("cosmology", help="Run Cosmological Analysis (MCMC)")
    parser_cosmo.add_argument("--quick", action="store_true", help="Fast verification run")

    # Subcommand: elevate (Run All)
    parser_elevate = subparsers.add_parser("elevate", help="Run the full 'Elevated' research suite")
    parser_elevate.add_argument("--quick", action="store_true", help="Skip long-running parameter studies")

    # Subcommand: total (High-Fidelity Total Analysis)
    parser_total = subparsers.add_parser("total", help="Run High-Fidelity analysis of all 1701 supernovae")

    args = parser.parse_args()

    if args.command == "synthesis":
        from spandrel.synthesis.unified_experiment import run_all_experiments
        print("[LAUNCH] Launching Unified Synthesis...")
        run_all_experiments()

    elif args.command == "total":
        from spandrel.analysis.total_analysis import run_total_fidelity_analysis
        print("[LAUNCH] Launching High-Fidelity Total Analysis...")
        run_total_fidelity_analysis()

    elif args.command == "ddt":
        from spandrel.ddt.main_zeldovich import SimulationConfig, ZeldovichDDTSolver
        print("[BOOM] Initializing DDT Simulation...")
        # Quick config vs Full config
        if args.quick:
            config = SimulationConfig(n_cells=128, max_steps=1000, verbose=True)
        else:
            config = SimulationConfig(n_cells=512, verbose=True)
        
        solver = ZeldovichDDTSolver(config)
        solver.run()

    elif args.command == "cosmology":
        from spandrel.analysis.run_analysis import run_analysis
        print("[OBS] Starting Cosmological Analysis...")
        run_analysis(quick_mode=args.quick)

    elif args.command == "elevate":
        from spandrel.elevated.run_all import main as run_elevated
        print("[NEW] Running Elevated Suite...")
        # We need to hack sys.argv for the elevated script's argparse
        sys.argv = [sys.argv[0]]
        if args.quick:
            sys.argv.append("--quick")
        else:
            sys.argv.append("--full")
        run_elevated()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
