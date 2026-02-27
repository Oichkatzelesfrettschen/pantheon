"""Tests for the Spandrel CLI (cli.py).

Verifies that each subcommand routes to the correct underlying function
without actually running the full (expensive) analysis. Underlying
functions are mocked so these tests run in milliseconds.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestCLIHelp:
    """CLI help and no-arg behaviour."""

    def test_no_args_prints_help(self, capsys):
        """Running with no subcommand should print help and exit cleanly."""
        import sys

        from spandrel.cli import main
        with patch.object(sys, 'argv', ['spandrel']):
            main()
        captured = capsys.readouterr()
        assert 'Available modules' in captured.out or 'usage' in captured.out.lower()


class TestCLISynthesis:
    """CLI synthesis subcommand routing."""

    def test_synthesis_calls_run_all_experiments(self):
        import sys

        from spandrel.cli import main
        with patch('spandrel.synthesis.unified_experiment.run_all_experiments') as mock_fn, \
             patch.object(sys, 'argv', ['spandrel', 'synthesis']):
            main()
        mock_fn.assert_called_once()


class TestCLIDDT:
    """CLI ddt subcommand routing."""

    def test_ddt_default_config(self):
        import sys

        from spandrel.cli import main
        mock_solver = MagicMock()
        with patch('spandrel.ddt.main_zeldovich.ZeldovichDDTSolver', return_value=mock_solver), \
             patch.object(sys, 'argv', ['spandrel', 'ddt']):
            main()
        mock_solver.run.assert_called_once()

    def test_ddt_quick_config(self):
        import sys

        from spandrel.cli import main
        mock_solver = MagicMock()
        with patch('spandrel.ddt.main_zeldovich.ZeldovichDDTSolver', return_value=mock_solver), \
             patch('spandrel.ddt.main_zeldovich.SimulationConfig') as mock_cfg, \
             patch.object(sys, 'argv', ['spandrel', 'ddt', '--quick']):
            main()
        # Quick mode uses n_cells=128
        call_kwargs = mock_cfg.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get('n_cells', call_kwargs.args[0] if call_kwargs.args else None) == 128


class TestCLICosmology:
    """CLI cosmology subcommand routing."""

    def test_cosmology_calls_run_analysis(self):
        import importlib
        import sys

        from spandrel.cli import main
        # spandrel.analysis.__init__ re-exports run_analysis, which shadows
        # the submodule in the spandrel.analysis namespace.  Obtain the real
        # module via importlib to ensure patch.object targets the module, not
        # the function object.
        ra_mod = importlib.import_module('spandrel.analysis.run_analysis')
        with patch.object(ra_mod, 'run_analysis') as mock_fn, \
             patch.object(sys, 'argv', ['spandrel', 'cosmology']):
            main()
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs.get('quick_mode') is False

    def test_cosmology_quick_flag(self):
        import importlib
        import sys

        from spandrel.cli import main
        ra_mod = importlib.import_module('spandrel.analysis.run_analysis')
        with patch.object(ra_mod, 'run_analysis') as mock_fn, \
             patch.object(sys, 'argv', ['spandrel', 'cosmology', '--quick']):
            main()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs.get('quick_mode') is True


class TestCLIElevate:
    """CLI elevate subcommand routing."""

    def test_elevate_calls_elevated_main(self):
        import sys

        from spandrel.cli import main
        with patch('spandrel.elevated.run_all.main') as mock_fn, \
             patch.object(sys, 'argv', ['spandrel', 'elevate']):
            main()
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs.get('quick') is False

    def test_elevate_quick_flag(self):
        import sys

        from spandrel.cli import main
        with patch('spandrel.elevated.run_all.main') as mock_fn, \
             patch.object(sys, 'argv', ['spandrel', 'elevate', '--quick']):
            main()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs.get('quick') is True
