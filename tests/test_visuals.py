"""Smoke tests for visualization utilities.

Uses the Agg (non-interactive) backend so tests run in CI without a display.
"""

import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TestShowOrClose:
    """Test the show_or_close guard utility."""

    def test_imports(self):
        from spandrel.visuals.utils import show_or_close
        assert callable(show_or_close)

    def test_closes_figure_with_agg(self):
        """With Agg backend, show_or_close should close the figure."""
        from spandrel.visuals.utils import show_or_close
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        # Should not raise; figure should be closed
        show_or_close(fig)
        assert not plt.fignum_exists(fig.number)

    def test_closes_with_none(self):
        """show_or_close(None) should close the current active figure."""
        from spandrel.visuals.utils import show_or_close
        fig = plt.figure()
        show_or_close(None)
        assert not plt.fignum_exists(fig.number)

    def test_no_figures_open_after(self):
        """After show_or_close, no figures should remain open."""
        from spandrel.visuals.utils import show_or_close
        figs = [plt.figure() for _ in range(3)]
        for f in figs:
            show_or_close(f)
        for f in figs:
            assert not plt.fignum_exists(f.number)


class TestVisualsPackage:
    """Test the visuals sub-package public API."""

    def test_package_exports_show_or_close(self):
        import spandrel.visuals as v
        assert hasattr(v, 'show_or_close')

    def test_all_list(self):
        import spandrel.visuals as v
        assert 'show_or_close' in v.__all__
