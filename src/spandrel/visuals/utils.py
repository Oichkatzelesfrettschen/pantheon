# SPDX-License-Identifier: GPL-2.0-only
"""Visualization utility helpers for Spandrel.

The show_or_close helper guards plt.show() calls so that the code runs
correctly in non-interactive environments (CI, batch scripts) as well as
interactive sessions. In non-interactive mode (MPLBACKEND=Agg or similar),
calling plt.show() would block or produce no output; we close the figure
instead so memory is freed.
"""

import matplotlib
import matplotlib.pyplot as plt


def show_or_close(fig=None) -> None:
    """Display a figure in interactive mode, or close it otherwise.

    Parameters
    ----------
    fig:
        The Figure to act on. When None, acts on the current active figure.

    Why:
        plt.show() blocks execution in non-interactive backends (Agg, Pdf,
        etc.) and is a no-op that leaves figures open in some environments.
        This helper chooses the right action based on the active backend.
    """
    backend = matplotlib.get_backend().lower()
    # Backends that render to screen and can actually display a window.
    interactive_backends = {
        'tkagg', 'qt5agg', 'qt4agg', 'wxagg', 'macosx',
        'gtk3agg', 'gtk4agg', 'webagg', 'nbagg', 'widget',
    }
    if backend in interactive_backends:
        plt.show()
    else:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close()
