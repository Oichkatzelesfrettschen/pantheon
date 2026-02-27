# SPDX-License-Identifier: GPL-2.0-only
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_spandrel_core_importable() -> None:
    """Make `spandrel_core` importable in the meta-repo checkout.

    Pantheon is a submodule in the OpenUniverse meta-repo. The canonical Spandrel
    core lives in the sibling submodule `spandrel-core/`. This helper avoids
    duplicating core primitives inside Pantheon.

    The sys.path manipulation here is intentional: it is the single place where
    we resolve the sibling-submodule layout. All other modules import via the
    installed `spandrel` package and must NOT add their own sys.path entries.
    """
    try:
        import spandrel_core  # noqa: F401
        return
    except ImportError:
        pass

    pantheon_dir = Path(__file__).resolve().parents[2]
    candidate = pantheon_dir.parent / "spandrel-core" / "src"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

    # One more attempt, but do not raise from here; callers will fail with the
    # natural ImportError if `spandrel_core` is truly unavailable.
    try:
        import spandrel_core  # noqa: F401
    except ImportError:
        return


_ensure_spandrel_core_importable()
