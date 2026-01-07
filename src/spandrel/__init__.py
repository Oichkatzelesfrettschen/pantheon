from __future__ import annotations

from pathlib import Path
import sys


def _ensure_spandrel_core_importable() -> None:
    """Make `spandrel_core` importable in the meta-repo checkout.

    Pantheon is a submodule in the OpenUniverse meta-repo. The canonical Spandrel
    core lives in the sibling submodule `spandrel-core/`. This helper avoids
    duplicating core primitives inside Pantheon.
    """
    try:
        import spandrel_core  # noqa: F401
        return
    except Exception:
        pass

    pantheon_dir = Path(__file__).resolve().parents[2]
    candidate = pantheon_dir.parent / "spandrel-core" / "src"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

    # One more attempt, but do not raise from here; callers will fail with the
    # natural ImportError if `spandrel_core` is truly unavailable.
    try:
        import spandrel_core  # noqa: F401
    except Exception:
        return


_ensure_spandrel_core_importable()
