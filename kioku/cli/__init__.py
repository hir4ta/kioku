"""kioku CLI entry point package.

Exposes the ``kioku`` console script declared in ``pyproject.toml``:

.. code-block:: text

    kioku rebuild   # full re-ETL: vault → SQLite
    kioku scaffold  # create the vault directory layout
    kioku search    # hybrid retrieval (stub in Phase 1; full in Phase 3+)
    kioku status    # vault + store statistics
    kioku version   # print kioku version

The CLI is intentionally thin: every subcommand resolves to a single
function in :mod:`kioku`. Hooks (``hooks/memory/*.sh``) invoke the same
functions via a shorter ``kioku-hook`` shim added in Phase 2.
"""

from kioku.cli.main import main

__all__ = ["main"]
