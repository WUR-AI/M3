"""Execution topology definitions for the multi-agent system.

This module defines ``EXECUTION_TOPOLOGIES``: named execution presets that
control how plans are executed.

Each topology contains:
    ``description``:
        Human-readable summary of the execution strategy used by the topology.
    ``players_per_step``:
        How many players work on each step in parallel.
    ``debate_rounds``:
        How many critique/revise cycles occur within each step.
    ``player_pool``:
        Which player roles can be assigned to steps. ``PLAYER_CONFIGS`` are
        defined in ``src/players/configs.py``.

Available topologies:
    ``default``:
        Standard execution with 3 parallel players per step, 2 debate rounds,
        and comprehensive player pool.

        - ``players_per_step``: 3
        - ``debate_rounds``: 2
        - ``player_pool``: ``data_analyst``, ``schema_expert``,
          ``metadata_specialist``
    ``fast``:
        Quick execution with 2 parallel players and minimal debate.

        - ``players_per_step``: 2
        - ``debate_rounds``: 1
        - ``player_pool``: ``data_analyst``, ``schema_expert``
    ``thorough``:
        Thorough execution with more players and extended debate.

        - ``players_per_step``: 4
        - ``debate_rounds``: 3
        - ``player_pool``: ``data_analyst``, ``schema_expert``,
          ``metadata_specialist``, ``critic``
    ``single``:
        Single-player execution with no debate. Fastest but least robust.

        - ``players_per_step``: 1
        - ``debate_rounds``: 0
        - ``player_pool``: ``data_analyst``

.. note::
   The orchestrator automatically adds ``relationship_analyst`` to the player
   pool for multi-context dataset analysis. No need for separate multi-context
   topologies.
"""
from typing import Dict, Any


# ===================================================================
#  EXECUTION TOPOLOGY DEFINITIONS
# ===================================================================
EXECUTION_TOPOLOGIES: Dict[str, Dict[str, Any]] = {
    "default": {
        "description": (
            "Standard execution with 3 parallel players per step, "
            "2 debate rounds, and comprehensive player pool."
        ),
        "players_per_step": 3,
        "debate_rounds": 2,
        "player_pool": ["data_analyst", "schema_expert", "metadata_specialist"],
    },
    "fast": {
        "description": (
            "Quick execution with 2 parallel players and minimal debate."
        ),
        "players_per_step": 2,
        "debate_rounds": 1,
        "player_pool": ["data_analyst", "schema_expert"],
    },
    "thorough": {
        "description": (
            "Thorough execution with more players and extended debate."
        ),
        "players_per_step": 4,
        "debate_rounds": 3,
        "player_pool": ["data_analyst", "schema_expert", "metadata_specialist", "critic"],
    },
    "single": {
        "description": (
            "Single player execution with no debate. Fastest but least robust."
        ),
        "players_per_step": 1,
        "debate_rounds": 0,
        "player_pool": ["data_analyst"],
    },
}
