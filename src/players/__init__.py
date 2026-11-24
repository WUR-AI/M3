"""
Players module for the multi-agent system.

This package exposes the public player API used by the orchestrator and other
multi-agent components.

Exports
-------
``Player``: 
    Unified player class that can execute tasks and participate in debates.
``create_player_from_config``: 
    Factory function for creating players from configuration dictionaries.
``PLAYER_CONFIGS``: 
    Configuration dictionaries for available player roles.

Examples
--------
Create a player from a role configuration and execute a task::

    from src.players import PLAYER_CONFIGS, create_player_from_config
    
    player = create_player_from_config(
        PLAYER_CONFIGS["data_analyst"],
        name="analyst_1"
    )
    
    result = player.execute_task(
        task="Analyze dataset structure",
        context_key="ctx_abc123",
        context_info={"name": "my_dataset", "resources": ["users"]},
        workspace={},
        inputs={}
    )
"""

from .player import Player, create_player_from_config
from .configs import METADATA_OUTPUT_PLAYER_NAMES, PLAYER_CONFIGS

__all__ = [
    "METADATA_OUTPUT_PLAYER_NAMES",
    "Player",
    "create_player_from_config",
    "PLAYER_CONFIGS",
]
