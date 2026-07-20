"""Tests for synthesis input formatting."""

from src.players.player import _format_results_for_synthesis


def test_format_results_includes_tool_results_when_present():
    text = _format_results_for_synthesis(
        [
            {
                "player": "spatial_temporal_specialist_1",
                "analysis": "Let me try other temporal columns...",
                "tool_results": {
                    "llm:4:get_spatial_extent:tc0": {
                        "bounding_box": {
                            "min_lat": 31.95,
                            "max_lat": 56.47,
                            "min_lon": -96.94,
                            "max_lon": 67.01,
                        }
                    }
                },
            }
        ]
    )
    assert "Analysis:\nLet me try other temporal columns" in text
    assert "Tool results:" in text
    assert "min_lat" in text


def test_format_results_omits_tool_section_when_empty():
    text = _format_results_for_synthesis(
        [{"player": "p1", "analysis": "done", "tool_results": {}}]
    )
    assert "Tool results:" not in text
