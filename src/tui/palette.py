"""
Color palette tokens for the TUI.
"""

import os
import re
import select
import termios
import tty

TITLE_COLOR = "#D78FEE"
DESCRIPTION_COLOR = "#9aa0a6"
INPUT_BACKGROUND = "#452E5A"
DIALOG_COLOR = "#e8eaed"
SUGGESTION_COLOR = "#9aa0a6"
COMMAND_NAME_COLOR = "#FFC85C"
NO_MATCH_COMMAND_COLOR = "#FFC85C"
REFERENCE_TEXT_COLOR = "#e8eaed"
REFERENCE_BACKGROUND = "#3E1E68"
STANDARD_REFERENCE_TEXT_COLOR = "#E9FFF1"
STANDARD_REFERENCE_BACKGROUND = "#1F6B47"
AT_SUGGESTION_SELECTION_COLOR = "#FFC85C"
INPUT_WARNING_COLOR = "#FF6B6B"

# Execution plan tree (TUI)
PLAN_TITLE_COLOR = "#56C8FF"
PLAN_STEP_COLOR = "#56C8FF"
PLAN_LABEL_COLOR = "#e8eaed"
PLAN_VALUE_COLOR = "#e8eaed"
PLAN_MUTED_COLOR = "#9aa0a6"
PLAN_TREE_GUIDE_COLOR = "#6e6e7a"


def _detect_terminal_background() -> str:
    """
    Best-effort terminal background detection.

    Detection order:
    1) OSC 11 query (modern terminals: GNOME Terminal, iTerm2, kitty, etc.)
    2) COLORFGBG environment variable when available
    3) ansi_default fallback
    """
    osc_background = _detect_terminal_background_osc11()
    if osc_background:
        return osc_background

    colorfgbg = os.environ.get("COLORFGBG", "").strip()
    if colorfgbg:
        # Format is usually "foreground;background", sometimes multiple values.
        bg_token = colorfgbg.split(";")[-1].strip()
        dark_map = {
            "0": "#000000",
            "1": "#800000",
            "2": "#008000",
            "3": "#808000",
            "4": "#000080",
            "5": "#800080",
            "6": "#008080",
            "7": "#c0c0c0",
            "8": "#808080",
            "9": "#ff0000",
            "10": "#00ff00",
            "11": "#ffff00",
            "12": "#0000ff",
            "13": "#ff00ff",
            "14": "#00ffff",
            "15": "#ffffff",
        }
        if bg_token in dark_map:
            return dark_map[bg_token]

    return "ansi_default"


def _detect_terminal_background_osc11(timeout_seconds: float = 0.15) -> str | None:
    """
    Query terminal background via OSC 11.

    Expected response format:
      ESC ] 11;rgb:RRRR/GGGG/BBBB BEL
    or
      ESC ] 11;rgb:RRRR/GGGG/BBBB ESC \\
    """
    if not (os.isatty(0) and os.isatty(1)):
        return None

    tty_file = None
    try:
        tty_file = open("/dev/tty", "rb+", buffering=0)
        fd = tty_file.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        # Ask terminal for background color.
        tty_file.write(b"\x1b]11;?\x07")

        buffer = b""
        while True:
            ready, _, _ = select.select([tty_file], [], [], timeout_seconds)
            if not ready:
                break
            chunk = os.read(fd, 256)
            if not chunk:
                break
            buffer += chunk
            if b"\x07" in buffer or b"\x1b\\" in buffer:
                break

        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        text = buffer.decode("utf-8", errors="ignore")
        match = re.search(r"11;rgb:([0-9a-fA-F]{2,4})/([0-9a-fA-F]{2,4})/([0-9a-fA-F]{2,4})", text)
        if not match:
            return None

        def normalize(channel: str) -> str:
            if len(channel) == 2:
                return channel.lower()
            if len(channel) == 4:
                return channel[:2].lower()
            return channel[0].lower() * 2

        r, g, b = (normalize(match.group(1)), normalize(match.group(2)), normalize(match.group(3)))
        return f"#{r}{g}{b}"
    except Exception:
        return None
    finally:
        try:
            if tty_file is not None:
                tty_file.close()
        except Exception:
            pass


GLOBAL_BACKGROUND = _detect_terminal_background()
