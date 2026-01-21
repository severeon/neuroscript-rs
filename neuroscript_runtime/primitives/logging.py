"""
Logging primitives for debugging neural network architectures.

Provides pass-through neurons that log tensor information with colored
terminal output using the Rich library.
"""

from typing import Optional
import torch
import torch.nn as nn

# Try to import rich for colored output, fallback to plain print
try:
    from rich.console import Console
    from rich.text import Text

    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None


# Log level configuration
_LEVELS = {
    "debug": 0,
    "info": 1,
    "warn": 2,
    "error": 3,
}

_LEVEL_STYLES = {
    "debug": {"icon": "\u2699\ufe0f ", "color": "dim", "style": "dim"},
    "info": {"icon": "\U0001f535", "color": "cyan", "style": "cyan"},
    "warn": {"icon": "\u26a0\ufe0f ", "color": "yellow", "style": "yellow bold"},
    "error": {"icon": "\u274c", "color": "red", "style": "red bold"},
}


class Log(nn.Module):
    """Debug logging neuron with colored terminal output.

    A pass-through neuron that logs tensor information (shape, dtype) to the
    terminal with colored output and icons. The tensor is returned unchanged.

    Parameters:
        label: Identifier for this log point (e.g., "after_attention")
        level: Log level - "debug", "info", "warn", "error" (default: "info")

    Input shape: [*shape] - any tensor
    Output shape: [*shape] - unchanged pass-through

    Class Attributes:
        enabled: Global enable/disable for all Log neurons (default: True)
        min_level: Minimum level to display (default: "debug")

    Example:
        >>> log = Log("input_layer")
        >>> x = torch.randn(32, 512)
        >>> y = log(x)  # Prints: [input_layer] shape=[32, 512] dtype=float32
        >>> torch.equal(x, y)
        True

        >>> Log.enabled = False  # Disable all logging globally
        >>> Log.min_level = "warn"  # Only show warnings and errors
    """

    # Class-level configuration
    enabled: bool = True
    min_level: str = "debug"

    def __init__(
        self,
        label: str,
        level: str = "info",
    ) -> None:
        super().__init__()

        if level not in _LEVELS:
            raise ValueError(
                f"Invalid log level '{level}'. Must be one of: {list(_LEVELS.keys())}"
            )

        self.label = label
        self.level = level
        self._level_num = _LEVELS[level]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Pass through tensor while logging its information.

        Args:
            input: Input tensor of any shape

        Returns:
            The same input tensor, unchanged
        """
        if not Log.enabled:
            return input

        # Check if this level should be displayed
        min_level_num = _LEVELS.get(Log.min_level, 0)
        if self._level_num < min_level_num:
            return input

        # Format tensor info
        shape_str = list(input.shape)
        dtype_str = str(input.dtype).replace("torch.", "")

        if RICH_AVAILABLE and _console is not None:
            self._log_rich(shape_str, dtype_str)
        else:
            self._log_plain(shape_str, dtype_str)

        return input

    def _log_rich(self, shape: list, dtype: str) -> None:
        """Log with Rich colored output."""
        style_info = _LEVEL_STYLES[self.level]
        icon = style_info["icon"]
        style = style_info["style"]

        text = Text()
        text.append(f"{icon} ", style=style)
        text.append("[", style="dim")
        text.append(self.label, style=style)
        text.append("] ", style="dim")
        text.append("shape=", style="dim")
        text.append(str(shape), style="white")
        text.append(" dtype=", style="dim")
        text.append(dtype, style="white")

        _console.print(text)

    def _log_plain(self, shape: list, dtype: str) -> None:
        """Log with plain text output (fallback)."""
        style_info = _LEVEL_STYLES[self.level]
        icon = style_info["icon"]
        level_tag = self.level.upper()

        print(f"{icon} [{level_tag}] [{self.label}] shape={shape} dtype={dtype}")

    def extra_repr(self) -> str:
        """String representation for printing the module."""
        return f'label="{self.label}", level="{self.level}"'
