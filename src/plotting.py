"""
Publication-quality plotting utilities.

Provides:
  - Matplotlib RC configuration for print-quality vector output
  - Colorblind-safe palette accessor
  - Common figure/axis helpers
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import rcParams

from config import COLORBLIND_PALETTE, DPI


def configure_matplotlib() -> None:
    """Apply publication-quality RC settings for vector (PDF) output."""
    rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "text.usetex": False,
        }
    )


def color(name: str) -> str:
    """Return a hex colour from the colorblind-safe palette."""
    return COLORBLIND_PALETTE[name]


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
    """Create a figure with standardised settings.

    Calls :func:`configure_matplotlib` on first invocation.
    """
    configure_matplotlib()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def save_and_close(fig: plt.Figure, path: str) -> None:
    """Tight-layout, save to *path*, and close the figure."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
