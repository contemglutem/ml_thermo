"""Estilos e configurações para plots."""

from enum import Enum
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib as mpl


class PlotTheme(Enum):
    """Temas disponíveis para plots."""
    PROFESSIONAL = "professional"
    SCIENTIFIC = "scientific"
    DARK = "dark"
    COLORBLIND = "colorblind"


# Configurações de estilo por tema
THEME_CONFIGS: Dict[PlotTheme, Dict[str, Any]] = {
    PlotTheme.PROFESSIONAL: {
        'style': 'seaborn-v0_8-darkgrid',
        'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
        'grid_alpha': 0.3,
        'figure_facecolor': 'white',
    },
    PlotTheme.SCIENTIFIC: {
        'style': 'seaborn-v0_8-paper',
        'colors': ['#0077BB', '#CC3311', '#009988', '#EE7733', '#33BBEE'],
        'grid_alpha': 0.2,
        'figure_facecolor': 'white',
    },
    PlotTheme.DARK: {
        'style': 'dark_background',
        'colors': ['#00D9FF', '#FF6B9D', '#FFC75F', '#845EC2', '#00C9A7'],
        'grid_alpha': 0.2,
        'figure_facecolor': '#1a1a1a',
    },
    PlotTheme.COLORBLIND: {
        'style': 'seaborn-v0_8-colorblind',
        'colors': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161'],
        'grid_alpha': 0.3,
        'figure_facecolor': 'white',
    },
}


def setup_plot_style(
        theme: PlotTheme = PlotTheme.PROFESSIONAL,
        font_size: int = 11,
        font_family: str = 'sans-serif',
) -> Dict[str, Any]:
    """
    Configura estilo global dos plots.

    Args:
        theme: Tema a ser aplicado
        font_size: Tamanho base da fonte
        font_family: Família da fonte

    Returns:
        Dicionário com configurações aplicadas
    """
    config = THEME_CONFIGS[theme]

    # Aplicar estilo base
    try:
        plt.style.use(config['style'])
    except Exception:
        plt.style.use('default')

    # Configurações de fonte e tamanho
    mpl.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'axes.labelsize': font_size + 1,
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size + 3,
        'axes.grid': True,
        'grid.alpha': config['grid_alpha'],
        'lines.linewidth': 2,
        'figure.facecolor': config['figure_facecolor'],
        'axes.prop_cycle': mpl.cycler(color=config['colors']),
    })

    return config


def get_color_palette(theme: PlotTheme = PlotTheme.PROFESSIONAL) -> list:
    """Retorna paleta de cores do tema."""
    return THEME_CONFIGS[theme]['colors']