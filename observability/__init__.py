"""
Módulo de observabilidade para modelos de Machine Learning.
Otimizado para ambientes Docker/headless.
"""

# Configurar matplotlib ANTES de importar pyplot
from . import config

from .visualizer import ModelVisualizer, PINNVisualizer
from .plots import (
    plot_convergence,
    plot_temperature_profile,
    plot_loss_history,
    plot_measurement_errors,
)
from .metrics import (
    calculate_errors,
    calculate_metrics,
    ModelMetrics,
)
from .dashboard import (
    create_pinn_dashboard,
)
from .export import (
    save_results,
    export_to_json,
    export_to_csv,
)
from .styles import setup_plot_style, PlotTheme

__version__ = "0.1.0"

__all__ = [
    # Classes principais
    'ModelVisualizer',
    'PINNVisualizer',

    # Plots individuais
    'plot_convergence',
    'plot_temperature_profile',
    'plot_loss_history',
    'plot_measurement_errors',

    # Métricas
    'calculate_errors',
    'calculate_metrics',
    'ModelMetrics',

    # Dashboards
    'create_pinn_dashboard',

    # Exportação
    'save_results',
    'export_to_json',
    'export_to_csv',

    # Estilos
    'setup_plot_style',
    'PlotTheme',
]
