"""Criação de dashboards completos."""

from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .plots import (
    plot_convergence,
    plot_temperature_profile,
    plot_loss_history,
    plot_measurement_errors,
)
from .styles import setup_plot_style, PlotTheme


def create_pinn_dashboard(
    # ... (mesmos parâmetros)
    q_flux_history: np.ndarray,
    q_flux_true: float,
    loss_history: np.ndarray,
    y_positions: np.ndarray,
    T_true: np.ndarray,
    T_pinn: np.ndarray,
    T_analytical: Optional[np.ndarray] = None,
    y_measurements: Optional[np.ndarray] = None,
    T_measurements: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    theme: PlotTheme = PlotTheme.PROFESSIONAL,
    dpi: int = 150,
    figsize: Tuple[int, int] = (14, 10),
    title: str = 'Dashboard - Problema Inverso PINN',
    close_fig: bool = True,  # 🆕 NOVO: fechar figura após salvar
) -> plt.Figure:
    """
    Cria dashboard completo para análise de PINN.

    Args:
        ... (mesmos args anteriores)
        close_fig: Se True, fecha figura após salvar (economiza memória)

    Returns:
        Figura matplotlib (ou None se close_fig=True)
    """
    # Configurar estilo
    setup_plot_style(theme)

    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Convergência do q_flux
    plot_convergence(
        ax=axes[0, 0],
        history=q_flux_history,
        true_value=q_flux_true,
        ylabel='q_flux (W/m²)',
        title='Convergência de q_flux',
        color='b',
        true_value_label='Real',
    )

    # Plot 2: Perfil de Temperatura
    measurements = None
    if y_measurements is not None and T_measurements is not None:
        measurements = (y_measurements, T_measurements)

    plot_temperature_profile(
        ax=axes[0, 1],
        y_positions=y_positions,
        T_true=T_true,
        T_pred=T_pinn,
        T_analytical=T_analytical,
        measurements=measurements,
    )

    # Plot 3: Loss
    plot_loss_history(
        ax=axes[1, 0],
        loss_history=loss_history,
        log_scale=True,
    )

    # Plot 4: Erro nas medições
    if y_measurements is not None and T_measurements is not None:
        errors = T_pinn[:len(T_measurements)] - T_measurements
        plot_measurement_errors(
            ax=axes[1, 1],
            positions=y_measurements,
            errors=errors,
        )
    else:
        axes[1, 1].text(
            0.5, 0.5,
            'Sem medições disponíveis',
            ha='center',
            va='center',
            transform=axes[1, 1].transAxes
        )
        axes[1, 1].set_title('Erro nas Medições')

    # Título geral
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Salvar se especificado
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Dashboard salvo em {save_path}")

    # 🆕 NOVO: Fechar figura para liberar memória
    if close_fig:
        plt.close(fig)
        return None

    return fig


def create_dashboard(
        data: Dict[str, Any],
        save_path: Optional[str] = None,
        **kwargs
) -> Optional[plt.Figure]:
    """
    Cria dashboard genérico a partir de dicionário de dados.

    Args:
        data: Dicionário com dados para plotar
        save_path: Caminho para salvar
        **kwargs: Argumentos adicionais para create_pinn_dashboard

    Returns:
        Figura matplotlib (ou None se close_fig=True)

    Examples:
        >>> data = {
        ...     'q_flux_history': history_q,
        ...     'q_flux_true': 5000.0,
        ...     'loss_history': history_loss,
        ...     # ...
        ... }
        >>> fig = create_dashboard(data, save_path='results.png')
    """
    # Wrapper genérico - pode ser expandido
    return create_pinn_dashboard(
        q_flux_history=data.get('q_flux_history'),
        q_flux_true=data.get('q_flux_true'),
        loss_history=data.get('loss_history'),
        y_positions=data.get('y_positions'),
        T_true=data.get('T_true'),
        T_pinn=data.get('T_pinn'),
        T_analytical=data.get('T_analytical'),
        y_measurements=data.get('y_measurements'),
        T_measurements=data.get('T_measurements'),
        save_path=save_path,
        **kwargs
    )
