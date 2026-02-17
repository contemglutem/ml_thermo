"""Funções de plotting individuais."""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_convergence(
        ax: Axes,
        history: np.ndarray,
        true_value: Optional[float] = None,
        ylabel: str = 'Valor',
        title: str = 'Convergência',
        color: str = 'blue',
        true_value_label: str = 'Real',
) -> None:
    """
    Plota convergência de uma variável ao longo das épocas.

    Args:
        ax: Eixo matplotlib
        history: Histórico de valores
        true_value: Valor verdadeiro (linha de referência)
        ylabel: Label do eixo Y
        title: Título do plot
        color: Cor da linha
        true_value_label: Label do valor verdadeiro
    """
    ax.plot(history, f'{color}-', alpha=0.7, linewidth=1.5)

    if true_value is not None:
        ax.axhline(
            y=true_value,
            color='red',
            linestyle='--',
            linewidth=2,
            label=true_value_label
        )
        ax.legend()

    ax.set_xlabel('Época')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_temperature_profile(
        ax: Axes,
        y_positions: np.ndarray,
        T_true: Optional[np.ndarray] = None,
        T_pred: Optional[np.ndarray] = None,
        T_analytical: Optional[np.ndarray] = None,
        measurements: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        xlabel: str = 'Distância (mm)',
        ylabel: str = 'Temperatura (K)',
        title: str = 'Perfil de Temperatura',
) -> None:
    """
    Plota perfil de temperatura comparando diferentes soluções.

    Args:
        ax: Eixo matplotlib
        y_positions: Posições espaciais
        T_true: Temperatura verdadeira
        T_pred: Temperatura predita pelo modelo
        T_analytical: Solução analítica
        measurements: Tupla (posições, temperaturas) das medições
        xlabel: Label do eixo X
        ylabel: Label do eixo Y
        title: Título do plot
    """
    if T_true is not None:
        ax.plot(y_positions, T_true, 'r--', label='Real', linewidth=2)

    if T_pred is not None:
        ax.plot(y_positions, T_pred, 'b-', label='PINN', alpha=0.7, linewidth=2)

    if T_analytical is not None:
        ax.plot(y_positions, T_analytical, 'g:', label='Analítica', linewidth=2)

    if measurements is not None:
        y_meas, T_meas = measurements
        ax.scatter(
            y_meas, T_meas,
            color='orange',
            s=100,
            marker='o',
            edgecolors='black',
            label='Medições',
            zorder=5
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_loss_history(
        ax: Axes,
        loss_history: np.ndarray,
        log_scale: bool = True,
        color: str = 'purple',
        title: str = 'Convergência da Loss',
) -> None:
    """
    Plota histórico de loss.

    Args:
        ax: Eixo matplotlib
        loss_history: Histórico de loss
        log_scale: Se True, usa escala logarítmica
        color: Cor da linha
        title: Título do plot
    """
    if log_scale:
        ax.semilogy(loss_history, color=color, linewidth=2)
        ylabel = 'Loss (log)'
    else:
        ax.plot(loss_history, color=color, linewidth=2)
        ylabel = 'Loss'

    ax.set_xlabel('Época')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_measurement_errors(
        ax: Axes,
        positions: np.ndarray,
        errors: np.ndarray,
        xlabel: str = 'Posição (mm)',
        ylabel: str = 'Erro (K)',
        title: str = 'Erro nas Medições',
        color: str = 'red',
) -> None:
    """
    Plota erros nas medições.

    Args:
        ax: Eixo matplotlib
        positions: Posições das medições
        errors: Erros calculados
        xlabel: Label do eixo X
        ylabel: Label do eixo Y
        title: Título do plot
        color: Cor dos pontos
    """
    ax.scatter(positions, errors, s=100, c=color, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)