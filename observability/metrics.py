"""Cálculo de métricas para avaliação de modelos."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ModelMetrics:
    """Métricas de desempenho do modelo."""

    # Erros básicos
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error

    # Erros relativos
    max_error: float
    min_error: float
    std_error: float

    # Qualidade do ajuste
    r2_score: float

    # Quantidade de dados
    n_samples: int

    def __str__(self) -> str:
        """Representação legível das métricas."""
        return (
            f"Métricas do Modelo\n"
            f"{'=' * 50}\n"
            f"Amostras: {self.n_samples}\n"
            f"\nErros Absolutos:\n"
            f"  MAE  = {self.mae:.6f}\n"
            f"  MSE  = {self.mse:.6f}\n"
            f"  RMSE = {self.rmse:.6f}\n"
            f"\nErro Percentual:\n"
            f"  MAPE = {self.mape:.2f}%\n"
            f"\nDistribuição de Erros:\n"
            f"  Max  = {self.max_error:.6f}\n"
            f"  Min  = {self.min_error:.6f}\n"
            f"  Std  = {self.std_error:.6f}\n"
            f"\nQualidade do Ajuste:\n"
            f"  R²   = {self.r2_score:.6f}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'mae': float(self.mae),
            'mse': float(self.mse),
            'rmse': float(self.rmse),
            'mape': float(self.mape),
            'max_error': float(self.max_error),
            'min_error': float(self.min_error),
            'std_error': float(self.std_error),
            'r2_score': float(self.r2_score),
            'n_samples': int(self.n_samples),
        }


def calculate_errors(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calcula erros ponto a ponto.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos

    Returns:
        Array com erros (y_pred - y_true)
    """
    return y_pred.flatten() - y_true.flatten()


def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epsilon: float = 1e-10,
) -> ModelMetrics:
    """
    Calcula todas as métricas de avaliação.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos
        epsilon: Valor pequeno para evitar divisão por zero

    Returns:
        ModelMetrics com todas as métricas calculadas
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Erros
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # Métricas básicas
    mae = np.mean(abs_errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    # MAPE (evitar divisão por zero)
    mape = np.mean(abs_errors / (np.abs(y_true) + epsilon)) * 100

    # Distribuição de erros
    max_error = np.max(abs_errors)
    min_error = np.min(abs_errors)
    std_error = np.std(errors)

    # R² score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))

    return ModelMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        mape=mape,
        max_error=max_error,
        min_error=min_error,
        std_error=std_error,
        r2_score=r2,
        n_samples=len(y_true),
    )