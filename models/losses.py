"""Funções de loss customizadas para PINNs."""

import tensorflow as tf
from typing import Dict, Any


class PINNLoss:
    """Classe base para losses de PINN."""

    def __init__(self, weights: Dict[str, float]):
        """
        Inicializa loss.

        Args:
            weights: Pesos para cada componente da loss
        """
        self.weights = weights

    def __call__(self, components: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Calcula loss ponderada.

        Args:
            components: Dicionário com componentes individuais

        Returns:
            Loss total
        """
        total = tf.constant(0.0, dtype=tf.float32)

        for key, value in components.items():
            weight = self.weights.get(key, 1.0)
            total += weight * value

        return total


class InverseProblemLoss(PINNLoss):
    """Loss específica para problema inverso."""

    def __init__(
            self,
            w_data: float = 100.0,
            w_physics: float = 0.1,
            w_bc: float = 50.0,
    ):
        """
        Inicializa loss para problema inverso.

        Args:
            w_data: Peso da loss de dados
            w_physics: Peso da loss de física
            w_bc: Peso da loss de boundary condition
        """
        weights = {
            'data': w_data,
            'physics': w_physics,
            'bc': w_bc,
        }
        super().__init__(weights)
