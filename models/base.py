"""Classe base abstrata para modelos."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import tensorflow as tf


class BaseModel(ABC, tf.keras.Model):
    """
    Classe base para todos os modelos.

    Define interface comum e funcionalidades compartilhadas.
    """

    def __init__(self, name: str = "BaseModel"):
        """
        Inicializa modelo base.

        Args:
            name: Nome do modelo
        """
        super(BaseModel, self).__init__(name=name)
        self._config: Dict[str, Any] = {}

    @abstractmethod
    def call(self, inputs, training=False):
        """
        Forward pass do modelo.

        Args:
            inputs: Entradas do modelo
            training: Se está em modo de treino

        Returns:
            Saída do modelo
        """
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        """
        Calcula loss do modelo.

        Returns:
            Loss total e componentes
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Retorna configuração do modelo."""
        return self._config

    def set_config(self, config: Dict[str, Any]) -> None:
        """Define configuração do modelo."""
        self._config = config

    def summary_custom(self) -> None:
        """Imprime resumo customizado do modelo."""
        print("=" * 80)
        print(f"MODEL: {self.name}")
        print("=" * 80)

        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        print(f"Total de parâmetros treináveis: {total_params:,}")

        if hasattr(self, 'trainable_params'):
            print("\nParâmetros Físicos Treináveis:")
            for param_name, param in self.trainable_params.items():
                print(f"  {param_name}: {param.numpy():.4f}")

        print("=" * 80)
        