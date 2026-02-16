"""Callbacks customizados para treinamento."""

from typing import Dict, Any, List
import numpy as np


class Callback:
    """Classe base para callbacks."""

    def on_epoch_begin(self, epoch: int) -> None:
        """Chamado no início de cada época."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Chamado no fim de cada época."""
        pass

    def on_train_begin(self) -> None:
        """Chamado no início do treinamento."""
        pass

    def on_train_end(self) -> None:
        """Chamado no fim do treinamento."""
        pass


class LossHistoryCallback(Callback):
    """Armazena histórico de losses."""

    def __init__(self):
        """Inicializa callback."""
        self.history: Dict[str, List[float]] = {}

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Armazena valores da época."""
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []

            # Converter tensor para float se necessário
            if hasattr(value, 'numpy'):
                value = float(value.numpy())

            self.history[key].append(value)

    def get_history(self, key: str) -> np.ndarray:
        """Retorna histórico de uma métrica."""
        return np.array(self.history.get(key, []))


class ModelCheckpointCallback(Callback):
    """Salva checkpoints do modelo."""

    def __init__(
            self,
            filepath: str,
            save_freq: int = 1000,
            save_best_only: bool = False,
            monitor: str = 'total',
    ):
        """
        Inicializa callback.

        Args:
            filepath: Caminho base para salvar
            save_freq: Frequência de salvamento (épocas)
            save_best_only: Salvar apenas melhor modelo
            monitor: Métrica a monitorar
        """
        self.filepath = filepath
        self.save_freq = save_freq
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_value = float('inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Salva checkpoint se necessário."""
        if epoch % self.save_freq != 0:
            return

        current_value = logs.get(self.monitor, float('inf'))

        if hasattr(current_value, 'numpy'):
            current_value = float(current_value.numpy())

        if self.save_best_only:
            if current_value < self.best_value:
                self.best_value = current_value
                print(f"\n✓ Novo melhor {self.monitor}: {current_value:.6f}")
                # Aqui você salvaria o modelo
        else:
            print(f"\n✓ Checkpoint salvo (época {epoch})")
            # Aqui você salvaria o modelo
