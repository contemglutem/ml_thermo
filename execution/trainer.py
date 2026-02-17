"""Classe principal de treinamento."""

import time
from typing import Dict, Any, List, Optional
import tensorflow as tf
import numpy as np

from models.base import BaseModel
from .callbacks import Callback


class PINNTrainer:
    """
    Trainer genérico para PINNs.

    Orquestra o treinamento com:
    - Otimizadores separados para rede e parâmetros
    - Gradient clipping
    - Callbacks
    - Logging
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer_network: tf.keras.optimizers.Optimizer,
        optimizer_params: Optional[tf.keras.optimizers.Optimizer] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Inicializa trainer.

        Args:
            model: Modelo a ser treinado
            optimizer_network: Otimizador para pesos da rede
            optimizer_params: Otimizador para parâmetros físicos (opcional)
            callbacks: Lista de callbacks
        """
        self.model = model
        self.optimizer_network = optimizer_network
        self.optimizer_params = optimizer_params
        self.callbacks = callbacks or []

        # Métricas
        self.start_time = None
        self.last_log_time = None

    @tf.function(jit_compile=False)
    def train_step(
        self,
        y_collocation: tf.Tensor,
        y_measurements: tf.Tensor,
        T_measurements: tf.Tensor,
        loss_weights: Dict[str, float],
        epoch: int,  # ✅ Agora recebe int Python diretamente
    ) -> Dict[str, tf.Tensor]:
        """
        Passo de treinamento otimizado.

        Args:
            y_collocation: Pontos de colocação
            y_measurements: Posições das medições
            T_measurements: Temperaturas medidas
            loss_weights: Pesos das losses
            epoch: Época atual (int Python)

        Returns:
            Dicionário com losses
        """
        with tf.GradientTape(persistent=True) as tape:
            total_loss, components = self.model.compute_loss(
                y_collocation=y_collocation,
                y_measurements=y_measurements,
                T_measurements=T_measurements,
                weights=loss_weights,
                epoch=epoch,  # ✅ Passar int diretamente
            )

        # Separar variáveis
        network_vars = [v for v in self.model.trainable_variables
                       if 'log_q_flux' not in v.name]

        # Calcular gradientes para rede
        grads_network = tape.gradient(total_loss, network_vars)
        grads_network, _ = tf.clip_by_global_norm(grads_network, 1.0)
        self.optimizer_network.apply_gradients(zip(grads_network, network_vars))

        # Calcular gradientes para parâmetros físicos (se aplicável)
        if self.optimizer_params is not None and hasattr(self.model, 'log_q_flux'):
            param_vars = [self.model.log_q_flux]
            grads_params = tape.gradient(total_loss, param_vars)
            grads_params, _ = tf.clip_by_global_norm(grads_params, 0.5)
            self.optimizer_params.apply_gradients(zip(grads_params, param_vars))

        del tape

        return components

    def train(
        self,
        y_collocation: np.ndarray,
        y_measurements: np.ndarray,
        T_measurements: np.ndarray,
        epochs: int,
        loss_weights: Dict[str, float],
        log_freq: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Loop de treinamento principal.

        Args:
            y_collocation: Pontos de colocação
            y_measurements: Medições (posições)
            T_measurements: Medições (temperaturas)
            epochs: Número de épocas
            loss_weights: Pesos das losses
            log_freq: Frequência de logging

        Returns:
            Histórico de treinamento
        """
        # Converter para tensores
        y_col_tensor = tf.constant(y_collocation, dtype=tf.float32)
        y_meas_tensor = tf.constant(y_measurements, dtype=tf.float32)
        T_meas_tensor = tf.constant(T_measurements, dtype=tf.float32)

        # Callbacks: início do treino
        for callback in self.callbacks:
            callback.on_train_begin()

        # Aquecimento (compilar grafos)
        print("Aquecendo GPU (compilando grafos)...")
        for _ in range(5):
            _ = self.train_step(
                y_col_tensor, y_meas_tensor, T_meas_tensor,
                loss_weights, 0  # ✅ Passar int Python diretamente
            )
        print("✓ Aquecimento concluído\n")

        # Início do treino
        self.start_time = time.time()
        self.last_log_time = self.start_time

        print("=" * 80)
        print("INICIANDO TREINAMENTO")
        print("=" * 80)

        for epoch in range(epochs):
            # Callbacks: início da época
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # Train step - passar epoch como int Python
            loss_components = self.train_step(
                y_col_tensor, y_meas_tensor, T_meas_tensor,
                loss_weights, epoch  # ✅ Passar int diretamente (não tensor)
            )

            # Adicionar parâmetros físicos ao log
            logs = {k: v for k, v in loss_components.items()}
            if hasattr(self.model, 'q_flux'):
                logs['q_flux'] = self.model.q_flux

            # Callbacks: fim da época
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)

            # Logging
            if epoch % log_freq == 0:
                self._log_progress(epoch, epochs, logs)

        # Callbacks: fim do treino
        for callback in self.callbacks:
            callback.on_train_end()

        # Sumário final
        self._print_final_summary(epochs)

        # Retornar histórico
        history = {}
        for callback in self.callbacks:
            if hasattr(callback, 'history'):
                history = callback.history
                break

        return history

    def _log_progress(
        self,
        epoch: int,
        total_epochs: int,
        logs: Dict[str, Any]
    ) -> None:
        """Imprime progresso do treinamento."""
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        speed = 100 / elapsed if elapsed > 0 else 0

        total_elapsed = current_time - self.start_time
        eta = (total_epochs - epoch) / speed if speed > 0 else 0

        print(f"Epoch {epoch:5d} | Total: {float(logs['total']):.2e} | "
              f"Data: {float(logs['data']):.2e} | Physics: {float(logs['physics']):.2e}")

        if 'q_flux' in logs:
            print(f"             | q_flux: {float(logs['q_flux']):.2f} W/m²")

        print(f"             | Speed: {speed:.1f} ep/s | "
              f"Elapsed: {total_elapsed:.1f}s | ETA: {eta:.0f}s")
        print("-" * 80)

        self.last_log_time = current_time

    def _print_final_summary(self, epochs: int) -> None:
        """Imprime sumário final do treinamento."""
        total_time = time.time() - self.start_time

        print(f"\n✓ Treinamento Concluído!")
        print(f"  Tempo total: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Velocidade média: {epochs / total_time:.1f} epochs/s")
        print(f"  Tempo por epoch: {total_time / epochs * 1000:.2f} ms")
