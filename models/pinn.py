"""Modelos PINN para problemas diretos e inversos."""

from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
import numpy as np

from .base import BaseModel
from .physics import TurbulentFlowPhysics


class InversePINN(BaseModel):
    """
    Physics-Informed Neural Network para problema inverso.

    Estima parâmetros físicos (q_flux) a partir de medições.
    """

    def __init__(
            self,
            physics: TurbulentFlowPhysics,
            hidden_layers: tuple = (64, 64, 32),
            activation: str = 'tanh',
            q_flux_init: float = 4500.0,
            T_wall_init: float = 320.0,
    ):
        """
        Inicializa PINN inverso.

        Args:
            physics: Objeto com equações físicas
            hidden_layers: Neurônios por camada oculta
            activation: Função de ativação
            q_flux_init: Valor inicial de q_flux
            T_wall_init: Temperatura da parede
        """
        super(InversePINN, self).__init__(name="InversePINN")

        self.physics = physics
        self.T_wall = tf.constant(T_wall_init, dtype=tf.float32)

        # Rede neural
        self.hidden_layers_list = []
        for i, units in enumerate(hidden_layers):
            self.hidden_layers_list.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer='glorot_normal',
                    dtype='float32',
                    name=f'hidden_{i}'
                )
            )

        self.output_layer = tf.keras.layers.Dense(
            1,
            bias_initializer=tf.keras.initializers.Constant(T_wall_init - 1.0),
            dtype='float32',
            name='output'
        )

        # Parâmetro físico a ser estimado (em log-space para estabilidade)
        self.log_q_flux = tf.Variable(
            np.log(q_flux_init),
            trainable=True,
            dtype=tf.float32,
            name='log_q_flux'
        )

        # Configuração
        self._config = {
            'hidden_layers': hidden_layers,
            'activation': activation,
            'q_flux_init': q_flux_init,
            'T_wall_init': T_wall_init,
        }

    @property
    def q_flux(self) -> tf.Tensor:
        """Retorna q_flux (sempre positivo via exponencial)."""
        return tf.exp(self.log_q_flux)

    @property
    def trainable_params(self) -> Dict[str, tf.Variable]:
        """Retorna parâmetros físicos treináveis."""
        return {
            'q_flux': self.q_flux,
        }

    def call(self, y, training=False):
        """
        Forward pass.

        Args:
            y: Posição adimensional [0, 1]
            training: Modo de treino

        Returns:
            Temperatura predita
        """
        x = y
        for layer in self.hidden_layers_list:
            x = layer(x, training=training)

        T = self.output_layer(x, training=training)
        return T

    def compute_physics_loss(
            self,
            y: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Calcula loss de física (resíduo da PDE).

        Args:
            y: Posições adimensionais

        Returns:
            Loss de física e componentes
        """
        with tf.GradientTape() as tape:
            tape.watch(y)
            T = self(y, training=True)

        # Gradiente em coordenadas adimensionais
        dT_dy_norm = tape.gradient(T, y)

        # Converter para coordenadas físicas
        dT_dy = dT_dy_norm / self.physics.L_max

        # Calcular resíduo
        residual, termo_fonte = self.physics.compute_residual(
            T=T,
            dT_dy=dT_dy,
            y=y,
            q_flux=self.q_flux,
        )

        # Loss normalizada
        loss_physics = tf.reduce_mean(tf.square(residual)) / (termo_fonte ** 2)

        components = {
            'residual_mean': tf.reduce_mean(tf.abs(residual)),
            'residual_max': tf.reduce_max(tf.abs(residual)),
        }

        return loss_physics, components

    def compute_data_loss(
            self,
            y_measurements: tf.Tensor,
            T_measurements: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calcula loss de dados (medições).

        Args:
            y_measurements: Posições das medições
            T_measurements: Temperaturas medidas

        Returns:
            Loss de dados
        """
        T_pred = self(y_measurements, training=True)
        loss_data = tf.reduce_mean(tf.square(T_pred - T_measurements))
        return loss_data

    def compute_boundary_loss(self) -> tf.Tensor:
        """
        Calcula loss de condição de contorno.

        Returns:
            Loss de BC
        """
        y_wall = tf.constant([[0.0]], dtype=tf.float32)
        T_pred_wall = self(y_wall, training=True)
        loss_bc = tf.square(T_pred_wall - self.T_wall)
        return loss_bc

    def compute_loss(
            self,
            y_collocation: tf.Tensor,
            y_measurements: tf.Tensor,
            T_measurements: tf.Tensor,
            weights: Dict[str, float],
            epoch: int = 0,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Calcula loss total com pesos adaptativos.

        Args:
            y_collocation: Pontos de colocação
            y_measurements: Posições das medições
            T_measurements: Temperaturas medidas
            weights: Pesos das losses
            epoch: Época atual (para pesos adaptativos)

        Returns:
            Loss total e dicionário de componentes
        """
        # Computar todas as losses
        loss_physics, physics_components = self.compute_physics_loss(y_collocation)
        loss_data = self.compute_data_loss(y_measurements, T_measurements)
        loss_bc = self.compute_boundary_loss()

        # Pesos adaptativos
        w_data = weights.get('data', 100.0)
        w_bc = weights.get('bc', 50.0)

        # Peso de física cresce com o tempo
        w_physics_base = weights.get('physics', 0.1)
        if epoch < 2000:
            w_physics = w_physics_base
        else:
            w_physics = tf.minimum(
                w_physics_base + (float(epoch) - 2000.0) / 400.0,
                5.0
            )

        # Loss total
        total_loss = (
                w_data * loss_data +
                w_physics * loss_physics +
                w_bc * loss_bc
        )

        # Componentes para logging
        components = {
            'total': total_loss,
            'data': loss_data,
            'physics': loss_physics,
            'bc': loss_bc,
            'w_physics': w_physics,
            **physics_components,
        }

        return total_loss, components


class DirectPINN(BaseModel):
    """
    PINN para problema direto.

    Resolve PDE dado q_flux conhecido.
    """

    def __init__(
            self,
            physics: TurbulentFlowPhysics,
            q_flux: float,
            hidden_layers: tuple = (64, 64, 32),
            activation: str = 'tanh',
            T_wall: float = 320.0,
    ):
        """
        Inicializa PINN direto.

        Args:
            physics: Objeto com equações físicas
            q_flux: Fluxo de calor conhecido
            hidden_layers: Neurônios por camada
            activation: Função de ativação
            T_wall: Temperatura da parede
        """
        super(DirectPINN, self).__init__(name="DirectPINN")

        self.physics = physics
        self.q_flux_value = tf.constant(q_flux, dtype=tf.float32)
        self.T_wall = tf.constant(T_wall, dtype=tf.float32)

        # Rede neural (mesma arquitetura do inverso)
        self.hidden_layers_list = []
        for i, units in enumerate(hidden_layers):
            self.hidden_layers_list.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer='glorot_normal',
                    dtype='float32',
                    name=f'hidden_{i}'
                )
            )

        self.output_layer = tf.keras.layers.Dense(
            1,
            bias_initializer=tf.keras.initializers.Constant(T_wall - 1.0),
            dtype='float32',
            name='output'
        )

    def call(self, y, training=False):
        """Forward pass."""
        x = y
        for layer in self.hidden_layers_list:
            x = layer(x, training=training)
        return self.output_layer(x, training=training)

    def compute_loss(
            self,
            y_collocation: tf.Tensor,
            weights: Dict[str, float],
            **kwargs
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Calcula loss para problema direto.

        Args:
            y_collocation: Pontos de colocação
            weights: Pesos das losses

        Returns:
            Loss total e componentes
        """
        with tf.GradientTape() as tape:
            tape.watch(y_collocation)
            T = self(y_collocation, training=True)

        dT_dy_norm = tape.gradient(T, y_collocation)
        dT_dy = dT_dy_norm / self.physics.L_max

        residual, termo_fonte = self.physics.compute_residual(
            T=T,
            dT_dy=dT_dy,
            y=y_collocation,
            q_flux=self.q_flux_value,
        )

        loss_physics = tf.reduce_mean(tf.square(residual))

        # BC
        y_wall = tf.constant([[0.0]], dtype=tf.float32)
        T_wall_pred = self(y_wall, training=True)
        loss_bc = tf.square(T_wall_pred - self.T_wall)

        w_physics = weights.get('physics', 1.0)
        w_bc = weights.get('bc', 50.0)

        total_loss = w_physics * loss_physics + w_bc * loss_bc

        components = {
            'total': total_loss,
            'physics': loss_physics,
            'bc': loss_bc,
        }

        return total_loss, components
