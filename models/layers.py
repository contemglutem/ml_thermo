"""Camadas customizadas para PINNs."""

import tensorflow as tf


class FourierFeatureLayer(tf.keras.layers.Layer):
    """
    Fourier Feature Layer para melhor representação de alta frequência.

    Referência: https://arxiv.org/abs/2006.10739
    """

    def __init__(self, num_features: int = 256, scale: float = 1.0, **kwargs):
        """
        Inicializa camada.

        Args:
            num_features: Número de features de Fourier
            scale: Escala das frequências
        """
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scale = scale

    def build(self, input_shape):
        """Constrói pesos da camada."""
        input_dim = input_shape[-1]

        # Matriz de frequências aleatórias (não treinável)
        self.B = self.add_weight(
            name='fourier_features',
            shape=(input_dim, self.num_features),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.scale),
            trainable=False,
        )

    def call(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Fourier features concatenados [cos(2πBx), sin(2πBx)]
        """
        x_proj = 2 * 3.14159265359 * tf.matmul(x, self.B)
        return tf.concat([tf.cos(x_proj), tf.sin(x_proj)], axis=-1)

    def get_config(self):
        """Retorna configuração da camada."""
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'scale': self.scale,
        })
        return config


class AdaptiveActivation(tf.keras.layers.Layer):
    """
    Função de ativação adaptativa com parâmetros aprendíveis.

    a(x) = tanh(α * x) onde α é aprendível
    """

    def __init__(self, alpha_init: float = 1.0, **kwargs):
        """
        Inicializa ativação adaptativa.

        Args:
            alpha_init: Valor inicial de α
        """
        super().__init__(**kwargs)
        self.alpha_init = alpha_init

    def build(self, input_shape):
        """Constrói parâmetro α."""
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.alpha_init),
            trainable=True,
        )

    def call(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            tanh(α * x)
        """
        return tf.nn.tanh(self.alpha * x)

    def get_config(self):
        """Retorna configuração da camada."""
        config = super().get_config()
        config.update({'alpha_init': self.alpha_init})
        return config
