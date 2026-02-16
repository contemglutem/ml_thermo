"""Equações físicas para PINNs."""

from typing import Tuple
import tensorflow as tf


class TurbulentFlowPhysics:
    """
    Física de escoamento turbulento com modelo de viscosidade turbulenta.
    """

    def __init__(
            self,
            rho: float,
            cp: float,
            nu: float,
            alpha: float,
            v_star: float,
            L_max: float,
    ):
        """
        Inicializa modelo físico.

        Args:
            rho: Densidade [kg/m³]
            cp: Calor específico [J/(kg·K)]
            nu: Viscosidade cinemática [m²/s]
            alpha: Difusividade térmica [m²/s]
            v_star: Velocidade de fricção [m/s]
            L_max: Comprimento característico [m]
        """
        self.rho = tf.constant(rho, dtype=tf.float32)
        self.cp = tf.constant(cp, dtype=tf.float32)
        self.nu = tf.constant(nu, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.v_star = tf.constant(v_star, dtype=tf.float32)
        self.L_max = tf.constant(L_max, dtype=tf.float32)

    def compute_turbulent_viscosity(self, y: tf.Tensor) -> tf.Tensor:
        """
        Calcula viscosidade turbulenta usando modelo algébrico.

        Args:
            y: Posição adimensional [0, 1]

        Returns:
            Viscosidade turbulenta [m²/s]
        """
        y_real = y * self.L_max
        nu_t = self.nu * tf.pow((y_real * self.v_star) / (14.5 * self.nu), 3)
        return nu_t

    def compute_residual(
            self,
            T: tf.Tensor,
            dT_dy: tf.Tensor,
            y: tf.Tensor,
            q_flux: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calcula resíduo da equação de energia.

        Equação: (α + ν_t) * dT/dy + q/(ρ*cp) = 0

        Args:
            T: Temperatura
            dT_dy: Derivada de T em relação a y (física)
            y: Posição adimensional
            q_flux: Fluxo de calor

        Returns:
            Resíduo da equação
        """
        nu_t = self.compute_turbulent_viscosity(y)
        termo_fonte = q_flux / (self.rho * self.cp)

        residual = (self.alpha + nu_t) * dT_dy + termo_fonte

        return residual, termo_fonte

    def get_properties_dict(self) -> dict:
        """Retorna dicionário com propriedades."""
        return {
            'rho': float(self.rho.numpy()),
            'cp': float(self.cp.numpy()),
            'nu': float(self.nu.numpy()),
            'alpha': float(self.alpha.numpy()),
            'v_star': float(self.v_star.numpy()),
            'L_max': float(self.L_max.numpy()),
        }
    