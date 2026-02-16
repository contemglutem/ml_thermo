"""
Módulo de modelos de Machine Learning para problemas termo-fluidos.

Contém:
    - PINNs (Physics-Informed Neural Networks)
    - Equações físicas
"""

from .base import BaseModel
from .pinn import InversePINN, DirectPINN
from .physics import TurbulentFlowPhysics

__version__ = "0.1.0"

__all__ = [
    'BaseModel',
    'InversePINN',
    'DirectPINN',
    'TurbulentFlowPhysics',
]