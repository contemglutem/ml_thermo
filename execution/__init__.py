"""
Módulo de execução e orquestração de treinamento.

Contém:
    - Trainer: Classe principal de treinamento
    - Pipeline: Pipeline completo de experimentos
    - Config: Gerenciamento de configurações
"""

from .trainer import PINNTrainer
from .pipeline import InverseProblemPipeline
from .config import ExperimentConfig, load_config
from .callbacks import LossHistoryCallback, ModelCheckpointCallback

__version__ = "0.1.0"

__all__ = [
    'PINNTrainer',
    'InverseProblemPipeline',
    'ExperimentConfig',
    'load_config',
    'LossHistoryCallback',
    'ModelCheckpointCallback',
]
