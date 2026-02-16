"""Gerenciamento de configurações."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class ExperimentConfig:
    """Configuração de experimento."""

    # Experimento
    experiment_name: str = "inverse_pinn"
    output_dir: str = "outputs"

    # Modelo
    hidden_layers: tuple = (64, 64, 32)
    activation: str = 'tanh'
    q_flux_init: float = 4500.0
    T_wall: float = 320.0

    # Física
    fluid: str = 'Ammonia'
    temperature: float = 300.0
    pressure: float = 1e6
    v_star: float = 0.05
    L_max: float = 0.001

    # Dados
    n_measurements: int = 12
    noise_std: float = 0.05
    n_collocation: int = 4096
    q_flux_real: float = 5000.0

    # Treinamento
    epochs: int = 10000
    learning_rate_network: float = 3e-3
    learning_rate_param: float = 5e-2

    # Pesos de loss
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'data': 100.0,
        'physics': 0.1,
        'bc': 50.0,
    })

    # Visualização
    plot_theme: str = 'professional'
    plot_dpi: int = 150

    # Hardware
    use_mixed_precision: bool = True
    use_xla: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'hidden_layers': list(self.hidden_layers),
            'activation': self.activation,
            'q_flux_init': self.q_flux_init,
            'T_wall': self.T_wall,
            'fluid': self.fluid,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'v_star': self.v_star,
            'L_max': self.L_max,
            'n_measurements': self.n_measurements,
            'noise_std': self.noise_std,
            'n_collocation': self.n_collocation,
            'q_flux_real': self.q_flux_real,
            'epochs': self.epochs,
            'learning_rate_network': self.learning_rate_network,
            'learning_rate_param': self.learning_rate_param,
            'loss_weights': self.loss_weights,
            'plot_theme': self.plot_theme,
            'plot_dpi': self.plot_dpi,
            'use_mixed_precision': self.use_mixed_precision,
            'use_xla': self.use_xla,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Cria a partir de dicionário."""
        if 'hidden_layers' in config_dict:
            config_dict['hidden_layers'] = tuple(config_dict['hidden_layers'])
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """Salva configuração em YAML."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

        print(f"✓ Configuração salva em {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Carrega configuração de YAML."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)


def load_config(filepath: Optional[str] = None) -> ExperimentConfig:
    """
    Carrega configuração de arquivo ou usa padrão.

    Args:
        filepath: Caminho do arquivo YAML (None = config padrão)

    Returns:
        ExperimentConfig
    """
    if filepath is None:
        return ExperimentConfig()

    return ExperimentConfig.load(filepath)
    