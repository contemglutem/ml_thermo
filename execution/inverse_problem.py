"""
Script principal para problema inverso.

Este é o equivalente refatorado do seu código original.
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ✅ ADICIONAR: Configurar ambiente ANTES de imports
from hardware.environment import setup_tensorflow_environment

setup_tensorflow_environment(
    log_level='ERROR',
    suppress_protobuf_warnings=True,
    suppress_cpp_logs=True,
)

# Agora importar o resto
from execution import InverseProblemPipeline, ExperimentConfig


def main():
    """Função principal."""

    # Criar configuração
    config = ExperimentConfig(
        experiment_name="inverse_problem_ammonia",
        output_dir="outputs/inverse_problem",

        # Modelo
        hidden_layers=(64, 64, 32),
        activation='tanh',
        q_flux_init=4500.0,
        T_wall=320.0,

        # Física
        fluid='Ammonia',
        temperature=300.0,
        pressure=1e6,
        v_star=0.05,
        L_max=0.001,

        # Dados
        n_measurements=12,
        noise_std=0.05,
        n_collocation=4096,
        q_flux_real=5000.0,

        # Treinamento
        epochs=10000,
        learning_rate_network=3e-3,
        learning_rate_param=5e-2,

        # Pesos de loss
        loss_weights={
            'data': 100.0,
            'physics': 0.1,
            'bc': 50.0,
        },

        # Visualização
        plot_theme='professional',
        plot_dpi=150,

        # Hardware
        use_mixed_precision=True,
        use_xla=False,
    )

    # Criar e executar pipeline
    pipeline = InverseProblemPipeline(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    results = main()
