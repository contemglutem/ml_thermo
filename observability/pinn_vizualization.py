"""Exemplo de uso do módulo observability para PINN."""

import numpy as np
from observability import PINNVisualizer, PlotTheme

# Simular dados de um experimento PINN
np.random.seed(42)

# Histórico de treinamento
n_epochs = 1000
q_flux_history = 50000 + np.cumsum(np.random.randn(n_epochs) * 100)
loss_history = 1000 * np.exp(-np.linspace(0, 5, n_epochs))

# Parâmetros verdadeiros
q_flux_true = 50000.0
L_max = 0.001

# Perfil de temperatura
y_positions = np.linspace(0, L_max, 200) * 1000  # em mm
T_parede = 350.0

# Simulando soluções
T_true = T_parede + q_flux_true * y_positions / 1000 / 1000
T_pinn = T_true + np.random.randn(len(y_positions)) * 0.5
T_analytical = T_parede + q_flux_history[-1] * y_positions / 1000 / 1000

# Medições com ruído
n_measurements = 10
idx_meas = np.linspace(0, len(y_positions) - 1, n_measurements, dtype=int)
y_measurements = y_positions[idx_meas]
T_measurements = T_true[idx_meas] + np.random.randn(n_measurements) * 1.0


def example1_basic_usage():
    """Exemplo 1: Uso básico."""
    print("\n" + "=" * 80)
    print("EXEMPLO 1: Uso Básico")
    print("=" * 80 + "\n")

    visualizer = PINNVisualizer(
        output_dir="outputs/example1",
        theme=PlotTheme.PROFESSIONAL,
        dpi=150,
    )

    fig = visualizer.visualize(
        q_flux_history=q_flux_history,
        q_flux_true=q_flux_true,
        loss_history=loss_history,
        y_positions=y_positions,
        T_true=T_true,
        T_pinn=T_pinn,
        T_analytical=T_analytical,
        y_measurements=y_measurements,
        T_measurements=T_measurements,
        experiment_name="basic_pinn",
    )

    visualizer.print_summary()


def example2_different_themes():
    """Exemplo 2: Diferentes temas visuais."""
    print("\n" + "=" * 80)
    print("EXEMPLO 2: Diferentes Temas")
    print("=" * 80 + "\n")

    themes = [PlotTheme.PROFESSIONAL, PlotTheme.SCIENTIFIC, PlotTheme.DARK]

    for theme in themes:
        print(f"Gerando dashboard com tema: {theme.value}")

        visualizer = PINNVisualizer(
            output_dir=f"outputs/example2_{theme.value}",
            theme=theme,
        )

        visualizer.visualize(
            q_flux_history=q_flux_history,
            q_flux_true=q_flux_true,
            loss_history=loss_history,
            y_positions=y_positions,
            T_true=T_true,
            T_pinn=T_pinn,
            experiment_name=f"pinn_{theme.value}",
        )


def example3_your_code_refactored():
    """Exemplo 3: Seu código original refatorado."""
    print("\n" + "=" * 80)
    print("EXEMPLO 3: Código Original Refatorado")
    print("=" * 80 + "\n")

    # Substituir todo o bloco de visualização por:
    visualizer = PINNVisualizer(output_dir="outputs/pinn_optimized")

    fig = visualizer.visualize(
        q_flux_history=q_flux_history,
        q_flux_true=q_flux_true,
        loss_history=loss_history,
        y_positions=y_positions,
        T_true=T_true,
        T_pinn=T_pinn,
        T_analytical=T_analytical,
        y_measurements=y_measurements,
        T_measurements=T_measurements,
        experiment_name="inverse_problem",
    )

    visualizer.print_summary()

    # Agora você tem:
    # - Dashboard salvo automaticamente
    # - Métricas calculadas e salvas
    # - Resultados exportados em JSON e CSV
    # - Tudo organizado em outputs/


if __name__ == "__main__":
    example1_basic_usage()
    example2_different_themes()
    example3_your_code_refactored()

    print("\n✓ Todos os exemplos concluídos!")
    print("✓ Verifique a pasta 'outputs/' para os resultados")
