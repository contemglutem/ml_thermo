"""Pipeline completo para experimentos."""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from scipy.integrate import quad
import tensorflow as tf

from hardware import check_hardware
from thermodynamics import FluidPropertiesCalculator
from models.physics import TurbulentFlowPhysics
from models.pinn import InversePINN
from observability import PINNVisualizer, PlotTheme
from .config import ExperimentConfig
from .trainer import PINNTrainer
from .callbacks import LossHistoryCallback, ModelCheckpointCallback


class InverseProblemPipeline:
    """
    Pipeline completo para problema inverso com PINN.

    Orquestra:
    1. Setup de hardware
    2. Propriedades termodinâmicas
    3. Geração de dados sintéticos
    4. Criação e treinamento do modelo
    5. Visualização de resultados
    """

    def __init__(self, config: ExperimentConfig):
        """
        Inicializa pipeline.

        Args:
            config: Configuração do experimento
        """
        self.config = config
        self.hw_info = None
        self.fluid_props = None
        self.physics = None
        self.model = None
        self.trainer = None
        self.visualizer = None
        self.results = {}

    def setup_hardware(self) -> None:
        """Configura hardware (GPU/CPU)."""
        print("=" * 80)
        print("ETAPA 1: CONFIGURAÇÃO DE HARDWARE")
        print("=" * 80 + "\n")

        # Verificar hardware
        self.hw_info = check_hardware(verbose=True)

        # Mixed precision se solicitado
        if self.config.use_mixed_precision:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✓ Mixed Precision (FP16) ativado\n")

        # XLA se solicitado
        if self.config.use_xla:
            tf.config.optimizer.set_jit(True)
            print("✓ XLA JIT compilation ativado\n")

    def setup_thermodynamics(self) -> None:
        """Calcula propriedades termodinâmicas."""
        print("=" * 80)
        print("ETAPA 2: PROPRIEDADES TERMODINÂMICAS")
        print("=" * 80 + "\n")

        # Calcular propriedades
        calc = FluidPropertiesCalculator(
            fluid=self.config.fluid,
            default_temperature=self.config.temperature,
            default_pressure=self.config.pressure,
        )

        self.fluid_props = calc.calculate_properties()

        print(f"Fluido: {self.config.fluid}")
        print(f"  ρ  = {self.fluid_props.density:.2f} kg/m³")
        print(f"  cp = {self.fluid_props.specific_heat:.2f} J/(kg·K)")
        print(f"  α  = {self.fluid_props.thermal_diffusivity:.2e} m²/s")
        print(f"  ν  = {self.fluid_props.kinematic_viscosity:.2e} m²/s")
        print()

        # Criar objeto de física
        self.physics = TurbulentFlowPhysics(
            rho=self.fluid_props.density,
            cp=self.fluid_props.specific_heat,
            nu=self.fluid_props.kinematic_viscosity,
            alpha=self.fluid_props.thermal_diffusivity,
            v_star=self.config.v_star,
            L_max=self.config.L_max,
        )

    def generate_synthetic_data(self) -> Dict[str, np.ndarray]:
        """
        Gera dados sintéticos usando solução analítica.

        Returns:
            Dicionário com dados sintéticos
        """
        print("=" * 80)
        print("ETAPA 3: GERAÇÃO DE DADOS SINTÉTICOS")
        print("=" * 80 + "\n")

        def integrando_analitico(y_val):
            """Integrando da solução analítica."""
            nu_t = self.fluid_props.kinematic_viscosity * (
                    (y_val * self.config.v_star) / (14.5 * self.fluid_props.kinematic_viscosity)
            ) ** 3
            return 1 / (self.fluid_props.thermal_diffusivity + nu_t)

        def solucao_analitica(y_array, T_wall, q_flux):
            """Calcula solução analítica."""
            T_analitica = []
            for y in y_array:
                integral, _ = quad(integrando_analitico, 0, y)
                temp = T_wall - (q_flux / (
                        self.fluid_props.density * self.fluid_props.specific_heat
                )) * integral
                T_analitica.append(temp)
            return np.array(T_analitica)

        # Gerar medições
        y_measurements = np.linspace(
            0.0001,
            self.config.L_max,
            self.config.n_measurements
        )

        T_measurements = solucao_analitica(
            y_measurements,
            self.config.T_wall,
            self.config.q_flux_real
        )

        # Adicionar ruído
        np.random.seed(42)
        noise = np.random.normal(0, self.config.noise_std, self.config.n_measurements)
        T_measurements_noisy = T_measurements + noise

        # Normalizar posições
        y_measurements_norm = y_measurements / self.config.L_max

        print(f"Fluxo de calor REAL: {self.config.q_flux_real} W/m²")
        print(f"Número de medições: {self.config.n_measurements}")
        print(f"Ruído (std): {self.config.noise_std} K")
        print()

        # Salvar para uso posterior
        self.results['data'] = {
            'y_measurements_physical': y_measurements,
            'y_measurements_normalized': y_measurements_norm,
            'T_measurements_clean': T_measurements,
            'T_measurements_noisy': T_measurements_noisy,
            'analytical_solution_func': solucao_analitica,
        }

        return self.results['data']

    def create_model(self) -> None:
        """Cria modelo PINN."""
        print("=" * 80)
        print("ETAPA 4: CRIAÇÃO DO MODELO")
        print("=" * 80 + "\n")

        self.model = InversePINN(
            physics=self.physics,
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
            q_flux_init=self.config.q_flux_init,
            T_wall_init=self.config.T_wall,
        )

        # Imprimir resumo
        self.model.summary_custom()

    def setup_training(self) -> None:
        """Configura treinamento."""
        print("\n" + "=" * 80)
        print("ETAPA 5: CONFIGURAÇÃO DE TREINAMENTO")
        print("=" * 80 + "\n")

        # Optimizers
        optimizer_network = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate_network
        )
        optimizer_params = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate_param
        )

        # Callbacks
        callbacks = [
            LossHistoryCallback(),
            ModelCheckpointCallback(
                filepath=str(Path(self.config.output_dir) / "checkpoints"),
                save_freq=1000,
            )
        ]

        # Criar trainer
        self.trainer = PINNTrainer(
            model=self.model,
            optimizer_network=optimizer_network,
            optimizer_params=optimizer_params,
            callbacks=callbacks,
        )

        print(f"✓ Optimizer (network): Adam (lr={self.config.learning_rate_network})")
        print(f"✓ Optimizer (params):  Adam (lr={self.config.learning_rate_param})")
        print(f"✓ Callbacks: {len(callbacks)}")
        print()

    def train(self) -> Dict[str, Any]:
        """
        Executa treinamento.

        Returns:
            Histórico de treinamento
        """
        # Preparar dados de colocação
        y_collocation = np.linspace(
            0, 1.0,
            self.config.n_collocation,
            dtype=np.float32
        ).reshape(-1, 1)

        # Dados de medições
        data = self.results['data']
        y_meas = data['y_measurements_normalized'].astype(np.float32).reshape(-1, 1)
        T_meas = data['T_measurements_noisy'].astype(np.float32).reshape(-1, 1)

        # Treinar
        history = self.trainer.train(
            y_collocation=y_collocation,
            y_measurements=y_meas,
            T_measurements=T_meas,
            epochs=self.config.epochs,
            loss_weights=self.config.loss_weights,
            log_freq=100,
        )

        self.results['history'] = history

        return history

    def evaluate_results(self) -> None:
        """Avalia resultados do treinamento."""
        print("\n" + "=" * 80)
        print("ETAPA 6: AVALIAÇÃO DE RESULTADOS")
        print("=" * 80 + "\n")

        # Extrair q_flux estimado
        q_estimated = float(self.model.q_flux.numpy())
        q_real = self.config.q_flux_real
        error = abs(q_estimated - q_real) / q_real * 100

        print(f"Fluxo de calor REAL:      {q_real:.2f} W/m²")
        print(f"Fluxo de calor ESTIMADO:  {q_estimated:.2f} W/m²")
        print(f"Erro relativo:            {error:.2f}%")
        print()

        # Calcular perfis de temperatura
        y_test_norm = np.linspace(0, 1, 200, dtype=np.float32).reshape(-1, 1)
        T_pinn = self.model.predict(y_test_norm, verbose=0).flatten()
        y_physical = y_test_norm.flatten() * self.config.L_max

        # Solução analítica
        analytical_func = self.results['data']['analytical_solution_func']
        T_true = analytical_func(y_physical, self.config.T_wall, q_real)
        T_estimated_analytical = analytical_func(y_physical, self.config.T_wall, q_estimated)

        # Calcular erros nas medições
        data = self.results['data']
        y_meas_norm = data['y_measurements_normalized'].astype(np.float32).reshape(-1, 1)
        T_pinn_meas = self.model.predict(y_meas_norm, verbose=0).flatten()
        T_meas_noisy = data['T_measurements_noisy']

        errors = T_pinn_meas - T_meas_noisy
        rmse = np.sqrt(np.mean(errors ** 2))
        max_error = np.max(np.abs(errors))

        print(f"RMSE nas medições: {rmse:.4f} K")
        print(f"Erro máximo:       {max_error:.4f} K")
        print()

        # Salvar para visualização
        self.results['evaluation'] = {
            'q_estimated': q_estimated,
            'q_real': q_real,
            'error_percent': error,
            'y_physical': y_physical,
            'T_true': T_true,
            'T_pinn': T_pinn,
            'T_estimated_analytical': T_estimated_analytical,
            'rmse': rmse,
            'max_error': max_error,
        }

    def visualize(self) -> None:
        """Cria visualizações."""
        print("=" * 80)
        print("ETAPA 7: VISUALIZAÇÃO")
        print("=" * 80 + "\n")

        # Criar visualizador
        theme_map = {
            'professional': PlotTheme.PROFESSIONAL,
            'scientific': PlotTheme.SCIENTIFIC,
            'dark': PlotTheme.DARK,
            'colorblind': PlotTheme.COLORBLIND,
        }

        self.visualizer = PINNVisualizer(
            output_dir=self.config.output_dir,
            theme=theme_map.get(self.config.plot_theme, PlotTheme.PROFESSIONAL),
            dpi=self.config.plot_dpi,
        )

        # Extrair dados
        history = self.results['history']
        evaluation = self.results['evaluation']
        data = self.results['data']

        # Criar visualização
        self.visualizer.visualize(
            q_flux_history=history['q_flux'],
            q_flux_true=self.config.q_flux_real,
            loss_history=history['data'],
            y_positions=evaluation['y_physical'] * 1000,  # mm
            T_true=evaluation['T_true'],
            T_pinn=evaluation['T_pinn'],
            T_analytical=evaluation['T_estimated_analytical'],
            y_measurements=data['y_measurements_physical'] * 1000,  # mm
            T_measurements=data['T_measurements_noisy'],
            experiment_name=self.config.experiment_name,
        )

        self.visualizer.print_summary()

    def save_config(self) -> None:
        """Salva configuração do experimento."""
        config_path = Path(self.config.output_dir) / "config.yaml"
        self.config.save(str(config_path))

    def run(self) -> Dict[str, Any]:
        """
        Executa pipeline completo.

        Returns:
            Dicionário com todos os resultados
        """
        start_time = time.time()

        print("\n" + "🚀" * 40)
        print(f"INICIANDO EXPERIMENTO: {self.config.experiment_name}")
        print("🚀" * 40 + "\n")

        # Pipeline
        self.setup_hardware()
        self.setup_thermodynamics()
        self.generate_synthetic_data()
        self.create_model()
        self.setup_training()
        self.train()
        self.evaluate_results()
        self.visualize()
        self.save_config()

        total_time = time.time() - start_time

        print("\n" + "✓" * 40)
        print(f"EXPERIMENTO CONCLUÍDO EM {total_time:.1f}s ({total_time / 60:.1f} min)")
        print("✓" * 40 + "\n")

        print(f"📁 Resultados salvos em: {self.config.output_dir}/")
        print(f"  - Figuras: {self.config.output_dir}/figures/")
        print(f"  - Métricas: {self.config.output_dir}/metrics/")
        print(f"  - Relatórios: {self.config.output_dir}/reports/")

        return self.results
