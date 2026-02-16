"""Classes principais para visualização."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .dashboard import create_pinn_dashboard
from .metrics import calculate_metrics, calculate_errors, ModelMetrics
from .export import save_results
from .styles import PlotTheme


class ModelVisualizer:
    """Classe base para visualização de modelos."""

    def __init__(
            self,
            output_dir: str = "outputs",
            theme: PlotTheme = PlotTheme.PROFESSIONAL,
            dpi: int = 150,
    ):
        """
        Inicializa visualizador.

        Args:
            output_dir: Diretório para salvar outputs
            theme: Tema visual
            dpi: Resolução das figuras
        """
        self.output_dir = Path(output_dir)
        self.theme = theme
        self.dpi = dpi

        # Criar subdiretórios
        self.figures_dir = self.output_dir / "figures"
        self.metrics_dir = self.output_dir / "metrics"
        self.reports_dir = self.output_dir / "reports"

        for dir_path in [self.figures_dir, self.metrics_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class PINNVisualizer(ModelVisualizer):
    """Visualizador especializado para PINNs."""

    def __init__(self, **kwargs):
        """Inicializa visualizador PINN."""
        super().__init__(**kwargs)
        self.results: Dict[str, Any] = {}
        self.metrics: Optional[ModelMetrics] = None

    def visualize(
            self,
            q_flux_history: np.ndarray,
            q_flux_true: float,
            loss_history: np.ndarray,
            y_positions: np.ndarray,
            T_true: np.ndarray,
            T_pinn: np.ndarray,
            T_analytical: Optional[np.ndarray] = None,
            y_measurements: Optional[np.ndarray] = None,
            T_measurements: Optional[np.ndarray] = None,
            experiment_name: str = "pinn_experiment",
    ) -> plt.Figure:
        """
        Cria visualização completa e salva resultados.

        Args:
            ... (mesmo que create_pinn_dashboard)
            experiment_name: Nome do experimento

        Returns:
            Figura matplotlib
        """
        # Salvar dados para referência
        self.results = {
            'q_flux_history': q_flux_history,
            'q_flux_true': q_flux_true,
            'q_flux_final': q_flux_history[-1],
            'loss_history': loss_history,
            'y_positions': y_positions,
            'T_true': T_true,
            'T_pinn': T_pinn,
            'T_analytical': T_analytical,
            'y_measurements': y_measurements,
            'T_measurements': T_measurements,
        }

        # Calcular métricas
        self.metrics = calculate_metrics(T_true, T_pinn)

        # Criar dashboard
        save_path = self.figures_dir / f"{experiment_name}_dashboard.png"
        fig = create_pinn_dashboard(
            q_flux_history=q_flux_history,
            q_flux_true=q_flux_true,
            loss_history=loss_history,
            y_positions=y_positions,
            T_true=T_true,
            T_pinn=T_pinn,
            T_analytical=T_analytical,
            y_measurements=y_measurements,
            T_measurements=T_measurements,
            save_path=str(save_path),
            theme=self.theme,
            dpi=self.dpi,
        )

        # Salvar métricas
        self.save_metrics(experiment_name)

        # Salvar resultados completos
        save_results(
            data=self.results,
            metrics=self.metrics,
            output_dir=str(self.output_dir),
            experiment_name=experiment_name,
        )

        return fig

    def save_metrics(self, experiment_name: str) -> None:
        """Salva métricas em arquivo texto."""
        if self.metrics is None:
            return

        metrics_file = self.metrics_dir / f"{experiment_name}_metrics.txt"

        with open(metrics_file, 'w') as f:
            f.write(str(self.metrics))
            f.write("\n\nResultados Finais:\n")
            f.write(f"q_flux estimado: {self.results['q_flux_final']:.2f} W/m²\n")
            f.write(f"q_flux real: {self.results['q_flux_true']:.2f} W/m²\n")
            f.write(
                f"Erro relativo: {abs(self.results['q_flux_final'] - self.results['q_flux_true']) / self.results['q_flux_true'] * 100:.2f}%\n")

        print(f"✓ Métricas salvas em {metrics_file}")

    def print_summary(self) -> None:
        """Imprime resumo dos resultados."""
        if self.metrics is None:
            print("Nenhuma métrica disponível")
            return

        print("\n" + "=" * 80)
        print("RESUMO DOS RESULTADOS")
        print("=" * 80)
        print(self.metrics)
        print("\nFluxo de Calor:")
        print(f"  Estimado: {self.results['q_flux_final']:.2f} W/m²")
        print(f"  Real:     {self.results['q_flux_true']:.2f} W/m²")
        erro_rel = abs(self.results['q_flux_final'] - self.results['q_flux_true']) / self.results['q_flux_true'] * 100
        print(f"  Erro:     {erro_rel:.2f}%")
        print("=" * 80)
