"""Monitoramento de recursos em tempo real."""

import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass
import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot de uso de recursos em um momento."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float


class ResourceMonitor:
    """
    Monitor de recursos do sistema em tempo real.

    Examples:
        >>> monitor = ResourceMonitor(interval=1.0)
        >>> monitor.start()
        >>> # ... seu código ...
        >>> monitor.stop()
        >>> monitor.print_summary()
    """

    def __init__(self, interval: float = 1.0):
        """
        Inicializa monitor de recursos.

        Args:
            interval: Intervalo de amostragem em segundos
        """
        self.interval = interval
        self.snapshots = []
        self.is_running = False
        self._start_time = None

    def take_snapshot(self) -> ResourceSnapshot:
        """Captura snapshot atual de recursos."""
        mem = psutil.virtual_memory()

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=mem.percent,
            memory_used_gb=mem.used / (1024 ** 3),
            memory_available_gb=mem.available / (1024 ** 3),
        )

    def start(self) -> None:
        """Inicia monitoramento."""
        self.is_running = True
        self._start_time = time.time()
        self.snapshots = []
        logger.info("Monitoramento de recursos iniciado")

    def stop(self) -> None:
        """Para monitoramento."""
        self.is_running = False
        logger.info(f"Monitoramento parado. {len(self.snapshots)} snapshots coletados")

    def monitor_while(self, condition: Callable[[], bool]) -> None:
        """
        Monitora recursos enquanto condição for verdadeira.

        Args:
            condition: Função que retorna True para continuar monitorando
        """
        self.start()

        try:
            while condition() and self.is_running:
                snapshot = self.take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.interval)
        finally:
            self.stop()

    def get_peak_usage(self) -> Optional[ResourceSnapshot]:
        """Retorna snapshot com maior uso de CPU."""
        if not self.snapshots:
            return None
        return max(self.snapshots, key=lambda s: s.cpu_percent)

    def get_average_cpu(self) -> float:
        """Retorna uso médio de CPU."""
        if not self.snapshots:
            return 0.0
        return sum(s.cpu_percent for s in self.snapshots) / len(self.snapshots)

    def get_average_memory(self) -> float:
        """Retorna uso médio de memória (%)."""
        if not self.snapshots:
            return 0.0
        return sum(s.memory_percent for s in self.snapshots) / len(self.snapshots)

    def print_summary(self) -> None:
        """Imprime resumo do monitoramento."""
        if not self.snapshots:
            print("Nenhum dado de monitoramento disponível")
            return

        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        peak = self.get_peak_usage()
        avg_cpu = self.get_average_cpu()
        avg_mem = self.get_average_memory()

        print("=" * 80)
        print("RESUMO DE MONITORAMENTO DE RECURSOS")
        print("=" * 80)
        print(f"Duração: {duration:.1f}s")
        print(f"Snapshots: {len(self.snapshots)}")
        print(f"\nCPU:")
        print(f"  Média: {avg_cpu:.1f}%")
        print(f"  Pico: {peak.cpu_percent:.1f}%")
        print(f"\nMemória:")
        print(f"  Média: {avg_mem:.1f}%")
        print(f"  Pico: {peak.memory_percent:.1f}%")
        print("=" * 80)