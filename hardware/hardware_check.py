"""Exemplos de uso do módulo hardware."""

import sys
import logging

# ============================================================================
# CRÍTICO: Configurar ambiente ANTES de importar qualquer coisa do hardware
# ============================================================================

# Adicionar o diretório pai ao path se necessário
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# PRIMEIRO: Configurar ambiente TensorFlow
from hardware.environment import setup_tensorflow_environment

setup_tensorflow_environment(
    log_level='ERROR',
    suppress_protobuf_warnings=True,
    enable_onednn=True,
    suppress_cpp_logs=True,
)

# DEPOIS: Importar o resto
from hardware import (
    check_hardware,
    HardwareManager,
    ResourceMonitor,
    print_system_info,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example1_basic_check():
    """Exemplo 1: Verificação básica de hardware."""
    print("\n" + "=" * 80)
    print("EXEMPLO 1: Verificação Básica")
    print("=" * 80 + "\n")

    # Verificar hardware
    info = check_hardware()

    if info.has_functional_gpu:
        print("\n✓ Sistema pronto para treinamento em GPU")
    else:
        print("\n⚠ Sistema usará CPU para treinamento")


def example2_custom_config():
    """Exemplo 2: Configuração customizada."""
    print("\n" + "=" * 80)
    print("EXEMPLO 2: Configuração Customizada")
    print("=" * 80 + "\n")

    manager = HardwareManager(
        enable_memory_growth=True,
        gpu_index=0,
        verbose=True,
        test_gpu=True,
    )

    info = manager.setup_and_verify()

    print(f"\nGPUs detectadas: {info.gpu_count}")
    print(f"GPU funcional: {info.has_functional_gpu}")


def example3_monitoring():
    """Exemplo 3: Monitoramento de recursos."""
    print("\n" + "=" * 80)
    print("EXEMPLO 3: Monitoramento de Recursos")
    print("=" * 80 + "\n")

    import time

    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    # Simular trabalho
    print("Simulando trabalho por 5 segundos...")
    for i in range(5):
        snapshot = monitor.take_snapshot()
        monitor.snapshots.append(snapshot)
        time.sleep(1)
        print(f"  {i+1}/5 - CPU: {snapshot.cpu_percent:.1f}%")

    monitor.stop()
    monitor.print_summary()


def example4_system_info():
    """Exemplo 4: Informações do sistema."""
    print("\n" + "=" * 80)
    print("EXEMPLO 4: Informações do Sistema")
    print("=" * 80 + "\n")

    print_system_info()


if __name__ == "__main__":
    example1_basic_check()
    example2_custom_config()
    example3_monitoring()
    example4_system_info()
