"""Utilidades auxiliares para hardware."""

import platform
import subprocess
from typing import Optional


def get_system_info() -> dict:
    """
    Retorna informações do sistema operacional.

    Returns:
        Dicionário com informações do sistema
    """
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
    }


def print_system_info() -> None:
    """Imprime informações do sistema."""
    info = get_system_info()

    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)

    for key, value in info.items():
        print(f"{key:20s}: {value}")

    print("=" * 80)


def is_nvidia_gpu_available() -> bool:
    """
    Verifica se há GPU NVIDIA disponível via nvidia-smi.

    Returns:
        True se nvidia-smi está disponível e detecta GPUs
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True,
            text=True,
            check=True
        )
        return 'GPU' in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_nvidia_driver_version() -> Optional[str]:
    """
    Obtém versão do driver NVIDIA.

    Returns:
        String com versão do driver ou None
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None