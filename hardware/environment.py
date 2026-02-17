"""Configuração de ambiente TensorFlow."""

import os
import warnings
import logging
from typing import Literal

logger = logging.getLogger(__name__)

LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR']


def setup_tensorflow_environment(
        log_level: LogLevel = 'ERROR',
        suppress_protobuf_warnings: bool = True,
        enable_onednn: bool = True,
        suppress_cpp_logs: bool = True,
        enable_xla: bool = False,
        gpu_memory_growth: bool = True,
) -> None:
    """
    Configura ambiente TensorFlow antes da importação.

    IMPORTANTE: Deve ser chamada ANTES de importar TensorFlow.

    Args:
        log_level: Nível de log TF ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        suppress_protobuf_warnings: Suprime warnings de versão do Protobuf
        enable_onednn: Habilita otimizações oneDNN (Intel)
        suppress_cpp_logs: Suprime logs C++ do TensorFlow
        enable_xla: Habilita compilação XLA (experimental)
        gpu_memory_growth: Permite crescimento dinâmico de memória GPU

    Examples:
        >>> from hardware import setup_tensorflow_environment
        >>> setup_tensorflow_environment(log_level='ERROR')
        >>> import tensorflow as tf  # Importar DEPOIS
    """
    # Suprimir logs C++ do TensorFlow
    if suppress_cpp_logs:
        # 0 = ALL, 1 = INFO, 2 = WARNING, 3 = ERROR
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        logger.debug("TensorFlow C++ logs suprimidos")

    # Controlar oneDNN
    if not enable_onednn:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        logger.info("oneDNN desabilitado (melhor reproduzibilidade)")
    else:
        logger.debug("oneDNN habilitado (melhor performance)")

    # Suprimir warnings do Protobuf
    if suppress_protobuf_warnings:
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            module='google.protobuf'
        )
        logger.debug("Warnings do Protobuf suprimidos")

    # Suprimir warnings de deprecação
    os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'

    # XLA (Accelerated Linear Algebra)
    if enable_xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
        logger.info("XLA habilitado (compilação JIT)")

    # GPU memory growth
    if gpu_memory_growth:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        logger.debug("GPU memory growth habilitado")

    logger.info(f"Ambiente TensorFlow configurado (log_level={log_level})")


def configure_tensorflow_logging(level: LogLevel = 'ERROR') -> None:
    """
    Configura logging do TensorFlow após importação.

    Args:
        level: Nível de log ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(level)
        logger.info(f"TensorFlow logging configurado: {level}")
    except ImportError:
        logger.warning("TensorFlow não está instalado")


def get_environment_info() -> dict:
    """
    Retorna informações sobre variáveis de ambiente relevantes.

    Returns:
        Dicionário com variáveis de ambiente TensorFlow
    """
    relevant_vars = [
        'TF_CPP_MIN_LOG_LEVEL',
        'TF_ENABLE_ONEDNN_OPTS',
        'TF_ENABLE_DEPRECATION_WARNINGS',
        'TF_XLA_FLAGS',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'CUDA_VISIBLE_DEVICES',
    ]

    return {var: os.environ.get(var, 'Not set') for var in relevant_vars}


def print_environment_info() -> None:
    """Imprime informações do ambiente TensorFlow."""
    info = get_environment_info()

    print("=" * 80)
    print("TENSORFLOW ENVIRONMENT")
    print("=" * 80)

    for var, value in info.items():
        print(f"{var:35s} = {value}")

    print("=" * 80)