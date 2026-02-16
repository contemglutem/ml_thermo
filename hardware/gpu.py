"""Funções específicas para operações com GPU."""

import logging
from typing import List, Optional, Tuple
from .types import GPUInfo, GPUVendor, GPUConfigurationError

logger = logging.getLogger(__name__)


def check_gpu_availability() -> bool:
    """
    Verifica se há GPUs disponíveis.

    Returns:
        True se há GPUs, False caso contrário
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus) > 0
    except ImportError:
        logger.warning("TensorFlow não está instalado")
        return False
    except Exception as e:
        logger.error(f"Erro ao verificar GPUs: {e}")
        return False


def get_gpu_devices() -> List[GPUInfo]:
    """
    Lista todas as GPUs disponíveis.

    Returns:
        Lista de GPUInfo com informações de cada GPU
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        gpu_infos = []

        for idx, gpu in enumerate(gpus):
            vendor = _detect_gpu_vendor(gpu.name)

            gpu_info = GPUInfo(
                index=idx,
                name=_extract_gpu_name(gpu.name),
                device_name=gpu.name,
                vendor=vendor,
            )

            gpu_infos.append(gpu_info)

        return gpu_infos

    except ImportError:
        logger.warning("TensorFlow não está instalado")
        return []
    except Exception as e:
        logger.error(f"Erro ao listar GPUs: {e}")
        return []


def _detect_gpu_vendor(device_name: str) -> GPUVendor:
    """Detecta fabricante da GPU pelo nome do dispositivo."""
    device_lower = device_name.lower()

    if 'nvidia' in device_lower or 'geforce' in device_lower or 'tesla' in device_lower:
        return GPUVendor.NVIDIA
    elif 'amd' in device_lower or 'radeon' in device_lower:
        return GPUVendor.AMD
    elif 'intel' in device_lower:
        return GPUVendor.INTEL
    elif 'apple' in device_lower or 'metal' in device_lower:
        return GPUVendor.APPLE
    else:
        return GPUVendor.UNKNOWN


def _extract_gpu_name(device_name: str) -> str:
    """Extrai nome limpo da GPU."""
    # Remove prefixos como "/physical_device:GPU:0"
    parts = device_name.split(':')
    return parts[-1] if parts else device_name


def set_gpu_memory_growth(enable: bool = True, gpu_index: Optional[int] = None) -> None:
    """
    Configura crescimento dinâmico de memória GPU.

    Args:
        enable: Se True, habilita memory growth
        gpu_index: Índice da GPU (None = todas)

    Raises:
        GPUConfigurationError: Se falhar ao configurar
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            logger.warning("Nenhuma GPU detectada para configurar memory growth")
            return

        if gpu_index is not None:
            if gpu_index >= len(gpus):
                raise GPUConfigurationError(
                    f"GPU index {gpu_index} inválido. Disponíveis: {len(gpus)}"
                )
            gpus_to_configure = [gpus[gpu_index]]
        else:
            gpus_to_configure = gpus

        for gpu in gpus_to_configure:
            tf.config.experimental.set_memory_growth(gpu, enable)

        status = "habilitado" if enable else "desabilitado"
        logger.info(f"Memory growth {status} para {len(gpus_to_configure)} GPU(s)")

    except RuntimeError as e:
        raise GPUConfigurationError(
            f"Erro ao configurar memory growth: {e}. "
            "Certifique-se de chamar antes de usar TensorFlow."
        ) from e


def select_gpu(gpu_index: int = 0) -> None:
    """
    Seleciona uma GPU específica para uso.

    Args:
        gpu_index: Índice da GPU a ser selecionada

    Raises:
        GPUConfigurationError: Se a GPU não existe ou falha ao selecionar
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            raise GPUConfigurationError("Nenhuma GPU detectada")

        if gpu_index >= len(gpus):
            raise GPUConfigurationError(
                f"GPU index {gpu_index} inválido. Disponíveis: {len(gpus)}"
            )

        selected_gpu = gpus[gpu_index]
        tf.config.set_visible_devices(selected_gpu, 'GPU')

        logger.info(f"GPU {gpu_index} selecionada: {selected_gpu.name}")

    except RuntimeError as e:
        raise GPUConfigurationError(f"Erro ao selecionar GPU: {e}") from e


def test_gpu_functionality(gpu_index: int = 0, matrix_size: int = 1000) -> bool:
    """
    Testa se uma GPU está funcional executando operação matricial.

    Args:
        gpu_index: Índice da GPU a testar
        matrix_size: Tamanho da matriz de teste

    Returns:
        True se GPU está funcional, False caso contrário
    """
    try:
        import tensorflow as tf

        with tf.device(f'/GPU:{gpu_index}'):
            test_matrix = tf.random.normal([matrix_size, matrix_size])
            result = tf.matmul(test_matrix, test_matrix)
            # Force execution
            _ = result.numpy()

        logger.info(f"GPU {gpu_index} testada com sucesso")
        return True

    except (RuntimeError, tf.errors.InvalidArgumentError) as e:
        logger.error(f"Falha no teste de GPU {gpu_index}: {e}")
        return False


def get_gpu_memory_info(gpu_index: int = 0) -> Optional[Tuple[int, int]]:
    """
    Obtém informações de memória da GPU (se disponível).

    Args:
        gpu_index: Índice da GPU

    Returns:
        Tupla (total_bytes, available_bytes) ou None se não disponível
    """
    try:
        # Tentar obter info via nvidia-smi
        import subprocess

        cmd = [
            'nvidia-smi',
            f'--id={gpu_index}',
            '--query-gpu=memory.total,memory.free',
            '--format=csv,noheader,nounits'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        total, free = map(int, result.stdout.strip().split(','))

        # Converter MB para bytes
        total_bytes = total * 1024 * 1024
        free_bytes = free * 1024 * 1024

        return (total_bytes, free_bytes)

    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.debug(f"Não foi possível obter info de memória GPU: {e}")
        return None


def get_cuda_version() -> Optional[str]:
    """
    Obtém versão do CUDA instalada.

    Returns:
        String com versão do CUDA ou None
    """
    try:
        import tensorflow as tf
        # TensorFlow build info
        build_info = tf.sysconfig.get_build_info()
        return build_info.get('cuda_version', None)
    except (ImportError, AttributeError):
        return None


def get_cudnn_version() -> Optional[str]:
    """
    Obtém versão do cuDNN instalada.

    Returns:
        String com versão do cuDNN ou None
    """
    try:
        import tensorflow as tf
        build_info = tf.sysconfig.get_build_info()
        return build_info.get('cudnn_version', None)
    except (ImportError, AttributeError):
        return None
