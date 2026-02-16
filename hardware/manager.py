"""Gerenciador principal de hardware."""

import logging
from typing import Optional
import psutil

try:
    import tensorflow as tf
except ImportError:
    tf = None

from .types import (
    HardwareInfo,
    GPUInfo,
    CPUInfo,
    MemoryInfo,
    GPUConfigurationError,
    TensorFlowNotAvailableError,
)
from .gpu import (
    get_gpu_devices,
    set_gpu_memory_growth,
    select_gpu,
    test_gpu_functionality,
    get_gpu_memory_info,
    get_cuda_version,
    get_cudnn_version,
)

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Gerencia configuração e verificação de hardware para TensorFlow.

    Examples:
        >>> from hardware import HardwareManager
        >>> manager = HardwareManager(enable_memory_growth=True)
        >>> info = manager.setup_and_verify()
        >>> print(info)
    """

    def __init__(
            self,
            enable_memory_growth: bool = True,
            gpu_index: int = 0,
            verbose: bool = True,
            test_gpu: bool = True,
    ):
        """
        Inicializa gerenciador de hardware.

        Args:
            enable_memory_growth: Habilita crescimento dinâmico de memória GPU
            gpu_index: Índice da GPU a ser utilizada (default: 0)
            verbose: Se True, exibe informações no console
            test_gpu: Se True, testa funcionalidade da GPU
        """
        if tf is None:
            raise TensorFlowNotAvailableError(
                "TensorFlow não está instalado. "
                "Instale com: pip install tensorflow"
            )

        self.enable_memory_growth = enable_memory_growth
        self.gpu_index = gpu_index
        self.verbose = verbose
        self.test_gpu = test_gpu

    def get_cpu_info(self) -> CPUInfo:
        """
        Coleta informações sobre CPU.

        Returns:
            CPUInfo com detalhes da CPU
        """
        physical_cores = psutil.cpu_count(logical=False) or 0
        logical_cores = psutil.cpu_count(logical=True) or 0

        try:
            cpu_freq = psutil.cpu_freq()
            frequency = cpu_freq.current if cpu_freq else None
        except Exception:
            frequency = None

        try:
            usage = psutil.cpu_percent(interval=0.1)
        except Exception:
            usage = None

        return CPUInfo(
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            frequency_mhz=frequency,
            usage_percent=usage,
        )

    def get_memory_info(self) -> MemoryInfo:
        """
        Coleta informações sobre memória RAM.

        Returns:
            MemoryInfo com detalhes da memória
        """
        mem = psutil.virtual_memory()

        return MemoryInfo(
            total=mem.total,
            available=mem.available,
            used=mem.used,
            percent=mem.percent,
        )

    def get_hardware_info(self) -> HardwareInfo:
        """
        Coleta informações completas sobre hardware.

        Returns:
            HardwareInfo com todos os detalhes
        """
        # GPU
        gpu_devices = get_gpu_devices()

        # Adicionar informações de memória GPU se disponível
        for gpu in gpu_devices:
            mem_info = get_gpu_memory_info(gpu.index)
            if mem_info:
                total, available = mem_info
                gpu.memory_total = total
                gpu.memory_available = available
                gpu.memory_used = total - available

        # CPU e Memória
        cpu_info = self.get_cpu_info()
        memory_info = self.get_memory_info()

        # TensorFlow versions
        tf_version = tf.__version__
        cuda_version = get_cuda_version()
        cudnn_version = get_cudnn_version()

        return HardwareInfo(
            gpu_devices=gpu_devices,
            gpu_available=len(gpu_devices) > 0,
            cpu_info=cpu_info,
            memory_info=memory_info,
            tensorflow_version=tf_version,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
        )

    def configure_gpu(self) -> Optional[GPUInfo]:
        """
        Configura GPU com as opções especificadas.

        Returns:
            GPUInfo da GPU selecionada ou None se não há GPU

        Raises:
            GPUConfigurationError: Se falhar ao configurar GPU
        """
        gpu_devices = get_gpu_devices()

        if not gpu_devices:
            logger.warning("Nenhuma GPU detectada para configuração")
            return None

        if self.gpu_index >= len(gpu_devices):
            raise GPUConfigurationError(
                f"GPU index {self.gpu_index} inválido. "
                f"Disponíveis: {len(gpu_devices)}"
            )

        # Configurar memory growth
        if self.enable_memory_growth:
            set_gpu_memory_growth(True, self.gpu_index)

        # Selecionar GPU
        select_gpu(self.gpu_index)

        selected_gpu = gpu_devices[self.gpu_index]

        # Testar GPU se solicitado
        if self.test_gpu:
            selected_gpu.is_functional = test_gpu_functionality(self.gpu_index)

        return selected_gpu

    def setup_and_verify(self) -> HardwareInfo:
        """
        Pipeline completo: configura GPU e verifica hardware.

        Returns:
            HardwareInfo com status completo do hardware
        """
        info = self.get_hardware_info()

        if info.gpu_available:
            try:
                selected_gpu = self.configure_gpu()
                info.selected_gpu = selected_gpu
            except GPUConfigurationError as e:
                info.error_message = str(e)
                logger.error(f"Erro na configuração: {e}")

        if self.verbose:
            print(info)

        return info


# ============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ============================================================================

def check_hardware(verbose: bool = True) -> HardwareInfo:
    """
    Função de conveniência para verificação completa de hardware.

    Args:
        verbose: Se True, exibe informações no console

    Returns:
        HardwareInfo com status do hardware

    Examples:
        >>> from hardware import check_hardware
        >>> info = check_hardware()
    """
    manager = HardwareManager(verbose=verbose)
    return manager.setup_and_verify()


def check_gpu(verbose: bool = True) -> HardwareInfo:
    """
    Função de conveniência para verificação de GPU (retrocompatibilidade).

    Args:
        verbose: Se True, exibe informações no console

    Returns:
        HardwareInfo com status do hardware
    """
    return check_hardware(verbose=verbose)
