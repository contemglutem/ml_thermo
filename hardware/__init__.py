"""
Gerenciamento de hardware e configuração TensorFlow.

Módulos:
    manager: Classe principal HardwareManager
    gpu: Funções específicas de GPU
    monitoring: Monitoramento de recursos em tempo real
    environment: Configuração de ambiente TensorFlow
    types: Tipos, dataclasses e exceções
    utils: Utilidades auxiliares
"""

from .manager import HardwareManager
from .types import (
    HardwareInfo,
    GPUInfo,
    CPUInfo,
    MemoryInfo,
    DeviceType,
    GPUVendor,
    GPUConfigurationError,
    HardwareNotFoundError,
    HardwareError,
    TensorFlowNotAvailableError,
)
from .environment import (
    setup_tensorflow_environment,
    configure_tensorflow_logging,
    get_environment_info,
    print_environment_info,
)
from .gpu import (
    check_gpu_availability,
    get_gpu_devices,
    get_gpu_memory_info,
    set_gpu_memory_growth,
    select_gpu,
    test_gpu_functionality,
    get_cuda_version,
    get_cudnn_version,
)
from .monitoring import ResourceMonitor, ResourceSnapshot
from .utils import (
    get_system_info,
    print_system_info,
    is_nvidia_gpu_available,
    get_nvidia_driver_version,
)

# Funções de conveniência
from .manager import check_hardware, check_gpu

__version__ = "0.1.0"

__all__ = [
    # Classes principais
    'HardwareManager',
    'ResourceMonitor',

    # Tipos e dataclasses
    'HardwareInfo',
    'GPUInfo',
    'CPUInfo',
    'MemoryInfo',
    'ResourceSnapshot',
    'DeviceType',
    'GPUVendor',

    # Exceções
    'GPUConfigurationError',
    'HardwareNotFoundError',
    'HardwareError',
    'TensorFlowNotAvailableError',

    # Configuração de ambiente
    'setup_tensorflow_environment',
    'configure_tensorflow_logging',
    'get_environment_info',
    'print_environment_info',

    # Funções de GPU
    'check_gpu_availability',
    'get_gpu_devices',
    'get_gpu_memory_info',
    'set_gpu_memory_growth',
    'select_gpu',
    'test_gpu_functionality',
    'get_cuda_version',
    'get_cudnn_version',

    # Monitoramento
    'ResourceMonitor',
    'ResourceSnapshot',

    # Utilidades
    'get_system_info',
    'print_system_info',
    'is_nvidia_gpu_available',
    'get_nvidia_driver_version',

    # Funções de conveniência
    'check_hardware',
    'check_gpu',
]