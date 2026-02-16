"""Tipos, dataclasses e exceções para hardware."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class DeviceType(Enum):
    """Tipos de dispositivos de computação."""
    CPU = "CPU"
    GPU = "GPU"
    TPU = "TPU"


class GPUVendor(Enum):
    """Fabricantes de GPU."""
    NVIDIA = "NVIDIA"
    AMD = "AMD"
    INTEL = "Intel"
    APPLE = "Apple"
    UNKNOWN = "Unknown"


@dataclass
class GPUInfo:
    """Informações detalhadas sobre uma GPU."""
    index: int
    name: str
    device_name: str
    vendor: GPUVendor = GPUVendor.UNKNOWN
    memory_total: Optional[int] = None  # bytes
    memory_available: Optional[int] = None  # bytes
    memory_used: Optional[int] = None  # bytes
    compute_capability: Optional[str] = None
    is_functional: bool = False

    @property
    def memory_total_gb(self) -> Optional[float]:
        """Memória total em GB."""
        return self.memory_total / (1024**3) if self.memory_total else None

    @property
    def memory_available_gb(self) -> Optional[float]:
        """Memória disponível em GB."""
        return self.memory_available / (1024**3) if self.memory_available else None

    @property
    def memory_usage_percent(self) -> Optional[float]:
        """Percentual de uso de memória."""
        if self.memory_total and self.memory_used:
            return (self.memory_used / self.memory_total) * 100
        return None

    def __str__(self) -> str:
        """Representação legível da GPU."""
        lines = [
            f"GPU {self.index}: {self.name}",
            f"  Vendor: {self.vendor.value}",
            f"  Device: {self.device_name}",
        ]

        if self.memory_total_gb:
            lines.append(f"  Memory: {self.memory_total_gb:.2f} GB total")

        if self.memory_available_gb:
            lines.append(f"  Available: {self.memory_available_gb:.2f} GB")

        if self.memory_usage_percent:
            lines.append(f"  Usage: {self.memory_usage_percent:.1f}%")

        if self.compute_capability:
            lines.append(f"  Compute: {self.compute_capability}")

        status = "✓ Functional" if self.is_functional else "⚠ Not tested"
        lines.append(f"  Status: {status}")

        return "\n".join(lines)


@dataclass
class CPUInfo:
    """Informações sobre CPU."""
    physical_cores: int
    logical_cores: int
    frequency_mhz: Optional[float] = None
    usage_percent: Optional[float] = None

    def __str__(self) -> str:
        """Representação legível da CPU."""
        lines = [
            f"CPU Information:",
            f"  Physical cores: {self.physical_cores}",
            f"  Logical cores: {self.logical_cores}",
        ]

        if self.frequency_mhz:
            lines.append(f"  Frequency: {self.frequency_mhz:.0f} MHz")

        if self.usage_percent is not None:
            lines.append(f"  Usage: {self.usage_percent:.1f}%")

        return "\n".join(lines)


@dataclass
class MemoryInfo:
    """Informações sobre memória RAM."""
    total: int  # bytes
    available: int  # bytes
    used: int  # bytes
    percent: float

    @property
    def total_gb(self) -> float:
        """Memória total em GB."""
        return self.total / (1024**3)

    @property
    def available_gb(self) -> float:
        """Memória disponível em GB."""
        return self.available / (1024**3)

    @property
    def used_gb(self) -> float:
        """Memória usada em GB."""
        return self.used / (1024**3)

    def __str__(self) -> str:
        """Representação legível da memória."""
        return (
            f"RAM: {self.used_gb:.1f}/{self.total_gb:.1f} GB "
            f"({self.percent:.1f}% used)"
        )


@dataclass
class HardwareInfo:
    """Informações completas sobre hardware disponível."""
    # GPU
    gpu_devices: List[GPUInfo] = field(default_factory=list)
    gpu_available: bool = False
    selected_gpu: Optional[GPUInfo] = None

    # CPU
    cpu_info: Optional[CPUInfo] = None

    # Memória
    memory_info: Optional[MemoryInfo] = None

    # TensorFlow
    tensorflow_version: str = ""
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None

    # Status
    error_message: Optional[str] = None

    @property
    def has_functional_gpu(self) -> bool:
        """Verifica se há GPU funcional."""
        return any(gpu.is_functional for gpu in self.gpu_devices)

    @property
    def gpu_count(self) -> int:
        """Número de GPUs detectadas."""
        return len(self.gpu_devices)

    def __str__(self) -> str:
        """Representação legível do hardware."""
        lines = [
            "=" * 80,
            "HARDWARE CONFIGURATION",
            "=" * 80,
            f"TensorFlow: {self.tensorflow_version}",
        ]

        if self.cuda_version:
            lines.append(f"CUDA: {self.cuda_version}")

        if self.cudnn_version:
            lines.append(f"cuDNN: {self.cudnn_version}")

        lines.append("")

        # GPU Info
        if self.gpu_available:
            lines.append(f"GPUs Detected: {self.gpu_count}")
            for gpu in self.gpu_devices:
                lines.append("")
                lines.append(str(gpu))

            if self.selected_gpu:
                lines.append(f"\n✓ Selected GPU: {self.selected_gpu.index}")
        else:
            lines.append("⚠ No GPU detected")

        lines.append("")

        # CPU Info
        if self.cpu_info:
            lines.append(str(self.cpu_info))

        lines.append("")

        # Memory Info
        if self.memory_info:
            lines.append(str(self.memory_info))

        if self.error_message:
            lines.append(f"\n⚠ Error: {self.error_message}")

        lines.append("=" * 80)

        return "\n".join(lines)


# ============================================================================
# EXCEÇÕES
# ============================================================================

class HardwareError(Exception):
    """Exceção base para erros de hardware."""
    pass


class GPUConfigurationError(HardwareError):
    """Exceção para erros de configuração de GPU."""
    pass


class HardwareNotFoundError(HardwareError):
    """Exceção quando hardware esperado não é encontrado."""
    pass


class TensorFlowNotAvailableError(HardwareError):
    """Exceção quando TensorFlow não está disponível."""
    pass