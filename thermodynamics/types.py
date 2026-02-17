"""Tipos, dataclasses e exceções para termodinâmica."""

from dataclasses import dataclass
from typing import Literal
from enum import Enum

# Type alias para fluidos
FluidType = Literal[
    "Water", "Air", "CO2", "R134a", "Ammonia",
    "Methane", "Nitrogen", "Oxygen", "Helium"
]


class TemperatureUnit(Enum):
    """Unidades de temperatura."""
    KELVIN = "K"
    CELSIUS = "C"
    FAHRENHEIT = "F"


class PressureUnit(Enum):
    """Unidades de pressão."""
    PASCAL = "Pa"
    BAR = "bar"
    ATM = "atm"
    PSI = "psi"


@dataclass
class ThermalProperties:
    """
    Propriedades termodinâmicas de um fluido em condições específicas.

    Todas as propriedades em unidades SI.
    """
    # Condições de estado
    temperature: float  # K
    pressure: float  # Pa
    fluid: str

    # Propriedades do fluido
    density: float  # kg/m³ (ρ)
    dynamic_viscosity: float  # Pa·s (μ)
    specific_heat: float  # J/(kg·K) (cp)
    thermal_conductivity: float  # W/(m·K) (k)

    # Propriedades calculadas
    kinematic_viscosity: float  # m²/s (ν)
    thermal_diffusivity: float  # m²/s (α)

    @property
    def prandtl_number(self) -> float:
        """Número de Prandtl (Pr = ν/α)."""
        return self.kinematic_viscosity / self.thermal_diffusivity

    def __str__(self) -> str:
        """Representação legível das propriedades."""
        return (
            f"Propriedades Termodinâmicas - {self.fluid}\n"
            f"{'=' * 50}\n"
            f"Condições:\n"
            f"  T = {self.temperature:.2f} K ({self.temperature - 273.15:.2f} °C)\n"
            f"  P = {self.pressure / 1e6:.2f} MPa ({self.pressure / 1e5:.2f} bar)\n"
            f"\nPropriedades:\n"
            f"  ρ (densidade)            = {self.density:.3f} kg/m³\n"
            f"  μ (viscosidade dinâmica) = {self.dynamic_viscosity:.6e} Pa·s\n"
            f"  cp (calor específico)    = {self.specific_heat:.2f} J/(kg·K)\n"
            f"  k (condutividade)        = {self.thermal_conductivity:.4f} W/(m·K)\n"
            f"  ν (viscosidade cinemática) = {self.kinematic_viscosity:.6e} m²/s\n"
            f"  α (difusividade térmica)   = {self.thermal_diffusivity:.6e} m²/s\n"
            f"  Pr (número de Prandtl)     = {self.prandtl_number:.4f}\n"
        )


class ThermodynamicPropertiesError(Exception):
    """Exceção para erros no cálculo de propriedades termodinâmicas."""
    pass


class FluidNotFoundError(ThermodynamicPropertiesError):
    """Exceção quando fluido não é encontrado."""
    pass


class InvalidConditionsError(ThermodynamicPropertiesError):
    """Exceção quando condições (T, P) são inválidas."""
    pass