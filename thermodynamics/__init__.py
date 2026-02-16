"""
Biblioteca para cálculo de propriedades termodinâmicas e números adimensionais.

Módulos:
    properties: Cálculo de propriedades usando CoolProp
    dimensionless: Números adimensionais (Re, Pr, Nu, etc)
    fluids: Constantes e dados de fluidos
    conversions: Conversões de unidades
"""

from .properties import (
    FluidPropertiesCalculator,
    ThermalProperties,
    get_water_properties,
)
from .types import (
    ThermodynamicPropertiesError,
    FluidType,
)
from .dimensionless import (
    reynolds_number,
    prandtl_number,
    nusselt_number,
    grashof_number,
)
from .fluids import CommonFluids, FluidDatabase

__version__ = "0.1.0"

__all__ = [
    # Classes principais
    'FluidPropertiesCalculator',
    'ThermalProperties',
    'CommonFluids',
    'FluidDatabase',

    # Exceções
    'ThermodynamicPropertiesError',

    # Tipos
    'FluidType',

    # Funções de conveniência
    'get_water_properties',

    # Números adimensionais
    'reynolds_number',
    'prandtl_number',
    'nusselt_number',
    'grashof_number',
]
