"""Calculadora de propriedades termodinâmicas usando CoolProp."""

from typing import Optional
import logging

try:
    import CoolProp.CoolProp as CP
except ImportError as e:
    raise ImportError(
        "CoolProp não está instalado. Instale com: pip install CoolProp"
    ) from e

from .types import (
    ThermalProperties,
    ThermodynamicPropertiesError,
    FluidNotFoundError,
    InvalidConditionsError,
)

logger = logging.getLogger(__name__)


class FluidPropertiesCalculator:
    """
    Calculadora de propriedades termodinâmicas usando CoolProp.

    Examples:
        >>> calc = FluidPropertiesCalculator("Water")
        >>> props = calc.calculate_at_celsius(25, 1)
        >>> print(props.density)
        997.05
    """

    def __init__(
            self,
            fluid: str = "Water",
            default_temperature: float = 300.0,  # K (≈27°C)
            default_pressure: float = 1e5,  # Pa (1 bar)
    ):
        """
        Inicializa calculadora de propriedades.

        Args:
            fluid: Nome do fluido CoolProp
            default_temperature: Temperatura padrão em Kelvin
            default_pressure: Pressão padrão em Pascal

        Raises:
            FluidNotFoundError: Se o fluido não é válido
        """
        self.fluid = fluid
        self.default_temperature = default_temperature
        self.default_pressure = default_pressure

        self._validate_fluid()

    def _validate_fluid(self) -> None:
        """Valida se o fluido existe no CoolProp."""
        try:
            CP.PropsSI('D', 'T', 300, 'P', 1e5, self.fluid)
        except ValueError as e:
            raise FluidNotFoundError(
                f"Fluido '{self.fluid}' não encontrado no CoolProp. "
                f"Veja: http://www.coolprop.org/fluid_properties/PurePseudoPure.html"
            ) from e

    def calculate_properties(
            self,
            temperature: Optional[float] = None,
            pressure: Optional[float] = None,
    ) -> ThermalProperties:
        """
        Calcula todas as propriedades termodinâmicas.

        Args:
            temperature: Temperatura em Kelvin (usa padrão se None)
            pressure: Pressão em Pascal (usa padrão se None)

        Returns:
            ThermalProperties com todas as propriedades

        Raises:
            InvalidConditionsError: Se T ou P são inválidos
        """
        T = temperature or self.default_temperature
        P = pressure or self.default_pressure

        # Validações básicas
        if T <= 0:
            raise InvalidConditionsError(f"Temperatura deve ser > 0K, recebido: {T}K")
        if P <= 0:
            raise InvalidConditionsError(f"Pressão deve ser > 0Pa, recebido: {P}Pa")

        try:
            # Propriedades diretas do CoolProp
            rho = CP.PropsSI('D', 'T', T, 'P', P, self.fluid)
            mu = CP.PropsSI('V', 'T', T, 'P', P, self.fluid)
            cp = CP.PropsSI('Cpmass', 'T', T, 'P', P, self.fluid)
            k = CP.PropsSI('L', 'T', T, 'P', P, self.fluid)

            # Propriedades calculadas
            nu = mu / rho
            alpha = k / (rho * cp)

            return ThermalProperties(
                temperature=T,
                pressure=P,
                fluid=self.fluid,
                density=rho,
                dynamic_viscosity=mu,
                specific_heat=cp,
                thermal_conductivity=k,
                kinematic_viscosity=nu,
                thermal_diffusivity=alpha,
            )

        except Exception as e:
            raise ThermodynamicPropertiesError(
                f"Erro ao calcular propriedades para {self.fluid} "
                f"em T={T}K, P={P}Pa: {e}"
            ) from e

    def calculate_at_celsius(
            self,
            temperature_c: float,
            pressure_bar: float,
    ) -> ThermalProperties:
        """
        Conveniência: calcula propriedades usando °C e bar.

        Args:
            temperature_c: Temperatura em Celsius
            pressure_bar: Pressão em bar

        Returns:
            ThermalProperties calculadas
        """
        T_kelvin = temperature_c + 273.15
        P_pascal = pressure_bar * 1e5

        return self.calculate_properties(T_kelvin, P_pascal)


def get_water_properties(
        temperature_c: float = 25.0,
        pressure_bar: float = 1.0,
) -> ThermalProperties:
    """
    Atalho para propriedades da água.

    Args:
        temperature_c: Temperatura em °C (padrão: 25°C)
        pressure_bar: Pressão em bar (padrão: 1 bar)

    Returns:
        Propriedades da água
    """
    calc = FluidPropertiesCalculator("Water")
    return calc.calculate_at_celsius(temperature_c, pressure_bar)