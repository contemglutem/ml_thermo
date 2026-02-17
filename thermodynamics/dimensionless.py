"""Cálculo de números adimensionais."""

from .types import ThermalProperties


def reynolds_number(
        velocity: float,
        characteristic_length: float,
        props: ThermalProperties,
        ) -> float:
    """
    Calcula número de Reynolds (Re = VL/ν).

    Args:
        velocity: Velocidade do fluido [m/s]
        characteristic_length: Comprimento característico [m]
        props: Propriedades termodinâmicas

    Returns:
        Número de Reynolds (adimensional)
    """
    return velocity * characteristic_length / props.kinematic_viscosity


def prandtl_number(props: ThermalProperties) -> float:
    """
    Calcula número de Prandtl (Pr = ν/α).

    Args:
        props: Propriedades termodinâmicas

    Returns:
        Número de Prandtl (adimensional)
    """
    return props.prandtl_number


def nusselt_number(
        heat_transfer_coefficient: float,
        characteristic_length: float,
        props: ThermalProperties,
        ) -> float:
    """
    Calcula número de Nusselt (Nu = hL/k).

    Args:
        heat_transfer_coefficient: Coeficiente de transferência [W/(m²·K)]
        characteristic_length: Comprimento característico [m]
        props: Propriedades termodinâmicas

    Returns:
        Número de Nusselt (adimensional)
    """
    return (heat_transfer_coefficient * characteristic_length /
            props.thermal_conductivity)


def grashof_number(
        gravity: float,
        beta: float,
        delta_temp: float,
        characteristic_length: float,
        props: ThermalProperties,
        ) -> float:
    """
    Calcula número de Grashof (Gr = gβΔTL³/ν²).

    Args:
        gravity: Aceleração da gravidade [m/s²]
        beta: Coeficiente de expansão térmica [1/K]
        delta_temp: Diferença de temperatura [K]
        characteristic_length: Comprimento característico [m]
        props: Propriedades termodinâmicas

    Returns:
        Número de Grashof (adimensional)
    """
    return (gravity * beta * delta_temp * characteristic_length ** 3 /
            props.kinematic_viscosity ** 2)
