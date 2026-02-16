"""Conversões de unidades."""


def celsius_to_kelvin(temp_c: float) -> float:
    """Converte Celsius para Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k: float) -> float:
    """Converte Kelvin para Celsius."""
    return temp_k - 273.15


def fahrenheit_to_kelvin(temp_f: float) -> float:
    """Converte Fahrenheit para Kelvin."""
    return (temp_f - 32) * 5/9 + 273.15


def bar_to_pascal(pressure_bar: float) -> float:
    """Converte bar para Pascal."""
    return pressure_bar * 1e5


def psi_to_pascal(pressure_psi: float) -> float:
    """Converte PSI para Pascal."""
    return pressure_psi * 6894.76


def atm_to_pascal(pressure_atm: float) -> float:
    """Converte atm para Pascal."""
    return pressure_atm * 101325