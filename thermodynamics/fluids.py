"""Banco de dados e constantes de fluidos."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class FluidInfo:
    """Informações sobre um fluido."""
    name: str
    formula: str
    description: str
    common_use: str
    critical_temp: float  # K
    critical_pressure: float  # Pa


class CommonFluids:
    """Fluidos comumente usados em aplicações de engenharia."""

    WATER = "Water"
    AIR = "Air"
    CO2 = "CO2"
    R134A = "R134a"
    AMMONIA = "Ammonia"
    METHANE = "Methane"
    NITROGEN = "Nitrogen"
    OXYGEN = "Oxygen"


class FluidDatabase:
    """Banco de dados com informações de fluidos."""

    FLUIDS: Dict[str, FluidInfo] = {
        "Water": FluidInfo(
            name="Water",
            formula="H₂O",
            description="Água",
            common_use="HVAC, processos químicos, refrigeração",
            critical_temp=647.096,
            critical_pressure=22.064e6,
        ),
        "Air": FluidInfo(
            name="Air",
            formula="N₂ + O₂ + ...",
            description="Ar atmosférico",
            common_use="Ventilação, pneumática, combustão",
            critical_temp=132.5,
            critical_pressure=3.77e6,
        ),
        "CO2": FluidInfo(
            name="CO2",
            formula="CO₂",
            description="Dióxido de carbono",
            common_use="Refrigeração transcrítica, processos",
            critical_temp=304.13,
            critical_pressure=7.377e6,
        ),
        "R134a": FluidInfo(
            name="R134a",
            formula="C₂H₂F₄",
            description="Refrigerante HFC-134a",
            common_use="Ar condicionado, refrigeração",
            critical_temp=374.21,
            critical_pressure=4.059e6,
        ),
    }

    @classmethod
    def get_info(cls, fluid: str) -> FluidInfo:
        """Retorna informações sobre um fluido."""
        return cls.FLUIDS.get(fluid)

    @classmethod
    def list_fluids(cls) -> None:
        """Lista todos os fluidos disponíveis."""
        print("Fluidos Disponíveis:")
        print("=" * 80)
        for fluid, info in cls.FLUIDS.items():
            print(f"{fluid:15s} ({info.formula:10s}) - {info.description}")
            print(f"  Uso: {info.common_use}")
            print()
