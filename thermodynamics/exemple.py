from thermo import (
    FluidPropertiesCalculator,
    get_water_properties,
    reynolds_number,
    CommonFluids)

# Método 1: Função de conveniência
props = get_water_properties(temperature_c=25, pressure_bar=1)
print(props)

# Método 2: Calculadora customizada
calc = FluidPropertiesCalculator(CommonFluids.AIR)
air_props = calc.calculate_at_celsius(20, 1.013)

# Cálculo de Reynolds
Re = reynolds_number(
    velocity=5.0,
    characteristic_length=0.05,
    props=air_props
)
print(f"Reynolds: {Re:.2f}")