"""
Core thermodynamics module for H2/CH4 blend modeling.

Equations:
- Peng-Robinson EOS for compressibility factor (Z)
- Wilke mixing rule for dynamic viscosity
- Ideal gas law with real-gas correction for density
- Mass/volumetric heating value blending
"""

from dataclasses import dataclass
import numpy as np

R_UNIVERSAL = 8.314462618  # J/(mol·K)


@dataclass(frozen=True)
class GasProperties:
    """Critical properties and reference values for a pure gas component."""
    name: str
    molecular_weight: float        # g/mol
    critical_temp: float           # K
    critical_press: float          # Pa
    acentric_factor: float         # -
    heating_value_mj_kg: float     # MJ/kg (LHV)
    heating_value_mj_m3_stp: float # MJ/m³ at STP (1.01325 bar, 273.15 K)
    viscosity_ref: float           # Pa·s at temp_ref
    temp_ref: float                # K


# Pure component properties (source: NIST, GPSA, ISO 6976)
CH4 = GasProperties(
    name="Methane",
    molecular_weight=16.043,
    critical_temp=190.56,
    critical_press=45.99e5,
    acentric_factor=0.011,
    heating_value_mj_kg=55.5,
    heating_value_mj_m3_stp=35.8,
    viscosity_ref=1.12e-5,
    temp_ref=273.15
)

H2 = GasProperties(
    name="Hydrogen",
    molecular_weight=2.016,
    critical_temp=33.19,
    critical_press=12.97e5,
    acentric_factor=-0.216,
    heating_value_mj_kg=141.8,
    heating_value_mj_m3_stp=12.7,
    viscosity_ref=8.92e-6,
    temp_ref=273.15
)


def peng_robinson_z(T: float, P: float, gas: GasProperties) -> float:
    """
    Peng-Robinson cubic EOS compressibility factor.

    Returns the largest real root (vapor phase).
    """
    Tr = T / gas.critical_temp
    Pr = P / gas.critical_press
    kappa = 0.37464 + 1.54226 * gas.acentric_factor - 0.26992 * gas.acentric_factor**2
    alpha = (1 + kappa * (1 - Tr**0.5))**2
    a = 0.45724 * (R_UNIVERSAL**2 * gas.critical_temp**2) / gas.critical_press * alpha
    b = 0.07780 * R_UNIVERSAL * gas.critical_temp / gas.critical_press
    A = a * P / (R_UNIVERSAL**2 * T**2)
    B = b * P / (R_UNIVERSAL * T)
    coeffs = [1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    return float(max(real_roots)) if len(real_roots) > 0 else 1.0


def viscosity_pure(T: float, gas: GasProperties) -> float:
    """Temperature-dependent viscosity via power law (valid 200-400 K)."""
    return gas.viscosity_ref * (T / gas.temp_ref)**(0.68 if gas.name == "Hydrogen" else 0.92)


def wilke_viscosity(y_h2: float, T: float) -> float:
    """
    Wilke mixing rule for binary mixture viscosity.

    μ_mix = Σ (y_i μ_i) / Σ (y_j φ_ij)
    φ_ij = (1 + (μ_i/μ_j)^0.5 (M_j/M_i)^0.25)^2 / (8(1 + M_i/M_j))^0.5
    """
    y_ch4 = 1 - y_h2
    mu_h2 = viscosity_pure(T, H2)
    mu_ch4 = viscosity_pure(T, CH4)

    phi_h2_ch4 = (1 + (mu_h2/mu_ch4)**0.5 * (CH4.molecular_weight/H2.molecular_weight)**0.25)**2 \
                 / (8 * (1 + H2.molecular_weight/CH4.molecular_weight))**0.5
    phi_ch4_h2 = (1 + (mu_ch4/mu_h2)**0.5 * (H2.molecular_weight/CH4.molecular_weight)**0.25)**2 \
                 / (8 * (1 + CH4.molecular_weight/H2.molecular_weight))**0.5

    return (y_h2 * mu_h2) / (y_h2 + y_ch4 * phi_h2_ch4) + \
           (y_ch4 * mu_ch4) / (y_ch4 + y_h2 * phi_ch4_h2)


def blend_properties(h2_frac: float, T: float, P: float) -> dict:
    """
    Compute all blend thermodynamic properties at given T, P.

    Returns dict with:
    - M_blend: molar mass (g/mol)
    - Z_blend: compressibility factor
    - rho_blend: density (kg/m³)
    - mu_blend: dynamic viscosity (Pa·s)
    - hv_mass_blend: mass heating value (MJ/kg)
    - hv_vol_blend: volumetric heating value at STP (MJ/m³)
    """
    y_h2 = h2_frac
    y_ch4 = 1 - h2_frac

    M_blend = y_h2 * H2.molecular_weight + y_ch4 * CH4.molecular_weight

    Z_h2 = peng_robinson_z(T, P, H2)
    Z_ch4 = peng_robinson_z(T, P, CH4)
    Z_blend = y_h2 * Z_h2 + y_ch4 * Z_ch4

    rho_blend = (P * M_blend) / (Z_blend * R_UNIVERSAL * T) / 1000.0  # kg/m³

    mu_blend = wilke_viscosity(y_h2, T)

    hv_mass_blend = y_h2 * H2.heating_value_mj_kg + y_ch4 * CH4.heating_value_mj_kg
    hv_vol_blend = y_h2 * H2.heating_value_mj_m3_stp + y_ch4 * CH4.heating_value_mj_m3_stp

    return {
        "M_blend": M_blend,
        "Z_blend": Z_blend,
        "rho_blend": rho_blend,
        "mu_blend": mu_blend,
        "hv_mass_blend": hv_mass_blend,
        "hv_vol_blend": hv_vol_blend,
        "Z_h2": Z_h2,
        "Z_ch4": Z_ch4,
        "mu_h2": viscosity_pure(T, H2),
        "mu_ch4": viscosity_pure(T, CH4),
    }