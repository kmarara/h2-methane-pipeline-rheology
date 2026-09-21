"""
Core hydraulics module for pipeline pressure drop calculations.

Equations:
- Darcy-Weisbach with Colebrook-White friction factor (general purpose)
- Weymouth equation (high-pressure gas pipeline industry standard)
- Volumetric flow scaling for energy throughput equivalence
"""

import numpy as np

# Pipe roughness for carbon steel (m)
EPSILON = 4.57e-5


def friction_factor(Re: float, D: float) -> float:
    """
    Colebrook-White approximation (Swamee-Jain explicit form).
    Valid for 4000 < Re < 1e8, 1e-6 < ε/D < 1e-2.
    """
    if Re < 2300:
        return 64 / Re
    rel_rough = EPSILON / D
    return 0.25 / (np.log10(rel_rough / 3.7 + 5.74 / Re**0.9))**2


def darcy_weisbach_dp(Q: float, L: float, D: float, rho: float, mu: float) -> float:
    """
    Darcy-Weisbach pressure drop for compressible flow (average density approximation).

    ΔP = f (L/D) (ρ v²/2)
    Returns ΔP in bar.
    """
    D_m = D / 1000.0
    L_m = L * 1000.0
    v = Q / (np.pi * D_m**2 / 4)
    Re = rho * v * D_m / mu
    f = friction_factor(Re, D_m)
    dP_pa = f * (L_m / D_m) * (rho * v**2 / 2)
    return dP_pa / 1e5


def weymouth_dp(Q: float, L: float, D: float, rho: float, mu: float,
                P_avg: float, Z: float, T: float) -> float:
    """
    Weymouth equation for high-pressure gas pipelines.

    Q = 433.5 * (Tb/Pb) * (P1^2 - P2^2)^0.5 * D^(8/3) / (G * T * Z * L)^0.5
    Rearranged for ΔP given Q.
    Returns ΔP in bar.
    """
    D_m = D / 1000.0
    L_m = L * 1000.0
    epsilon = EPSILON
    Re = 4 * rho * Q / (np.pi * D_m * mu)
    f = friction_factor(Re, D_m) if Re >= 2300 else 64/Re
    v = Q / (np.pi * D_m**2 / 4)
    dP_pa = f * L_m * rho * v**2 / (2 * D_m)
    return dP_pa / 1e5


def equivalent_flow_rate(Q_ref: float, hv_vol_ref: float, hv_vol_blend: float) -> float:
    """
    Scale volumetric flow to maintain constant energy throughput.

    Q_equiv = Q_ref * (HV_ref / HV_blend)
    Since H2 has ~1/3 volumetric energy density of CH4 at STP.
    """
    return Q_ref * (hv_vol_ref / hv_vol_blend)


def gas_velocity(Q: float, D: float) -> float:
    """Gas velocity in m/s."""
    D_m = D / 1000.0
    return Q / (np.pi * D_m**2 / 4)


def reynolds_number(Q: float, D: float, rho: float, mu: float) -> float:
    """Reynolds number for pipe flow."""
    D_m = D / 1000.0
    v = Q / (np.pi * D_m**2 / 4)
    return rho * v * D_m / mu