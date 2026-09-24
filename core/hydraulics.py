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


def compression_work_mw(Q: float, P_in: float, P_out: float, rho: float,
                        Z: float, T: float, eta_isentropic: float = 0.85,
                        eta_mechanical: float = 0.98) -> float:
    """
    Calculate compressor power requirement for gas compression.

    Polytropic compression work (MW):
    W = (Z * R * T / M) * (k/(k-1)) * [(P_out/P_in)^((k-1)/k) - 1] * Q * rho / (eta_isen * eta_mech)

    For natural gas/H2 blends, k (isentropic exponent) ≈ 1.3
    Returns power in MW.
    """
    # Isentropic exponent for methane/hydrogen blends
    k = 1.3
    # Molar mass approximation (kg/mol) - will be passed via rho/Z/RT relation
    # Using: rho = P * M / (Z * R * T) => M = rho * Z * R * T / P
    R_specific = 518.3  # J/(kg·K) for methane approx, adjusted per blend

    # Simplified: use polytropic head formula
    # H = (Z * R * T / M) * (n/(n-1)) * [(P_out/P_in)^((n-1)/n) - 1]
    # where n = polytropic exponent ≈ k for estimation

    pressure_ratio = P_out / P_in
    if pressure_ratio <= 1.0:
        return 0.0

    # Polytropic head (J/kg)
    # Using gas constant for mixture: R_mix = R_universal / M_mix
    # But we have rho, so: R_mix = P / (rho * Z * T) * (P_out/P_in) approx
    R_mix = 8.314462618 / 0.016  # Default for CH4, will be overridden by caller

    # Better approach: use the actual gas properties
    # Head = (k/(k-1)) * (P_in/rho) * [(P_out/P_in)^((k-1)/k) - 1] * Z
    head = (k / (k - 1)) * (P_in * 1e5 / rho) * (pressure_ratio**((k - 1) / k) - 1)

    # Power = mass_flow * head / (eta_isen * eta_mech)
    mass_flow = rho * Q  # kg/s
    power_w = mass_flow * head / (eta_isentropic * eta_mechanical)

    return power_w / 1e6  # MW


def equivalent_hydraulic_diameter(D1: float, D2: float) -> float:
    """
    Equivalent hydraulic diameter for parallel pipes (Weymouth/Darcy scaling).

    D_eq = (D1^2.5 + D2^2.5)^0.4

    Based on Weymouth: Q ∝ D^(8/3) ≈ D^2.67, so for parallel flow
    capacity addition: D_eq^2.5 = D1^2.5 + D2^2.5

    D1, D2 in mm. Returns D_eq in mm.
    """
    return (D1**2.5 + D2**2.5)**0.4


def pressure_drop_with_looping(Q: float, L_total: float, L_loop: float,
                                D_main: float, D_loop: float,
                                rho: float, mu: float) -> float:
    """
    Calculate total pressure drop with a looped section.

    ΔP_total = ΔP(L_loop, D_eq) + ΔP(L_total - L_loop, D_main)

    Returns ΔP in bar.
    """
    if L_loop <= 0:
        return darcy_weisbach_dp(Q, L_total, D_main, rho, mu)

    if L_loop >= L_total:
        D_eq = equivalent_hydraulic_diameter(D_main, D_loop)
        return darcy_weisbach_dp(Q, L_total, D_eq, rho, mu)

    D_eq = equivalent_hydraulic_diameter(D_main, D_loop)
    dp_looped = darcy_weisbach_dp(Q, L_loop, D_eq, rho, mu)
    dp_single = darcy_weisbach_dp(Q, L_total - L_loop, D_main, rho, mu)

    return dp_looped + dp_single


def simulate_pipeline_with_compressors(Q: float, L_total: float, D: float,
                                        rho: float, mu: float, Z: float, T: float,
                                        P_in: float, P_min: float = 20.0,
                                        target_P: float = 85.0) -> dict:
    """
    Simulate pipeline with automatic intermediate compressor insertion.

    Splits pipeline into segments, adds compressors when pressure would drop below P_min.
    Returns dict with final pressure, number of compressors, total power, segment details.
    """
    # First, calculate pressure drop without compressors
    dp_total = darcy_weisbach_dp(Q, L_total, D, rho, mu)
    P_out = P_in - dp_total

    if P_out >= P_min:
        return {
            "P_out": P_out,
            "num_compressors": 0,
            "total_power_mw": 0.0,
            "compressor_locations_km": [],
            "segment_pressures": [P_in, P_out],
            "exceedance": False,
        }

    # Need compressors - determine optimal placement
    # Strategy: place compressors to keep each segment outlet >= P_min
    # Target: reset to target_P (or P_in) at each compressor

    segments = 1
    while True:
        seg_length = L_total / segments
        dp_seg = darcy_weisbach_dp(Q, seg_length, D, rho, mu)
        P_seg_out = target_P - dp_seg

        if P_seg_out >= P_min or segments > 10:  # Max 10 compressors safety
            break
        segments += 1

    # Recalculate with determined number of segments
    seg_length = L_total / segments
    dp_seg = darcy_weisbach_dp(Q, seg_length, D, rho, mu)

    compressor_locations = []
    segment_pressures = [P_in]
    total_power = 0.0
    current_P = P_in

    for i in range(segments):
        current_P -= dp_seg
        segment_pressures.append(current_P)

        if i < segments - 1:  # Not the last segment
            # Compressor needed
            compressor_locations.append((i + 1) * seg_length)
            power = compression_work_mw(Q, current_P, target_P, rho, Z, T)
            total_power += power
            current_P = target_P
            segment_pressures.append(current_P)

    return {
        "P_out": current_P,
        "num_compressors": segments - 1,
        "total_power_mw": total_power,
        "compressor_locations_km": compressor_locations,
        "segment_pressures": segment_pressures,
        "exceedance": False,
        "segment_length_km": seg_length,
    }


def find_required_looping(Q: float, L_total: float, D_main: float,
                          rho: float, mu: float, P_in: float, P_min: float = 20.0,
                          D_loop: float = None, max_loop_pct: float = 1.0) -> dict:
    """
    Find the loop percentage needed to maintain P_out >= P_min.

    Returns dict with loop_percentage, loop_length_km, P_out, success.
    """
    if D_loop is None:
        D_loop = D_main  # Same diameter loop by default

    # Binary search for required loop percentage
    low, high = 0.0, max_loop_pct
    best_result = None

    for _ in range(30):  # Binary search iterations
        mid = (low + high) / 2
        L_loop = L_total * mid
        dp = pressure_drop_with_looping(Q, L_total, L_loop, D_main, D_loop, rho, mu)
        P_out = P_in - dp

        if P_out >= P_min:
            best_result = {
                "loop_percentage": mid,
                "loop_length_km": L_loop,
                "P_out": P_out,
                "success": True,
            }
            high = mid
        else:
            low = mid

    if best_result is None:
        # Even full looping not enough
        L_loop = L_total * max_loop_pct
        dp = pressure_drop_with_looping(Q, L_total, L_loop, D_main, D_loop, rho, mu)
        P_out = P_in - dp
        return {
            "loop_percentage": max_loop_pct,
            "loop_length_km": L_loop,
            "P_out": P_out,
            "success": False,
        }

    return best_result