"""
Materials risk assessment for hydrogen blending in steel pipelines.

Thresholds based on:
- ASME B31.12 (Hydrogen Piping and Pipelines)
- IGEM TD/13 (Hydrogen in Gas Networks - UK)
- API 5L / ISO 3183 (Line pipe steel grades)
- EPRG / PRCI hydrogen embrittlement guidelines
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class RiskTier(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class MaterialAlert:
    tier: RiskTier
    title: str
    message: str
    threshold_pct: float


# Embrittlement risk tiers (volumetric H2 %)
EMBRITTLEMENT_TIERS = [
    (0, 10, RiskTier.LOW,
     "Low Risk",
     "Within safe operational envelope for API 5L Gr. B/X42-X70. "
     "Standard monitoring per ASME B31.8 / IGEM TD/1."),
    (10, 25, RiskTier.MODERATE,
     "Moderate Risk",
     "Increased fatigue crack growth rates (da/dN) per ASME B31.12 Fig. 5.2. "
     "Enhanced ILI intervals, pressure cycling monitoring, fracture mechanics assessment recommended."),
    (25, 100, RiskTier.HIGH,
     "High Risk",
     "Significant embrittlement risk. Requires: "
     "derating per B31.12 Para. 5.2, mandatory fracture toughness testing (K_IC/J_IC), "
     "potential pipeline looping or dedicated H2 infrastructure."),
]

# Velocity limits (m/s)
VELOCITY_LIMITS = [
    (0, 15, RiskTier.LOW, "Nominal", "Within erosional limits for carbon steel."),
    (15, 20, RiskTier.MODERATE, "Elevated", "Monitor for erosion at bends, tees, valve trims."),
    (20, 100, RiskTier.HIGH, "Erosional Limit Exceeded", "Compressor re-sizing, trim upgrades, or flow restriction required."),
]

# Pressure drop limits (fraction of inlet pressure)
DP_LIMITS = [
    (0, 0.15, RiskTier.LOW, "Acceptable", "Within typical compressor station capacity margins."),
    (0.15, 0.30, RiskTier.MODERATE, "Elevated", "Verify compressor capacity; consider intermediate compression."),
    (0.30, 1.0, RiskTier.HIGH, "Exceedance", "Intermediate compression or pipeline looping required."),
]


def embrittlement_tier(h2_pct: float) -> Tuple[RiskTier, str, str]:
    """Return (tier, title, message) for given H2 blend percentage."""
    for low, high, tier, title, msg in EMBRITTLEMENT_TIERS:
        if low <= h2_pct < high:
            return tier, title, msg
    return RiskTier.HIGH, "High Risk", "Above validated range."


def velocity_tier(velocity: float) -> Tuple[RiskTier, str, str]:
    """Return (tier, title, message) for given gas velocity."""
    for low, high, tier, title, msg in VELOCITY_LIMITS:
        if low <= velocity < high:
            return tier, title, msg
    return RiskTier.HIGH, "Erosional Limit Exceeded", "Velocity exceeds 20 m/s."


def pressure_drop_tier(dp_frac: float) -> Tuple[RiskTier, str, str]:
    """Return (tier, title, message) for ΔP/P_inlet fraction."""
    for low, high, tier, title, msg in DP_LIMITS:
        if low <= dp_frac < high:
            return tier, title, msg
    return RiskTier.HIGH, "Exceedance", "ΔP > 30% of inlet pressure."


def assess_all(h2_pct: float, velocity: float, dp_bar: float, p_inlet_bar: float) -> List[MaterialAlert]:
    """Generate all material/hydraulic alerts for a scenario."""
    alerts = []

    tier, title, msg = embrittlement_tier(h2_pct)
    alerts.append(MaterialAlert(tier, f"Embrittlement: {title}", msg, h2_pct))

    tier, title, msg = velocity_tier(velocity)
    alerts.append(MaterialAlert(tier, f"Velocity: {title}", msg, velocity))

    dp_frac = dp_bar / p_inlet_bar if p_inlet_bar > 0 else 0
    tier, title, msg = pressure_drop_tier(dp_frac)
    alerts.append(MaterialAlert(tier, f"Pressure Drop: {title}", msg, dp_frac * 100))

    return alerts


def risk_color(tier: RiskTier) -> str:
    """Hex color for risk tier visualization."""
    return {
        RiskTier.LOW: "#27AE60",
        RiskTier.MODERATE: "#F39C12",
        RiskTier.HIGH: "#E74C3C",
    }[tier]