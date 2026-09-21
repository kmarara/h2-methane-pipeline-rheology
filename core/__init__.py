"""
Core physics engine for H2/CH4 pipeline digital twin.

Modules:
- thermodynamics: Peng-Robinson EOS, Wilke viscosity, blend properties
- hydraulics: Darcy-Weisbach, Weymouth, flow scaling
- materials: Embrittlement risk tiers, velocity limits, pressure drop limits
- pipeline: Geographic network definitions, Folium mapping
"""

from .thermodynamics import (
    GasProperties, CH4, H2, R_UNIVERSAL,
    peng_robinson_z, viscosity_pure, wilke_viscosity, blend_properties
)
from .hydraulics import (
    friction_factor, darcy_weisbach_dp, weymouth_dp,
    equivalent_flow_rate, gas_velocity, reynolds_number,
    EPSILON
)
from .materials import (
    RiskTier, MaterialAlert,
    embrittlement_tier, velocity_tier, pressure_drop_tier,
    assess_all, risk_color,
    EMBRITTLEMENT_TIERS, VELOCITY_LIMITS, DP_LIMITS
)
from .pipeline import (
    PipelineSegment, CompressorStation, PipelineNetwork,
    create_ireland_network
)

__all__ = [
    # Thermodynamics
    "GasProperties", "CH4", "H2", "R_UNIVERSAL",
    "peng_robinson_z", "viscosity_pure", "wilke_viscosity", "blend_properties",
    # Hydraulics
    "friction_factor", "darcy_weisbach_dp", "weymouth_dp",
    "equivalent_flow_rate", "gas_velocity", "reynolds_number", "EPSILON",
    # Materials
    "RiskTier", "MaterialAlert",
    "embrittlement_tier", "velocity_tier", "pressure_drop_tier",
    "assess_all", "risk_color",
    "EMBRITTLEMENT_TIERS", "VELOCITY_LIMITS", "DP_LIMITS",
    # Pipeline
    "PipelineSegment", "CompressorStation", "PipelineNetwork",
    "create_ireland_network",
]