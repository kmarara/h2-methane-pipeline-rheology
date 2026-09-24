"""
Core physics engine for H2/CH4 pipeline digital twin.

Modules:
- thermodynamics: Peng-Robinson EOS, Wilke viscosity, blend properties
- hydraulics: Darcy-Weisbach, Weymouth, flow scaling, compression, looping
- materials: Embrittlement risk tiers, velocity limits, pressure drop limits
- pipeline: Geographic network definitions, Folium mapping
- network: Graph-based steady-state network solver
- gis: Shapefile/GeoJSON ingestion for real pipeline routes
- compressor: Compressor performance maps and station models
- transient: Time-dependent simulation (linepack, blending transients)
- economics: CAPEX/OPEX modeling and optimization
"""

from .thermodynamics import (
    GasProperties, CH4, H2, R_UNIVERSAL,
    peng_robinson_z, viscosity_pure, wilke_viscosity, blend_properties
)
from .hydraulics import (
    friction_factor, darcy_weisbach_dp, weymouth_dp,
    equivalent_flow_rate, gas_velocity, reynolds_number,
    compression_work_mw, equivalent_hydraulic_diameter,
    pressure_drop_with_looping, simulate_pipeline_with_compressors,
    find_required_looping,
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
from .network.solver import (
    GasNetwork, NetworkNode, NetworkEdge, NodeType, EdgeType, CompressorStation,
    create_ireland_network
)
from .gis.importer import GISImporter, GISLayerConfig, create_sample_geojson
from .compressor.maps import (
    CompressorMap, DriverModel, CompressorStationModel,
    DriverType, create_standard_compressor
)
from .transient.simulator import (
    TransientSimulator, TransientScenario, TransientResults,
    PipeSegment, TransientMethod, create_blending_ramp_scenario
)
from .economics.cost_model import (
    PipelineCAPEX, CompressorCAPEX, OPEXModel, EconomicAnalysis,
    CostBasis, estimate_ireland_h2_project
)
from .uq.monte_carlo import (
    MonteCarloUQ, UncertainParameter, UQResults, run_quick_uq_demo
)

__all__ = [
    # Thermodynamics
    "GasProperties", "CH4", "H2", "R_UNIVERSAL",
    "peng_robinson_z", "viscosity_pure", "wilke_viscosity", "blend_properties",
    # Hydraulics
    "friction_factor", "darcy_weisbach_dp", "weymouth_dp",
    "equivalent_flow_rate", "gas_velocity", "reynolds_number",
    "compression_work_mw", "equivalent_hydraulic_diameter",
    "pressure_drop_with_looping", "simulate_pipeline_with_compressors",
    "find_required_looping", "EPSILON",
    # Materials
    "RiskTier", "MaterialAlert",
    "embrittlement_tier", "velocity_tier", "pressure_drop_tier",
    "assess_all", "risk_color",
    "EMBRITTLEMENT_TIERS", "VELOCITY_LIMITS", "DP_LIMITS",
    # Pipeline (original)
    "PipelineSegment", "CompressorStation", "PipelineNetwork",
    "create_ireland_network",
    # Network Solver
    "GasNetwork", "NetworkNode", "NetworkEdge", "NodeType", "EdgeType", "CompressorStation",
    "create_ireland_network",
    # GIS
    "GISImporter", "GISLayerConfig", "create_sample_geojson",
    # Compressor
    "CompressorMap", "DriverModel", "CompressorStationModel",
    "DriverType", "create_standard_compressor",
    # Transient
    "TransientSimulator", "TransientScenario", "TransientResults",
    "PipeSegment", "TransientMethod", "create_blending_ramp_scenario",
    # Economics
    "PipelineCAPEX", "CompressorCAPEX", "OPEXModel", "EconomicAnalysis",
    "CostBasis", "estimate_ireland_h2_project",
    # UQ
    "MonteCarloUQ", "UncertainParameter", "UQResults", "run_quick_uq_demo",
]