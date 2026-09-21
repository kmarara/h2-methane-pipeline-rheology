"""
Ireland (GNI) case study wrapper.

Loads configuration from config.yaml and provides case-specific
parameter defaults, network definition, and policy context.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from core.pipeline import create_ireland_network, PipelineNetwork


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load Ireland case configuration from YAML."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


CONFIG = load_config()


def get_defaults() -> Dict[str, Any]:
    """Return simulation defaults for Ireland case."""
    d = CONFIG["defaults"]
    return {
        "h2_pct": d["h2_blend_pct"],
        "P_bar": d["pressure_bar"],
        "D_mm": d["diameter_mm"],
        "L_km": d["length_km"],
        "T_amb_c": d["ambient_temp_c"],
        "reference_hv_vol": d["reference_hv_vol_mj_m3"],
    }


def get_network() -> PipelineNetwork:
    """Return the GNI transmission corridor network."""
    return create_ireland_network()


def get_corridors() -> List[Dict[str, Any]]:
    """Return corridor definitions for UI selection."""
    return CONFIG["corridors"]


def get_policy_context() -> Dict[str, Any]:
    """Return policy context for documentation/display."""
    return CONFIG["policy_context"]


def get_steel_grades() -> List[str]:
    """Return applicable steel grades for this case."""
    return CONFIG["defaults"]["steel_grades"]


def get_risk_thresholds() -> Dict[str, Any]:
    """Return risk threshold configuration."""
    return {
        "embrittlement": CONFIG["defaults"]["embrittlement_tiers"],
        "velocity": CONFIG["defaults"]["velocity_limits_m_s"],
        "pressure_drop": CONFIG["defaults"]["pressure_drop_fraction"],
    }


# Case metadata for UI
CASE_METADATA = {
    "name": CONFIG["case"]["name"],
    "region": CONFIG["case"]["region"],
    "operator": CONFIG["case"]["operator"],
    "description": CONFIG["case"]["description"],
    "default_corridor": "Dublin-Cork (Inchicore → Whitegate)",
}