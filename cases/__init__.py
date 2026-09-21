"""
Case studies package.

Each case provides:
- config.yaml: Region-specific parameters, corridors, policy context
- __init__.py: Loader functions (get_defaults, get_network, get_corridors, etc.)
- Optional: case-specific extensions (e.g., UK NTS, EU backbone)

Available cases:
- ireland: Gas Networks Ireland transmission corridors (reference implementation)
"""

from .ireland import (
    load_config as load_ireland_config,
    get_defaults as get_ireland_defaults,
    get_network as get_ireland_network,
    get_corridors as get_ireland_corridors,
    get_policy_context as get_ireland_policy,
    get_steel_grades as get_ireland_steel_grades,
    get_risk_thresholds as get_ireland_risk_thresholds,
    CASE_METADATA as IRELAND_METADATA,
)

AVAILABLE_CASES = {
    "ireland": {
        "loader": load_ireland_config,
        "defaults": get_ireland_defaults,
        "network": get_ireland_network,
        "corridors": get_ireland_corridors,
        "policy": get_ireland_policy,
        "steel_grades": get_ireland_steel_grades,
        "risk_thresholds": get_ireland_risk_thresholds,
        "metadata": IRELAND_METADATA,
    },
}

DEFAULT_CASE = "ireland"


def list_cases() -> List[str]:
    """Return list of available case study keys."""
    return list(AVAILABLE_CASES.keys())


def get_case(case_key: str) -> Dict[str, Any]:
    """Return case loader dict for given key."""
    if case_key not in AVAILABLE_CASES:
        raise ValueError(f"Unknown case: {case_key}. Available: {list_cases()}")
    return AVAILABLE_CASES[case_key]