"""
Economic modeling for hydrogen pipeline projects.

CAPEX: Pipeline materials, construction, compressors, valves, commissioning
OPEX: Compression fuel/power, maintenance, monitoring, emissions costs
Optimization: Least-cost pathway for H2 blending targets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
from scipy.optimize import minimize, differential_evolution


class CostBasis(Enum):
    EUR_2024 = "EUR_2024"
    USD_2024 = "USD_2024"


@dataclass
class PipelineCAPEX:
    """Pipeline capital cost model."""
    # Base costs (EUR/m or EUR/km)
    base_cost_per_km: Dict[int, float] = field(default_factory=lambda: {
        200: 800_000,    # 8" 
        300: 1_200_000,  # 12"
        400: 1_800_000,  # 16"
        500: 2_500_000,  # 20"
        600: 3_300_000,  # 24"
        800: 5_500_000,  # 32"
        1000: 8_000_000, # 40"
    })
    
    # H2-ready premium (materials, welding, testing)
    h2_ready_premium_pct: float = 0.15  # 15% for H2-compatible steel, NDT
    
    # Terrain multipliers
    terrain_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "flat_rural": 1.0,
        "rolling_rural": 1.2,
        "mountainous": 1.6,
        "urban": 2.5,
        "river_crossing": 3.0,
        "offshore": 4.0,
        "bog_peat": 1.8,  # Very relevant for Ireland
    })
    
    # Additional costs
    valve_station_cost: float = 500_000  # per station
    pigging_facility_cost: float = 300_000
    cathodic_protection_per_km: float = 15_000
    fiber_optic_per_km: float = 25_000
    land_acquisition_per_km: Dict[str, float] = field(default_factory=lambda: {
        "rural": 50_000,
        "urban": 500_000,
    })
    
    def estimate(self, length_km: float, diameter_mm: int, 
                 terrain: str = "flat_rural", h2_ready: bool = True,
                 num_valves: int = 0, include_fiber: bool = True) -> float:
        """Estimate pipeline CAPEX in EUR."""
        # Base pipeline cost
        base = self.base_cost_per_km.get(diameter_mm, 
                                          self.base_cost_per_km[500] * (diameter_mm/500)**1.2)
        cost = base * length_km
        
        # Terrain
        cost *= self.terrain_multipliers.get(terrain, 1.0)
        
        # H2 ready premium
        if h2_ready:
            cost *= (1 + self.h2_ready_premium_pct)
        
        # Valve stations
        cost += num_valves * self.valve_station_cost
        
        # Pigging (every 50 km)
        cost += int(length_km / 50) * self.pigging_facility_cost
        
        # Cathodic protection
        cost += length_km * self.cathodic_protection_per_km
        
        # Fiber optic
        if include_fiber:
            cost += length_km * self.fiber_optic_per_km
        
        # Land acquisition (assume rural)
        cost += length_km * self.land_acquisition_per_km["rural"]
        
        # Engineering, procurement, construction management (15%)
        cost *= 1.15
        
        # Contingency (20%)
        cost *= 1.20
        
        return cost


@dataclass
class CompressorCAPEX:
    """Compressor station capital cost."""
    # Cost curve: EUR/kW installed
    base_cost_per_kw: float = 800  # Gas turbine driven
    
    # Driver type multipliers
    driver_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "gas_turbine": 1.0,
        "electric_motor": 0.7,
        "reciprocating": 1.3,
    })
    
    # H2 service premium
    h2_service_premium: float = 0.20  # Seals, materials, controls
    
    # Balance of plant (buildings, piping, controls, cooling)
    bop_factor: float = 1.4
    
    def estimate(self, power_mw: float, driver_type: str = "gas_turbine",
                 h2_service: bool = True, num_units: int = 1) -> float:
        """Estimate compressor station CAPEX in EUR."""
        cost_per_kw = self.base_cost_per_kw * self.driver_multipliers.get(driver_type, 1.0)
        if h2_service:
            cost_per_kw *= (1 + self.h2_service_premium)
        
        cost = cost_per_kw * power_mw * 1000 * self.bop_factor
        
        # Multiple units: some savings
        if num_units > 1:
            cost *= (0.85 + 0.15 / num_units)
        
        return cost


@dataclass
class OPEXModel:
    """Operating cost model."""
    # Energy costs
    gas_price_eur_mwh: float = 35.0      # Wholesale gas
    electricity_price_eur_mwh: float = 80.0  # Industrial electricity
    
    # Carbon costs
    carbon_price_eur_tonne: float = 85.0  # EU ETS
    
    # Maintenance
    pipeline_maintenance_pct_capex: float = 0.015  # 1.5%/year
    compressor_maintenance_pct_capex: float = 0.03  # 3%/year
    
    # Monitoring & integrity
    ilu_cost_per_km: float = 5_000  # In-line inspection
    leak_survey_per_km: float = 2_000
    cathodic_protection_maint_per_km: float = 5_000
    
    # H2-specific
    h2_monitoring_premium_pct: float = 0.25  # Extra sensors, labs
    embrittlement_mitigation_annual: float = 100_000  # Per station
    
    def compression_cost(self, power_mw: float, hours: float, 
                         driver_type: str = "gas_turbine",
                         gas_price: Optional[float] = None,
                         electricity_price: Optional[float] = None) -> float:
        """Annual compression energy cost."""
        gas_p = gas_price or self.gas_price_eur_mwh
        elec_p = electricity_price or self.electricity_price_eur_mwh
        
        if driver_type == "gas_turbine":
            # ~35% efficiency, fuel cost
            thermal_input_mwh = power_mw / 0.35 * hours
            return thermal_input_mwh * gas_p
        elif driver_type == "electric_motor":
            # ~95% efficiency
            elec_mwh = power_mw / 0.95 * hours
            return elec_mwh * elec_p
        else:
            return power_mw * hours * gas_p  # Approximate
    
    def carbon_cost(self, power_mw: float, hours: float, 
                    driver_type: str = "gas_turbine",
                    carbon_price: Optional[float] = None) -> float:
        """Annual carbon cost (EU ETS)."""
        cp = carbon_price or self.carbon_price_eur_tonne
        
        if driver_type == "gas_turbine":
            # ~0.45 tCO2/MWh thermal, thermal = power/0.35
            co2_tonnes = power_mw / 0.35 * hours * 0.45
        elif driver_type == "electric_motor":
            # Grid average ~0.3 tCO2/MWh (Ireland 2024)
            co2_tonnes = power_mw / 0.95 * hours * 0.3
        else:
            co2_tonnes = power_mw * hours * 0.4
        
        return co2_tonnes * cp
    
    def annual_opex(self, pipeline_capex: float, compressor_capex: float,
                    length_km: float, num_compressors: int,
                    compression_power_mw: float, hours: float,
                    driver_type: str = "gas_turbine",
                    h2_blend_pct: float = 0.0) -> Dict[str, float]:
        """Total annual OPEX breakdown."""
        opex = {}
        
        # Maintenance
        opex["pipeline_maintenance"] = pipeline_capex * self.pipeline_maintenance_pct_capex
        opex["compressor_maintenance"] = compressor_capex * self.compressor_maintenance_pct_capex
        
        # Integrity
        opex["ilu"] = length_km * self.ilu_cost_per_km * (1 + self.h2_monitoring_premium_pct * (h2_blend_pct / 100))
        opex["leak_survey"] = length_km * self.leak_survey_per_km
        opex["cathodic_protection"] = length_km * self.cathodic_protection_maint_per_km
        
        # Compression energy
        opex["compression_energy"] = self.compression_cost(compression_power_mw, hours, driver_type)
        
        # Carbon
        opex["carbon_cost"] = self.carbon_cost(compression_power_mw, hours, driver_type)
        
        # H2-specific
        if h2_blend_pct > 10:
            opex["embrittlement_mitigation"] = num_compressors * self.embrittlement_mitigation_annual
        else:
            opex["embrittlement_mitigation"] = 0
        
        opex["total"] = sum(opex.values())
        return opex


@dataclass
class EconomicAnalysis:
    """Full project economic analysis."""
    # Project parameters
    project_life_years: int = 25
    discount_rate: float = 0.06  # 6% WACC
    capacity_factor: float = 0.9  # Utilization
    
    # Models
    pipeline_capex: PipelineCAPEX = field(default_factory=PipelineCAPEX)
    compressor_capex: CompressorCAPEX = field(default_factory=CompressorCAPEX)
    opex: OPEXModel = field(default_factory=OPEXModel)
    
    def npv(self, capex: float, annual_opex: float) -> float:
        """Net Present Value."""
        annuity_factor = (1 - (1 + self.discount_rate)**(-self.project_life_years)) / self.discount_rate
        return -capex - annual_opex * annuity_factor
    
    def levelized_cost(self, capex: float, annual_opex: float, 
                       annual_throughput_gj: float) -> float:
        """Levelized cost of transport (EUR/GJ)."""
        annuity = (1 - (1 + self.discount_rate)**(-self.project_life_years)) / self.discount_rate
        total_pv_cost = capex + annual_opex * annuity
        total_throughput = annual_throughput_gj * annuity
        return total_pv_cost / total_throughput if total_throughput > 0 else 0
    
    def optimize_blending_pathway(self, network_results: Dict,
                                   target_h2_pct: float = 20.0,
                                   years_to_target: int = 5) -> Dict:
        """
        Optimize least-cost pathway to reach target H2 blend.
        
        Decision variables:
        - Compressor additions (when, where, how many)
        - Pipeline looping (when, where, how much)
        - H2 blend ramp schedule
        """
        # Simplified: evaluate discrete options
        options = []
        
        # Option 1: Compressors only
        comp_capex = self.compressor_capex.estimate(
            network_results.get("compression_power_mw", 0) * 1.2,
            h2_service=True
        )
        pipeline_capex = 0
        
        # Option 2: Looping only
        loop_km = network_results.get("required_loop_km", 0)
        pipeline_capex = self.pipeline_capex.estimate(
            loop_km, network_results.get("diameter_mm", 500), h2_ready=True
        )
        comp_capex = 0
        
        # Option 3: Hybrid
        # ...
        
        return {
            "compressors_only": {"capex": comp_capex, "description": "Add intermediate compression"},
            "looping_only": {"capex": pipeline_capex, "description": "Parallel pipeline looping"},
            "recommended": "compressors_only" if comp_capex < pipeline_capex else "looping_only",
        }


def estimate_ireland_h2_project(network_results: Dict,
                                 h2_target_pct: float = 20.0) -> Dict:
    """
    Estimate costs for Ireland H2 blending project based on network results.
    """
    eco = EconomicAnalysis()
    
    # Extract network parameters
    length_km = network_results.get("total_length_km", 220)
    diameter_mm = network_results.get("diameter_mm", 500)
    compression_power = network_results.get("compression_power_mw", 15)
    num_compressors = network_results.get("num_compressors", 1)
    
    # CAPEX
    pipe_capex = eco.pipeline_capex.estimate(
        length_km, diameter_mm, terrain="bog_peat", h2_ready=True,
        num_valves=int(length_km / 30)
    )
    
    comp_capex = eco.compressor_capex.estimate(
        compression_power, driver_type="gas_turbine", h2_service=True,
        num_units=num_compressors
    )
    
    total_capex = pipe_capex + comp_capex
    
    # OPEX (annual)
    hours = 8760 * eco.capacity_factor
    opex_breakdown = eco.opex.annual_opex(
        pipe_capex, comp_capex, length_km, num_compressors,
        compression_power, hours, "gas_turbine", h2_target_pct
    )
    
    # Throughput
    # Assume 20% H2 blend, 70 bar, 500mm, ~10 m/s -> ~200 GWh/day
    annual_throughput_gj = 200 * 3600 * 365 * eco.capacity_factor  # GJ/year
    
    # Economics
    npv = eco.npv(total_capex, opex_breakdown["total"])
    lcot = eco.levelized_cost(total_capex, opex_breakdown["total"], annual_throughput_gj)
    
    return {
        "capex_breakdown": {
            "pipeline_eur": pipe_capex,
            "compressors_eur": comp_capex,
            "total_eur": total_capex,
        },
        "opex_breakdown_eur_year": opex_breakdown,
        "economic_indicators": {
            "npv_eur": npv,
            "levelized_cost_eur_gj": lcot,
            "levelized_cost_eur_mwh": lcot * 3.6,
            "project_life_years": eco.project_life_years,
            "discount_rate": eco.discount_rate,
        },
        "annual_throughput_gj": annual_throughput_gj,
        "assumptions": {
            "gas_price_eur_mwh": eco.opex.gas_price_eur_mwh,
            "carbon_price_eur_t": eco.opex.carbon_price_eur_tonne,
            "capacity_factor": eco.capacity_factor,
        }
    }