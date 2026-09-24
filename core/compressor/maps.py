"""
Compressor station performance modeling.

Implements:
- Polytropic head/flow curves (universal compressor map)
- Speed lines, surge/stonewall limits
- Driver characteristics (gas turbine, electric motor, reciprocating)
- Part-load efficiency
- ISO 5389 / ASME PTC 10 performance testing basis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


class DriverType(Enum):
    GAS_TURBINE = "gas_turbine"
    ELECTRIC_MOTOR = "electric_motor"
    RECIPROCATING = "reciprocating"
    STEAM_TURBINE = "steam_turbine"


@dataclass
class CompressorMap:
    """
    Universal compressor performance map.
    
    Based on non-dimensional parameters:
    - Flow coefficient: Φ = Q / (N * D^3)
    - Head coefficient: Ψ = (g * H) / (N^2 * D^2)  or  Δh / (N^2)
    - Speed ratio: N / N_design
    
    Map boundaries:
    - Surge line (left boundary)
    - Stonewall/choke line (right boundary)
    - Max speed line (top)
    - Min speed line (bottom)
    """
    # Design point
    design_flow_m3h: float      # Actual m3/h at suction conditions
    design_head_kJ_kg: float    # Polytropic head
    design_speed_rpm: float     # Design speed
    design_efficiency: float    # Polytropic efficiency at design
    
    # Map curves (as interpolators)
    # Speed lines: head vs flow at constant speed
    speed_ratios: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1])
    flow_coeffs: List[float] = field(default_factory=lambda: [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    head_coeffs: List[List[float]] = field(default_factory=list)  # [speed][flow]
    efficiency_map: List[List[float]] = field(default_factory=list)  # [speed][flow]
    
    # Surge/stonewall boundaries
    surge_flow_coeffs: List[float] = field(default_factory=lambda: [0.02, 0.025, 0.03, 0.035, 0.04])
    surge_speed_ratios: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 1.0, 1.1])
    stonewall_flow_coeffs: List[float] = field(default_factory=lambda: [0.075, 0.08, 0.082, 0.084, 0.085])
    stonewall_speed_ratios: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 1.0, 1.1])
    
    # Impeller geometry
    impeller_diameter_m: float = 0.5
    impeller_tip_speed_m_s: float = 450.0
    
    # Interpolators (built on init)
    _head_interp: Optional[RegularGridInterpolator] = field(default=None, init=False)
    _eff_interp: Optional[RegularGridInterpolator] = field(default=None, init=False)
    _surge_interp: Optional[interp1d] = field(default=None, init=False)
    _stonewall_interp: Optional[interp1d] = field(default=None, init=False)
    
    def __post_init__(self):
        self._build_interpolators()
    
    def _build_interpolators(self):
        """Build 2D interpolators for head and efficiency."""
        if not self.head_coeffs or not self.efficiency_map:
            # Generate default map if not provided
            self._generate_default_map()
        
        # Head interpolator
        X, Y = np.meshgrid(self.flow_coeffs, self.speed_ratios)
        self._head_interp = RegularGridInterpolator(
            (self.speed_ratios, self.flow_coeffs), 
            np.array(self.head_coeffs), 
            bounds_error=False, fill_value=None
        )
        self._eff_interp = RegularGridInterpolator(
            (self.speed_ratios, self.flow_coeffs),
            np.array(self.efficiency_map),
            bounds_error=False, fill_value=None
        )
        
        # Surge/stonewall interpolators
        self._surge_interp = interp1d(
            self.surge_speed_ratios, self.surge_flow_coeffs, 
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        self._stonewall_interp = interp1d(
            self.stonewall_speed_ratios, self.stonewall_flow_coeffs,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
    
    def _generate_default_map(self):
        """Generate a realistic default compressor map (centrifugal)."""
        # Typical centrifugal compressor map
        # Head coefficient decreases with flow, peaks near surge
        # Efficiency peaks at design point
        self.head_coeffs = []
        self.efficiency_map = []
        
        for sr in self.speed_ratios:
            head_row = []
            eff_row = []
            for fc in self.flow_coeffs:
                # Head: parabolic-ish, higher at low flow
                # Normalized: at design (sr=1, fc=0.05) head_coeff = 1.0
                design_fc = 0.05
                design_sr = 1.0
                
                # Affinity laws: head ∝ speed^2, flow ∝ speed
                fc_scaled = fc / sr
                
                # Characteristic curve shape
                if fc_scaled <= 0.025:
                    h = 1.1 * sr**2
                elif fc_scaled <= design_fc:
                    h = (1.1 - 0.4 * (fc_scaled - 0.025) / (design_fc - 0.025)) * sr**2
                else:
                    h = (1.0 - 1.5 * (fc_scaled - design_fc) / (0.08 - design_fc)) * sr**2
                
                head_row.append(max(h, 0.1))
                
                # Efficiency: peaks at design
                eff = self.design_efficiency * (
                    1 - 0.3 * abs(fc_scaled - design_fc) / design_fc
                )
                eff_row.append(max(eff, 0.5))
            
            self.head_coeffs.append(head_row)
            self.efficiency_map.append(eff_row)
    
    def get_performance(self, flow_m3h: float, speed_rpm: float, 
                        suction_density_kgm3: float) -> Tuple[float, float, float, bool, bool]:
        """
        Get compressor performance at operating point.
        
        Returns: (head_kJ_kg, efficiency, power_kW, is_surge, is_stonewall)
        """
        # Flow coefficient
        N = speed_rpm / 60.0  # rps
        D = self.impeller_diameter_m
        fc = flow_m3h / 3600.0 / (N * D**3)
        
        # Speed ratio
        sr = speed_rpm / self.design_speed_rpm
        
        # Interpolate head and efficiency
        try:
            head_coeff = float(self._head_interp([sr, fc]))
            eff = float(self._eff_interp([sr, fc]))
        except:
            head_coeff = 0.8
            eff = 0.75
        
        # Dimensional head
        head = head_coeff * (N * D)**2 / 1000.0  # kJ/kg
        
        # Power
        mass_flow = flow_m3h / 3600.0 * suction_density_kgm3  # kg/s
        power_kW = mass_flow * head / eff  # kW
        
        # Check boundaries
        surge_fc = float(self._surge_interp(sr))
        stonewall_fc = float(self._stonewall_interp(sr))
        
        is_surge = fc <= surge_fc * 1.02  # 2% margin
        is_stonewall = fc >= stonewall_fc * 0.98
        
        return head, eff, power_kW, is_surge, is_stonewall
    
    def find_operating_point(self, target_head: float, flow_m3h: float,
                             suction_density: float,
                             speed_range: Tuple[float, float] = (0.5, 1.1)) -> Optional[Dict]:
        """Find speed required to deliver target head at given flow."""
        from scipy.optimize import minimize_scalar
        
        def error(sr):
            speed = sr * self.design_speed_rpm
            head, eff, power, surge, choke = self.get_performance(
                flow_m3h, speed, suction_density
            )
            if surge or choke:
                return 1e6
            return abs(head - target_head)
        
        result = minimize_scalar(error, bounds=speed_range, method='bounded')
        if result.success:
            sr = result.x
            speed = sr * self.design_speed_rpm
            head, eff, power, surge, choke = self.get_performance(
                flow_m3h, speed, suction_density
            )
            return {
                "speed_rpm": speed,
                "speed_ratio": sr,
                "head_kJ_kg": head,
                "efficiency": eff,
                "power_kW": power,
                "is_surge": surge,
                "is_stonewall": choke,
            }
        return None


@dataclass
class DriverModel:
    """Driver (prime mover) performance model."""
    driver_type: DriverType
    rated_power_kW: float
    rated_speed_rpm: float
    
    # Performance curves
    # Part-load efficiency curve: eff = f(load_fraction)
    part_load_efficiency: List[Tuple[float, float]] = field(default_factory=list)
    
    # For gas turbines: heat rate curve
    heat_rate_curve: List[Tuple[float, float]] = field(default_factory=list)  # (load, kJ/kWh)
    
    # Emissions
    co2_factor_kg_per_mwh: float = 0.0  # For gas turbine
    nox_factor_g_per_kwh: float = 0.0
    
    def __post_init__(self):
        if not self.part_load_efficiency:
            # Default curves
            if self.driver_type == DriverType.GAS_TURBINE:
                self.part_load_efficiency = [
                    (0.0, 0.0), (0.2, 0.25), (0.4, 0.32), (0.6, 0.36),
                    (0.8, 0.38), (1.0, 0.39), (1.1, 0.38)
                ]
                self.heat_rate_curve = [
                    (0.2, 14000), (0.4, 11000), (0.6, 9500),
                    (0.8, 9000), (1.0, 8800), (1.1, 9000)
                ]
                self.co2_factor_kg_per_mwh = 450  # kg CO2/MWh thermal
            elif self.driver_type == DriverType.ELECTRIC_MOTOR:
                self.part_load_efficiency = [
                    (0.0, 0.0), (0.1, 0.85), (0.25, 0.92), (0.5, 0.96),
                    (0.75, 0.97), (1.0, 0.965), (1.15, 0.95)
                ]
            elif self.driver_type == DriverType.RECIPROCATING:
                self.part_load_efficiency = [
                    (0.0, 0.0), (0.2, 0.35), (0.4, 0.40), (0.6, 0.42),
                    (0.8, 0.43), (1.0, 0.43), (1.1, 0.42)
                ]
    
    def get_efficiency(self, load_fraction: float) -> float:
        """Get driver efficiency at load fraction."""
        if not self.part_load_efficiency:
            return 0.95
        loads, effs = zip(*self.part_load_efficiency)
        interp = interp1d(loads, effs, kind='linear', bounds_error=False, 
                          fill_value=(effs[0], effs[-1]))
        return float(interp(np.clip(load_fraction, loads[0], loads[-1])))
    
    def get_fuel_consumption(self, power_kW: float) -> float:
        """Get fuel consumption in kg/h (for gas turbine)."""
        if self.driver_type != DriverType.GAS_TURBINE:
            return 0.0
        load = power_kW / self.rated_power_kW
        loads, hr = zip(*self.heat_rate_curve)
        interp = interp1d(loads, hr, kind='linear', bounds_error=False,
                          fill_value=(hr[0], hr[-1]))
        heat_rate = float(interp(np.clip(load, loads[0], loads[-1])))  # kJ/kWh
        # Fuel: power (kW) * heat_rate (kJ/kWh) / LHV (kJ/kg)
        LHV_NG = 48000  # kJ/kg approx
        return power_kW * heat_rate / LHV_NG  # kg/h


@dataclass
class CompressorStationModel:
    """
    Complete compressor station model combining compressor map + driver.
    """
    station_id: str
    name: str
    compressor_map: CompressorMap
    driver: DriverModel
    
    # Station configuration
    num_units: int = 1
    units_in_parallel: bool = True
    suction_cooler: bool = True
    discharge_cooler: bool = False
    
    # Piping losses
    suction_loss_factor: float = 0.01  # Fraction of suction pressure
    discharge_loss_factor: float = 0.01
    
    # Control
    control_mode: str = "discharge_pressure"  # discharge_pressure, suction_pressure, flow
    setpoint: float = 70.0  # bar
    
    def simulate(self, suction_pressure_bar: float, suction_temp_k: float,
                 flow_sm3h: float, gas_props: Dict,
                 target_discharge_pressure: Optional[float] = None) -> Dict:
        """
        Simulate compressor station performance.
        
        Returns dict with discharge conditions, power, fuel, emissions.
        """
        # Suction density (actual conditions)
        Z = gas_props.get("Z_blend", 0.9)
        M = gas_props.get("M_blend", 16.0) / 1000.0  # kg/mol
        R = 8.314462618
        suction_density = suction_pressure_bar * 1e5 * M / (Z * R * suction_temp_k)  # kg/m3
        
        # Convert standard flow to actual suction flow
        # Q_actual = Q_std * (P_std/P_suc) * (T_suc/T_std) * (Z_suc/Z_std)
        P_std = 1.01325
        T_std = 273.15
        flow_actual_m3h = flow_sm3h * (P_std / suction_pressure_bar) * \
                          (suction_temp_k / T_std) * (Z / 1.0)
        
        # Per unit flow
        flow_per_unit = flow_actual_m3h / self.num_units
        
        # Target discharge pressure
        if target_discharge_pressure is None:
            target_discharge_pressure = self.setpoint
        
        # Required head
        # Polytropic head: H = (Z*R*T/M) * (n/(n-1)) * [(Pd/Ps)^((n-1)/n) - 1]
        n = 1.3  # Polytropic exponent
        pressure_ratio = target_discharge_pressure / suction_pressure_bar
        head_required = (Z * R * suction_temp_k / M) * (n / (n - 1)) * \
                        (pressure_ratio**((n - 1) / n) - 1) / 1000.0  # kJ/kg
        
        # Find operating point
        op = self.compressor_map.find_operating_point(
            head_required, flow_per_unit, suction_density
        )
        
        if op is None:
            return {"error": "No valid operating point found", "success": False}
        
        # Apply piping losses
        P_disch_actual = op["head_kJ_kg"] * 1000.0 * (n - 1) / n * M / (Z * R * suction_temp_k)
        P_disch_actual = suction_pressure_bar * (1 + P_disch_actual)**(n / (n - 1))
        P_disch_actual *= (1 - self.discharge_loss_factor)
        
        # Driver
        load_fraction = op["power_kW"] / self.driver.rated_power_kW
        driver_eff = self.driver.get_efficiency(load_fraction)
        shaft_power = op["power_kW"] / driver_eff
        
        # Fuel/emissions
        fuel_kg_h = self.driver.get_fuel_consumption(shaft_power)
        co2_kg_h = fuel_kg_h * self.driver.co2_factor_kg_per_mwh / 1000.0 * shaft_power if self.driver.co2_factor_kg_per_mwh else 0
        
        return {
            "success": True,
            "suction_pressure_bar": suction_pressure_bar,
            "discharge_pressure_bar": P_disch_actual,
            "pressure_ratio": P_disch_actual / suction_pressure_bar,
            "flow_sm3h": flow_sm3h,
            "flow_per_unit_m3h": flow_per_unit,
            "speed_rpm": op["speed_rpm"],
            "speed_ratio": op["speed_ratio"],
            "polytropic_head_kJ_kg": op["head_kJ_kg"],
            "compressor_efficiency": op["efficiency"],
            "compressor_power_kW": op["power_kW"],
            "driver_efficiency": driver_eff,
            "shaft_power_kW": shaft_power,
            "total_shaft_power_kW": shaft_power * self.num_units,
            "fuel_kg_h": fuel_kg_h * self.num_units,
            "co2_kg_h": co2_kg_h * self.num_units,
            "is_surge": op["is_surge"],
            "is_stonewall": op["is_stonewall"],
            "surge_margin": (flow_per_unit - self._surge_flow_at_speed(op["speed_rpm"], suction_density)) 
                           / flow_per_unit if not op["is_surge"] else 0,
        }
    
    def _surge_flow_at_speed(self, speed_rpm: float, suction_density: float) -> float:
        """Get surge flow at given speed."""
        sr = speed_rpm / self.compressor_map.design_speed_rpm
        surge_fc = float(self.compressor_map._surge_interp(sr))
        N = speed_rpm / 60.0
        D = self.compressor_map.impeller_diameter_m
        return surge_fc * N * D**3 * 3600.0  # m3/h actual


def create_standard_compressor(station_id: str, name: str, 
                               design_flow: float, design_head: float,
                               driver_type: DriverType = DriverType.GAS_TURBINE,
                               driver_power: float = 50000) -> CompressorStationModel:
    """Create a standard compressor station model."""
    comp_map = CompressorMap(
        design_flow_m3h=design_flow,
        design_head_kJ_kg=design_head,
        design_speed_rpm=10000,
        design_efficiency=0.82,
    )
    driver = DriverModel(
        driver_type=driver_type,
        rated_power_kW=driver_power,
        rated_speed_rpm=10000,
    )
    return CompressorStationModel(
        station_id=station_id,
        name=name,
        compressor_map=comp_map,
        driver=driver,
    )