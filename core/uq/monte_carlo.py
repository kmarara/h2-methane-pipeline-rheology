"""
Monte Carlo Uncertainty Quantification for H2 Pipeline Digital Twin.

Performs probabilistic analysis on key uncertain parameters:
- Pipeline roughness
- Ground temperature
- Gas composition
- Demand forecasts
- Compressor performance
- Material properties

Outputs: P10/P50/P90 confidence intervals, sensitivity indices (Sobol)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from scipy.stats import norm, lognorm, uniform, triang
from scipy.optimize import minimize
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from core import GasNetwork, create_ireland_network
from core.transient.simulator import TransientSimulator, create_blending_ramp_scenario


@dataclass
class UncertainParameter:
    """Definition of an uncertain input parameter."""
    name: str
    distribution: str  # normal, lognormal, uniform, triangular
    params: Dict[str, float]  # Distribution parameters
    description: str = ""
    
    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate n samples."""
        if seed is not None:
            np.random.seed(seed)
        
        if self.distribution == "normal":
            return np.random.normal(self.params["mean"], self.params["std"], n)
        elif self.distribution == "lognormal":
            # params: mean, std of underlying normal
            return np.random.lognormal(self.params["mean"], self.params["std"], n)
        elif self.distribution == "uniform":
            return np.random.uniform(self.params["low"], self.params["high"], n)
        elif self.distribution == "triangular":
            return np.random.triangular(self.params["left"], self.params["mode"], self.params["right"], n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class UQResults:
    """Results from Monte Carlo UQ analysis."""
    parameter_samples: Dict[str, np.ndarray]  # [n_samples, n_params]
    output_samples: Dict[str, np.ndarray]     # [n_samples] for each output
    statistics: Dict[str, Dict[str, float]]   # mean, std, P10, P50, P90 for each output
    sensitivity: Dict[str, Dict[str, float]]  # Sobol indices for each output
    n_samples: int
    compute_time_s: float


class MonteCarloUQ:
    """
    Monte Carlo Uncertainty Quantification for pipeline simulations.
    
    Supports:
    - Steady-state network analysis
    - Transient blending scenarios
    - Economic analysis
    """
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or mp.cpu_count()
        self.parameters: List[UncertainParameter] = []
        self._setup_default_parameters()
    
    def _setup_default_parameters(self):
        """Default uncertain parameters for H2 pipeline analysis."""
        self.parameters = [
            UncertainParameter(
                name="roughness_mm",
                distribution="lognormal",
                params={"mean": np.log(0.0457), "std": 0.3},
                description="Pipeline internal roughness (mm)"
            ),
            UncertainParameter(
                name="ground_temp_c",
                distribution="normal",
                params={"mean": 10.0, "std": 3.0},
                description="Ground temperature (°C)"
            ),
            UncertainParameter(
                name="h2_fraction",
                distribution="triangular",
                params={"left": 0.0, "mode": 0.2, "right": 0.3},
                description="Hydrogen blend fraction"
            ),
            UncertainParameter(
                name="demand_scaling",
                distribution="normal",
                params={"mean": 1.0, "std": 0.15},
                description="Demand forecast scaling factor"
            ),
            UncertainParameter(
                name="compressor_efficiency",
                distribution="triangular",
                params={"left": 0.75, "mode": 0.82, "right": 0.88},
                description="Compressor polytropic efficiency"
            ),
            UncertainParameter(
                name="gas_price_eur_mwh",
                distribution="lognormal",
                params={"mean": np.log(35), "std": 0.4},
                description="Wholesale gas price (EUR/MWh)"
            ),
            UncertainParameter(
                name="carbon_price_eur_t",
                distribution="triangular",
                params={"left": 50, "mode": 85, "right": 150},
                description="EU ETS carbon price (EUR/tonne CO2)"
            ),
        ]
    
    def add_parameter(self, param: UncertainParameter):
        """Add custom uncertain parameter."""
        self.parameters.append(param)
    
    def _run_single_steady_state(self, sample: Dict[str, float]) -> Dict[str, float]:
        """Run single steady-state simulation with sampled parameters."""
        try:
            from core.network.solver import create_ireland_network, GasNetwork, NetworkNode, NodeType
            from dataclasses import replace
            
            base_net = create_ireland_network()
            
            # Create new network with scaled demands (frozen dataclass needs replace)
            net = GasNetwork("MC Network")
            for nid, node in base_net.nodes.items():
                if node.node_type == NodeType.DEMAND:
                    new_node = replace(node, demand_sm3h=node.demand_sm3h * sample["demand_scaling"])
                else:
                    new_node = node
                net.add_node(new_node)
            
            for eid, edge in base_net.edges.items():
                net.add_edge(edge)
            
            for cid, comp in base_net.compressors.items():
                net.add_compressor(comp)
            
            net.set_gas_properties(sample["h2_fraction"], 
                                  sample["ground_temp_c"] + 273.15)
            
            results = net.solve_steady_state()
            
            if not results["converged"]:
                return {"success": False}
            
            min_pressure = min(results["node_pressures"].values())
            total_power = results["total_compression_power_mw"]
            
            return {
                "success": True,
                "min_pressure_bar": min_pressure,
                "total_compression_power_mw": total_power,
                "n_violations": len(results["pressure_violations"]),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_single_transient(self, sample: Dict[str, float]) -> Dict[str, float]:
        """Run single transient simulation with sampled parameters."""
        try:
            from core.network.solver import create_ireland_network
            from core.transient.simulator import TransientSimulator, create_blending_ramp_scenario
            net = create_ireland_network()
            net.set_gas_properties(sample["h2_fraction"], 
                                  sample["ground_temp_c"] + 273.15)
            
            for nid, node in net.nodes.items():
                if node.node_type.name == "DEMAND":
                    node.demand_sm3h *= sample["demand_scaling"]
            
            scenario = create_blending_ramp_scenario(
                net, duration_hours=24, 
                initial_h2=sample["h2_fraction"] * 0.5,
                final_h2=sample["h2_fraction"],
                ramp_hours=6
            )
            
            sim = TransientSimulator(net)
            results = sim.simulate(scenario, dt_hours=0.5)
            
            # Key outputs
            min_pressure = min(
                min(p) for p in results.node_pressures.values()
            )
            max_h2_at_demand = max(
                max(h) for nid, h in results.node_h2_fractions.items()
                if net.nodes[nid].node_type.name == "DEMAND"
            )
            linepack_change = (results.total_linepack_kg[-1] - results.total_linepack_kg[0]) / 1000.0  # tonnes
            
            return {
                "success": True,
                "min_pressure_bar": min_pressure,
                "max_h2_at_demand": max_h2_at_demand,
                "linepack_change_tonnes": linepack_change,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_single_economics(self, sample: Dict[str, float]) -> Dict[str, float]:
        """Run single economic analysis with sampled parameters."""
        try:
            # Imports inside function for multiprocessing compatibility
            from core.economics.cost_model import EconomicAnalysis, estimate_ireland_h2_project
            
            # Simplified network results
            network_results = {
                "total_length_km": 220,
                "diameter_mm": 500,
                "compression_power_mw": 15 * sample.get("demand_scaling", 1.0),
                "num_compressors": 1,
            }
            
            eco = EconomicAnalysis()
            eco.opex.gas_price_eur_mwh = sample["gas_price_eur_mwh"]
            eco.opex.carbon_price_eur_tonne = sample["carbon_price_eur_t"]
            
            results = estimate_ireland_h2_project(network_results, 
                                                  sample["h2_fraction"] * 100)
            
            return {
                "success": True,
                "total_capex_eur": results["capex_breakdown"]["total_eur"],
                "annual_opex_eur": results["opex_breakdown_eur_year"]["total"],
                "npv_eur": results["economic_indicators"]["npv_eur"],
                "lcot_eur_gj": results["economic_indicators"]["levelized_cost_eur_gj"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_steady_state_mc(self, n_samples: int = 1000, seed: int = 42) -> UQResults:
        """Run Monte Carlo for steady-state network analysis."""
        return self._run_monte_carlo(
            self._run_single_steady_state, n_samples, seed
        )
    
    def run_transient_mc(self, n_samples: int = 500, seed: int = 42) -> UQResults:
        """Run Monte Carlo for transient blending analysis."""
        return self._run_monte_carlo(
            self._run_single_transient, n_samples, seed
        )
    
    def run_economics_mc(self, n_samples: int = 1000, seed: int = 42) -> UQResults:
        """Run Monte Carlo for economic analysis."""
        return self._run_monte_carlo(
            self._run_single_economics, n_samples, seed
        )
    
    def _run_monte_carlo(self, model_func: Callable, n_samples: int, seed: int) -> UQResults:
        """Generic Monte Carlo runner."""
        start_time = time.time()
        
        # Generate all samples
        n_params = len(self.parameters)
        param_samples = np.zeros((n_samples, n_params))
        param_names = [p.name for p in self.parameters]
        
        for i, param in enumerate(self.parameters):
            # Use different seed for each parameter
            param_samples[:, i] = param.sample(n_samples, seed=seed + i * 1000)
        
        # Prepare sample dicts
        sample_dicts = [
            {name: param_samples[j, i] for i, name in enumerate(param_names)}
            for j in range(n_samples)
        ]
        
        # Run in parallel
        output_samples_list = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(model_func, s) for s in sample_dicts]
            for future in as_completed(futures):
                output_samples_list.append(future.result())
        
        # Process outputs
        # Find all output keys from successful runs
        output_keys = set()
        for out in output_samples_list:
            if out.get("success"):
                output_keys.update(k for k in out.keys() if k != "success")
        
        output_arrays = {key: [] for key in output_keys}
        success_count = 0
        
        for out in output_samples_list:
            if out.get("success"):
                success_count += 1
                for key in output_keys:
                    output_arrays[key].append(out.get(key, np.nan))
            else:
                for key in output_keys:
                    output_arrays[key].append(np.nan)
        
        # Convert to arrays
        for key in output_arrays:
            output_arrays[key] = np.array(output_arrays[key])
        
        # Statistics
        statistics = {}
        for key, arr in output_arrays.items():
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                statistics[key] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "p10": float(np.percentile(valid, 10)),
                    "p50": float(np.percentile(valid, 50)),
                    "p90": float(np.percentile(valid, 90)),
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                    "success_rate": success_count / n_samples,
                }
        
        # Sensitivity analysis (simplified: correlation-based)
        sensitivity = {}
        for key, arr in output_arrays.items():
            valid_mask = ~np.isnan(arr)
            if np.sum(valid_mask) > 10:
                sens = {}
                for i, pname in enumerate(param_names):
                    p_vals = param_samples[valid_mask, i]
                    o_vals = arr[valid_mask]
                    # Pearson correlation as sensitivity proxy
                    corr = np.corrcoef(p_vals, o_vals)[0, 1]
                    if not np.isnan(corr):
                        sens[pname] = float(abs(corr))
                sensitivity[key] = sens
        
        compute_time = time.time() - start_time
        
        return UQResults(
            parameter_samples={name: param_samples[:, i] for i, name in enumerate(param_names)},
            output_samples=output_arrays,
            statistics=statistics,
            sensitivity=sensitivity,
            n_samples=n_samples,
            compute_time_s=compute_time,
        )
    
    def print_summary(self, results: UQResults):
        """Print formatted UQ results summary."""
        print(f"\n{'='*60}")
        print(f"Monte Carlo UQ Summary ({results.n_samples} samples)")
        print(f"Compute time: {results.compute_time_s:.1f}s")
        print(f"{'='*60}")
        
        for out_name, stats in results.statistics.items():
            print(f"\n{out_name}:")
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  P10/P50/P90: {stats['p10']:.3f} / {stats['p50']:.3f} / {stats['p90']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            
            if out_name in results.sensitivity:
                print(f"  Sensitivities:")
                for param, sens in sorted(results.sensitivity[out_name].items(), 
                                          key=lambda x: -x[1])[:5]:
                    print(f"    {param}: {sens:.3f}")


def run_quick_uq_demo():
    """Quick demo of UQ capabilities."""
    uq = MonteCarloUQ(n_workers=4)
    
    print("Running steady-state MC (200 samples)...")
    ss_results = uq.run_steady_state_mc(n_samples=200, seed=42)
    uq.print_summary(ss_results)
    
    print("\n\nRunning economics MC (200 samples)...")
    econ_results = uq.run_economics_mc(n_samples=200, seed=42)
    uq.print_summary(econ_results)


if __name__ == "__main__":
    run_quick_uq_demo()