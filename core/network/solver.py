"""
Graph-based steady-state hydraulic network solver for gas transmission systems.

Solves for nodal pressures and pipe flows using Newton-Raphson on the
non-linear system: A^T * Q = d (mass balance) + Weymouth/Darcy-Weisbach (pressure-flow).

References:
- Osiadacz, A.J. (1987). Simulation and Analysis of Gas Networks.
- Kiuchi, T. (1994). An implicit method for transient gas flow in pipe networks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from core.hydraulics import (
    darcy_weisbach_dp, weymouth_dp, gas_velocity, friction_factor,
    equivalent_hydraulic_diameter, pressure_drop_with_looping
)
from core.thermodynamics import blend_properties, GasProperties, CH4, H2


class NodeType(Enum):
    SUPPLY = "supply"           # Entry point (gas field, import, compressor discharge)
    DEMAND = "demand"           # Exit point (city gate, power plant, industrial)
    JUNCTION = "junction"       # Pipeline intersection
    COMPRESSOR = "compressor"   # Compressor station node
    STORAGE = "storage"         # Underground storage / linepack


class EdgeType(Enum):
    PIPE = "pipe"
    COMPRESSOR = "compressor"
    VALVE = "valve"
    SHORT_PIPE = "short_pipe"   # Negligible pressure drop


@dataclass(frozen=True)
class NetworkNode:
    node_id: str
    name: str
    node_type: NodeType
    lat: float = 0.0
    lon: float = 0.0
    elevation_m: float = 0.0
    # For supply nodes
    pressure_bar: Optional[float] = None      # Fixed pressure (slack node)
    flow_sm3h: Optional[float] = None         # Fixed flow
    # For demand nodes
    demand_sm3h: float = 0.0                  # Known demand
    min_pressure_bar: float = 20.0            # Minimum delivery pressure
    # For compressor nodes
    compressor_id: Optional[str] = None


@dataclass(frozen=True)
class NetworkEdge:
    edge_id: str
    from_node: str
    to_node: str
    edge_type: EdgeType = EdgeType.PIPE
    length_km: float = 1.0
    diameter_mm: float = 500.0
    roughness_mm: float = 0.0457
    # Looping
    is_looped: bool = False
    loop_diameter_mm: Optional[float] = None
    loop_length_km: Optional[float] = None
    # Compressor
    compressor_id: Optional[str] = None
    # Valve/regulator
    pressure_setpoint_bar: Optional[float] = None
    # Status
    is_active: bool = True


@dataclass(frozen=True)
class CompressorStation:
    compressor_id: str
    name: str
    suction_node: str
    discharge_node: str
    # Performance curve coefficients (polynomial fit: power = f(flow, pressure_ratio))
    # Power (MW) = a0 + a1*Q + a2*Q^2 + a3*PR + a4*PR^2 + a5*Q*PR
    curve_coeffs: Dict[str, float] = field(default_factory=dict)
    # Operating limits
    max_flow_sm3h: float = 1_000_000
    min_flow_sm3h: float = 50_000
    max_pressure_ratio: float = 2.0
    min_pressure_ratio: float = 1.0
    max_suction_pressure_bar: float = 80.0
    min_suction_pressure_bar: float = 20.0
    max_discharge_pressure_bar: float = 85.0
    # Efficiency
    isentropic_efficiency: float = 0.85
    mechanical_efficiency: float = 0.98
    # Driver
    driver_type: str = "gas_turbine"  # gas_turbine, electric, reciprocating
    max_power_mw: float = 50.0


class GasNetwork:
    """
    Gas transmission network with hydraulic solver.
    
    Nodes: supply, demand, junction, compressor, storage
    Edges: pipes, compressors, valves
    
    Solves: Nodal pressures (bar) and edge flows (Sm3/h)
    """
    
    def __init__(self, name: str = "Gas Network"):
        self.name = name
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: Dict[str, NetworkEdge] = {}
        self.compressors: Dict[str, CompressorStation] = {}
        self.graph = nx.DiGraph()
        
        # Gas properties (set per scenario)
        self.h2_fraction: float = 0.0
        self.temperature_k: float = 283.15
        self.base_pressure_bar: float = 1.01325
        self.base_temperature_k: float = 273.15
        
        # Solver settings
        self.tolerance: float = 1e-4
        self.max_iterations: int = 50
        self.relaxation: float = 0.7
        
        # Results
        self.node_pressures: Dict[str, float] = {}
        self.edge_flows: Dict[str, float] = {}
        self.converged: bool = False
        self.iterations: int = 0
        self.residual_norm: float = 0.0
    
    def add_node(self, node: NetworkNode):
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **node.__dict__)
    
    def add_edge(self, edge: NetworkEdge):
        self.edges[edge.edge_id] = edge
        edge_dict = edge.__dict__.copy()
        edge_dict.pop('edge_id', None)  # Remove edge_id to avoid duplicate
        self.graph.add_edge(edge.from_node, edge.to_node, edge_id=edge.edge_id, **edge_dict)
    
    def add_compressor(self, compressor: CompressorStation):
        self.compressors[compressor.compressor_id] = compressor
        # Add compressor as an edge
        comp_edge = NetworkEdge(
            edge_id=f"comp_{compressor.compressor_id}",
            from_node=compressor.suction_node,
            to_node=compressor.discharge_node,
            edge_type=EdgeType.COMPRESSOR,
            compressor_id=compressor.compressor_id,
        )
        self.add_edge(comp_edge)
    
    def set_gas_properties(self, h2_fraction: float, temperature_k: float = 283.15):
        """Set gas composition for the simulation."""
        self.h2_fraction = h2_fraction
        self.temperature_k = temperature_k
    
    def _get_gas_props(self, pressure_bar: float) -> dict:
        """Get gas properties at given pressure."""
        P_pa = pressure_bar * 1e5
        return blend_properties(self.h2_fraction, self.temperature_k, P_pa)
    
    def _pipe_flow_equation(self, P_from: float, P_to: float, edge: NetworkEdge,
                             rho: float, mu: float, Z: float) -> float:
        """
        Calculate flow through a pipe given upstream/downstream pressures.
        Uses Weymouth equation (standard for high-pressure gas).
        Returns flow in Sm3/h.
        """
        if not edge.is_active:
            return 0.0
        
        D_m = edge.diameter_mm / 1000.0
        L_m = edge.length_km * 1000.0
        
        if edge.is_looped and edge.loop_diameter_mm and edge.loop_length_km:
            # Use equivalent diameter for looped section
            D_eq = equivalent_hydraulic_diameter(edge.diameter_mm, edge.loop_diameter_mm)
            # Simplified: treat as equivalent diameter for looped portion
            # For full accuracy, would split into two segments
            D_m = D_eq / 1000.0
        
        # Weymouth: Q = C * (P1^2 - P2^2)^0.5 * D^(8/3) / (G * T * Z * L)^0.5
        # In SI units with Sm3/h:
        P1 = P_from * 1e5
        P2 = P_to * 1e5
        
        if P1 <= P2:
            return 0.0
        
        # Gas gravity relative to air
        props = self._get_gas_props((P_from + P_to) / 2)
        G = props["M_blend"] / 28.97  # Air MW = 28.97
        
        # Weymouth constant (SI -> Sm3/h)
        # Q = 353.6 * (Pb/Tb) * (P1^2 - P2^2)^0.5 * D^(8/3) / (G * T * Z * L)^0.5
        # Where Pb=1.01325 bar, Tb=273.15 K, D in m, L in km, P in bar
        Pb = 1.01325
        Tb = 273.15
        
        C = 353.6 * (Pb / Tb) * (D_m ** (8/3)) / (G * self.temperature_k * Z * edge.length_km) ** 0.5
        Q = C * ((P_from ** 2 - P_to ** 2) ** 0.5)
        
        return max(Q, 0.0)
    
    def _pipe_pressure_drop(self, Q: float, edge: NetworkEdge, rho: float, mu: float) -> float:
        """Calculate pressure drop for given flow (Darcy-Weisbach). Returns delta P in bar."""
        if not edge.is_active or Q <= 0:
            return 0.0
        
        if edge.is_looped and edge.loop_diameter_mm and edge.loop_length_km:
            return pressure_drop_with_looping(
                Q / 3600.0,  # Convert Sm3/h to m3/s (approximate)
                edge.length_km,
                edge.loop_length_km,
                edge.diameter_mm,
                edge.loop_diameter_mm,
                rho, mu
            )
        else:
            return darcy_weisbach_dp(Q / 3600.0, edge.length_km, edge.diameter_mm, rho, mu)
    
    def _compressor_relation(self, P_suction: float, Q: float, comp: CompressorStation) -> float:
        """
        Compressor pressure-flow relationship.
        Returns discharge pressure for given suction pressure and flow.
        Uses polynomial performance curve if available, else isentropic model.
        """
        if comp.curve_coeffs:
            # Polynomial: P_disch = f(Q, P_suct)
            coeffs = comp.curve_coeffs
            PR = coeffs.get('a0', 1.0) + coeffs.get('a1', 0)*Q + coeffs.get('a2', 0)*Q**2
            PR = max(comp.min_pressure_ratio, min(comp.max_pressure_ratio, PR))
            return P_suction * PR
        else:
            # Simple isentropic model with efficiency
            # Assume target pressure ratio based on flow (simplified)
            # Real implementation would use compressor map
            target_PR = 1.5  # Default
            return min(P_suction * target_PR, comp.max_discharge_pressure_bar)
    
    def _compressor_power(self, P_suction: float, P_discharge: float, Q: float, 
                          comp: CompressorStation, rho: float, Z: float) -> float:
        """Calculate compressor power in MW."""
        from core.hydraulics import compression_work_mw
        return compression_work_mw(
            Q / 3600.0,  # m3/s
            P_suction, P_discharge,
            rho, Z, self.temperature_k,
            comp.isentropic_efficiency, comp.mechanical_efficiency
        )
    
    def solve_steady_state(self) -> Dict:
        """
        Solve steady-state hydraulic network using Newton-Raphson.
        
        Unknowns: nodal pressures (except slack nodes)
        Equations: 
        - Mass balance at each node: sum(Q_in) - sum(Q_out) = demand
        - Pressure-flow relations for each edge
        - Compressor relations
        
        Returns dict with results.
        """
        # Identify unknown pressure nodes (non-slack)
        slack_nodes = [nid for nid, node in self.nodes.items() 
                       if node.node_type == NodeType.SUPPLY and node.pressure_bar is not None]
        unknown_nodes = [nid for nid in self.nodes.keys() if nid not in slack_nodes]
        
        n_unknown = len(unknown_nodes)
        if n_unknown == 0:
            # All pressures fixed
            self.node_pressures = {nid: node.pressure_bar for nid, node in self.nodes.items()}
            self._compute_flows_from_pressures()
            self.converged = True
            return self._get_results()
        
        node_to_idx = {nid: i for i, nid in enumerate(unknown_nodes)}
        
        # Initial guess: linear interpolation between min/max known pressures
        known_pressures = [self.nodes[nid].pressure_bar for nid in slack_nodes if self.nodes[nid].pressure_bar]
        P_avg = np.mean(known_pressures) if known_pressures else 50.0
        P = np.full(n_unknown, P_avg)
        
        # Newton-Raphson iteration
        for iteration in range(self.max_iterations):
            # Compute residuals: mass balance at each unknown node
            F = np.zeros(n_unknown)
            J = np.zeros((n_unknown, n_unknown))
            
            # Current pressures (all nodes)
            P_all = {}
            for nid in slack_nodes:
                P_all[nid] = self.nodes[nid].pressure_bar
            for i, nid in enumerate(unknown_nodes):
                P_all[nid] = P[i]
            
            # Compute flows for all edges
            flows = {}
            for edge_id, edge in self.edges.items():
                if not edge.is_active:
                    flows[edge_id] = 0.0
                    continue
                
                P_fr = P_all[edge.from_node]
                P_to = P_all[edge.to_node]
                
                if edge.edge_type == EdgeType.PIPE:
                    # Get gas properties at average pressure
                    P_avg_edge = (P_fr + P_to) / 2
                    props = self._get_gas_props(P_avg_edge)
                    Q = self._pipe_flow_equation(P_fr, P_to, edge, props["rho_blend"], 
                                                  props["mu_blend"], props["Z_blend"])
                    flows[edge_id] = Q
                elif edge.edge_type == EdgeType.COMPRESSOR:
                    comp = self.compressors.get(edge.compressor_id)
                    if comp:
                        P_disch = self._compressor_relation(P_fr, 0, comp)  # Initial guess
                        # Will be solved implicitly
                        flows[edge_id] = 0.0  # Placeholder
            
            # Mass balance residuals
            for i, nid in enumerate(unknown_nodes):
                node = self.nodes[nid]
                net_flow = 0.0
                
                # Inflows
                for pred in self.graph.predecessors(nid):
                    edge_data = self.graph[pred][nid]
                    eid = edge_data.get('edge_id')
                    if eid in flows:
                        net_flow += flows[eid]
                
                # Outflows
                for succ in self.graph.successors(nid):
                    edge_data = self.graph[nid][succ]
                    eid = edge_data.get('edge_id')
                    if eid in flows:
                        net_flow -= flows[eid]
                
                # Demand (positive = consumption)
                net_flow -= node.demand_sm3h
                F[i] = net_flow
            
            # Compute Jacobian numerically (simpler, robust)
            h = 1e-4
            for j in range(n_unknown):
                P_pert = P.copy()
                P_pert[j] += h * max(1.0, abs(P[j]))
                
                P_all_pert = P_all.copy()
                P_all_pert[unknown_nodes[j]] = P_pert[j]
                
                # Recompute flows with perturbed pressure
                F_pert = np.zeros(n_unknown)
                for i, nid in enumerate(unknown_nodes):
                    net_flow = 0.0
                    for pred in self.graph.predecessors(nid):
                        edge_data = self.graph[pred][nid]
                        eid = edge_data.get('edge_id')
                        edge = self.edges.get(eid)
                        if edge and edge.edge_type == EdgeType.PIPE and edge.is_active:
                            P_fr = P_all_pert[edge.from_node]
                            P_to = P_all_pert[edge.to_node]
                            P_avg_e = (P_fr + P_to) / 2
                            props = self._get_gas_props(P_avg_e)
                            Q = self._pipe_flow_equation(P_fr, P_to, edge, props["rho_blend"], 
                                                          props["mu_blend"], props["Z_blend"])
                            net_flow += Q
                    for succ in self.graph.successors(nid):
                        edge_data = self.graph[nid][succ]
                        eid = edge_data.get('edge_id')
                        edge = self.edges.get(eid)
                        if edge and edge.edge_type == EdgeType.PIPE and edge.is_active:
                            P_fr = P_all_pert[edge.from_node]
                            P_to = P_all_pert[edge.to_node]
                            P_avg_e = (P_fr + P_to) / 2
                            props = self._get_gas_props(P_avg_e)
                            Q = self._pipe_flow_equation(P_fr, P_to, edge, props["rho_blend"], 
                                                          props["mu_blend"], props["Z_blend"])
                            net_flow -= Q
                    net_flow -= self.nodes[nid].demand_sm3h
                    F_pert[i] = net_flow
                
                J[:, j] = (F_pert - F) / (h * max(1.0, abs(P[j])))
            
            # Solve linear system
            try:
                delta_P = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # Singular - use pseudo-inverse
                delta_P = np.linalg.lstsq(J, -F, rcond=None)[0]
            
            # Apply with relaxation
            P_new = P + self.relaxation * delta_P
            
            # Enforce physical bounds
            P_new = np.maximum(P_new, 1.0)  # Min 1 bar
            P_new = np.minimum(P_new, 100.0)  # Max 100 bar
            
            # Check convergence
            residual_norm = np.linalg.norm(F)
            self.residual_norm = residual_norm
            
            if residual_norm < self.tolerance:
                P = P_new
                self.iterations = iteration + 1
                self.converged = True
                break
            
            P = P_new
        
        # Store results
        for i, nid in enumerate(unknown_nodes):
            self.node_pressures[nid] = P[i]
        for nid in slack_nodes:
            self.node_pressures[nid] = self.nodes[nid].pressure_bar
        
        self._compute_flows_from_pressures()
        return self._get_results()
    
    def _compute_flows_from_pressures(self):
        """Compute all edge flows from converged pressures."""
        self.edge_flows = {}
        for edge_id, edge in self.edges.items():
            if not edge.is_active:
                self.edge_flows[edge_id] = 0.0
                continue
            
            P_fr = self.node_pressures[edge.from_node]
            P_to = self.node_pressures[edge.to_node]
            
            if edge.edge_type == EdgeType.PIPE:
                P_avg = (P_fr + P_to) / 2
                props = self._get_gas_props(P_avg)
                Q = self._pipe_flow_equation(P_fr, P_to, edge, props["rho_blend"], 
                                              props["mu_blend"], props["Z_blend"])
                self.edge_flows[edge_id] = Q
            elif edge.edge_type == EdgeType.COMPRESSOR:
                # Flow determined by network balance
                # In steady state, compressor flow = flow through connecting pipes
                # Simplified: use upstream pipe flow
                self.edge_flows[edge_id] = 0.0  # Will be set by continuity
    
    def _get_results(self) -> dict:
        """Return solver results."""
        # Check pressure violations
        violations = []
        for nid, node in self.nodes.items():
            if node.node_type == NodeType.DEMAND:
                p = self.node_pressures.get(nid, 0)
                if p < node.min_pressure_bar:
                    violations.append({
                        "node": nid,
                        "name": node.name,
                        "pressure": p,
                        "min_required": node.min_pressure_bar,
                        "deficit": node.min_pressure_bar - p,
                    })
        
        # Compressor power summary
        compressor_power = {}
        for cid, comp in self.compressors.items():
            edge_id = f"comp_{cid}"
            P_suc = self.node_pressures.get(comp.suction_node, 0)
            P_dis = self.node_pressures.get(comp.discharge_node, 0)
            Q = self.edge_flows.get(edge_id, 0)
            if Q > 0 and P_dis > P_suc:
                props = self._get_gas_props(P_suc)
                power = self._compressor_power(P_suc, P_dis, Q, comp, 
                                               props["rho_blend"], props["Z_blend"])
                compressor_power[cid] = power
        
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
            "node_pressures": self.node_pressures,
            "edge_flows": self.edge_flows,
            "pressure_violations": violations,
            "compressor_power_mw": compressor_power,
            "total_compression_power_mw": sum(compressor_power.values()),
        }
    
    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Export results as DataFrames."""
        import pandas as pd
        
        nodes_df = pd.DataFrame([{
            "node_id": nid,
            "name": node.name,
            "type": node.node_type.value,
            "pressure_bar": self.node_pressures.get(nid, np.nan),
            "demand_sm3h": node.demand_sm3h,
            "min_pressure_bar": node.min_pressure_bar,
            "lat": node.lat,
            "lon": node.lon,
        } for nid, node in self.nodes.items()])
        
        edges_df = pd.DataFrame([{
            "edge_id": eid,
            "from_node": edge.from_node,
            "to_node": edge.to_node,
            "type": edge.edge_type.value,
            "length_km": edge.length_km,
            "diameter_mm": edge.diameter_mm,
            "flow_sm3h": self.edge_flows.get(eid, 0),
            "pressure_drop_bar": (self.node_pressures.get(edge.from_node, 0) - 
                                  self.node_pressures.get(edge.to_node, 0)),
            "velocity_m_s": gas_velocity(self.edge_flows.get(eid, 0) / 3600.0, edge.diameter_mm) 
                           if self.edge_flows.get(eid, 0) > 0 else 0,
            "is_looped": edge.is_looped,
        } for eid, edge in self.edges.items()])
        
        return nodes_df, edges_df


def create_ireland_network() -> GasNetwork:
    """Create the Gas Networks Ireland transmission network topology."""
    net = GasNetwork("GNI Transmission Network")
    
    # Major nodes (simplified topology)
    # Entry points
    net.add_node(NetworkNode("inchicore", "Inchicore (Dublin)", NodeType.SUPPLY, 
                            lat=53.3298, lon=-6.3158, pressure_bar=70.0, flow_sm3h=2_000_000))
    net.add_node(NetworkNode("whitegate", "Whitegate (Cork)", NodeType.SUPPLY,
                            lat=51.8333, lon=-8.2500, pressure_bar=70.0, flow_sm3h=1_500_000))
    net.add_node(NetworkNode("galway", "Galway Entry", NodeType.SUPPLY,
                            lat=53.2707, lon=-9.0568, pressure_bar=65.0, flow_sm3h=800_000))
    net.add_node(NetworkNode("killarney", "Killarney Entry", NodeType.SUPPLY,
                            lat=52.0599, lon=-9.5043, pressure_bar=60.0, flow_sm3h=300_000))
    
    # Demand centers
    net.add_node(NetworkNode("dublin_cg", "Dublin City Gate", NodeType.DEMAND,
                            lat=53.3498, lon=-6.2603, demand_sm3h=1_800_000, min_pressure_bar=35.0))
    net.add_node(NetworkNode("cork_cg", "Cork City Gate", NodeType.DEMAND,
                            lat=51.8985, lon=-8.4756, demand_sm3h=600_000, min_pressure_bar=30.0))
    net.add_node(NetworkNode("limerick_cg", "Limerick City Gate", NodeType.DEMAND,
                            lat=52.6638, lon=-8.6267, demand_sm3h=400_000, min_pressure_bar=30.0))
    net.add_node(NetworkNode("galway_cg", "Galway City Gate", NodeType.DEMAND,
                            lat=53.2707, lon=-9.0568, demand_sm3h=300_000, min_pressure_bar=30.0))
    net.add_node(NetworkNode("waterford_cg", "Waterford City Gate", NodeType.DEMAND,
                            lat=52.2593, lon=-7.1101, demand_sm3h=250_000, min_pressure_bar=28.0))
    net.add_node(NetworkNode("sligo_cg", "Sligo City Gate", NodeType.DEMAND,
                            lat=54.2766, lon=-8.4761, demand_sm3h=150_000, min_pressure_bar=25.0))
    
    # Junctions
    net.add_node(NetworkNode("j1", "Junction 1 (Midlands)", NodeType.JUNCTION,
                            lat=53.2000, lon=-7.5000))
    net.add_node(NetworkNode("j2", "Junction 2 (South Midlands)", NodeType.JUNCTION,
                            lat=52.5000, lon=-8.0000))
    net.add_node(NetworkNode("j3", "Junction 3 (West)", NodeType.JUNCTION,
                            lat=53.0000, lon=-8.5000))
    
    # Compressor stations
    net.add_compressor(CompressorStation(
        compressor_id="comp_inchicore",
        name="Inchicore Compressor Station",
        suction_node="inchicore",
        discharge_node="j1",
        max_power_mw=45.0,
        max_pressure_ratio=1.8,
    ))
    net.add_compressor(CompressorStation(
        compressor_id="comp_ballytrasna",
        name="Ballytrasna Compressor Station",
        suction_node="j2",
        discharge_node="j1",
        max_power_mw=30.0,
        max_pressure_ratio=1.6,
    ))
    net.add_compressor(CompressorStation(
        compressor_id="comp_galway",
        name="Galway Compressor Station",
        suction_node="galway",
        discharge_node="j3",
        max_power_mw=25.0,
        max_pressure_ratio=1.5,
    ))
    
    # Pipeline segments (simplified topology)
    pipes = [
        # Main Dublin-Cork corridor
        ("pipe_1", "inchicore", "j1", 50, 500),
        ("pipe_2", "j1", "j2", 80, 500),
        ("pipe_3", "j2", "whitegate", 90, 500),
        ("pipe_4", "j2", "cork_cg", 20, 400),
        ("pipe_5", "j2", "limerick_cg", 30, 300),
        # Galway-Dublin
        ("pipe_6", "galway", "j3", 40, 400),
        ("pipe_7", "j3", "j1", 70, 400),
        ("pipe_8", "j3", "galway_cg", 10, 300),
        # Spurs
        ("pipe_9", "killarney", "whitegate", 50, 300),
        ("pipe_10", "j2", "waterford_cg", 60, 300),
        ("pipe_11", "j3", "sligo_cg", 80, 300),
    ]
    
    for eid, from_n, to_n, length, dia in pipes:
        net.add_edge(NetworkEdge(
            edge_id=eid,
            from_node=from_n,
            to_node=to_n,
            length_km=length,
            diameter_mm=dia,
        ))
    
    return net