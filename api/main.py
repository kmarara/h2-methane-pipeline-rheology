"""
FastAPI backend for programmatic access to H2 Pipeline Digital Twin.

Provides REST API for:
- Scenario simulation
- Network analysis
- Optimization
- Results export
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid
import asyncio
from datetime import datetime
import json

# Import core modules
from core import blend_properties, simulate_pipeline_with_compressors, find_required_looping
from core.network.solver import GasNetwork, create_ireland_network, NetworkNode, NetworkEdge, NodeType, EdgeType, CompressorStation
from core.transient.simulator import TransientSimulator, TransientScenario, create_blending_ramp_scenario
from core.economics.cost_model import estimate_ireland_h2_project, EconomicAnalysis
from core.compressor.maps import create_standard_compressor, DriverType


app = FastAPI(
    title="H₂ Pipeline Digital Twin API",
    description="Programmatic interface for hydrogen blending analysis in gas transmission networks",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (use Redis in production)
jobs: Dict[str, Dict] = {}


# ──────────────────────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────────────────────

class GasComposition(BaseModel):
    h2_fraction: float = Field(..., ge=0, le=1, description="H2 molar fraction (0-1)")
    temperature_c: float = Field(10, ge=-20, le=50, description="Gas temperature (°C)")
    pressure_bar: float = Field(70, ge=1, le=100, description="Reference pressure (bar)")


class PipelineGeometry(BaseModel):
    length_km: float = Field(..., gt=0, le=1000)
    diameter_mm: float = Field(..., gt=0, le=2000)
    roughness_mm: float = Field(0.0457, ge=0, le=1)
    elevation_change_m: float = Field(0, ge=-1000, le=1000)


class SteadyStateRequest(BaseModel):
    composition: GasComposition
    pipeline: PipelineGeometry
    inlet_pressure_bar: float = Field(70, ge=10, le=100)
    min_outlet_pressure_bar: float = Field(20, ge=5, le=50)
    reference_flow_sm3h: float = Field(1_000_000, gt=0)
    reference_hv_vol_mj_m3: float = Field(35.8, gt=0)  # CH4 at STP


class SteadyStateResponse(BaseModel):
    success: bool
    outlet_pressure_bar: float
    pressure_drop_bar: float
    equivalent_flow_sm3h: float
    flow_increase_pct: float
    velocity_m_s: float
    exceedance: bool
    compressor_solution: Optional[Dict] = None
    looping_solution: Optional[Dict] = None


class NetworkNodeRequest(BaseModel):
    node_id: str
    name: str
    node_type: str  # supply, demand, junction, compressor, storage
    lat: float = 0
    lon: float = 0
    pressure_bar: Optional[float] = None
    demand_sm3h: float = 0
    min_pressure_bar: float = 20


class NetworkEdgeRequest(BaseModel):
    edge_id: str
    from_node: str
    to_node: str
    edge_type: str = "pipe"  # pipe, compressor, valve
    length_km: float = 1
    diameter_mm: float = 500
    is_looped: bool = False
    loop_diameter_mm: Optional[float] = None
    loop_length_km: Optional[float] = None
    compressor_id: Optional[str] = None


class CompressorRequest(BaseModel):
    compressor_id: str
    name: str
    suction_node: str
    discharge_node: str
    max_power_mw: float = 50
    max_pressure_ratio: float = 1.8


class NetworkSolveRequest(BaseModel):
    nodes: List[NetworkNodeRequest]
    edges: List[NetworkEdgeRequest]
    compressors: List[CompressorRequest] = []
    h2_fraction: float = Field(0, ge=0, le=1)
    temperature_c: float = 10


class TransientRequest(BaseModel):
    network_id: str = "ireland"
    duration_hours: float = Field(24, gt=0, le=168)
    initial_h2: float = Field(0, ge=0, le=1)
    final_h2: float = Field(0.2, ge=0, le=0.5)
    ramp_hours: float = Field(6, gt=0)
    dt_hours: float = Field(0.25, gt=0, le=1)


class EconomicRequest(BaseModel):
    network_results: Dict
    h2_target_pct: float = Field(20, ge=0, le=50)
    project_life_years: int = Field(25, ge=10, le=50)
    discount_rate: float = Field(0.06, ge=0, le=0.2)
    gas_price_eur_mwh: float = Field(35, ge=10, le=200)
    carbon_price_eur_t: float = Field(85, ge=0, le=500)


class JobResponse(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: str
    result: Optional[Dict] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "H₂ Pipeline Digital Twin API",
        "version": "2.0.0",
        "description": "Hydrogen blending analysis for gas transmission networks",
        "endpoints": {
            "thermodynamics": "/api/v1/thermodynamics/blend",
            "steady_state": "/api/v1/simulate/steady-state",
            "network": "/api/v1/network/solve",
            "transient": "/api/v1/simulate/transient",
            "economics": "/api/v1/economics/analyze",
            "jobs": "/api/v1/jobs/{job_id}",
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ──────────────────────────────────────────────────────────────
# Thermodynamics
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/thermodynamics/blend")
async def calculate_blend_properties(composition: GasComposition):
    """Calculate blend thermodynamic properties."""
    try:
        T_k = composition.temperature_c + 273.15
        P_pa = composition.pressure_bar * 1e5
        props = blend_properties(composition.h2_fraction, T_k, P_pa)
        
        return {
            "success": True,
            "molecular_weight_g_mol": props["M_blend"],
            "compressibility_factor": props["Z_blend"],
            "density_kg_m3": props["rho_blend"],
            "viscosity_pas": props["mu_blend"],
            "viscosity_uPas": props["mu_blend"] * 1e6,
            "heating_value_mass_mj_kg": props["hv_mass_blend"],
            "heating_value_vol_mj_m3_stp": props["hv_vol_blend"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Steady-State Pipeline Simulation
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/simulate/steady-state", response_model=SteadyStateResponse)
async def simulate_steady_state(req: SteadyStateRequest):
    """Simulate single pipeline steady-state hydraulics with H2 blend."""
    try:
        T_k = req.composition.temperature_c + 273.15
        P_pa = req.inlet_pressure_bar * 1e5
        h2_frac = req.composition.h2_fraction
        
        # Blend properties at inlet
        props = blend_properties(h2_frac, T_k, P_pa)
        
        # Energy-equivalent flow
        Q_ref = req.reference_flow_sm3h / 3600.0  # m3/s
        Q_equiv = Q_ref * (req.reference_hv_vol_mj_m3 / props["hv_vol_blend"])
        
        # Pressure drop (Darcy-Weisbach)
        dp = darcy_weisbach_dp(
            Q_equiv, req.pipeline.length_km, req.pipeline.diameter_mm,
            props["rho_blend"], props["mu_blend"]
        )
        
        P_out = req.inlet_pressure_bar - dp
        exceedance = P_out < req.min_outlet_pressure_bar
        
        # Velocity
        velocity = gas_velocity(Q_equiv, req.pipeline.diameter_mm)
        
        response = SteadyStateResponse(
            success=True,
            outlet_pressure_bar=round(P_out, 2),
            pressure_drop_bar=round(dp, 2),
            equivalent_flow_sm3h=round(Q_equiv * 3600, 0),
            flow_increase_pct=round((Q_equiv / Q_ref - 1) * 100, 1),
            velocity_m_s=round(velocity, 2),
            exceedance=exceedance,
        )
        
        # If exceedance, compute mitigation options
        if exceedance:
            # Compressor solution
            comp = simulate_pipeline_with_compressors(
                Q=Q_equiv, L_total=req.pipeline.length_km, D=req.pipeline.diameter_mm,
                rho=props["rho_blend"], mu=props["mu_blend"], Z=props["Z_blend"],
                T=T_k, P_in=req.inlet_pressure_bar, P_min=req.min_outlet_pressure_bar
            )
            response.compressor_solution = {
                "num_compressors": comp["num_compressors"],
                "total_power_mw": round(comp["total_power_mw"], 2),
                "locations_km": [round(x, 1) for x in comp["compressor_locations_km"]],
            }
            
            # Looping solution
            loop = find_required_looping(
                Q=Q_equiv, L_total=req.pipeline.length_km, D_main=req.pipeline.diameter_mm,
                rho=props["rho_blend"], mu=props["mu_blend"],
                P_in=req.inlet_pressure_bar, P_min=req.min_outlet_pressure_bar
            )
            response.looping_solution = {
                "loop_percentage": round(loop["loop_percentage"] * 100, 1),
                "loop_length_km": round(loop["loop_length_km"], 1),
                "success": loop["success"],
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Network Solver
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/network/solve")
async def solve_network(req: NetworkSolveRequest):
    """Solve full network hydraulics."""
    try:
        net = GasNetwork("API Network")
        net.set_gas_properties(req.h2_fraction, req.temperature_c + 273.15)
        
        # Add nodes
        for n in req.nodes:
            ntype = NodeType(n.node_type.lower())
            node = NetworkNode(
                node_id=n.node_id,
                name=n.name,
                node_type=ntype,
                lat=n.lat,
                lon=n.lon,
                pressure_bar=n.pressure_bar,
                demand_sm3h=n.demand_sm3h,
                min_pressure_bar=n.min_pressure_bar,
            )
            net.add_node(node)
        
        # Add edges
        for e in req.edges:
            etype = EdgeType(e.edge_type.lower())
            edge = NetworkEdge(
                edge_id=e.edge_id,
                from_node=e.from_node,
                to_node=e.to_node,
                edge_type=etype,
                length_km=e.length_km,
                diameter_mm=e.diameter_mm,
                is_looped=e.is_looped,
                loop_diameter_mm=e.loop_diameter_mm,
                loop_length_km=e.loop_length_km,
                compressor_id=e.compressor_id,
            )
            net.add_edge(edge)
        
        # Add compressors
        for c in req.compressors:
            comp = CompressorStation(
                compressor_id=c.compressor_id,
                name=c.name,
                suction_node=c.suction_node,
                discharge_node=c.discharge_node,
                max_power_mw=c.max_power_mw,
                max_pressure_ratio=c.max_pressure_ratio,
            )
            net.add_compressor(comp)
        
        # Solve
        results = net.solve_steady_state()
        nodes_df, edges_df = net.to_dataframe()
        
        return {
            "success": results["converged"],
            "iterations": results["iterations"],
            "residual": results["residual_norm"],
            "node_pressures": {k: round(v, 2) for k, v in results["node_pressures"].items()},
            "edge_flows": {k: round(v, 0) for k, v in results["edge_flows"].items()},
            "violations": results["pressure_violations"],
            "compressor_power_mw": {k: round(v, 2) for k, v in results["compressor_power_mw"].items()},
            "nodes": nodes_df.to_dict("records"),
            "edges": edges_df.to_dict("records"),
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/network/ireland")
async def get_ireland_network():
    """Get pre-built Ireland network."""
    net = create_ireland_network()
    nodes_df, edges_df = net.to_dataframe()
    return {
        "nodes": nodes_df.to_dict("records"),
        "edges": edges_df.to_dict("records"),
    }


# ──────────────────────────────────────────────────────────────
# Transient Simulation (Async Job)
# ──────────────────────────────────────────────────────────────

async def run_transient_job(job_id: str, req: TransientRequest):
    """Background task for transient simulation."""
    jobs[job_id]["status"] = "running"
    
    try:
        net = create_ireland_network()
        net.set_gas_properties(req.initial_h2, 283.15)
        
        scenario = create_blending_ramp_scenario(
            net, req.duration_hours, req.initial_h2, req.final_h2, req.ramp_hours
        )
        
        sim = TransientSimulator(net)
        results = sim.simulate(scenario, dt_hours=req.dt_hours)
        
        # Convert to serializable format
        jobs[job_id]["result"] = {
            "time_hours": results.time_hours.tolist(),
            "node_pressures": {k: v.tolist() for k, v in results.node_pressures.items()},
            "node_h2_fractions": {k: v.tolist() for k, v in results.node_h2_fractions.items()},
            "edge_flows": {k: v.tolist() for k, v in results.edge_flows.items()},
            "total_linepack_kg": results.total_linepack_kg.tolist(),
        }
        jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/api/v1/simulate/transient", response_model=JobResponse)
async def start_transient_simulation(req: TransientRequest, background_tasks: BackgroundTasks):
    """Start transient simulation as background job."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "type": "transient",
        "request": req.dict(),
    }
    
    background_tasks.add_task(run_transient_job, job_id, req)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"],
    )


# ──────────────────────────────────────────────────────────────
# Economics
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/economics/analyze")
async def analyze_economics(req: EconomicRequest):
    """Economic analysis of H2 blending project."""
    try:
        eco = EconomicAnalysis(
            project_life_years=req.project_life_years,
            discount_rate=req.discount_rate,
        )
        eco.opex.gas_price_eur_mwh = req.gas_price_eur_mwh
        eco.opex.carbon_price_eur_tonne = req.carbon_price_eur_t
        
        results = estimate_ireland_h2_project(req.network_results, req.h2_target_pct)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Job Management
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobResponse(**job)


@app.get("/api/v1/jobs")
async def list_jobs():
    """List all jobs."""
    return {"jobs": [JobResponse(**j) for j in jobs.values()]}


# ──────────────────────────────────────────────────────────────
# Ireland-Specific Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/ireland/corridors")
async def get_ireland_corridors():
    """Get Ireland pipeline corridors."""
    from cases.ireland import get_corridors
    return {"corridors": get_corridors()}


@app.get("/api/v1/ireland/policy")
async def get_ireland_policy():
    """Get Ireland policy context."""
    from cases.ireland import get_policy_context
    return {"policy": get_policy_context()}


# ──────────────────────────────────────────────────────────────
# Run Server
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)