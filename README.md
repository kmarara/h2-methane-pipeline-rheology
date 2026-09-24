# H₂ Pipeline Digital Twin — Core Engine + Regional Case Studies

A modular computational platform for hydrogen blending analysis in natural gas transmission networks. The **core physics engine** is geography-agnostic; **case studies** provide region-specific configuration (corridors, steel grades, policy context, risk thresholds).

**New in v2.0:** Interactive Streamlit dashboard (6 tabs), FastAPI REST backend, Monte Carlo UQ, Network Solver, Transient Simulator, Economics Dashboard.

---

## Why This Exists

Gas networks globally face the same physics challenge: **H₂ has 1/3 the volumetric energy density of CH₄ but lower viscosity and different compressibility**. Blending changes pressure drop, compressor power, materials degradation, and flow requirements — all simultaneously. This tool quantifies those coupled effects using industry-standard equations.

---

## Architecture

```
h2-pipeline-digital-twin/
├── app.py                           # Streamlit dashboard (6 tabs)
├── api/main.py                      # FastAPI REST backend
├── core/                            # Physics engine (zero geography)
│   ├── thermodynamics.py            # Peng-Robinson EOS, Wilke viscosity
│   ├── hydraulics.py                # Darcy-Weisbach, Weymouth, compression, looping
│   ├── materials.py                 # ASME B31.12 / IGEM TD/13 risk tiers
│   ├── pipeline.py                  # Geographic network + Folium mapping
│   ├── network/solver.py            # Graph-based Newton-Raphson solver
│   ├── gis/importer.py              # Shapefile/GeoJSON ingestion
│   ├── compressor/maps.py           # Universal Φ-Ψ maps, driver models
│   ├── transient/simulator.py       # Implicit FD, linepack, H₂ transport
│   ├── economics/cost_model.py      # CAPEX/OPEX, NPV, LCOT
│   ├── uq/monte_carlo.py            # Parallel Monte Carlo UQ
│   └── __init__.py                  # Unified API
├── cases/ireland/                   # Ireland (GNI) reference case
│   ├── config.yaml                  # Corridors, steel grades, policy, thresholds
│   └── __init__.py                  # Case loader
├── tests/test_core.py               # 21 validation tests (all passing)
└── requirements.txt                 # All dependencies
```

---

## Quick Start

```bash
git clone https://github.com/<your-org>/h2-pipeline-digital-twin.git
cd h2-pipeline-digital-twin

# Virtual environment
python -m venv venv && source venv/bin/activate  # Linux/macOS
# venv\Scripts\Activate.ps1                      # Windows PowerShell

pip install -r requirements.txt

# Option 1: Interactive Dashboard (Streamlit)
streamlit run app.py

# Option 2: REST API (FastAPI)
uvicorn api.main:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs

# Option 3: Python API
python -c "
from core import MonteCarloUQ
uq = MonteCarloUQ()
r = uq.run_economics_mc(200)
uq.print_summary(r)
"
```

Open `http://localhost:8501` for Streamlit UI or `http://localhost:8000/docs` for API docs.

---

## Dashboard Tabs (Streamlit)

| Tab | Features |
|-----|----------|
| **📊 Hydraulics** | Pressure drop vs blend %, energy vs flow trade-off, risk alerts, mitigation (compressors/looping) with live plots |
| **🗺️ Network** | Folium map of GNI corridors, network solver results, node pressures table |
| **⚡ Transient** | Time-dependent blending ramps, linepack tracking, H₂ front propagation |
| **🎲 Uncertainty (MC)** | Monte Carlo UQ with P10/P50/P90, sensitivity tornado charts, distribution histograms |
| **💰 Economics** | CAPEX/OPEX breakdown, NPV, LCOT, sensitivity analysis, pie/bar charts |
| **📋 Export** | CSV/JSON download of full scenario |

---

## Terminal Capabilities (What You See in CLI)

```bash
# Monte Carlo Economics (200 samples, 4 workers)
from core import MonteCarloUQ
uq = MonteCarloUQ(n_workers=4)
r = uq.run_economics_mc(200)
uq.print_summary(r)

# Output:
# ============================================================
# Monte Carlo UQ Summary (200 samples)
# Compute time: 0.3s
# ============================================================
#
# annual_opex_eur:
#   Mean: 51924451.024 ± 4008113.802
#   P10/P50/P90: 46766666.932 / 52116419.818 / 57051920.953
#   Sensitivities: demand_scaling: 0.897, ground_temp_c: 0.122...
#
# npv_eur:
#   Mean: -2288644127.385 ± 54419210.630
#   P10/P50/P90: -2358266490.912 / -2291233622.746 / -2218593412.800
#   Sensitivities: demand_scaling: 0.897, ...
```

```bash
# Steady-State Single Pipeline
from core import simulate_pipeline_with_compressors, find_required_looping
comp = simulate_pipeline_with_compressors(Q=Q_equiv, L_total=220, D=500, ...)
# Returns: num_compressors, total_power_mw, compressor_locations_km, pressure_profile

loop = find_required_looping(Q=Q_equiv, L_total=220, D_main=500, ...)
# Returns: loop_percentage, loop_length_km, P_out, success
```

```bash
# Network Solver (Multi-node)
from core import GasNetwork, create_ireland_network
net = create_ireland_network()
net.set_gas_properties(0.2, 283.15)
results = net.solve_steady_state()
# Returns: node_pressures, edge_flows, pressure_violations, compressor_power_mw
```

```bash
# Transient Blending
from core import TransientSimulator, create_blending_ramp_scenario
scenario = create_blending_ramp_scenario(net, duration_hours=24, final_h2=0.2)
sim = TransientSimulator(net)
results = sim.simulate(scenario)
# Returns: time series of pressures, H2 fractions, linepack, flows
```

```bash
# Economics
from core.economics.cost_model import estimate_ireland_h2_project
results = estimate_ireland_h2_project(network_results, h2_target_pct=20)
# Returns: CAPEX/OPEX breakdown, NPV, LCOT, sensitivity
```

---

## Why Ireland as Reference Case Study?

| Reason | Detail |
|--------|--------|
| **Defined decarbonization mandate** | Climate Action Plan 2024: 5 GW offshore wind by 2030, 2 GW green H₂. Blending is a near-term action. |
| **Single-operator simplicity** | Gas Networks Ireland (GNI) operates the entire ~2,000 km transmission system. One asset base, one steel grade family (API 5L X52/X65), one regulatory framework. |
| **Well-characterized corridors** | Dublin-Cork (220 km, 20"), Galway-Dublin (180 km, 16") documented in GNI Network Development Plan. Real coordinates, real design pressures. |
| **Regulatory alignment** | ASME B31.12 (US) and IGEM TD/13 (UK) both referenced in Irish practice. Risk tiers in this tool map directly to both. |

**This does not limit the tool to Ireland.** The core engine accepts any `PipelineNetwork` definition. The Ireland case demonstrates *how* to configure a real system — contributors replicate the pattern for their region.

---

## Extending to a New Region (Contributor Guide)

### 1. Create Case Folder
```bash
mkdir -p cases/my_region/
```

### 2. Add `config.yaml`
```yaml
case:
  name: "My Region - Operator Name"
  region: "Country/Region"
  operator: "TSO Name"
  description: "Network description..."

defaults:
  h2_blend_pct: 20
  pressure_bar: 70
  diameter_mm: 500
  length_km: 200
  ambient_temp_c: 10
  reference_hv_vol_mj_m3: 35.8
  steel_grades: ["API 5L X65", "API 5L X52"]

corridors:
  - name: "Corridor A → B"
    start: [lat, lon]
    end: [lat, lon]
    length_km: 200
    diameter_mm: 500
    design_pressure_bar: 70
    steel_grade: "API 5L X65"

policy_context:
  # Your regional policies, targets, references
```

### 3. Add `__init__.py`
```python
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())

def get_defaults(): return CONFIG["defaults"]
def get_corridors(): return CONFIG["corridors"]
# ... etc
```

### 4. Register in `cases/__init__.py`
```python
from .my_region import get_defaults as get_my_region_defaults, ...

AVAILABLE_CASES = {
    "ireland": {...},
    "my_region": {
        "defaults": get_my_region_defaults,
        "corridors": get_my_region_corridors,
        # ...
    },
}
```

### 5. Open PR
- Include validation: compare ΔP predictions against operator's hydraulic model or field data
- Document sources for steel grades, design pressures, roughness values
- Reference local standards (ASME B31.8, CSA Z662, EN 1594)

---

## API Reference (FastAPI)

Start server: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/thermodynamics/blend` | POST | Blend properties (Z, ρ, μ, HV) |
| `/api/v1/simulate/steady-state` | POST | Single pipeline hydraulics + mitigation |
| `/api/v1/network/solve` | POST | Full network Newton-Raphson solve |
| `/api/v1/simulate/transient` | POST | Async transient job |
| `/api/v1/economics/analyze` | POST | Economic analysis |
| `/api/v1/jobs/{job_id}` | GET | Job status/results |
| `/api/v1/ireland/corridors` | GET | Ireland corridors config |
| `/api/v1/ireland/policy` | GET | Ireland policy context |

---

## Core Physics (What the Engine Computes)

| Module | Equations | Outputs |
|--------|-----------|---------|
| **Thermodynamics** | Peng-Robinson cubic EOS for Z; Wilke mixing rule for μ; ρ = PM/ZRT; linear HV blending | `M_blend, Z_blend, ρ, μ, HV_mass, HV_vol` |
| **Hydraulics** | Darcy-Weisbach (Colebrook-White f) + Weymouth; energy-equivalent flow scaling | `ΔP_weymouth, ΔP_darcy, Q_equiv, velocity, Re` |
| **Materials** | Tiered thresholds from ASME B31.12 / IGEM TD/13 / API 5L | `embrittlement_tier, velocity_tier, dp_tier` |
| **Network** | Graph-based mass balance + pressure-flow relations (Newton-Raphson) | `node_pressures, edge_flows, violations` |
| **Transient** | Implicit FD for continuity + momentum; linepack + composition transport | `P(t), H₂(t), linepack(t), Q(t)` |
| **Economics** | CAPEX (terrain, H₂-ready), OPEX (fuel, carbon, maintenance), NPV, LCOT | `capex, opex, npv, lcot` |
| **UQ** | Monte Carlo with parallel execution; correlation sensitivities | `P10/P50/P90, tornado charts` |

---

## Validation Tests

```bash
python -m pytest tests/ -v
```

All 21 tests pass:
- Peng-Robinson Z-factors at STP and 70 bar
- Wilke viscosity mixing (including non-monotonic peak)
- Blend properties (MW, density, HV)
- Darcy-Weisbach / Weymouth positive ΔP
- Flow scaling, velocity calculation
- Risk tier thresholds (embrittlement, velocity, ΔP)
- Known validation cases (20% H₂ density, viscosity, flow increase)

---

## Scaling Roadmap

| Horizon | Scope | Additions |
|---------|-------|-----------|
| **Near** | Single corridor optimization | Multi-objective (CAPEX/OPEX/risk), compressor map integration, cost model |
| **Mid** | Full network (2,000+ km) | Graph-based steady-state solver, GIS shapefile ingestion, transient simulation |
| **Long** | Integrated energy system | Electrolyzer dispatch, salt cavern storage, power-to-gas optimization, PyPSA export |

---

## Contributing

1. **Fork → branch:** `git checkout -b feat/your-change`
2. **Physics changes:** Add to `core/` with:
   - Equation reference (paper/standard/section)
   - Validation case (analytical or experimental data)
   - Type hints + docstrings
3. **Case additions:** Follow *Extending to a New Region* above
4. **Run checks:** `python -m py_compile app.py` + `pytest tests/ -v`
5. **PR template:** What, why, validation, docs updated

### Priority Physics Gaps
- [ ] Panhandle A/B for distribution pressures (<20 bar)
- [ ] Transient (time-dependent) blending ramps — **DONE in v2.0**
- [ ] Compressor polytropic head/flow maps (blend-dependent) — **DONE in v2.0**
- [ ] Fracture mechanics: Paris law with H₂-enhanced da/dN
- [ ] Uncertainty quantification (Monte Carlo on ε, T, composition) — **DONE in v2.0**
- [ ] Export adapters: PyPSA, OpenModelica, SAInt

---

## License

MIT — free for research, consulting, commercial use. Attribution appreciated.

---

## References

- **Peng-Robinson EOS**: Peng & Robinson (1976), *Ind. Eng. Chem. Fundam.* 15:59
- **Wilke Viscosity**: Wilke (1950), *J. Chem. Phys.* 18:517
- **Weymouth Equation**: Weymouth (1912), *Trans. ASME* 34:105
- **Colebrook-White**: Colebrook & White (1937), *J. ICE* 11:133
- **ASME B31.12**: Hydrogen Piping and Pipelines (2023)
- **IGEM TD/13**: Hydrogen in Gas Networks (2023, UK)
- **API 5L / ISO 3183**: Line Pipe specifications
- **GNI**: Network Development Plan 2023-2032
- **Government of Ireland**: Climate Action Plan 2024, National Hydrogen Strategy 2023

---

*Modular physics. Regional reality. Open for every gas network on the planet.*