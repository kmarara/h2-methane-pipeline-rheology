# H₂ Pipeline Digital Twin — Core Engine + Regional Case Studies

A modular computational platform for hydrogen blending analysis in natural gas transmission networks. The **core physics engine** is geography-agnostic; **case studies** provide region-specific configuration (corridors, steel grades, policy context, risk thresholds).
## Why This Exists

Gas networks globally face the same physics challenge: **H₂ has 1/3 the volumetric energy density of CH₄ but lower viscosity and different compressibility**. Blending changes pressure drop, compressor power, materials degradation, and flow requirements — all simultaneously. This tool quantifies those coupled effects using industry-standard equations.

---

## Architecture

```
h2-pipeline-digital-twin/
├── core/                          # Pure physics engine (zero geography)
│   ├── thermodynamics.py          # Peng-Robinson EOS, Wilke viscosity, blend properties
│   ├── hydraulics.py              # Darcy-Weisbach, Weymouth, flow scaling
│   ├── materials.py               # Embrittlement tiers, velocity limits, ΔP limits
│   ├── pipeline.py                # Geographic network definitions, Folium mapping
│   └── __init__.py                # Public API
├── cases/                         # Regional configurations (add yours)
│   ├── ireland/                   # Gas Networks Ireland (reference implementation)
│   │   ├── config.yaml            # Corridors, defaults, policy context, risk thresholds
│   │   └── __init__.py            # Loader functions
│   ├── __init__.py                # Case registry & loader
│   └── (add: uk/, eu/, us/, ...)
├── app.py                         # Streamlit UI (thin wrapper: core + selected case)
├── requirements.txt
└── README.md
```

**Key principle:** Physics lives in `core/`. Geography/policy lives in `cases/`. Contributors add new regions by creating a `cases/<region>/` folder — no physics changes needed.

---

## Why Ireland as the Reference Case Study?

Ireland was chosen as the **reference implementation** for four reasons:

| Reason | Detail |
|--------|--------|
| **Defined decarbonization mandate** | Climate Action Plan 2024: 5 GW offshore wind by 2030, 2 GW green H₂ production. Blending is a near-term action. |
| **Single-operator simplicity** | Gas Networks Ireland (GNI) operates the entire ~2,000 km transmission system. One asset base, one steel grade family (API 5L X52/X65), one regulatory framework. |
| **Well-characterized corridors** | Dublin-Cork (220 km, 20"), Galway-Dublin (180 km, 16") are documented in GNI Network Development Plan. Real coordinates, real design pressures. |
| **Regulatory alignment** | ASME B31.12 (US) and IGEM TD/13 (UK) are both referenced in Irish practice. Risk tiers in this tool map directly to both. |

**This does not limit the tool to Ireland.** The core engine accepts any `PipelineNetwork` definition. The Ireland case demonstrates *how* to configure a real system — contributors replicate the pattern for their region.

---

## Quick Start

```bash
git clone https://github.com/<your-org>/h2-pipeline-digital-twin.git
cd h2-pipeline-digital-twin

# Virtual environment
python -m venv venv && source venv/bin/activate  # Linux/macOS
# venv\Scripts\Activate.ps1                      # Windows PowerShell

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. The UI loads the **Ireland (GNI)** case by default. Select corridor, adjust sliders, download CSV.

---

## Core Physics (What the Engine Computes)

| Module | Equations | Outputs |
|--------|-----------|---------|
| **Thermodynamics** | Peng-Robinson cubic EOS for Z; Wilke mixing rule for μ; ρ = PM/ZRT; linear heating value blending | `M_blend, Z_blend, ρ, μ, HV_mass, HV_vol` |
| **Hydraulics** | Darcy-Weisbach (Colebrook-White f) + Weymouth (high-P gas standard); energy-equivalent flow scaling | `ΔP_weymouth, ΔP_darcy, Q_equiv, velocity, Re` |
| **Materials** | Tiered thresholds from ASME B31.12 / IGEM TD/13 / API 5L | `embrittlement_tier, velocity_tier, dp_tier` |
| **Pipeline** | Segment + compressor station definitions → Folium map with risk coloring | Interactive geospatial visualization |

All functions are pure, typed, and unit-tested (see `tests/`).

---

## Running the Application

```bash
cd h2-pipeline-digital-twin

# Activate environment
source venv/bin/activate          # Linux/macOS
# venv\Scripts\Activate.ps1       # Windows PowerShell
# venv\Scripts\activate.bat       # Windows CMD

# Launch
streamlit run app.py
```

- **Stop:** `Ctrl+C`
- **Deactivate:** `deactivate`
- **Port conflict:** `streamlit run app.py --server.port 8502`

---

## Key Parameters (Ireland Case Defaults)

| Parameter | Range | Ireland Default | Source |
|-----------|-------|-----------------|--------|
| H₂ Blend | 0-50% | 20% | EU/UK blending trials, GNI pilot targets |
| Pressure | 10-80 bar | 70 bar | GNI transmission design pressure |
| Diameter | 200-1000 mm | 500 mm (20") | Dublin-Cork trunk line |
| Length | 10-500 km | 220 km | Inchicore → Whitegate |
| Ground Temp | 0-20°C | 10°C | Irish sub-surface annual average |
| Steel Grade | — | API 5L X65/X52 | GNI asset register |

*All defaults configurable in `cases/ireland/config.yaml`.*

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

### 5. (Optional) Custom Network Class
If your topology is complex (meshed, multiple entry/exit points), extend `core.pipeline.PipelineNetwork` in a new `cases/my_region/network.py`.

### 6. Open PR
- Include validation: compare your ΔP predictions against operator's hydraulic model or field data
- Document sources for steel grades, design pressures, roughness values
- Reference local standards (e.g., ASME B31.8, CSA Z662, EN 1594)

---

## Scaling Roadmap

| Horizon | Scope | Additions |
|---------|-------|-----------|
| **Near** | Single corridor optimization | Multi-objective (CAPEX/OPEX/risk), compressor maps, cost model |
| **Mid** | Full network (2,000+ km) | Graph-based steady-state solver, GIS shapefile ingestion, transient simulation |
| **Long** | Integrated energy system | Electrolyzer dispatch, salt cavern storage, power-to-gas optimization, PyPSA/OpenModelica export |

---

## Contributing

1. **Fork → branch:** `git checkout -b feat/your-change`
2. **Physics changes:** Add to `core/` with:
   - Equation reference (paper/standard/section)
   - Validation case (analytical or experimental data)
   - Type hints + docstrings
3. **Case additions:** Follow *Extending to a New Region* above
4. **Run checks:** `python -m py_compile app.py` + any pytest suite
5. **PR template:** What, why, validation, docs updated

### Priority Physics Gaps
- [ ] Panhandle A/B for distribution pressures (<20 bar)
- [ ] Transient (time-dependent) blending ramps
- [ ] Compressor polytropic head/flow maps (blend-dependent)
- [ ] Fracture mechanics: Paris law with H₂-enhanced da/dN
- [ ] Uncertainty quantification (Monte Carlo on ε, T, composition)
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