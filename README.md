# Ireland Green H₂ Pipeline Optimizer & Digital Twin

A computational fluid dynamics and thermodynamic simulation platform for modeling hydrogen blending into Ireland's natural gas transmission network. Built for energy transition analysis, infrastructure planning, and regulatory assessment under the Climate Action Plan.

## Problem Space

Ireland's gas network (operated by Gas Networks Ireland) spans ~2,000 km of transmission pipelines. As the country targets 5 GW of offshore wind by 2030 and 2 GW of green hydrogen production, the existing infrastructure must be assessed for hydrogen compatibility. This tool quantifies the thermodynamic, hydraulic, and materials implications of blending H₂ into CH₄ streams at scales relevant to Irish transmission corridors.

## What This Solves

| Challenge | Solution |
|-----------|----------|
| **Real-gas behavior at 70+ bar** | Peng-Robinson EOS with mixture rules for Z-factor, density, viscosity |
| **Pressure drop prediction** | Weymouth (industry standard) + Darcy-Weisbach with Colebrook-White friction |
| **Energy throughput equivalence** | Volumetric flow scaling to maintain MJ/s delivery as H₂ fraction increases |
| **Materials risk assessment** | Tiered alerts aligned with ASME B31.12 / IGEM TD/13 embrittlement thresholds |
| **Geospatial context** | Interactive corridor mapping (Dublin-Cork, Galway-Dublin) with risk visualization |
| **Scenario portability** | CSV export for integration with GIS, hydraulic models, or regulatory submissions |

## Architecture

```
app.py                 # Single-file Streamlit application
├── GasProperties      # CH₄/H₂ critical constants, heating values, viscosity refs
├── peng_robinson_z()  # Compressibility factor via cubic EOS
├── blend_properties() # Mixture MW, Z, ρ, μ (Wilke), heating values
├── weymouth_pressure_drop()    # High-pressure gas pipeline equation
├── darcy_weisbach_pressure_drop() # General frictional loss
├── calculate_scenario()        # Full physics pipeline for one blend case
├── generate_blend_curves()     # Sweep 0-50% H₂ for plotting
├── create_pressure_drop_plot() # Matplotlib: ΔP vs blend %
├── create_energy_flow_plot()   # Dual-axis: energy density + flow increase
├── create_folium_map()         # Irish corridors with risk-tier coloring
└── main()                      # UI, alerts, CSV export
```

## Quick Start

```bash
# Clone
git clone https://github.com/<your-org>/h2-methane-pipeline-rheology.git
cd h2-methane-pipeline-rheology

# Environment (venv or conda)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open `http://localhost:8501` — adjust sliders for blend %, pressure, diameter, length, temperature. Charts and map update live. Download CSV for offline analysis.

## Running the Application

```bash
# Navigate to project root
cd h2-methane-pipeline-rheology

# Activate virtual environment (Linux/macOS)
source venv/bin/activate

# Activate virtual environment (Windows PowerShell)
# venv\Scripts\Activate.ps1

# Activate virtual environment (Windows CMD)
# venv\Scripts\activate.bat

# Launch dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`. Stop with `Ctrl+C`. Deactivate env with `deactivate`.

## Key Parameters & Defaults

| Parameter | Range | Default | Basis |
|-----------|-------|---------|-------|
| H₂ Blend | 0-50% | 20% | Current EU/UK blending trials |
| Pressure | 10-80 bar | 70 bar | GNI transmission standard |
| Diameter | 200-1000 mm | 500 mm | Typical Irish trunk lines |
| Length | 10-500 km | 220 km | Cork-Dublin corridor |
| Ground Temp | 0-20°C | 10°C | Irish sub-surface annual avg |

## Extending the Model

### Add New Corridors
Edit `create_folium_map()` — append coordinate pairs to the `folium.PolyLine` calls. Risk tier logic is in the same function.

### Swap Pressure Drop Correlation
Replace `weymouth_pressure_drop()` or `darcy_weisbach_pressure_drop()` with Panhandle A/B, AGA, or CFD-coupled models. The interface expects `(Q, L, D, ρ, μ, P, Z, T) → ΔP_bar`.

### Integrate Real Pipeline Data
Load GNI GIS shapefiles via `geopandas` → compute segment-by-segment ΔP → aggregate for network-level studies.

### Compressor Station Modeling
Add polytropic head/flow curves in `calculate_scenario()` to simulate re-compression requirements at blend-dependent flow rates.

### Materials Degradation Kinetics
Couple with fracture mechanics models (Paris law with H₂-enhanced da/dN) for remaining-life assessment per ASME B31.12.

## Scaling Pathways

| Horizon | Scope | Technical Additions |
|---------|-------|---------------------|
| **Near-term** | Single corridor optimization | Multi-objective (CAPEX vs OPEX vs risk), compressor map integration |
| **Mid-term** | Full network (2,000 km) | Graph-based hydraulic solver, GIS layer ingestion, transient simulation |
| **Long-term** | All-island energy system | Electrolyzer dispatch coupling, storage cavern modeling, power-to-gas optimization |

## Contributing

1. Fork → feature branch (`git checkout -b feat/your-change`)
2. Add tests for new physics functions (pytest style)
3. Ensure `python -m py_compile app.py` passes
4. Open PR with:
   - Equation references (paper/standard)
   - Validation case (analytical or experimental)
   - Updated docstrings

### Priority Areas
- [ ] Panhandle A/B correlations for low-pressure distribution
- [ ] Transient (time-dependent) blending scenarios
- [ ] Cost model: compressor power vs pipeline looping vs derating
- [ ] Uncertainty quantification (Monte Carlo on roughness, T, composition)
- [ ] Export to OpenModelica / PyPSA for system-level studies

## License

MIT — use freely in research, consulting, or commercial projects. Attribution appreciated.

## References

- **Peng-Robinson EOS**: Peng, D.Y., Robinson, D.B. (1976). *Ind. Eng. Chem. Fundam.*
- **Wilke Viscosity**: Wilke, C.R. (1950). *J. Chem. Phys.*
- **Weymouth Equation**: Weymouth, T.R. (1912). *Trans. Am. Inst. Mech. Eng.*
- **ASME B31.12**: Hydrogen Piping & Pipelines (2023)
- **IGEM TD/13**: Hydrogen in Gas Networks (UK)
- **Gas Networks Ireland**: Network Development Plan, Climate Action Plan alignment

---

*Built for the Irish energy transition. Physics-first. Deployment-ready.*