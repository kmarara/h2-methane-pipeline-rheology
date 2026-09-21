#!/usr/bin/env python3
"""
Ireland Green H₂ Pipeline Optimizer & Digital Twin
Reference implementation using the core physics engine.

Architecture:
- core/: Pure physics (thermodynamics, hydraulics, materials, pipeline)
- cases/ireland/: GNI-specific config, corridors, policy context
- This file: Streamlit UI wrapper
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import folium
from streamlit_folium import st_folium
from pathlib import Path

# Core physics engine
from core import (
    blend_properties,
    darcy_weisbach_dp,
    weymouth_dp,
    equivalent_flow_rate,
    gas_velocity,
    assess_all,
    risk_color,
    RiskTier,
    create_ireland_network,
)

# Ireland case study
from cases.ireland import (
    get_defaults,
    get_network,
    get_corridors,
    get_policy_context,
    CASE_METADATA,
)

st.set_page_config(
    page_title="Ireland Green H₂ Pipeline Optimizer & Digital Twin",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1B4F72; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4A4A4A; margin-bottom: 1.5rem; }
    .metric-card { background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .alert-warning { background: #FFF3CD; border: 1px solid #FFC107; border-radius: 8px; padding: 1rem; color: #856404; }
    .alert-danger { background: #F8D7DA; border: 1px solid #DC3545; border-radius: 8px; padding: 1rem; color: #721C24; }
    .alert-success { background: #D4EDDA; border: 1px solid #28A745; border-radius: 8px; padding: 1rem; color: #155724; }
    .stSlider > div > div > div > div { background: #1B4F72; }
    .case-badge { display: inline-block; background: #1B4F72; color: white; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Sidebar: Case selection & Simulation parameters
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="case-badge">🇮🇪 Case: Ireland (GNI)</div>', unsafe_allow_html=True)
    
    # Corridor selector
    corridors = get_corridors()
    corridor_names = [c["name"] for c in corridors]
    selected_corridor_name = st.selectbox("Pipeline Corridor", corridor_names, index=0)
    corridor = next(c for c in corridors if c["name"] == selected_corridor_name)

    st.markdown("### ⚙️ Simulation Parameters")
    defaults = get_defaults()

    h2_pct = st.slider("Hydrogen Blending Ratio (%)", 0, 50, defaults["h2_pct"], 1,
                      help="Volumetric percentage of H₂ in natural gas blend")
    P_bar = st.slider("Operating Pressure (bar)", 10, 80, defaults["P_bar"], 1,
                     help=f"GNI transmission standard: {defaults['P_bar']} bar")
    D_mm = st.slider("Pipeline Internal Diameter (mm)", 200, 1000, corridor["diameter_mm"], 50,
                    help=f"Selected corridor: {corridor['diameter_mm']} mm ({corridor['steel_grade']})")
    L_km = st.number_input("Pipeline Length (km)", 10, 500, corridor["length_km"], 10,
                          help=f"Selected corridor: {corridor['length_km']:.0f} km")
    T_amb_c = st.slider("Ambient Ground Temperature (°C)", 0, 20, defaults["T_amb_c"], 1,
                       help="Irish sub-surface annual average ≈ 10°C")

    st.markdown("---")
    st.markdown("### 📊 Quick Metrics")
    
    # Pre-compute for sidebar metrics
    T_k = T_amb_c + 273.15
    P_pa = P_bar * 1e5
    h2_frac = h2_pct / 100.0
    props = blend_properties(h2_frac, T_k, P_pa)
    scenario = {
        "h2_pct": h2_pct, "P_bar": P_bar, "D_mm": D_mm, "L_km": L_km, "T_amb_c": T_amb_c,
        **props,
    }
    Q_ref = 1.0
    dP_weymouth = weymouth_dp(Q_ref, L_km, D_mm, props["rho_blend"], props["mu_blend"], P_pa, props["Z_blend"], T_k)
    dP_darcy = darcy_weisbach_dp(Q_ref, L_km, D_mm, props["rho_blend"], props["mu_blend"])
    Q_equiv = equivalent_flow_rate(Q_ref, defaults["reference_hv_vol"], props["hv_vol_blend"])
    velocity = gas_velocity(Q_equiv, D_mm)

    st.metric("Blend Density", f"{props['rho_blend']:.3f} kg/m³")
    st.metric("Blend Viscosity", f"{props['mu_blend']*1e6:.3f} μPa·s")
    st.metric("Compressibility Z", f"{props['Z_blend']:.4f}")
    st.metric("Energy Density", f"{props['hv_vol_blend']:.2f} MJ/m³")
    st.metric("Flow Increase Req.", f"{(Q_equiv/Q_ref - 1)*100:.1f}%")
    st.metric("Gas Velocity", f"{velocity:.2f} m/s")

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown(f'<div class="main-header">🇮🇪 {CASE_METADATA["name"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{CASE_METADATA["description"]}</div>', unsafe_allow_html=True)
st.caption(f"Operator: {CASE_METADATA['operator']} | Engineered by Kudakwashe D. Marara | Petroleum Chemistry & Geospatial Analytics")

# ──────────────────────────────────────────────────────────────
# Main Layout: Map + Charts
# ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🗺️ Geospatial Pipeline Corridor Map")
    network = get_network()
    risk_colors = {RiskTier.LOW: "#27AE60", RiskTier.MODERATE: "#F39C12", RiskTier.HIGH: "#E74C3C"}
    m = network.to_folium(h2_pct, risk_colors)
    st_folium(m, width=700, height=500, returned_objects=[])

with col2:
    st.markdown("### 📈 Engineering Analysis Charts")
    
    # Generate blend sweep curves
    blends = np.linspace(0, 50, 51)
    curve_data = []
    for b in blends:
        h2_frac = b / 100.0
        props = blend_properties(h2_frac, T_k, P_pa)
        dP_w = weymouth_dp(Q_ref, L_km, D_mm, props["rho_blend"], props["mu_blend"], P_pa, props["Z_blend"], T_k)
        dP_d = darcy_weisbach_dp(Q_ref, L_km, D_mm, props["rho_blend"], props["mu_blend"])
        Q_eq = equivalent_flow_rate(Q_ref, defaults["reference_hv_vol"], props["hv_vol_blend"])
        flow_inc = (Q_eq / Q_ref - 1) * 100
        curve_data.append({
            "h2_pct": b,
            "dP_weymouth": dP_w,
            "dP_darcy": dP_d,
            "hv_vol_blend": props["hv_vol_blend"],
            "flow_increase_pct": flow_inc,
        })
    df_curves = pd.DataFrame(curve_data)

    # Plot 1: Pressure Drop vs Blend %
    fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
    ax1.plot(df_curves['h2_pct'], df_curves['dP_weymouth'], 'b-', linewidth=2.5, label='Weymouth Equation', color='#1B4F72')
    ax1.plot(df_curves['h2_pct'], df_curves['dP_darcy'], 'r--', linewidth=2, label='Darcy-Weisbach', color='#E74C3C')
    ax1.axvline(x=h2_pct, color='#F39C12', linestyle=':', linewidth=2, label=f'Current: {h2_pct:.1f}% H₂')
    ax1.set_xlabel('Hydrogen Blending Ratio (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pressure Drop (bar)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Pipeline Pressure Drop vs. H₂ Blend — {selected_corridor_name}', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax1.set_facecolor('#F8F9FA')
    fig1.tight_layout()
    st.pyplot(fig1)

    # Plot 2: Energy Density vs Flow Increase
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
    ax2.plot(df_curves['h2_pct'], df_curves['hv_vol_blend'], 'g-', linewidth=2.5, label='Volumetric Energy Density (MJ/m³@STP)', color='#27AE60')
    ax2.set_xlabel('Hydrogen Blending Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Energy Density (MJ/m³)', fontsize=12, fontweight='bold', color='#27AE60')
    ax2.tick_params(axis='y', labelcolor='#27AE60')
    ax2.axvline(x=h2_pct, color='#F39C12', linestyle=':', linewidth=2)
    ax3 = ax2.twinx()
    ax3.plot(df_curves['h2_pct'], df_curves['flow_increase_pct'], 'm-', linewidth=2.5, label='Volumetric Flow Increase (%)', color='#8E44AD')
    ax3.set_ylabel('Flow Increase Required (%)', fontsize=12, fontweight='bold', color='#8E44AD')
    ax3.tick_params(axis='y', labelcolor='#8E44AD')
    ax2.set_title('Energy Density Loss vs. Volumetric Flow Requirements', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines2 + lines3, labels2 + labels3, loc='center right', framealpha=0.9, fontsize=10)
    ax2.set_facecolor('#F8F9FA')
    fig2.tight_layout()
    st.pyplot(fig2)

# ──────────────────────────────────────────────────────────────
# Risk Assessment Alerts
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚠️ Engineering Risk Assessment & Alerts")

alerts = assess_all(h2_pct, velocity, dP_weymouth, P_bar)

for alert in alerts:
    color_map = {RiskTier.LOW: "alert-success", RiskTier.MODERATE: "alert-warning", RiskTier.HIGH: "alert-danger"}
    icon_map = {RiskTier.LOW: "🟢", RiskTier.MODERATE: "🟡", RiskTier.HIGH: "🔴"}
    css_class = color_map[alert.tier]
    icon = icon_map[alert.tier]
    st.markdown(f'<div class="{css_class}"><strong>{icon} {alert.title}:</strong> {alert.message}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Simulation Results Export")

export_df = pd.DataFrame({
    'Parameter': [
        'Case Study', 'Corridor', 'Hydrogen Blend (%)', 'Operating Pressure', 'Pipeline Diameter',
        'Pipeline Length', 'Ambient Temperature', 'Steel Grade', 'Blend Molecular Weight',
        'Compressibility Factor (Z)', 'Blend Density', 'Blend Viscosity',
        'Mass Heating Value', 'Volumetric Heating Value (STP)',
        'Pressure Drop (Weymouth)', 'Pressure Drop (Darcy-Weisbach)',
        'Reference Flow Rate', 'Equivalent Energy Flow Rate',
        'Flow Increase Required', 'Gas Velocity', 'Energy Throughput (ref)',
    ],
    'Value': [
        CASE_METADATA["name"], selected_corridor_name,
        f"{h2_pct:.1f}", f"{P_bar:.1f}", f"{D_mm:.0f}",
        f"{L_km:.0f}", f"{T_amb_c:.1f}", corridor["steel_grade"],
        f"{props['M_blend']:.4f}", f"{props['Z_blend']:.4f}",
        f"{props['rho_blend']:.4f}", f"{props['mu_blend']*1e6:.4f}",
        f"{props['hv_mass_blend']:.2f}", f"{props['hv_vol_blend']:.2f}",
        f"{dP_weymouth:.2f}", f"{dP_darcy:.2f}",
        f"{Q_ref:.4f}", f"{Q_equiv:.4f}",
        f"{(Q_equiv/Q_ref - 1)*100:.1f}", f"{velocity:.2f}",
        f"{Q_ref * props['hv_vol_blend'] * P_bar / 1.01325 / props['Z_blend'] * T_k / 273.15:.2f}",
    ],
    'Unit': [
        '-', '-', '%', 'bar', 'mm', 'km', '°C', '-',
        'g/mol', '-', 'kg/m³', 'μPa·s', 'MJ/kg', 'MJ/m³',
        'bar', 'bar', 'm³/s', 'm³/s', '%', 'm/s', 'MJ/s'
    ]
})

csv = export_df.to_csv(index=False)
st.download_button(
    label="📥 Download Simulation Results (CSV)",
    data=csv,
    file_name=f"simulation_results_{selected_corridor_name.replace(' ', '_').replace('(', '').replace(')', '').replace('→', 'to')}.csv",
    mime="text/csv",
    use_container_width=True
)

st.dataframe(export_df, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <strong>H₂ Pipeline Digital Twin — Core Physics Engine + Ireland (GNI) Case Study</strong><br>
    <em>Modular architecture: <code>core/</code> (physics) + <code>cases/</code> (regional config)</em><br>
    Equations: Peng-Robinson EOS • Wilke Viscosity • Weymouth/Darcy-Weisbach • ASME B31.12 Risk Tiers
</div>
""", unsafe_allow_html=True)