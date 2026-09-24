#!/usr/bin/env python3
"""
H₂ Pipeline Digital Twin — Interactive Streamlit Dashboard
Professional dark theme with Plotly visualizations and real-time analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from pathlib import Path
import time

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
    simulate_pipeline_with_compressors,
    find_required_looping,
    pressure_drop_with_looping,
    equivalent_hydraulic_diameter,
    MonteCarloUQ,
    GasNetwork,
    NodeType,
    create_ireland_network as create_network_solver,
    TransientSimulator,
    create_blending_ramp_scenario,
    EconomicAnalysis,
    estimate_ireland_h2_project,
)

# Ireland case study
from cases.ireland import (
    get_defaults,
    get_network,
    get_corridors,
    get_policy_context,
    CASE_METADATA,
)

# Page config
st.set_page_config(
    page_title="H₂ Pipeline Digital Twin",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# Custom CSS - Dark Professional Theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme variables */
    :root {
        --bg-primary: #0E1117;
        --bg-secondary: #1E222A;
        --bg-tertiary: #262B35;
        --accent-primary: #00D4AA;
        --accent-secondary: #1B4F72;
        --accent-warning: #F39C12;
        --accent-danger: #E74C3C;
        --text-primary: #FAFAFA;
        --text-secondary: #A0A0A0;
        --border-color: #333840;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4AA 0%, #1B4F72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,212,170,0.15);
        border-color: var(--accent-primary);
    }
    
    [data-testid="metric-container"] > div {
        color: var(--text-primary) !important;
    }
    
    [data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: var(--accent-primary) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: var(--bg-tertiary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
        color: var(--bg-primary) !important;
        box-shadow: 0 4px 16px rgba(0,212,170,0.3);
    }
    
    /* Alert boxes */
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        border-left: 4px solid;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .alert-success {
        background: rgba(39, 174, 96, 0.15);
        border-left-color: #27AE60;
        color: #2ECC71;
    }
    
    .alert-warning {
        background: rgba(243, 156, 18, 0.15);
        border-left-color: #F39C12;
        color: #F39C12;
    }
    
    .alert-danger {
        background: rgba(231, 76, 60, 0.15);
        border-left-color: #E74C3C;
        color: #E74C3C;
    }
    
    .alert-info {
        background: rgba(27, 79, 114, 0.15);
        border-left-color: #1B4F72;
        color: #3498DB;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-primary);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-header::before {
        content: "";
        width: 8px;
        height: 8px;
        background: var(--accent-primary);
        border-radius: 50%;
    }
    
    /* Card containers */
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        color: var(--bg-primary);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(0,212,170,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,212,170,0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: var(--bg-tertiary) !important;
        color: var(--accent-primary) !important;
        border: 1px solid var(--accent-primary) !important;
    }
    
    .stDownloadButton > button:hover {
        background: var(--accent-primary) !important;
        color: var(--bg-primary) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        gap: 1rem;
    }
    
    .stRadio label {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .stRadio label:hover {
        border-color: var(--accent-primary);
        background: var(--bg-tertiary);
    }
    
    .stRadio [data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.8rem;
        padding: 2rem;
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
    }
    
    /* Case badge */
    .case-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, var(--accent-secondary) 0%, #0A2A4A 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(27, 79, 114, 0.4);
    }
    
    .case-badge::before {
        content: "🇮🇪";
        font-size: 1rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Sidebar: Configuration
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="case-badge">Ireland (GNI)</div>', unsafe_allow_html=True)
    
    # Corridor selector
    corridors = get_corridors()
    corridor_names = [c["name"] for c in corridors]
    selected_corridor_name = st.selectbox("📍 Pipeline Corridor", corridor_names, index=0)
    corridor = next(c for c in corridors if c["name"] == selected_corridor_name)
    
    st.markdown("---")
    st.markdown("### ⚙️ Simulation Parameters")
    defaults = get_defaults()
    
    col1, col2 = st.columns(2)
    with col1:
        h2_pct = st.slider("H₂ Blend (%)", 0, 50, defaults["h2_pct"], 1,
                          help="Volumetric H₂ percentage")
        P_bar = st.slider("Inlet Pressure (bar)", 10, 85, defaults["P_bar"], 1)
    with col2:
        D_mm = st.slider("Diameter (mm)", 200, 1000, corridor["diameter_mm"], 50)
        L_km = st.number_input("Length (km)", 10, 500, corridor["length_km"], 10)
    
    T_amb_c = st.slider("Ground Temp (°C)", -5, 25, defaults["T_amb_c"], 1)
    
    # Advanced options expander
    with st.expander("🔧 Advanced Options"):
        P_min = st.number_input("Min Outlet Pressure (bar)", 5, 50, 20, 1)
        Q_ref_sm3h = st.number_input("Reference Flow (Sm³/h)", 100_000, 5_000_000, 1_000_000, 50_000)
        ref_hv_vol = st.number_input("Ref HV Vol (MJ/m³@STP)", 30.0, 40.0, 35.8, 0.1)
        n_samples = st.number_input("MC Samples", 50, 2000, 200, 50)
    
    st.markdown("---")
    st.markdown("### 📊 Live Metrics")
    
    # Compute sidebar metrics
    T_k = T_amb_c + 273.15
    P_pa = P_bar * 1e5
    h2_frac = h2_pct / 100.0
    props = blend_properties(h2_frac, T_k, P_pa)
    Q_ref = Q_ref_sm3h / 3600.0
    Q_equiv = equivalent_flow_rate(Q_ref, ref_hv_vol, props["hv_vol_blend"])
    velocity = gas_velocity(Q_equiv, D_mm)
    dp_darcy = darcy_weisbach_dp(Q_equiv, L_km, D_mm, props["rho_blend"], props["mu_blend"])
    P_out = P_bar - dp_darcy
    
    # Metric cards in sidebar
    c1, c2 = st.columns(2)
    c1.metric("Density", f"{props['rho_blend']:.2f} kg/m³")
    c2.metric("Viscosity", f"{props['mu_blend']*1e6:.1f} μPa·s")
    c1.metric("Compressibility Z", f"{props['Z_blend']:.4f}")
    c2.metric("Energy Density", f"{props['hv_vol_blend']:.1f} MJ/m³")
    c1.metric("Flow Increase", f"{(Q_equiv/Q_ref-1)*100:.1f}%")
    c2.metric("Velocity", f"{velocity:.1f} m/s", 
              delta="HIGH" if velocity >= 20 else "OK",
              delta_color="inverse" if velocity >= 20 else "normal")
    c1.metric("ΔP (Darcy)", f"{dp_darcy:.1f} bar")
    c2.metric("Outlet P", f"{P_out:.1f} bar",
              delta=f"{P_out - P_min:.1f} vs min",
              delta_color="normal" if P_out >= P_min else "inverse")
    
    st.markdown("---")
    st.caption("H₂ Pipeline Digital Twin v2.0")
    st.caption("Core Physics • Network Solver • UQ • Economics")

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown(f'<div class="main-header">🇮🇪 H₂ Pipeline Digital Twin</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Ireland Gas Networks Ireland • {CASE_METADATA["description"]}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Main Tabs
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Hydraulics", 
    "🗺️ Network", 
    "⚡ Transient", 
    "🎲 Uncertainty (MC)", 
    "💰 Economics", 
    "📋 Export"
])

# ──────────────────────────────────────────────────────────────
# TAB 1: HYDRAULICS
# ──────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown('<div class="section-header">📈 Pressure Drop vs H₂ Blend</div>', unsafe_allow_html=True)
        
        # Generate blend sweep
        blends = np.linspace(0, 50, 51)
        curve_data = []
        for b in blends:
            h2_f = b / 100.0
            p = blend_properties(h2_f, T_k, P_pa)
            dp_w = weymouth_dp(Q_ref, L_km, D_mm, p["rho_blend"], p["mu_blend"], P_pa, p["Z_blend"], T_k)
            dp_d = darcy_weisbach_dp(Q_ref, L_km, D_mm, p["rho_blend"], p["mu_blend"])
            Q_eq = equivalent_flow_rate(Q_ref, ref_hv_vol, p["hv_vol_blend"])
            flow_inc = (Q_eq / Q_ref - 1) * 100
            curve_data.append({
                "h2_pct": b, "dP_weymouth": dp_w, "dP_darcy": dp_d,
                "hv_vol": p["hv_vol_blend"], "flow_inc": flow_inc,
                "velocity": gas_velocity(Q_eq, D_mm), "density": p["rho_blend"]
            })
        df = pd.DataFrame(curve_data)
        
        # Plotly figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(
            x=df['h2_pct'], y=df['dP_weymouth'],
            name='Weymouth', line=dict(color='#1B4F72', width=3),
            hovertemplate='%{x:.1f}% H₂ → %{y:.2f} bar<extra></extra>'
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=df['h2_pct'], y=df['dP_darcy'],
            name='Darcy-Weisbach', line=dict(color='#E74C3C', width=2, dash='dash'),
            hovertemplate='%{x:.1f}% H₂ → %{y:.2f} bar<extra></extra>'
        ), secondary_y=False)
        
        # Current point
        fig.add_trace(go.Scatter(
            x=[h2_pct], y=[dp_darcy],
            mode='markers', name='Current',
            marker=dict(size=14, color='#F39C12', symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate=f'{h2_pct:.1f}% H₂ → {dp_darcy:.2f} bar<extra></extra>'
        ), secondary_y=False)
        
        # Min pressure line
        fig.add_hline(y=P_bar - P_min, line_dash="dot", line_color="#27AE60",
                     annotation_text=f"Min Safe Outlet ({P_min} bar)", 
                     annotation_position="top right")
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E222A',
            font=dict(color='#FAFAFA', size=12),
            title=dict(text=f"Pressure Drop — {selected_corridor_name}", font=dict(size=16, color='#FAFAFA')),
            xaxis=dict(title="H₂ Blend (%)", gridcolor='#333840'),
            yaxis=dict(title="Pressure Drop (bar)", gridcolor='#333840'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown('<div class="section-header">⚡ Energy Density & Flow Trade-off</div>', unsafe_allow_html=True)
        
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig2.add_trace(go.Scatter(
            x=df['h2_pct'], y=df['hv_vol'],
            name='Energy Density', line=dict(color='#27AE60', width=3),
            hovertemplate='%{x:.1f}% H₂ → %{y:.1f} MJ/m³<extra></extra>'
        ), secondary_y=False)
        
        fig2.add_trace(go.Scatter(
            x=df['h2_pct'], y=df['flow_inc'],
            name='Flow Increase %', line=dict(color='#8E44AD', width=3),
            hovertemplate='%{x:.1f}% H₂ → %{y:.1f}% flow increase<extra></extra>'
        ), secondary_y=True)
        
        fig2.add_trace(go.Scatter(
            x=[h2_pct], y=[props["hv_vol_blend"]],
            mode='markers', name='Current',
            marker=dict(size=14, color='#F39C12', symbol='diamond', line=dict(width=2, color='white')),
        ), secondary_y=False)
        
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E222A',
            font=dict(color='#FAFAFA', size=12),
            title=dict(text="Energy vs Flow Requirements", font=dict(size=16)),
            xaxis=dict(title="H₂ Blend (%)", gridcolor='#333840'),
            yaxis=dict(title="Energy Density (MJ/m³)", gridcolor='#333840', side='left'),
            yaxis2=dict(title="Flow Increase (%)", gridcolor='#333840', side='right'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # ──────────────────────────────────────────────────────────────
    # Risk Assessment Alerts
    # ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Engineering Risk Assessment</div>', unsafe_allow_html=True)
    
    alerts = assess_all(h2_pct, velocity, dp_darcy, P_bar)
    
    alert_cols = st.columns(len(alerts))
    for i, alert in enumerate(alerts):
        with alert_cols[i]:
            color_map = {RiskTier.LOW: "alert-success", RiskTier.MODERATE: "alert-warning", RiskTier.HIGH: "alert-danger"}
            icon_map = {RiskTier.LOW: "🟢", RiskTier.MODERATE: "🟡", RiskTier.HIGH: "🔴"}
            st.markdown(f'<div class="alert-box {color_map[alert.tier]}"><strong>{icon_map[alert.tier]} {alert.title}</strong><br>{alert.message}</div>', unsafe_allow_html=True)
    
    # ──────────────────────────────────────────────────────────────
    # Pressure Exceedance Mitigation
    # ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔧 Pressure Exceedance Mitigation</div>', unsafe_allow_html=True)
    
    exceedance = P_out < P_min
    
    if exceedance:
        st.markdown(f'<div class="alert-box alert-danger"><strong>⚠️ PRESSURE EXCEEDANCE</strong><br>At {h2_pct:.1f}% H₂, energy-equivalent flow ({Q_equiv*3600:,.0f} Sm³/h) causes ΔP = {dp_darcy:.1f} bar. Outlet pressure = <strong>{P_out:.1f} bar</strong> (below {P_min:.1f} bar minimum).</div>', unsafe_allow_html=True)
        
        mitigation = st.radio("Mitigation Strategy:", 
                             ["🔧 Midpoint Compressor Station(s)", "🔄 Pipeline Looping (Parallel)"],
                             horizontal=True, key="mitigation_choice")
        
        if "Compressor" in mitigation:
            T_k2 = T_amb_c + 273.15
            comp = simulate_pipeline_with_compressors(
                Q=Q_equiv, L_total=L_km, D=D_mm,
                rho=props["rho_blend"], mu=props["mu_blend"], Z=props["Z_blend"],
                T=T_k2, P_in=P_bar, P_min=P_min, target_P=P_bar
            )
            
            if comp["success"] and comp["num_compressors"] > 0:
                st.markdown(f'<div class="alert-box alert-success"><strong>✅ RESOLVED</strong><br>{comp["num_compressors"]} compressor(s) added. Outlet restored to <strong>{comp["P_out"]:.1f} bar</strong>. Total power: <strong>{comp["total_power_mw"]:.2f} MW</strong>.</div>', unsafe_allow_html=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Compressors", comp["num_compressors"])
                m2.metric("Total Power", f"{comp['total_power_mw']:.2f} MW")
                m3.metric("Segment Length", f"{comp['segment_length_km']:.1f} km")
                
                if comp['compressor_locations_km']:
                    st.info("📍 Locations: " + ", ".join([f"Km {loc:.1f}" for loc in comp['compressor_locations_km']]))
                
                # Pressure profile plot
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=list(range(len(comp['segment_pressures']))),
                    y=comp['segment_pressures'],
                    mode='lines+markers', line=dict(color='#00D4AA', width=3),
                    marker=dict(size=8, color='#00D4AA'),
                    name='Pressure'
                ))
                fig_p.add_hline(y=P_min, line_dash="dash", line_color="#E74C3C", annotation_text=f"Min Safe ({P_min} bar)")
                fig_p.add_hline(y=P_bar, line_dash="dash", line_color="#27AE60", annotation_text=f"Inlet ({P_bar} bar)")
                fig_p.update_layout(template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#1E222A',
                                   title="Pressure Profile with Compression", height=300,
                                   xaxis_title="Station", yaxis_title="Pressure (bar)")
                st.plotly_chart(fig_p, use_container_width=True)
        
        else:  # Looping
            loop = find_required_looping(Q_equiv, L_km, D_mm, props["rho_blend"], props["mu_blend"], P_bar, P_min)
            
            if loop["success"]:
                st.markdown(f'<div class="alert-box alert-success"><strong>✅ RESOLVED</strong><br>Parallel {D_mm:.0f}mm loop over <strong>{loop["loop_length_km"]:.1f} km</strong> ({loop["loop_percentage"]*100:.1f}%). Outlet restored to <strong>{loop["P_out"]:.1f} bar</strong>.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-box alert-danger"><strong>❌ INSUFFICIENT</strong><br>Even 100% looping only achieves {loop["P_out"]:.1f} bar. Needs larger diameter or compression.</div>', unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Loop %", f"{loop['loop_percentage']*100:.1f}%")
            m2.metric("Loop Length", f"{loop['loop_length_km']:.1f} km")
            m3.metric("Eq. Diameter", f"{equivalent_hydraulic_diameter(D_mm, D_mm):.0f} mm")
            
            # Bar comparison
            dp_looped = pressure_drop_with_looping(Q_equiv, L_km, loop["loop_length_km"], D_mm, D_mm, props["rho_blend"], props["mu_blend"])
            fig_l = go.Figure()
            fig_l.add_trace(go.Bar(x=["Single Pipe", f"Looped ({loop['loop_percentage']*100:.0f}%)"],
                                  y=[P_bar - dp_darcy, P_bar - dp_looped],
                                  marker_color=['#E74C3C' if P_bar - dp_darcy < P_min else '#27AE60', '#27AE60'],
                                  text=[f"{P_bar - dp_darcy:.1f}", f"{P_bar - dp_looped:.1f}"],
                                  textposition='outside'))
            fig_l.add_hline(y=P_min, line_dash="dash", line_color="#E74C3C", annotation_text=f"Min Safe")
            fig_l.update_layout(template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#1E222A',
                               title="Outlet Pressure Comparison", height=300, yaxis_title="Pressure (bar)")
            st.plotly_chart(fig_l, use_container_width=True)
    else:
        st.markdown(f'<div class="alert-box alert-success"><strong>✅ PRESSURE INTEGRITY OK</strong><br>At {h2_pct:.1f}% H₂, outlet pressure = <strong>{P_out:.1f} bar</strong> (ΔP = {dp_darcy:.1f} bar). Well above {P_min:.1f} bar minimum.</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 2: NETWORK SOLVER
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🗺️ Network Topology & Map</div>', unsafe_allow_html=True)
    
    col_map, col_data = st.columns([2, 1])
    
    with col_map:
        network = get_network()
        risk_colors_map = {RiskTier.LOW: "#27AE60", RiskTier.MODERATE: "#F39C12", RiskTier.HIGH: "#E74C3C"}
        m = network.to_folium(h2_pct, risk_colors_map)
        st_folium(m, width=900, height=550, returned_objects=[])
    
    with col_data:
        st.markdown("### Network Statistics")
        
        net = create_network_solver()
        net.set_gas_properties(h2_frac, T_k)
        
        # Scale demands
        for nid, node in net.nodes.items():
            if node.node_type == NodeType.DEMAND:
                from dataclasses import replace
                new_node = replace(node, demand_sm3h=node.demand_sm3h * 1.0)
                # Can't modify frozen, so skip for now
                pass
        
        results = net.solve_steady_state()
        
        if results["converged"]:
            st.success(f"✅ Network converged in {results['iterations']} iterations")
            
            # Summary metrics
            total_demand = sum(n.demand_sm3h for n in net.nodes.values() if n.node_type == NodeType.DEMAND)
            total_supply = sum(n.flow_sm3h or 0 for n in net.nodes.values() if n.node_type == NodeType.SUPPLY)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Nodes", len(net.nodes))
            m2.metric("Edges", len(net.edges))
            m3.metric("Compressors", len(net.compressors))
            
            m1.metric("Total Demand", f"{total_demand/1e6:.1f} M Sm³/h")
            m2.metric("Total Supply", f"{total_supply/1e6:.1f} M Sm³/h")
            m3.metric("Comp Power", f"{results['total_compression_power_mw']:.1f} MW")
            
            # Node pressures table
            if results["node_pressures"]:
                pressure_data = []
                for nid, node in net.nodes.items():
                    p = results["node_pressures"].get(nid, 0)
                    status = "✅" if p >= node.min_pressure_bar else "❌"
                    pressure_data.append({
                        "Node": node.name,
                        "Type": node.node_type.value,
                        "Pressure (bar)": f"{p:.1f}",
                        "Min Req": f"{node.min_pressure_bar:.1f}",
                        "Status": status
                    })
                st.dataframe(pd.DataFrame(pressure_data), use_container_width=True, hide_index=True)
        
        else:
            st.error("Network did not converge")

# ──────────────────────────────────────────────────────────────
# TAB 3: TRANSIENT
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">⚡ Transient Blending Simulation</div>', unsafe_allow_html=True)
    
    col_controls, col_results = st.columns([1, 2])
    
    with col_controls:
        st.markdown("#### Scenario Setup")
        duration = st.slider("Duration (hours)", 1, 72, 24)
        initial_h2 = st.slider("Initial H₂ (%)", 0, 30, 0) / 100
        final_h2 = st.slider("Final H₂ (%)", 0, 50, 20) / 100
        ramp_h = st.slider("Ramp Duration (hours)", 1, 24, 6)
        dt_h = st.select_slider("Time Step", [0.1, 0.25, 0.5, 1.0], 0.25)
        
        run_transient = st.button("🚀 Run Transient Simulation", type="primary", use_container_width=True)
    
    if run_transient:
        with st.spinner("Running transient simulation..."):
            net = create_network_solver()
            net.set_gas_properties(initial_h2, T_k)
            
            scenario = create_blending_ramp_scenario(
                net, duration_hours=duration, 
                initial_h2=initial_h2, final_h2=final_h2, ramp_hours=ramp_h
            )
            
            sim = TransientSimulator(net)
            start = time.time()
            results = sim.simulate(scenario, dt_hours=dt_h)
            elapsed = time.time() - start
        
        st.success(f"✅ Completed in {elapsed:.1f}s")
        
        with col_results:
            # Time series plots
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=("Node Pressures", "H₂ Fraction at Demand", 
                                              "Linepack", "Edge Flows"),
                               vertical_spacing=0.12)
            
            times = results.time_hours
            
            # Pressures
            for nid, pressures in results.node_pressures.items():
                node = net.nodes.get(nid)
                if node and node.node_type == NodeType.DEMAND:
                    fig.add_trace(go.Scatter(x=times, y=pressures, name=node.name, 
                                           line=dict(width=2)), row=1, col=1)
            fig.add_hline(y=P_min, line_dash="dash", line_color="red", row=1, col=1)
            
            # H2 fractions
            for nid, h2s in results.node_h2_fractions.items():
                node = net.nodes.get(nid)
                if node and node.node_type == NodeType.DEMAND:
                    fig.add_trace(go.Scatter(x=times, y=np.array(h2s)*100, name=node.name, 
                                           line=dict(width=2), showlegend=False), row=1, col=2)
            
            # Linepack
            fig.add_trace(go.Scatter(x=times, y=np.array(results.total_linepack_kg)/1000, 
                                   name="Linepack", line=dict(color='#00D4AA', width=3), 
                                   showlegend=False), row=2, col=1)
            
            # Edge flows (total)
            total_flow = np.zeros_like(times)
            for flows in results.edge_flows.values():
                total_flow += np.array(flows)
            fig.add_trace(go.Scatter(x=times, y=total_flow/1e6, name="Total Flow", 
                                   line=dict(color='#F39C12', width=3), showlegend=False), row=2, col=2)
            
            fig.update_layout(template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#1E222A',
                             font=dict(color='#FAFAFA'), height=600, showlegend=True,
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_xaxes(title_text="Time (hours)")
            fig.update_yaxes(title_text="Pressure (bar)", row=1, col=1)
            fig.update_yaxes(title_text="H₂ (%)", row=1, col=2)
            fig.update_yaxes(title_text="Linepack (tonnes)", row=2, col=1)
            fig.update_yaxes(title_text="Flow (M Sm³/h)", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 4: UNCERTAINTY QUANTIFICATION (MC)
# ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🎲 Monte Carlo Uncertainty Quantification</div>', unsafe_allow_html=True)
    
    col_mc_ctrl, col_mc_viz = st.columns([1, 2])
    
    with col_mc_ctrl:
        st.markdown("#### Configuration")
        mc_type = st.selectbox("Analysis Type", ["Economics", "Steady-State", "Transient"])
        n_mc = st.slider("Samples", 50, 1000, n_samples, 50)
        n_workers = st.slider("Workers", 1, 8, 4)
        seed = st.number_input("Random Seed", 0, 9999, 42)
        
        run_mc = st.button("🎲 Run Monte Carlo", type="primary", use_container_width=True)
    
    if run_mc:
        with st.spinner(f"Running {n_mc} samples with {n_workers} workers..."):
            uq = MonteCarloUQ(n_workers=n_workers)
            
            progress = st.progress(0)
            status = st.empty()
            
            if mc_type == "Economics":
                start = time.time()
                results = uq.run_economics_mc(n_samples=n_mc, seed=seed)
                elapsed = time.time() - start
            elif mc_type == "Steady-State":
                start = time.time()
                results = uq.run_steady_state_mc(n_samples=n_mc, seed=seed)
                elapsed = time.time() - start
            else:
                status.warning("Transient MC not yet implemented in UI")
                results = None
        
        if results:
            progress.progress(1.0)
            status.success(f"✅ Completed {results.n_samples} samples in {results.compute_time_s:.1f}s")
            
            with col_mc_viz:
                st.markdown(f"### Results: {mc_type} Monte Carlo ({results.n_samples} samples)")
                
                # Statistics table
                stats_data = []
                for out_name, stats in results.statistics.items():
                    stats_data.append({
                        "Output": out_name,
                        "Mean": f"{stats['mean']:,.2f}",
                        "Std": f"{stats['std']:,.2f}",
                        "P10": f"{stats['p10']:,.2f}",
                        "P50 (Median)": f"{stats['p50']:,.2f}",
                        "P90": f"{stats['p90']:,.2f}",
                        "Success %": f"{stats['success_rate']*100:.0f}%"
                    })
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                
                # Sensitivity tornado chart
                if results.sensitivity:
                    st.markdown("#### Sensitivity Analysis (Correlation)")
                    
                    # Pick first output for tornado
                    first_out = list(results.sensitivity.keys())[0]
                    sens_data = results.sensitivity[first_out]
                    
                    sens_df = pd.DataFrame(list(sens_data.items()), columns=["Parameter", "Correlation"])
                    sens_df = sens_df.sort_values("Correlation", ascending=True)
                    
                    fig_sens = go.Figure(go.Bar(
                        x=sens_df["Correlation"], y=sens_df["Parameter"],
                        orientation='h', marker_color='#00D4AA',
                        text=sens_df["Correlation"].apply(lambda x: f"{x:.3f}"),
                        textposition='outside'
                    ))
                    fig_sens.update_layout(
                        template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#1E222A',
                        title=f"Sensitivity: {first_out}", height=300,
                        xaxis_title="|Correlation|", yaxis_title=""
                    )
                    st.plotly_chart(fig_sens, use_container_width=True)
                
                # P10/P50/P90 chart
                st.markdown("#### P10 / P50 / P90 Intervals")
                
                out_names = list(results.statistics.keys())
                if len(out_names) > 1:
                    selected_out = st.selectbox("Select Output", out_names)
                else:
                    selected_out = out_names[0]
                
                stats = results.statistics[selected_out]
                samples = results.output_samples[selected_out]
                valid_samples = samples[~np.isnan(samples)]
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=valid_samples, nbinsx=30, 
                                               marker_color='#00D4AA', opacity=0.7, name="Samples"))
                fig_dist.add_vline(x=stats['p10'], line_dash="dash", line_color="#F39C12", 
                                  annotation_text=f"P10: {stats['p10']:,.0f}")
                fig_dist.add_vline(x=stats['p50'], line_dash="dash", line_color="#27AE60", 
                                  annotation_text=f"P50: {stats['p50']:,.0f}")
                fig_dist.add_vline(x=stats['p90'], line_dash="dash", line_color="#00D4AA", 
                                  annotation_text=f"P90: {stats['p90']:,.0f}")
                fig_dist.add_vline(x=stats['mean'], line_dash="dot", line_color="#E74C3C", 
                                  annotation_text=f"Mean: {stats['mean']:,.0f}")
                
                fig_dist.update_layout(template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#1E222A',
                                      title=f"Distribution: {selected_out}", height=300,
                                      showlegend=False, xaxis_title=selected_out, yaxis_title="Frequency")
                st.plotly_chart(fig_dist, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 5: ECONOMICS
# ──────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">💰 Economic Analysis Dashboard</div>', unsafe_allow_html=True)
    
    col_econ_ctrl, col_econ_results = st.columns([1, 2])
    
    with col_econ_ctrl:
        st.markdown("#### Economic Parameters")
        project_life = st.slider("Project Life (years)", 10, 50, 25)
        discount_rate = st.slider("Discount Rate (%)", 2, 15, 6) / 100
        gas_price = st.slider("Gas Price (EUR/MWh)", 10, 200, 35)
        carbon_price = st.slider("Carbon Price (EUR/t CO₂)", 0, 300, 85)
        h2_target = st.slider("H₂ Target (%)", 0, 50, 20)
        capacity_factor = st.slider("Capacity Factor", 0.5, 1.0, 0.9, 0.05)
        
        run_econ = st.button("💰 Run Economic Analysis", type="primary", use_container_width=True)
    
    if run_econ:
        with st.spinner("Computing economics..."):
            eco = EconomicAnalysis(
                project_life_years=project_life,
                discount_rate=discount_rate
            )
            eco.opex.gas_price_eur_mwh = gas_price
            eco.opex.carbon_price_eur_tonne = carbon_price
            
            network_results = {
                "total_length_km": L_km,
                "diameter_mm": D_mm,
                "compression_power_mw": 15 * 1.0,
                "num_compressors": 1,
            }
            
            results = estimate_ireland_h2_project(network_results, h2_target)
        
        with col_econ_results:
            st.success("✅ Economic analysis complete")
            
            # KPI cards
            k1, k2, k3, k4 = st.columns(4)
            
            capex_total = results["capex_breakdown"]["total_eur"]
            opex_total = results["opex_breakdown_eur_year"]["total"]
            npv = results["economic_indicators"]["npv_eur"]
            lcot = results["economic_indicators"]["levelized_cost_eur_gj"]
            
            k1.metric("Total CAPEX", f"€{capex_total/1e9:.2f}B")
            k2.metric("Annual OPEX", f"€{opex_total/1e6:.1f}M")
            k3.metric("NPV", f"€{npv/1e9:.2f}B", delta_color="inverse" if npv < 0 else "normal")
            k4.metric("LCOT", f"€{lcot:.3f}/GJ")
            
            # CAPEX breakdown
            st.markdown("#### CAPEX Breakdown")
            capex_df = pd.DataFrame({
                "Component": ["Pipeline", "Compressors"],
                "Cost (€M)": [
                    results["capex_breakdown"]["pipeline_eur"]/1e6,
                    results["capex_breakdown"]["compressors_eur"]/1e6
                ]
            })
            
            fig_capex = go.Figure(go.Bar(
                x=capex_df["Component"], y=capex_df["Cost (€M)"],
                marker_color=['#1B4F72', '#00D4AA'],
                text=capex_df["Cost (€M)"].apply(lambda x: f"€{x:,.0f}M"),
                textposition='outside'
            ))
            fig_capex.update_layout(template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#1E222A',
                                   height=300, showlegend=False, yaxis_title="Cost (€M)")
            st.plotly_chart(fig_capex, use_container_width=True)
            
            # OPEX breakdown
            st.markdown("#### OPEX Breakdown (Annual)")
            opex_items = {k: v for k, v in results["opex_breakdown_eur_year"].items() if k != "total"}
            opex_df = pd.DataFrame(list(opex_items.items()), columns=["Component", "Cost (€M)"])
            opex_df["Cost (€M)"] = opex_df["Cost (€M)"] / 1e6
            
            fig_opex = go.Figure(go.Pie(
                labels=opex_df["Component"], values=opex_df["Cost (€M)"],
                hole=0.4, marker_colors=['#00D4AA', '#1B4F72', '#F39C12', '#E74C3C', '#8E44AD', '#27AE60', '#3498DB'],
                textinfo='label+percent+value', texttemplate='%{label}<br>%{percent}<br>€%{value:.1f}M'
            ))
            fig_opex.update_layout(template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#1E222A',
                                  height=350, showlegend=False)
            st.plotly_chart(fig_opex, use_container_width=True)
            
            # Sensitivity: NPV vs key parameters
            st.markdown("#### NPV Sensitivity")
            
            gas_prices = np.linspace(15, 80, 10)
            npvs = []
            for gp in gas_prices:
                eco.opex.gas_price_eur_mwh = gp
                r = estimate_ireland_h2_project(network_results, h2_target)
                npvs.append(r["economic_indicators"]["npv_eur"]/1e9)
            
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(x=gas_prices, y=npvs, mode='lines+markers',
                                         line=dict(color='#00D4AA', width=3), marker=dict(size=8)))
            fig_sens.add_hline(y=0, line_dash="dash", line_color="#E74C3C")
            fig_sens.update_layout(template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#1E222A',
                                  title="NPV vs Gas Price", xaxis_title="Gas Price (EUR/MWh)", 
                                  yaxis_title="NPV (€B)", height=300)
            st.plotly_chart(fig_sens, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# TAB 6: EXPORT
# ──────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">📋 Simulation Results Export</div>', unsafe_allow_html=True)
    
    export_df = pd.DataFrame({
        'Parameter': [
            'Case Study', 'Corridor', 'Hydrogen Blend (%)', 'Operating Pressure (bar)', 
            'Pipeline Diameter (mm)', 'Pipeline Length (km)', 'Ambient Temperature (°C)',
            'Steel Grade', 'Blend Molecular Weight (g/mol)', 'Compressibility Factor (Z)',
            'Blend Density (kg/m³)', 'Blend Viscosity (μPa·s)',
            'Mass Heating Value (MJ/kg)', 'Volumetric Heating Value (MJ/m³@STP)',
            'Pressure Drop Weymouth (bar)', 'Pressure Drop Darcy (bar)',
            'Reference Flow (m³/s)', 'Equivalent Energy Flow (m³/s)',
            'Flow Increase Required (%)', 'Gas Velocity (m/s)',
            'Outlet Pressure (bar)', 'Min Allowed Pressure (bar)',
            'Pressure Exceedance', 'Mitigation Required'
        ],
        'Value': [
            CASE_METADATA["name"], selected_corridor_name,
            f"{h2_pct:.1f}", f"{P_bar:.1f}", f"{D_mm:.0f}", f"{L_km:.0f}", f"{T_amb_c:.1f}",
            corridor["steel_grade"], f"{props['M_blend']:.4f}", f"{props['Z_blend']:.4f}",
            f"{props['rho_blend']:.4f}", f"{props['mu_blend']*1e6:.4f}",
            f"{props['hv_mass_blend']:.2f}", f"{props['hv_vol_blend']:.2f}",
            f"{dP_weymouth:.2f}", f"{dp_darcy:.2f}",
            f"{Q_ref:.4f}", f"{Q_equiv:.4f}",
            f"{(Q_equiv/Q_ref-1)*100:.1f}", f"{velocity:.2f}",
            f"{P_out:.2f}", f"{P_min:.1f}",
            "YES" if exceedance else "NO", "YES" if exceedance else "NO"
        ],
        'Unit': ['', '', '%', 'bar', 'mm', 'km', '°C', '', 'g/mol', '', 'kg/m³', 'μPa·s',
                 'MJ/kg', 'MJ/m³', 'bar', 'bar', 'm³/s', 'm³/s', '%', 'm/s', 'bar', 'bar', '', '']
    })
    
    st.dataframe(export_df, use_container_width=True, hide_index=True)
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv = export_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"h2_simulation_{selected_corridor_name.replace(' ', '_').replace('(', '').replace(')', '').replace('→', 'to')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        # Full scenario JSON
        import json
        scenario_json = {
            "metadata": {"version": "2.0", "case": "Ireland-GNI", "timestamp": pd.Timestamp.now().isoformat()},
            "parameters": {
                "h2_blend_pct": h2_pct, "inlet_pressure_bar": P_bar,
                "diameter_mm": D_mm, "length_km": L_km, "ground_temp_c": T_amb_c
            },
            "results": {
                "blend_properties": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                   for k, v in props.items()},
                "hydraulics": {
                    "reference_flow_m3s": Q_ref, "equivalent_flow_m3s": float(Q_equiv),
                    "velocity_ms": velocity, "dp_darcy_bar": dp_darcy,
                    "dp_weymouth_bar": dP_weymouth, "outlet_pressure_bar": P_out
                },
                "risk_assessment": [{
                    "tier": a.tier.value, "title": a.title, "message": a.message, "threshold": a.threshold_pct
                } for a in alerts],
                "exceedance": exceedance,
                "mitigation": "compressor" if exceedance else "none"
            }
        }
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(scenario_json, indent=2),
            file_name=f"h2_scenario_{selected_corridor_name.replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>H₂ Pipeline Digital Twin v2.0</strong> | Core Physics Engine + Ireland (GNI) Case Study<br>
    <em>Peng-Robinson EOS • Wilke Viscosity • Weymouth/Darcy-Weisbach • Network Solver • Monte Carlo UQ • Economics</em><br>
    Engineered by Kudakwashe D. Marara | Petroleum Chemistry & Geospatial Analytics
</div>
""", unsafe_allow_html=True)