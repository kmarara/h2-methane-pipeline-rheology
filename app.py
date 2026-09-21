import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import folium
from streamlit_folium import st_folium
import io
from dataclasses import dataclass
from typing import Tuple

st.set_page_config(
    page_title="Ireland Green H2 Pipeline Optimizer & Digital Twin",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A4A4A;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: #FFF3CD;
        border: 1px solid #FFC107;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
    .alert-danger {
        background: #F8D7DA;
        border: 1px solid #DC3545;
        border-radius: 8px;
        padding: 1rem;
        color: #721C24;
    }
    .alert-success {
        background: #D4EDDA;
        border: 1px solid #28A745;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    .stSlider > div > div > div > div {
        background: #1B4F72;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class GasProperties:
    name: str
    molecular_weight: float
    critical_temp: float
    critical_press: float
    acentric_factor: float
    heating_value_mj_kg: float
    heating_value_mj_m3_stp: float
    viscosity_ref: float
    temp_ref: float

CH4 = GasProperties(
    name="Methane",
    molecular_weight=16.043,
    critical_temp=190.56,
    critical_press=45.99e5,
    acentric_factor=0.011,
    heating_value_mj_kg=55.5,
    heating_value_mj_m3_stp=35.8,
    viscosity_ref=1.12e-5,
    temp_ref=273.15
)

H2 = GasProperties(
    name="Hydrogen",
    molecular_weight=2.016,
    critical_temp=33.19,
    critical_press=12.97e5,
    acentric_factor=-0.216,
    heating_value_mj_kg=141.8,
    heating_value_mj_m3_stp=12.7,
    viscosity_ref=8.92e-6,
    temp_ref=273.15
)

R_UNIVERSAL = 8.314462618

def peng_robinson_z(T: float, P: float, gas: GasProperties) -> float:
    Tr = T / gas.critical_temp
    Pr = P / gas.critical_press
    kappa = 0.37464 + 1.54226 * gas.acentric_factor - 0.26992 * gas.acentric_factor**2
    alpha = (1 + kappa * (1 - Tr**0.5))**2
    a = 0.45724 * (R_UNIVERSAL**2 * gas.critical_temp**2) / gas.critical_press * alpha
    b = 0.07780 * R_UNIVERSAL * gas.critical_temp / gas.critical_press
    A = a * P / (R_UNIVERSAL**2 * T**2)
    B = b * P / (R_UNIVERSAL * T)
    coeffs = [1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    return float(max(real_roots)) if len(real_roots) > 0 else 1.0

def blend_properties(h2_frac: float, T: float, P: float) -> Tuple[float, float, float, float, float]:
    y_h2 = h2_frac
    y_ch4 = 1 - h2_frac
    M_blend = y_h2 * H2.molecular_weight + y_ch4 * CH4.molecular_weight
    Z_h2 = peng_robinson_z(T, P, H2)
    Z_ch4 = peng_robinson_z(T, P, CH4)
    Z_blend = y_h2 * Z_h2 + y_ch4 * Z_ch4
    rho_blend = (P * M_blend) / (Z_blend * R_UNIVERSAL * T) / 1000.0
    mu_h2 = H2.viscosity_ref * (T / H2.temp_ref)**0.68
    mu_ch4 = CH4.viscosity_ref * (T / CH4.temp_ref)**0.92
    phi_h2_ch4 = (1 + (mu_h2/mu_ch4)**0.5 * (CH4.molecular_weight/H2.molecular_weight)**0.25)**2 / (8 * (1 + H2.molecular_weight/CH4.molecular_weight))**0.5
    phi_ch4_h2 = (1 + (mu_ch4/mu_h2)**0.5 * (H2.molecular_weight/CH4.molecular_weight)**0.25)**2 / (8 * (1 + CH4.molecular_weight/H2.molecular_weight))**0.5
    mu_blend = (y_h2 * mu_h2) / (y_h2 + y_ch4 * phi_h2_ch4) + (y_ch4 * mu_ch4) / (y_ch4 + y_h2 * phi_ch4_h2)
    hv_mass_blend = y_h2 * H2.heating_value_mj_kg + y_ch4 * CH4.heating_value_mj_kg
    hv_vol_blend = y_h2 * H2.heating_value_mj_m3_stp + y_ch4 * CH4.heating_value_mj_m3_stp
    return M_blend, Z_blend, rho_blend, mu_blend, hv_mass_blend, hv_vol_blend

def weymouth_pressure_drop(Q: float, L: float, D: float, rho: float, mu: float, P_avg: float, Z: float, T: float) -> float:
    D_m = D / 1000.0
    L_m = L * 1000.0
    epsilon = 4.57e-5
    Re = 4 * rho * Q / (np.pi * D_m * mu)
    if Re < 2300:
        f = 64 / Re
    else:
        f = 0.25 / (np.log10(epsilon/D_m/3.7 + 5.74/Re**0.9))**2
    dP = (f * L_m * rho * (Q/(np.pi*D_m**2/4))**2) / (2 * D_m) / 1e5
    return dP

def darcy_weisbach_pressure_drop(Q: float, L: float, D: float, rho: float, mu: float) -> float:
    D_m = D / 1000.0
    L_m = L * 1000.0
    epsilon = 4.57e-5
    v = Q / (np.pi * D_m**2 / 4)
    Re = rho * v * D_m / mu
    if Re < 2300:
        f = 64 / Re
    else:
        f = 0.25 / (np.log10(epsilon/D_m/3.7 + 5.74/Re**0.9))**2
    dP = f * (L_m / D_m) * (rho * v**2 / 2) / 1e5
    return dP

def calculate_scenario(h2_pct: float, P_bar: float, D_mm: float, L_km: float, T_amb_c: float) -> dict:
    h2_frac = h2_pct / 100.0
    P_pa = P_bar * 1e5
    T_k = T_amb_c + 273.15
    M_blend, Z_blend, rho_blend, mu_blend, hv_mass_blend, hv_vol_blend = blend_properties(h2_frac, T_k, P_pa)
    Q_ref = 1.0
    dP_weymouth = weymouth_pressure_drop(Q_ref, L_km, D_mm, rho_blend, mu_blend, P_pa, Z_blend, T_k)
    dP_darcy = darcy_weisbach_pressure_drop(Q_ref, L_km, D_mm, rho_blend, mu_blend)
    energy_throughput_mj_s = Q_ref * hv_vol_blend * P_bar / 1.01325 / Z_blend * T_k / 273.15
    Q_equiv = Q_ref * (CH4.heating_value_mj_m3_stp / hv_vol_blend)
    velocity = Q_equiv / (np.pi * (D_mm/1000/2)**2)
    return {
        "h2_pct": h2_pct,
        "P_bar": P_bar,
        "D_mm": D_mm,
        "L_km": L_km,
        "T_amb_c": T_amb_c,
        "M_blend": M_blend,
        "Z_blend": Z_blend,
        "rho_blend": rho_blend,
        "mu_blend": mu_blend,
        "hv_mass_blend": hv_mass_blend,
        "hv_vol_blend": hv_vol_blend,
        "dP_weymouth_per_km": dP_weymouth / L_km,
        "dP_darcy_per_km": dP_darcy / L_km,
        "dP_total_weymouth": dP_weymouth,
        "dP_total_darcy": dP_darcy,
        "Q_ref": Q_ref,
        "Q_equiv": Q_equiv,
        "velocity": velocity,
        "energy_throughput_mj_s": energy_throughput_mj_s,
        "flow_increase_pct": (Q_equiv / Q_ref - 1) * 100
    }

def generate_blend_curves(P_bar: float, D_mm: float, L_km: float, T_amb_c: float) -> pd.DataFrame:
    blends = np.linspace(0, 50, 51)
    results = []
    for b in blends:
        res = calculate_scenario(b, P_bar, D_mm, L_km, T_amb_c)
        results.append(res)
    return pd.DataFrame(results)

def create_pressure_drop_plot(df: pd.DataFrame, current_h2: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
    ax.plot(df['h2_pct'], df['dP_total_weymouth'], 'b-', linewidth=2.5, label='Weymouth Equation', color='#1B4F72')
    ax.plot(df['h2_pct'], df['dP_total_darcy'], 'r--', linewidth=2, label='Darcy-Weisbach', color='#E74C3C')
    ax.axvline(x=current_h2, color='#F39C12', linestyle=':', linewidth=2, label=f'Current: {current_h2:.1f}% H₂')
    ax.set_xlabel('Hydrogen Blending Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure Drop (bar)', fontsize=12, fontweight='bold')
    ax.set_title('Pipeline Pressure Drop vs. Hydrogen Blend Ratio', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.set_facecolor('#F8F9FA')
    fig.tight_layout()
    return fig

def create_energy_flow_plot(df: pd.DataFrame, current_h2: float) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='#FFFFFF')
    ax1.plot(df['h2_pct'], df['hv_vol_blend'], 'g-', linewidth=2.5, label='Volumetric Energy Density (MJ/m³@STP)', color='#27AE60')
    ax1.set_xlabel('Hydrogen Blending Ratio (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy Density (MJ/m³)', fontsize=12, fontweight='bold', color='#27AE60')
    ax1.tick_params(axis='y', labelcolor='#27AE60')
    ax1.axvline(x=current_h2, color='#F39C12', linestyle=':', linewidth=2)
    ax2 = ax1.twinx()
    ax2.plot(df['h2_pct'], df['flow_increase_pct'], 'm-', linewidth=2.5, label='Volumetric Flow Increase (%)', color='#8E44AD')
    ax2.set_ylabel('Flow Increase Required (%)', fontsize=12, fontweight='bold', color='#8E44AD')
    ax2.tick_params(axis='y', labelcolor='#8E44AD')
    ax1.set_title('Energy Density Loss vs. Volumetric Flow Requirements', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', framealpha=0.9, fontsize=10)
    ax1.set_facecolor('#F8F9FA')
    fig.tight_layout()
    return fig

def create_folium_map(h2_pct: float) -> folium.Map:
    m = folium.Map(location=[53.1424, -7.6921], zoom_start=7, tiles='CartoDB positron')
    inchicore = [53.3298, -6.3158]
    whitegate = [51.8333, -8.2500]
    galway = [53.2707, -9.0568]
    dublin = [53.3498, -6.2603]
    if h2_pct < 10:
        color = '#27AE60'
        risk = 'Low Risk (<10%)'
    elif h2_pct < 25:
        color = '#F39C12'
        risk = 'Moderate Risk (10-25%)'
    else:
        color = '#E74C3C'
        risk = 'High Risk (>25%) - Infrastructure Upgrade Required'
    folium.PolyLine([inchicore, whitegate], color=color, weight=6, opacity=0.8,
                    tooltip=f'Dublin-Cork Corridor: {risk}', popup=f'H₂ Blend: {h2_pct:.1f}%').add_to(m)
    folium.PolyLine([galway, dublin], color=color, weight=6, opacity=0.8,
                    tooltip=f'Galway-Dublin Corridor: {risk}', popup=f'H₂ Blend: {h2_pct:.1f}%').add_to(m)
    folium.CircleMarker(inchicore, radius=8, color='#1B4F72', fill=True, fill_color='#1B4F72',
                        popup='Inchicore (Dublin) - Compressor Station').add_to(m)
    folium.CircleMarker(whitegate, radius=8, color='#1B4F72', fill=True, fill_color='#1B4F72',
                        popup='Whitegate Refinery (Cork) - Entry Point').add_to(m)
    folium.CircleMarker(galway, radius=8, color='#1B4F72', fill=True, fill_color='#1B4F72',
                        popup='Galway - Entry Point').add_to(m)
    folium.CircleMarker(dublin, radius=8, color='#1B4F72', fill=True, fill_color='#1B4F72',
                        popup='Dublin - Demand Center').add_to(m)
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 280px; height: 120px;
                background-color: white; border:2px solid grey; z-index:9999; font-size:13px;
                padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
    <b>H₂ Embrittlement Risk Tier</b><br>
    <i style="background:{color};width:18px;height:18px;display:inline-block;margin-right:8px;"></i> {risk}<br>
    <i style="background:#27AE60;width:18px;height:18px;display:inline-block;margin-right:8px;"></i> Low Risk (<10%)<br>
    <i style="background:#F39C12;width:18px;height:18px;display:inline-block;margin-right:8px;"></i> Moderate (10-25%)<br>
    <i style="background:#E74C3C;width:18px;height:18px;display:inline-block;margin-right:8px;"></i> High Risk (>25%)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def main():
    st.markdown('<div class="main-header">🇮🇪 Ireland Green H₂ Pipeline Optimizer & Digital Twin</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Engineered by Kudakwashe D. Marara | Petroleum Chemistry & Geospatial Analytics</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Simulation Parameters")
        h2_pct = st.slider("Hydrogen Blending Ratio (%)", 0, 50, 20, 1,
                          help="Volumetric percentage of H₂ in natural gas blend")
        P_bar = st.slider("Operating Pressure (bar)", 10, 80, 70, 1,
                         help="Standard Irish transmission pressure: 70 bar")
        D_mm = st.slider("Pipeline Internal Diameter (mm)", 200, 1000, 500, 50,
                        help="Typical Irish transmission pipeline diameters")
        L_km = st.number_input("Pipeline Length (km)", 10, 500, 220, 10,
                              help="Cork-Dublin corridor ≈ 220 km")
        T_amb_c = st.slider("Ambient Ground Temperature (°C)", 0, 20, 10, 1,
                           help="Irish sub-surface annual average ≈ 10°C")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Metrics")
        T_k = T_amb_c + 273.15
        P_pa = P_bar * 1e5
        h2_frac = h2_pct / 100.0
        M_blend, Z_blend, rho_blend, mu_blend, hv_mass_blend, hv_vol_blend = blend_properties(h2_frac, T_k, P_pa)
        scenario = calculate_scenario(h2_pct, P_bar, D_mm, L_km, T_amb_c)
        
        st.metric("Blend Density", f"{rho_blend:.3f} kg/m³")
        st.metric("Blend Viscosity", f"{mu_blend*1e6:.3f} μPa·s")
        st.metric("Compressibility Z", f"{Z_blend:.4f}")
        st.metric("Energy Density", f"{hv_vol_blend:.2f} MJ/m³")
        st.metric("Flow Increase Req.", f"{scenario['flow_increase_pct']:.1f}%")
        st.metric("Gas Velocity", f"{scenario['velocity']:.2f} m/s")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🗺️ Geospatial Pipeline Corridor Map")
        m = create_folium_map(h2_pct)
        st_folium(m, width=700, height=500, returned_objects=[])
    
    with col2:
        st.markdown("### 📈 Engineering Analysis Charts")
        df_curves = generate_blend_curves(P_bar, D_mm, L_km, T_amb_c)
        
        fig1 = create_pressure_drop_plot(df_curves, h2_pct)
        st.pyplot(fig1)
        
        fig2 = create_energy_flow_plot(df_curves, h2_pct)
        st.pyplot(fig2)
    
    st.markdown("---")
    st.markdown("### ⚠️ Engineering Risk Assessment & Alerts")
    
    alerts = []
    if h2_pct > 20:
        alerts.append(("danger", f"HIGH RISK: {h2_pct:.0f}% H₂ exceeds 20% threshold for API 5L Grade X52/X65 steel pipelines. "
                        "Hydrogen embrittlement risk significantly elevated per ASME B31.12 & IGEM/TD/13. "
                        "Requires fracture mechanics assessment and potential derating."))
    elif h2_pct > 10:
        alerts.append(("warning", f"MODERATE RISK: {h2_pct:.0f}% H₂ in 10-25% range. "
                        "Increased fatigue crack growth rates expected. "
                        "Enhanced inspection intervals (ILI) and pressure cycling monitoring recommended per GNI standards."))
    else:
        alerts.append(("success", f"LOW RISK: {h2_pct:.0f}% H₂ within safe operational envelope for existing Irish transmission assets. "
                        "Standard monitoring protocols sufficient per current GNI operating procedures."))
    
    if scenario['velocity'] > 20:
        alerts.append(("warning", f"COMPRESSOR ALERT: Gas velocity {scenario['velocity']:.1f} m/s exceeds 20 m/s erosional limit. "
                        "Compressor station re-sizing and valve trim upgrades required."))
    elif scenario['velocity'] > 15:
        alerts.append(("warning", f"Elevated velocity: {scenario['velocity']:.1f} m/s. Monitor for erosion at bends/tees."))
    
    if scenario['dP_total_weymouth'] > P_bar * 0.3:
        alerts.append(("danger", f"PRESSURE DROP EXCEEDANCE: ΔP = {scenario['dP_total_weymouth']:.1f} bar "
                        f"({scenario['dP_total_weymouth']/P_bar*100:.1f}% of inlet). "
                        "Intermediate compression or looping required for this corridor."))
    elif scenario['dP_total_weymouth'] > P_bar * 0.15:
        alerts.append(("warning", f"Elevated pressure drop: {scenario['dP_total_weymouth']:.1f} bar "
                        f"({scenario['dP_total_weymouth']/P_bar*100:.1f}% of inlet). "
                        "Verify compressor capacity margins."))
    
    for alert_type, msg in alerts:
        if alert_type == "danger":
            st.markdown(f'<div class="alert-danger"><strong>🔴 CRITICAL:</strong> {msg}</div>', unsafe_allow_html=True)
        elif alert_type == "warning":
            st.markdown(f'<div class="alert-warning"><strong>🟡 WARNING:</strong> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><strong>🟢 NOMINAL:</strong> {msg}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Simulation Results Export")
    
    export_df = pd.DataFrame([{
        'Parameter': 'Hydrogen Blend (%)', 'Value': f"{h2_pct:.1f}", 'Unit': '%',
        'Parameter': 'Operating Pressure', 'Value': f"{P_bar:.1f}", 'Unit': 'bar',
        'Parameter': 'Pipeline Diameter', 'Value': f"{D_mm:.0f}", 'Unit': 'mm',
        'Parameter': 'Pipeline Length', 'Value': f"{L_km:.0f}", 'Unit': 'km',
        'Parameter': 'Ambient Temperature', 'Value': f"{T_amb_c:.1f}", 'Unit': '°C',
        'Parameter': 'Blend Molecular Weight', 'Value': f"{M_blend:.4f}", 'Unit': 'g/mol',
        'Parameter': 'Compressibility Factor (Z)', 'Value': f"{Z_blend:.4f}", 'Unit': '-',
        'Parameter': 'Blend Density', 'Value': f"{rho_blend:.4f}", 'Unit': 'kg/m³',
        'Parameter': 'Blend Viscosity', 'Value': f"{mu_blend*1e6:.4f}", 'Unit': 'μPa·s',
        'Parameter': 'Mass Heating Value', 'Value': f"{hv_mass_blend:.2f}", 'Unit': 'MJ/kg',
        'Parameter': 'Volumetric Heating Value (STP)', 'Value': f"{hv_vol_blend:.2f}", 'Unit': 'MJ/m³',
        'Parameter': 'Pressure Drop (Weymouth)', 'Value': f"{scenario['dP_total_weymouth']:.2f}", 'Unit': 'bar',
        'Parameter': 'Pressure Drop (Darcy-Weisbach)', 'Value': f"{scenario['dP_total_darcy']:.2f}", 'Unit': 'bar',
        'Parameter': 'Reference Flow Rate', 'Value': f"{scenario['Q_ref']:.4f}", 'Unit': 'm³/s',
        'Parameter': 'Equivalent Energy Flow Rate', 'Value': f"{scenario['Q_equiv']:.4f}", 'Unit': 'm³/s',
        'Parameter': 'Flow Increase Required', 'Value': f"{scenario['flow_increase_pct']:.1f}", 'Unit': '%',
        'Parameter': 'Gas Velocity', 'Value': f"{scenario['velocity']:.2f}", 'Unit': 'm/s',
        'Parameter': 'Energy Throughput', 'Value': f"{scenario['energy_throughput_mj_s']:.2f}", 'Unit': 'MJ/s'
    }])
    
    export_df = pd.DataFrame({
        'Parameter': ['Hydrogen Blend (%)', 'Operating Pressure', 'Pipeline Diameter', 'Pipeline Length',
                      'Ambient Temperature', 'Blend Molecular Weight', 'Compressibility Factor (Z)',
                      'Blend Density', 'Blend Viscosity', 'Mass Heating Value',
                      'Volumetric Heating Value (STP)', 'Pressure Drop (Weymouth)',
                      'Pressure Drop (Darcy-Weisbach)', 'Reference Flow Rate',
                      'Equivalent Energy Flow Rate', 'Flow Increase Required',
                      'Gas Velocity', 'Energy Throughput'],
        'Value': [f"{h2_pct:.1f}", f"{P_bar:.1f}", f"{D_mm:.0f}", f"{L_km:.0f}",
                  f"{T_amb_c:.1f}", f"{M_blend:.4f}", f"{Z_blend:.4f}",
                  f"{rho_blend:.4f}", f"{mu_blend*1e6:.4f}", f"{hv_mass_blend:.2f}",
                  f"{hv_vol_blend:.2f}", f"{scenario['dP_total_weymouth']:.2f}",
                  f"{scenario['dP_total_darcy']:.2f}", f"{scenario['Q_ref']:.4f}",
                  f"{scenario['Q_equiv']:.4f}", f"{scenario['flow_increase_pct']:.1f}",
                  f"{scenario['velocity']:.2f}", f"{scenario['energy_throughput_mj_s']:.2f}"],
        'Unit': ['%', 'bar', 'mm', 'km', '°C', 'g/mol', '-', 'kg/m³', 'μPa·s',
                 'MJ/kg', 'MJ/m³', 'bar', 'bar', 'm³/s', 'm³/s', '%', 'm/s', 'MJ/s']
    })
    
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Simulation Results (CSV)",
        data=csv,
        file_name="simulation_results.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.dataframe(export_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <strong>Ireland Green H₂ Pipeline Optimizer & Digital Twin</strong> | 
        Built for Gas Networks Ireland Climate Action Plan Alignment | 
        Computational Fluid Dynamics • Thermodynamics • Geospatial Analytics<br>
        <em>Equations: Peng-Robinson EOS • Wilke Viscosity Mixing • Weymouth/Darcy-Weisbach Pressure Drop</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()