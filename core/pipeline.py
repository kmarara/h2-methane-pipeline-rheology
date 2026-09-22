"""
Pipeline segment and network definitions for geospatial modeling.
"""

from dataclasses import dataclass
from typing import List, Optional
import folium


@dataclass(frozen=True)
class PipelineSegment:
    """A single pipeline segment with geographic coordinates."""
    name: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    length_km: float
    diameter_mm: int
    design_pressure_bar: float
    steel_grade: str = "API 5L X65"
    notes: str = ""


@dataclass(frozen=True)
class CompressorStation:
    """Compressor station location and basic specs."""
    name: str
    lat: float
    lon: float
    power_mw: float
    suction_pressure_bar: float
    discharge_pressure_bar: float


class PipelineNetwork:
    """Collection of segments and stations for a geographic region."""

    def __init__(self, name: str, region: str):
        self.name = name
        self.region = region
        self.segments: List[PipelineSegment] = []
        self.stations: List[CompressorStation] = []

    def add_segment(self, segment: PipelineSegment):
        self.segments.append(segment)

    def add_station(self, station: CompressorStation):
        self.stations.append(station)

    def total_length_km(self) -> float:
        return sum(s.length_km for s in self.segments)

    def to_folium(self, h2_pct: float, risk_colors: dict) -> folium.Map:
        """Generate Folium map with risk-tier coloring."""
        # Center on network centroid
        lats = [s.start_lat for s in self.segments] + [s.end_lat for s in self.segments]
        lons = [s.start_lon for s in self.segments] + [s.end_lon for s in self.segments]
        center = [sum(lats)/len(lats), sum(lons)/len(lons)]

        m = folium.Map(location=center, zoom_start=7, tiles='OpenStreetMap')

        # Determine risk tier for this H2 blend
        from core.materials import embrittlement_tier, RiskTier
        tier, title, _ = embrittlement_tier(h2_pct)
        color = risk_colors.get(tier, risk_colors[RiskTier.LOW])

        # Add segments
        for seg in self.segments:
            folium.PolyLine(
                locations=[[seg.start_lat, seg.start_lon], [seg.end_lat, seg.end_lon]],
                color=color, weight=6, opacity=0.8,
                tooltip=f"{seg.name}: {title} ({h2_pct:.1f}% H₂)",
                popup=f"{seg.name}\nLength: {seg.length_km:.0f} km\nDia: {seg.diameter_mm} mm\n"
                      f"Design P: {seg.design_pressure_bar:.0f} bar\nGrade: {seg.steel_grade}\n"
                      f"H₂ Blend: {h2_pct:.1f}% → {title}"
            ).add_to(m)

        # Add compressor stations
        for st in self.stations:
            folium.CircleMarker(
                location=[st.lat, st.lon], radius=8,
                color='#1B4F72', fill=True, fill_color='#1B4F72',
                popup=f"{st.name}\nPower: {st.power_mw:.0f} MW\n"
                      f"Suction: {st.suction_pressure_bar:.0f} bar\n"
                      f"Discharge: {st.discharge_pressure_bar:.0f} bar"
            ).add_to(m)

        # Legend
        legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: 140px;
                    background-color: white; border:2px solid grey; z-index:9999; font-size:13px;
                    padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <b>H₂ Embrittlement Risk Tier</b><br>
        <i style="background:{risk_colors[RiskTier.LOW]};width:18px;height:18px;display:inline-block;margin-right:8px;"></i> Low Risk (<10%)<br>
        <i style="background:{risk_colors[RiskTier.MODERATE]};width:18px;height:18px;display:inline-block;margin-right:8px;"></i> Moderate (10-25%)<br>
        <i style="background:{risk_colors[RiskTier.HIGH]};width:18px;height:18px;display:inline-block;margin-right:8px;"></i> High Risk (>25%)<br>
        <hr style="margin:5px 0;">
        <b>Current Blend: {h2_pct:.1f}% H₂</b> → {title}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        return m


def create_ireland_network() -> PipelineNetwork:
    """Build the Gas Networks Ireland transmission corridor network."""
    net = PipelineNetwork("GNI Transmission Corridors", "Ireland")

    # Dublin-Cork corridor (Inchicore to Whitegate)
    net.add_segment(PipelineSegment(
        name="Dublin-Cork (Inchicore → Whitegate)",
        start_lat=53.3298, start_lon=-6.3158,
        end_lat=51.8333, end_lon=-8.2500,
        length_km=220, diameter_mm=500, design_pressure_bar=70,
        steel_grade="API 5L X65",
        notes="Primary east-south corridor, 20\" mainline"
    ))

    # Galway-Dublin corridor
    net.add_segment(PipelineSegment(
        name="Galway-Dublin",
        start_lat=53.2707, start_lon=-9.0568,
        end_lat=53.3498, end_lon=-6.2603,
        length_km=180, diameter_mm=400, design_pressure_bar=70,
        steel_grade="API 5L X52",
        notes="West-east corridor, 16\" mainline"
    ))

    # Cork-Limerick (spur)
    net.add_segment(PipelineSegment(
        name="Cork-Limerick",
        start_lat=51.8333, start_lon=-8.2500,
        end_lat=52.6638, end_lon=-8.6267,
        length_km=95, diameter_mm=300, design_pressure_bar=50,
        steel_grade="API 5L X52",
        notes="Southwest spur"
    ))

    # Compressor stations (major)
    net.add_station(CompressorStation(
        name="Inchicore Compressor Station",
        lat=53.3298, lon=-6.3158, power_mw=45,
        suction_pressure_bar=45, discharge_pressure_bar=70
    ))
    net.add_station(CompressorStation(
        name="Ballytrasna Compressor Station",
        lat=52.1000, lon=-8.4000, power_mw=30,
        suction_pressure_bar=40, discharge_pressure_bar=65
    ))

    return net