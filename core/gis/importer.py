"""
GIS integration for gas pipeline networks.

Handles:
- Shapefile ingestion (GNI, ENTSOG, custom)
- Coordinate reference system management
- Network topology extraction from GIS
- Export to network solver format
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Point, MultiLineString
    from shapely.ops import linemerge, unary_union
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None

from core.network.solver import GasNetwork, NetworkNode, NetworkEdge, NodeType, EdgeType, CompressorStation


@dataclass
class GISLayerConfig:
    """Configuration for a GIS layer."""
    path: str
    layer_name: Optional[str] = None
    crs: str = "EPSG:4326"  # WGS84
    # Column mappings
    id_field: str = "id"
    name_field: str = "name"
    diameter_field: Optional[str] = "diameter_mm"
    pressure_field: Optional[str] = "design_pressure_bar"
    material_field: Optional[str] = "material"
    length_field: Optional[str] = "length_km"
    # Node layers
    node_id_field: str = "node_id"
    node_type_field: str = "type"
    demand_field: Optional[str] = "demand_sm3h"
    pressure_setpoint_field: Optional[str] = "pressure_bar"


class GISImporter:
    """
    Import pipeline network from GIS shapefiles/GeoJSON.
    
    Expected layers:
    - pipelines: LineString geometries with attributes
    - nodes: Point geometries (supply, demand, compressor, junction)
    - compressors: Point or separate table
    """
    
    def __init__(self, config: GISLayerConfig):
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas required for GIS import. Install with: pip install geopandas")
        self.config = config
        self.pipes_gdf = None
        self.nodes_gdf = None
        self.compressors_gdf = None
    
    def load_pipelines(self, path: Optional[str] = None) -> 'GISImporter':
        """Load pipeline layer."""
        p = path or self.config.path
        self.pipes_gdf = gpd.read_file(p, layer=self.config.layer_name)
        if self.pipes_gdf.crs is None:
            self.pipes_gdf.set_crs(self.config.crs, inplace=True)
        else:
            self.pipes_gdf.to_crs(self.config.crs, inplace=True)
        return self
    
    def load_nodes(self, path: str, layer_name: Optional[str] = None) -> 'GISImporter':
        """Load nodes layer."""
        self.nodes_gdf = gpd.read_file(path, layer=layer_name)
        if self.nodes_gdf.crs is None:
            self.nodes_gdf.set_crs(self.config.crs, inplace=True)
        else:
            self.nodes_gdf.to_crs(self.config.crs, inplace=True)
        return self
    
    def load_compressors(self, path: str, layer_name: Optional[str] = None) -> 'GISImporter':
        """Load compressor stations layer."""
        self.compressors_gdf = gpd.read_file(path, layer=layer_name)
        if self.compressors_gdf.crs is not None:
            self.compressors_gdf.to_crs(self.config.crs, inplace=True)
        return self
    
    def build_network(self, network_name: str = "GIS Network") -> GasNetwork:
        """Build GasNetwork from loaded GIS layers."""
        net = GasNetwork(network_name)
        
        if self.nodes_gdf is not None:
            self._import_nodes(net)
        
        if self.pipes_gdf is not None:
            self._import_pipes(net)
        
        if self.compressors_gdf is not None:
            self._import_compressors(net)
        
        return net
    
    def _import_nodes(self, net: GasNetwork):
        """Import nodes from GeoDataFrame."""
        for idx, row in self.nodes_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            lat, lon = geom.y, geom.x
            
            node_id = str(row.get(self.config.node_id_field, f"node_{idx}"))
            name = str(row.get(self.config.name_field, node_id))
            node_type_str = str(row.get(self.config.node_type_field, "junction")).lower()
            
            # Map type
            type_map = {
                "supply": NodeType.SUPPLY,
                "entry": NodeType.SUPPLY,
                "demand": NodeType.DEMAND,
                "exit": NodeType.DEMAND,
                "city_gate": NodeType.DEMAND,
                "junction": NodeType.JUNCTION,
                "compressor": NodeType.COMPRESSOR,
                "storage": NodeType.STORAGE,
            }
            node_type = type_map.get(node_type_str, NodeType.JUNCTION)
            
            # Pressure/flow attributes
            pressure = row.get(self.config.pressure_setpoint_field) if self.config.pressure_setpoint_field else None
            demand = row.get(self.config.demand_field, 0) if self.config.demand_field else 0
            
            node = NetworkNode(
                node_id=node_id,
                name=name,
                node_type=node_type,
                lat=lat,
                lon=lon,
                pressure_bar=float(pressure) if pressure is not None else None,
                demand_sm3h=float(demand) if demand else 0.0,
            )
            net.add_node(node)
    
    def _import_pipes(self, net: GasNetwork):
        """Import pipes from GeoDataFrame."""
        for idx, row in self.pipes_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            # Handle MultiLineString
            if geom.geom_type == 'MultiLineString':
                geom = linemerge(geom)
            
            if geom.geom_type != 'LineString':
                continue
            
            # Get coordinates
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            
            # Start/end points
            start_lat, start_lon = coords[0][1], coords[0][0]
            end_lat, end_lon = coords[-1][1], coords[-1][0]
            
            # Length (calculate from geometry if not in attributes)
            if self.config.length_field and self.config.length_field in row:
                length_km = float(row[self.config.length_field])
            else:
                # Approximate from geometry (degrees to km)
                length_km = self._geometry_length_km(geom)
            
            # Diameter
            if self.config.diameter_field and self.config.diameter_field in row:
                diameter_mm = float(row[self.config.diameter_field])
            else:
                diameter_mm = 500.0  # Default
            
            # Pressure
            design_pressure = 70.0
            if self.config.pressure_field and self.config.pressure_field in row:
                design_pressure = float(row[self.config.pressure_field])
            
            edge_id = str(row.get(self.config.id_field, f"pipe_{idx}"))
            
            # For topology, we need to match to nodes
            # This is simplified - real implementation would snap to nearest nodes
            from_node = f"node_{idx}_start"
            to_node = f"node_{idx}_end"
            
            # Create implicit nodes if not in node layer
            if from_node not in net.nodes:
                net.add_node(NetworkNode(
                    node_id=from_node,
                    name=f"{edge_id}_start",
                    node_type=NodeType.JUNCTION,
                    lat=start_lat,
                    lon=start_lon,
                ))
            if to_node not in net.nodes:
                net.add_node(NetworkNode(
                    node_id=to_node,
                    name=f"{edge_id}_end",
                    node_type=NodeType.JUNCTION,
                    lat=end_lat,
                    lon=end_lon,
                ))
            
            edge = NetworkEdge(
                edge_id=edge_id,
                from_node=from_node,
                to_node=to_node,
                length_km=length_km,
                diameter_mm=diameter_mm,
            )
            net.add_edge(edge)
    
    def _import_compressors(self, net: GasNetwork):
        """Import compressor stations."""
        for idx, row in self.compressors_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            lat, lon = geom.y, geom.x
            comp_id = str(row.get(self.config.id_field, f"comp_{idx}"))
            name = str(row.get(self.config.name_field, comp_id))
            
            # Find nearest nodes for suction/discharge
            # Simplified - would need topology matching
            suction_node = f"{comp_id}_suction"
            discharge_node = f"{comp_id}_discharge"
            
            comp = CompressorStation(
                compressor_id=comp_id,
                name=name,
                suction_node=suction_node,
                discharge_node=discharge_node,
                max_power_mw=float(row.get('max_power_mw', 50.0)),
            )
            net.add_compressor(comp)
    
    def _geometry_length_km(self, geom: LineString) -> float:
        """Approximate length in km from WGS84 geometry."""
        # Simple approximation: 1 deg lat ≈ 111 km, 1 deg lon ≈ 111*cos(lat) km
        coords = list(geom.coords)
        total_km = 0.0
        for i in range(len(coords) - 1):
            lat1, lon1 = coords[i][1], coords[i][0]
            lat2, lon2 = coords[i+1][1], coords[i+1][0]
            dlat = (lat2 - lat1) * 111.0
            dlon = (lon2 - lon1) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
            total_km += np.sqrt(dlat**2 + dlon**2)
        return total_km


def create_sample_geojson(output_path: str):
    """Create sample GeoJSON for Ireland network (for testing without shapefiles)."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    
    # Pipelines
    pipes = [
        {
            "id": "pipe_1", "name": "Inchicore-J1",
            "geometry": LineString([(-6.3158, 53.3298), (-7.0, 53.2)]),
            "diameter_mm": 500, "design_pressure_bar": 70, "length_km": 50
        },
        {
            "id": "pipe_2", "name": "J1-J2",
            "geometry": LineString([(-7.0, 53.2), (-8.0, 52.5)]),
            "diameter_mm": 500, "design_pressure_bar": 70, "length_km": 80
        },
        {
            "id": "pipe_3", "name": "J2-Whitegate",
            "geometry": LineString([(-8.0, 52.5), (-8.25, 51.8333)]),
            "diameter_mm": 500, "design_pressure_bar": 70, "length_km": 90
        },
        {
            "id": "pipe_4", "name": "J2-Cork",
            "geometry": LineString([(-8.0, 52.5), (-8.4756, 51.8985)]),
            "diameter_mm": 400, "design_pressure_bar": 70, "length_km": 20
        },
    ]
    
    pipes_gdf = gpd.GeoDataFrame(pipes, crs="EPSG:4326")
    pipes_gdf.to_file(output_path.replace('.geojson', '_pipes.geojson'), driver='GeoJSON')
    
    # Nodes
    nodes = [
        {"node_id": "inchicore", "name": "Inchicore", "type": "supply", 
         "geometry": Point(-6.3158, 53.3298), "pressure_bar": 70},
        {"node_id": "whitegate", "name": "Whitegate", "type": "supply",
         "geometry": Point(-8.25, 51.8333), "pressure_bar": 70},
        {"node_id": "dublin_cg", "name": "Dublin City Gate", "type": "demand",
         "geometry": Point(-6.2603, 53.3498), "demand_sm3h": 1800000},
        {"node_id": "cork_cg", "name": "Cork City Gate", "type": "demand",
         "geometry": Point(-8.4756, 51.8985), "demand_sm3h": 600000},
        {"node_id": "j1", "name": "Junction 1", "type": "junction",
         "geometry": Point(-7.0, 53.2)},
        {"node_id": "j2", "name": "Junction 2", "type": "junction",
         "geometry": Point(-8.0, 52.5)},
    ]
    
    nodes_gdf = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
    nodes_gdf.to_file(output_path.replace('.geojson', '_nodes.geojson'), driver='GeoJSON')
    
    print(f"Sample GeoJSON created at {output_path}_pipes.geojson and {output_path}_nodes.geojson")


# Add numpy import for the geometry length calculation
import numpy as np