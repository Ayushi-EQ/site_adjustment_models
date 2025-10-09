import typer
from pathlib import Path
import pandas as pd
import pygmt
from pygmt_helper import plotting
import shapely
import geojson
import numpy as np
import geopandas as gpd

def load_outline(outline_file: Path) -> list[shapely.Polygon]:
    from shapely.geometry import shape, Polygon, MultiPolygon

    with open(outline_file, "r") as f:
        geojson_data = geojson.load(f)

    polygons = []
    for feature in geojson_data["features"]:
        geom = feature.get("geometry")
        if geom is None:
            continue
        shapely_geom = shape(geom)
        if isinstance(shapely_geom, Polygon):
            polygons.append(shapely_geom)
        elif isinstance(shapely_geom, MultiPolygon):
            polygons.extend(shapely_geom.geoms)

    return polygons


def plot_polygon(
    fig: pygmt.Figure,
    polygon: shapely.LineString
    | shapely.MultiLineString
    | shapely.Polygon
    | shapely.MultiPolygon,
    **kwargs,
) -> None:

    if isinstance(
        polygon, (shapely.MultiPolygon, shapely.MultiLineString)
    ):  # Simplified isinstance check
        for part in polygon.geoms:
            plot_polygon(fig, part, **kwargs)
    elif isinstance(polygon, shapely.LineString):
        coords = np.array(polygon.coords)
        fig.plot(
            x=coords[:, 0],
            y=coords[:, 1],
            **kwargs,
        )
    else:  # Assumes shapely.Polygon
        polygon_coords = np.array(polygon.exterior.coords)
        fig.plot(
            x=polygon_coords[:, 0],
            y=polygon_coords[:, 1],
            **kwargs,
        )
        # Plotting interior rings if any (optional, but good for completeness)
        for interior in polygon.interiors:
            interior_coords = np.array(interior.coords)
            fig.plot(x=interior_coords[:, 0], y=interior_coords[:, 1], **kwargs)

def main(output_path: Path, width: int = 17):
    main_region = [165.00, 180.00, -47.50, -34.00]

    fig = plotting.gen_region_fig(
        region = main_region,
        plot_kwargs={
            "water_color": "white",
            "topo_cmap_min": -900,
            "topo_cmap_max": 3100,
            'topo_cmap': 'gray'
        },
        plot_highways=False, 
        high_res_topo = True,
        config_options=dict(
            MAP_FRAME_TYPE="fancy",
            FORMAT_GEO_MAP="ddd",
            MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="WSne",
            FONT_ANNOT_PRIMARY="14p,Helvetica,black",   
            FONT_LABEL="18p,Helvetica,black",           
        ),
    )
    # Load your 3 core files
    im_obs = pd.read_csv("im_obs.csv")
    station_data = pd.read_csv("Features.csv")
    event_data = pd.read_csv("events.csv")
    plateBoundaryNZ_ffp = Path(r"plateboundary2.ll")
    nz_geometry = gpd.read_file('nz coastlines/nz-coastlines-and-islands-polygons-topo-150k.shp')
    nz_polygon = shapely.MultiPolygon(nz_geometry.geometry)
        
    basin_outlines = {}

    geojson_path = Path('NZVM2p02n')

    for regional_model_path in geojson_path.glob('*.geojson'):  # Renamed for clarity basin_name = regional_model_path.stem
        # This is the original outline from GeoJSON, assumed to be WGS84
        polygons_for_regionmask = load_outline(regional_model_path)
        basin_name = regional_model_path.stem
        basin_outlines[basin_name] = polygons_for_regionmask

    # Intersect basins with NZ geometry
    clipped_basins = {}
    for basin_name, polygons in basin_outlines.items():
        clipped_polygons = [shapely.intersection(nz_polygon, p) for p in polygons if not p.is_empty]
        clipped_basins[basin_name] = [p for p in clipped_polygons if not p.is_empty]

    # Plot clipped basins
    for basin_name, polygons in clipped_basins.items():
        print(f"{basin_name}: {len(polygons)} polygons loaded")
        for poly in polygons:
            if basin_name == "BPVOutcrops_WGS84":
                plot_polygon(fig, poly, fill="#FF6347", pen="0.1p,black", transparency=50)  # Tomato red
            else:
                plot_polygon(fig, poly, fill="#228B22",pen="0.1p,black",transparency=50)
    # Filter to used stations and events
    used_stat_ids = im_obs["stat_id"].unique()
    used_event_ids = im_obs["event_id"].unique()
    stations_used = station_data[station_data["stat_id"].isin(used_stat_ids)]
    events_used = event_data[event_data["event_id"].isin(used_event_ids)]

    # Plot lines between sources and stations
    merged = im_obs.merge(
        station_data[["stat_id", "Latitude", "Longitude"]],
        on="stat_id", how="left"
    ).merge(
        event_data[["event_id", "hlat", "hlon"]],
        on="event_id", how="left"
    )

    for _, row in merged.iterrows():
        fig.plot(
            x=[row["hlon"], row["Longitude"]],
            y=[row["hlat"], row["Latitude"]],
            pen="0.05p,black,solid"
        )

    # Plot events as beachballs scaled by magnitude
    for _, row in events_used.iterrows():
        focal_mechanism = {
            "strike": row["strike"],
            "dip": row["dip"],
            "rake": row["rake"],
            "magnitude": row["mag"],
        }
        scale = 0.06* row["mag"]
        fig.meca(
            spec=focal_mechanism,
            scale=f"{scale}c",
            compressionfill = "firebrick",
            pen="0.05p,black,solid",
            longitude=row["hlon"],
            latitude=row["hlat"],
            depth=row["hdepth"],
    )
    # Plot stations by category
    category_styles = {
        "Basin": {"symbol": "c0.2c", "fill": "#1E90FF", "pen": "0.2p,black"},
        "Basin Edge": {"symbol": "d0.25c", "fill": "#66C266", "pen": "0.2p,black"},
        "Valley": {"symbol": "s0.25c", "fill": "#9370DB", "pen": "0.2p,black"},
        "Hill": {"symbol": "t0.25c", "fill": "#993333", "pen": "0.2p,black"},
    }

    for category, style in category_styles.items():
        subset = stations_used[stations_used["Geomorphology"] == category]
        fig.plot(
            x=subset["Longitude"],
            y=subset["Latitude"],
            style=style["symbol"],
            fill=style["fill"],
            pen=style["pen"],
            label=category
        )





    # Plot plate boundary
    fig.plot(data = plateBoundaryNZ_ffp, pen="1.0p,black")
    fig.text(text="Australian-Pacific", x=177.0, y=-42.35, justify="LM", font="11p,Helvetica,black")
    fig.text(text="Plate Boundary", x=177.0, y=-42.65, justify="LM", font="11p,Helvetica,black")
    fig.plot(x=177.2, y=-41.65, style="v0.6c+eA", direction=[[-75], [1]], pen="0.7p,black")
    
    
    # Christchurch Inset
    inset_region1 = [171.68, 173.0, -43.85, -43.12]

    # Filter stations to those in the inset
    stations_inset = stations_used[
        (stations_used["Longitude"] > inset_region1[0]) &
        (stations_used["Longitude"] < inset_region1[1]) &
        (stations_used["Latitude"] > inset_region1[2]) &
        (stations_used["Latitude"] < inset_region1[3])
    ]

    # Draw inset
    inset_width = 7.5
    with fig.inset(
        position="jBR",
        region=inset_region1,
        projection=f"M{inset_width}c", 
        margin=0,
        box="+p0.7p,black", 
    ):
        # Draw basemap like main map
        plotting.gen_region_fig(
            region=inset_region1,
            plot_kwargs={
                "water_color": "white",
                "topo_cmap_min": -900,
                "topo_cmap_max": 3100,
                'topo_cmap': 'gray'
            },
            plot_highways=False, 
            high_res_topo=True,
            config_options=dict(
                MAP_FRAME_TYPE="fancy",
                FORMAT_GEO_MAP="ddd",
                MAP_GRID_PEN="0.5p,gray",
                MAP_TICK_PEN_PRIMARY="1p,black",
                MAP_FRAME_PEN="1p,black",
                MAP_FRAME_AXES="WSne",
                FONT_ANNOT_PRIMARY="14p,Helvetica,black",  
                FONT_LABEL="18p,Helvetica,black",          
            ),
            fig=fig
        )
        for regional_model_path in geojson_path.glob('*.geojson'):  # Renamed for clarity basin_name = regional_model_path.stem
            # This is the original outline from GeoJSON, assumed to be WGS84
            polygons_for_regionmask = load_outline(regional_model_path)
            basin_name = regional_model_path.stem
            basin_outlines[basin_name] = polygons_for_regionmask

        # Intersect basins with NZ geometry
        clipped_basins = {}
        for basin_name, polygons in basin_outlines.items():
            clipped_polygons = [shapely.intersection(nz_polygon, p) for p in polygons if not p.is_empty]
            clipped_basins[basin_name] = [p for p in clipped_polygons if not p.is_empty]

        # Plot clipped basins
        for basin_name, polygons in clipped_basins.items():
            print(f"{basin_name}: {len(polygons)} polygons loaded")
            for poly in polygons:
                if basin_name == "BPVOutcrops_WGS84":
                    plot_polygon(fig, poly, fill="#FF6347", pen="0.1p,black", transparency=50)  # Tomato red
                else:
                    plot_polygon(fig, poly, fill="#228B22",pen="0.1p,black",transparency=50)

        # Draw inset stations by category
        for category, style in category_styles.items():
            subset = stations_inset[stations_inset["Geomorphology"] == category]
            fig.plot(
                x=subset["Longitude"],
                y=subset["Latitude"],
                style=style["symbol"],
                fill=style["fill"],
                pen=style["pen"],
            )

        # Optional: inset label
        fig.text(text="Christchurch", x=172.38, y=-43.46, font="10p,Helvetica-Bold,black")

    fig.plot(
    data=[[inset_region1[0], inset_region1[2], inset_region1[1], inset_region1[3]]],
    style="r+s",
    pen="2p,black",
    )
    #inset_region1
    fig.plot(x=[173.38,171.68], y=[-47.50, -43.85], pen="0.7p,black,--")
    fig.plot(x=[180.0, 173.0], y=[-43.97, -43.12], pen="0.7p,black,--")
    
      # Wellington inset region
    inset_region2 = [174.68, 175.1, -41.4, -41.1]

    # Filter stations to those in the inset
    stations_inset = stations_used[
        (stations_used["Longitude"] > inset_region2[0]) &
        (stations_used["Longitude"] < inset_region2[1]) &
        (stations_used["Latitude"] > inset_region2[2]) &
        (stations_used["Latitude"] < inset_region2[3])
    ]

    # Draw inset
    inset_width = 7.5
    with fig.inset(
        position="jTL",  
        region=inset_region2,
        projection=f"M{inset_width}c", 
        margin=0,
        box="+p0.7p,black",  
    ):
        # Draw basemap like main map
        plotting.gen_region_fig(
            region=inset_region2,
            plot_kwargs={
                "water_color": "white",
                "topo_cmap_min": -900,
                "topo_cmap_max": 3100,
                'topo_cmap': 'gray'
            },
            plot_highways=False, 
            high_res_topo=True,
            config_options=dict(
                MAP_FRAME_TYPE="fancy",
                FORMAT_GEO_MAP="ddd",
                MAP_GRID_PEN="0.5p,gray",
                MAP_TICK_PEN_PRIMARY="1p,black",
                MAP_FRAME_PEN="1p,black",
                MAP_FRAME_AXES="WSne",
                FONT_ANNOT_PRIMARY="14p,Helvetica,black",   
                FONT_LABEL="18p,Helvetica,black",           
            ),
            fig=fig
        )
        for regional_model_path in geojson_path.glob('*.geojson'):  # Renamed for clarity basin_name = regional_model_path.stem
            # This is the original outline from GeoJSON, assumed to be WGS84
            polygons_for_regionmask = load_outline(regional_model_path)
            basin_name = regional_model_path.stem
            basin_outlines[basin_name] = polygons_for_regionmask

        # Intersect basins with NZ geometry
        clipped_basins = {}
        for basin_name, polygons in basin_outlines.items():
            clipped_polygons = [shapely.intersection(nz_polygon, p) for p in polygons if not p.is_empty]
            clipped_basins[basin_name] = [p for p in clipped_polygons if not p.is_empty]

        # Plot clipped basins
        for basin_name, polygons in clipped_basins.items():
            print(f"{basin_name}: {len(polygons)} polygons loaded")
            for poly in polygons:
                if basin_name == "BPVOutcrops_WGS84":
                    plot_polygon(fig, poly, fill="#FF6347", pen="0.1p,black", transparency=50)  # Tomato red
                else:
                    plot_polygon(fig, poly, fill="#228B22",pen="0.1p,black",transparency=50)

        # Draw inset stations by category
        for category, style in category_styles.items():
            subset = stations_inset[stations_inset["Geomorphology"] == category]
            fig.plot(
                x=subset["Longitude"],
                y=subset["Latitude"],
                style=style["symbol"],
                fill=style["fill"],
                pen=style["pen"],
            )

        # Optional: inset label
        fig.text(text="Wellington", x=174.8, y=-41.25, font="10p,Helvetica-Bold,black")

    fig.plot(
    data=[[inset_region2[0], inset_region2[2], inset_region2[1], inset_region2[3]]],
    style="r+s",
    pen="2p,black",
    )
    #inset_region2
    fig.plot(x=[165,174.68], y=[-39.05, -41.4], pen="0.7p,black,--")
    fig.plot(x=[171.63, 175.1], y=[-34, -41.1], pen="0.7p,black,--")

    with pygmt.config(FONT_ANNOT_PRIMARY="16p"):
        fig.legend(position="JTR+jTR+o0.2c",box="+gwhite+p0.5p,black,solid")
        

    fig.show()
    print(f"Saving figure to file... {output_path}")
    fig.savefig(output_path, dpi=900, anti_alias=True)


main("Figure_1.png")