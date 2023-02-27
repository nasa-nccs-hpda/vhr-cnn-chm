import os
import datetime
import pandas as pd
import geopandas as gpd
import branca.colormap as cm
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature

from shapely.geometry import shape
from geopandas import GeoDataFrame
from folium import (
    Map,
    TileLayer,
    GeoJson,
    LayerControl,
    features,
    Figure,
    CircleMarker,
    plugins
)
from rasterio.transform import from_origin
from matplotlib.colors import LinearSegmentedColormap


def MAP_ATL08(
            ATL08_GDF_SUBSET,
            SUBSET_NAME,
            YEAR,
            DIR_MAP,
            TITLE="Canopy height from ICESat-2",
            MAP_COL='h_can',
            YEAR_COL='y',
            proj="+proj=laea +lat_0=60 +lon_0=-100 +x_0=90 +y_0=0 +ellps=GRS80"
        ):

    # Plot obs from night and day
    # My cmap
    forest_ht_cmap = LinearSegmentedColormap.from_list(
        'forest_ht',
        [
            '#636363', '#fc8d59', '#fee08b', '#ffffbf',
            '#d9ef8b', '#91cf60', '#1a9850', '#005a32'
        ],
        12
    )

    print(f"There are {ATL08_GDF_SUBSET.shape[0]} ATL08 observations.")
    print(len(ATL08_GDF_SUBSET.columns))
    xmin, ymin, xmax, ymax = ATL08_GDF_SUBSET.total_bounds

    # Define a projection for the maps and the geodataframe

    # Clip world to ATL08 gdf
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_atl08 = world.cx[xmin:xmax, ymin:ymax]

    atl08_gdf_chull = ATL08_GDF_SUBSET.unary_union.convex_hull

    world_atl08 = world[world.intersects(atl08_gdf_chull)]
    # NA = world[world['continent'] == 'North America'].to_crs(boreal_alb)

    atl08_gdf_proj = ATL08_GDF_SUBSET.to_crs(proj)
    world_atl08_proj = world_atl08.to_crs(proj)

    if 'all' in YEAR:
        YEAR = f'{ATL08_GDF_SUBSET[YEAR_COL].min()}' + \
            f'- {ATL08_GDF_SUBSET[YEAR_COL].max()}'
    ax_map_title = f'{TITLE}: {YEAR}'
    cbar_map_title = f'Canopy height [m] (ATL08: {MAP_COL})'

    d = datetime.date.today().strftime("%Y%b%d")
    # Set up correct number of subplots, space them out.
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), sharex=True, sharey=True)

    bbox = atl08_gdf_proj.total_bounds

    world_atl08_proj.plot(
        ax=ax, facecolor='grey', edgecolor='black',  alpha=0.5)
    hb = ax.hexbin(
        atl08_gdf_proj.geometry.x, atl08_gdf_proj.geometry.y,
        C=atl08_gdf_proj[MAP_COL],
        reduce_C_function=np.median, gridsize=750,
        cmap=forest_ht_cmap, vmax=25, mincnt=1, alpha=0.7)
    world_atl08_proj.plot(
        ax=ax, facecolor='None', edgecolor='black',  alpha=0.9)

    cbar = plt.colorbar(
        hb, extend='max', spacing='proportional',
        orientation='vertical', shrink=0.7, format="%.0f")
    cbar.set_label(label=cbar_map_title, size=16)

    ax.set_xlim(bbox[[0, 2]])
    ax.set_ylim(bbox[[1, 3]])

    ax.set_title(ax_map_title, size=20, loc='left')
    ax.axis('off')
    ax.annotate(
        f'Filtered ATL08 obs: {len(ATL08_GDF_SUBSET)}',
        xy=(0.1, .08), xycoords='figure fraction',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=12,
        color='#555555'
    )
    fig_fn = os.path.join(
        DIR_MAP, 'atl08_alb_'+MAP_COL+'_'+SUBSET_NAME+'_'+YEAR+'_'+d+'.png')
    print(fig_fn)
    plt.savefig(fig_fn)


def MAP_ATL08_POLAR(atl08_gdf, YEAR=2021, MAP_COL='h_can'):

    cbar_map_title = f'Canopy height [m] (ATL08: {MAP_COL})'
    forest_ht_cmap = LinearSegmentedColormap.from_list(
        'forest_ht', [
            '#636363', '#fc8d59',
            '#fee08b', '#ffffbf',
            '#d9ef8b', '#91cf60',
            '#1a9850', '#005a32'
        ], 12
    )

    atl08_gdf = atl08_gdf.to_crs(ccrs.NorthPolarStereo())

    fig = plt.figure(figsize=[10, 5])
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.NorthPolarStereo(),
                      sharex=ax1, sharey=ax1)
    fig.subplots_adjust(bottom=0.05, top=0.95,
                        left=0.04, right=0.95, wspace=0.02)

    # Limit the map to -60 degrees latitude and below.
    ax1.set_extent([-180, 180, 90, 45], ccrs.PlateCarree())

    ax1.add_feature(cartopy.feature.OCEAN)
    ax1.add_feature(cartopy.feature.LAND)

    ax2.add_feature(cartopy.feature.OCEAN, color='white')
    ax2.add_feature(cartopy.feature.LAND, color='grey')

    hb = ax2.hexbin(
        atl08_gdf.geometry.x,
        atl08_gdf.geometry.y,
        C=atl08_gdf[MAP_COL],
        reduce_C_function=np.median,
        gridsize=750,
        cmap=forest_ht_cmap,
        vmax=25, mincnt=1, alpha=0.7
    )

    ax1.gridlines(color='black')
    ax2.gridlines(color='black')

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax2.set_boundary(circle, transform=ax2.transAxes)

    cbar = plt.colorbar(
        hb,
        extend='max',
        spacing='proportional',
        orientation='vertical',
        shrink=0.7,
        format="%.0f"
    )
    cbar.set_label(label=cbar_map_title, size=16)

    plt.show()


basemaps = {
    'Google Terrain': TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Terrain',
        overlay=False,
        control=True
    ),
    'basemap_gray': TileLayer(
        tiles=(
            'http://services.arcgisonline.com/ArcGIS/rest/services/'
            'Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
        ),
        opacity=1,
        name="World gray basemap",
        attr="ESRI",
        overlay=False
    ),
    'Imagery': TileLayer(
        tiles=(
            'https://services.arcgisonline.com/arcgis/rest/services/'
            'World_Imagery/MapServer/tile/{z}/{y}/{x}'
        ),
        opacity=1,
        name="World Imagery",
        attr="ESRI",
        overlay=False
    ),
    'ESRINatGeo': TileLayer(
        tiles=(
            'https://server.arcgisonline.com/ArcGIS/rest/services/'
            'NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
        ),
        opacity=1,
        name='ESRI NatGeo',
        attr='ESRI',
        overlay=False
    )
}


def ADD_ATL08_OBS_TO_MAP(
            atl08_gdf,
            MAP_COL,
            DO_NIGHT,
            NIGHT_FLAG_NAME,
            foliumMap,
            RADIUS=10
        ):

    pal_height_cmap = cm.LinearColormap(
        colors=[
            'black',
            '#636363',
            '#fc8d59',
            '#fee08b',
            '#ffffbf',
            '#d9ef8b',
            '#91cf60',
            '#1a9850'
        ], vmin=0, vmax=25
    )
    pal_height_cmap.caption = f'Vegetation height from  ATL08 ({MAP_COL})'

    night_flg_label = 'day/night'
    if DO_NIGHT:
        atl08_gdf = atl08_gdf[
            atl08_gdf[NIGHT_FLAG_NAME] == 1]
        night_flg_label = 'night'
    print(f'Mapping {len(atl08_gdf)} {night_flg_label} ATL08 of {MAP_COL}')

    # https://stackoverflow.com/questions/61263787/folium-featuregroup-in-python
    # feature_group = folium.FeatureGroup('ATL08')

    atl08_cols_zip_list = [atl08_gdf.lat, atl08_gdf.lon, atl08_gdf[MAP_COL]]

    for lat, lon, ht in zip(*atl08_cols_zip_list):
        ATL08_obs_night = CircleMarker(
            location=[lat, lon],
            radius=RADIUS,
            weight=0.75,
            tooltip=str(round(ht, 2)) + " m",
            fill=True,
            # fill_color=getfill(h_can),
            color=pal_height_cmap(ht),
            opacity=1,
            name=f"ATL08 {night_flg_label} obs"
        )
        ATL08_obs_night.add_to(foliumMap)
    foliumMap.add_child(pal_height_cmap)
    # LayerControl().add_to(foliumMap)
    return foliumMap


def ADD_ATL03_OBS_TO_MAP(atl03_gdf, foliumMap):
    for lat, lon, phclassname, phcolor, elev in \
        zip(
            atl03_gdf.lat,
            atl03_gdf.lon,
            atl03_gdf['class_name'],
            atl03_gdf['color'],
            atl03_gdf['elev']
            ):
        ATL03_obs = CircleMarker(
            location=[lat, lon],
            radius=5,
            weight=1,
            tooltip=phclassname + ": "+str(round(elev, 2)) + " m",
            fill=True,
            # fill_color=getfill(h_can),
            color=phcolor,
            opacity=1,
            name="ATL03 classified photons"
        )
        ATL03_obs.add_to(foliumMap)

    return foliumMap


def MAP_LAYER_FOLIUM(
        LAYER=None,
        LAYER_COL_NAME=None,
        fig_w=1000,
        fig_h=400,
        lat_start=60,
        lon_start=-120,
        zoom_start=8):

    # Map the Layers
    Map_Figure = Figure(width=fig_w, height=fig_h)
    foliumMap = Map(
        tiles=None,
        location=(lat_start, lon_start),
        zoom_start=zoom_start,
        control_scale=True
    )
    Map_Figure.add_child(foliumMap)

    if LAYER is not None:
        GEOJSON_LAYER = GeoJson(
            LAYER,
            name='footprints',
            style_function=lambda x:
                {
                    'fillColor': 'gray',
                    'color': 'red',
                    'weight': 0.75,
                    'opacity': 1,
                    'fillOpacity': 0.5
                },
                tooltip=features.GeoJsonTooltip(
                    fields=[LAYER_COL_NAME],
                    aliases=[f'{LAYER_COL_NAME}:'],
                )
        )
        GEOJSON_LAYER.add_to(foliumMap)

    basemaps['Imagery'].add_to(foliumMap)
    basemaps['basemap_gray'].add_to(foliumMap)
    basemaps['ESRINatGeo'].add_to(foliumMap)

    LayerControl().add_to(foliumMap)
    plugins.Geocoder().add_to(foliumMap)
    plugins.MousePosition().add_to(foliumMap)
    minimap = plugins.MiniMap()
    plugins.Fullscreen().add_to(foliumMap)
    foliumMap.add_child(minimap)

    return foliumMap


def MAP_FOLIUM(
            ADD_LAYER=False,
            LAYER_FN=None,
            basemaps=basemaps,
            fig_w=1000,
            fig_h=400,
            lat_start=5,
            lon_start=-17,
            zoom_start=8,
            LAYER_NAME="HRSI CHM footprints"
        ):
    Map_Figure = Figure(width=fig_w, height=fig_h)
    foliumMap = Map(
        location=(lat_start, lon_start),
        zoom_start=zoom_start, control_scale=True, tiles=None
    )
    Map_Figure.add_child(foliumMap)

    basemaps['Imagery'].add_to(foliumMap)
    basemaps['basemap_gray'].add_to(foliumMap)
    basemaps['ESRINatGeo'].add_to(foliumMap)

    if ADD_LAYER:
        lyr = gpd.read_file(LAYER_FN)
        lyrs = gpd.GeoDataFrame(
            pd.concat([lyr], ignore_index=True))
        lyr_style = {
            'fillColor': 'gray',
            'color': 'red',
            'weight': 0.75,
            'opacity': 1,
            'fillOpacity': 0.5}
        GeoJson(
            lyrs, name=LAYER_NAME,
            style_function=lambda x: lyr_style).add_to(foliumMap)

    LayerControl().add_to(foliumMap)
    plugins.Geocoder().add_to(foliumMap)
    plugins.MousePosition().add_to(foliumMap)
    minimap = plugins.MiniMap()
    plugins.Fullscreen().add_to(foliumMap)
    foliumMap.add_child(minimap)

    return foliumMap


def MAP_ATL08_FOLIUM(
            atl08_gdf,
            MAP_COL='h_can',
            DO_NIGHT=True,
            NIGHT_FLAG_NAME='night_flg',
            ADD_LAYER=True,
            LAYER_FN=None,
            basemaps=basemaps,
            fig_w=1000,
            fig_h=400,
            RADIUS=10
        ):

    if LAYER_FN is None:
        ADD_LAYER = False

    # Map the Layers
    Map_Figure = Figure(width=fig_w, height=fig_h)

    # ------------------
    foliumMap = Map(
        location=(atl08_gdf.lat.mean(), atl08_gdf.lon.mean()),
        zoom_start=8, control_scale=True, tiles=None
    )
    Map_Figure.add_child(foliumMap)

    basemaps['Imagery'].add_to(foliumMap)
    basemaps['basemap_gray'].add_to(foliumMap)
    basemaps['ESRINatGeo'].add_to(foliumMap)

    if ADD_LAYER:
        lyr = gpd.read_file(LAYER_FN)
        lyrs = gpd.GeoDataFrame(
            pd.concat([lyr], ignore_index=True))
        lyr_style = {
            'fillColor': 'gray',
            'color': 'gray',
            'weight': 0.75,
            'opacity': 1,
            'fillOpacity': 0.5
        }
        GeoJson(
            lyrs,
            name="HRSI CHM footprints",
            style_function=lambda x: lyr_style
        ).add_to(foliumMap)

    foliumMap = ADD_ATL08_OBS_TO_MAP(
        atl08_gdf, MAP_COL=MAP_COL, DO_NIGHT=DO_NIGHT,
        NIGHT_FLAG_NAME=NIGHT_FLAG_NAME, foliumMap=foliumMap, RADIUS=RADIUS)
    # foliumMap.add_child(LayerControl()) #LayerControl().add_to(foliumMap)

    LayerControl().add_to(foliumMap)
    # Add fullscreen button
    plugins.Fullscreen().add_to(foliumMap)
    # plugins.Geocoder().add_to(foliumMap)
    plugins.MousePosition().add_to(foliumMap)
    minimap = plugins.MiniMap()
    foliumMap.add_child(minimap)

    return foliumMap


def MAP_ATL03_FOLIUM(
            atl03_gdf,
            basemaps=basemaps,
            fig_w=1000,
            fig_h=400,
            zoom_start=10
        ):

    # Map the Layers
    Map_Figure = Figure(
        width=fig_w, height=fig_h)

    foliumMap = Map(
        # tiles="Stamen Terrain",
        location=(atl03_gdf.lat.mean(), atl03_gdf.lon.mean()),
        zoom_start=zoom_start, control_scale=True, tiles=None
    )
    Map_Figure.add_child(foliumMap)

    foliumMap = ADD_ATL03_OBS_TO_MAP(atl03_gdf, foliumMap)

    basemaps['Imagery'].add_to(foliumMap)
    basemaps['basemap_gray'].add_to(foliumMap)
    LayerControl().add_to(foliumMap)
    plugins.Geocoder().add_to(foliumMap)
    plugins.MousePosition().add_to(foliumMap)
    minimap = plugins.MiniMap()
    foliumMap.add_child(minimap)

    return foliumMap


def GET_FOOTPRINT(r_fn, out_gpkg_fn):
    # TODO get this from raster
    dx = 1
    XY = [504675.55, 7695881.9]
    nx = 1000
    west = XY[0] - (nx * dx)/2
    north = XY[1] + (nx * dx)/2
    Transform = from_origin(west, north, dx, dx)

    # TODO read r_fn with rasterio
    Array = np.zeros(shape=(10, 10))
    Array[2:4, 2:4] = 1
    Array[6:9, 6:7] = 2

    d = {}
    d['val'] = list()
    geometry = list()

    for shp, val in features.shapes(
            Array.astype('int16'), transform=Transform):
        d['val'].append(val)
        geometry.append(shape(shp))
        print('%s: %s' % (val, shape(shp)))
    df = pd.DataFrame(data=d)

    # TODO get crs from raster
    geo_df = GeoDataFrame(
        df, crs={'init': 'EPSG:32608'}, geometry=geometry)
    geo_df['area'] = geo_df.area
    geo_df.to_file(
        'JustSomeRectanglesInTheNWT.shp', driver='GPKG')  # 'ESRI Shapefile'

# TODO: add histograms here
# TODO: add dashboards here
