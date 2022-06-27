import os
import time
import glob
import pandas as pd
import geopandas as gpd

# retrived from https://github.com/pahbs/geoscitools/blob/master/atl08lib.py


def atl08_io(atl08_csv_output_dir, year_search, do_pickle=True):
    '''Read all ATL08 from CSVs of a given year after extract_filter_atl08.py
        Write to a pickle file by year
        Return a geodataframe
    '''
    dir_pickle = atl08_csv_output_dir
    # print("Building list of ATL08 csvs...")

    all_atl08_csvs = glob.glob(
        os.path.join(
            atl08_csv_output_dir, year_search, 'ATL08*100m.csv'
        ), recursive=True
    )

    atl08_gdf = pd.concat(
        (pd.read_csv(f) for f in all_atl08_csvs),
        sort=False, ignore_index=True)  # <--generator is (), list is []

    atl08_gdf = gpd.GeoDataFrame(
        atl08_gdf, geometry=gpd.points_from_xy(
            atl08_gdf.lon, atl08_gdf.lat), crs='epsg:4326')

    if do_pickle:
        # Pickle the file
        if year_search == "**":
            year_search = 'allyears'
        cur_time = time.strftime("%Y%m%d%H%M")
        out_pickle_fn = os.path.join(
            dir_pickle, f"atl08_{year_search}_filt_gdf_{cur_time}.pkl")
        atl08_gdf.to_pickle(out_pickle_fn)
    return atl08_gdf
