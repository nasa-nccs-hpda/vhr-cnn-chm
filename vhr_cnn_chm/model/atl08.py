import os
import re
import glob
import time
import logging
import rasterio
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from pathlib import Path
from itertools import repeat
from shapely.geometry import box
from multiprocessing import Pool, cpu_count


class ATL08(object):

    # -------------------------------------------------------------------------
    # IO
    # -------------------------------------------------------------------------
    def atl08_io(
                self,
                atl08_csv_output_dir,
                year_search, do_pickle=True,
                filename_regex='ATL08*.csv'
            ):
        """
        Read all ATL08 from CSVs of a given year after extract_filter_atl08.py
        Write to a pickle file by year
        Return a geodataframe
        Modified https://github.com/pahbs/geoscitools/blob/master/atl08lib.py
        """
        dir_pickle = atl08_csv_output_dir
        filename_regex = os.path.join(
            atl08_csv_output_dir, year_search, filename_regex)

        all_atl08_csvs = glob.glob(filename_regex, recursive=True)
        if len(all_atl08_csvs) < 1:
            logging.info(f"No ATL08 CSVs were found under {filename_regex}")
            return
        logging.info(f"Processing ATL08 CSV: {filename_regex}")

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

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def get_atl08_gdf(
                self,
                data_dir: str,
                start_year: int = 2018,
                end_year: int = 2022,
                reset_index: bool = True,
                crs: str = None
            ):
        """
        Get ATL08 from extracted CSVs.
        """
        dataframes = []
        for year in range(start_year, end_year):
            dataframe = self.atl08_io(data_dir, str(year), do_pickle=False)
            if dataframe is not None:
                dataframes.append(dataframe)
        atl08_gdf = pd.concat(dataframes)

        if crs is not None:
            logging.info(f'No CRS found, setting up: {crs}')
            atl08_gdf = atl08_gdf.to_crs(epsg=crs.split(':')[-1])
        return atl08_gdf.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # utilities
    # -------------------------------------------------------------------------
    def filter_raster_filenames_by_year(
                self, filenames: list,
                start_year: int,
                end_year: int
            ):
        """
        Filter raster filenames list by year.
        """
        new_list = []
        years = [str(year) for year in range(start_year, end_year+1)]
        for f in filenames:
            date_match = re.search(
                r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})', f)
            if date_match['year'] in years:
                new_list.append(f)
        return sorted(new_list)

    def filter_polygon_in_raster(self, filename, atl08_gdf, output_dir):
        """
        Multiprocessing wrapper to process a single point/polygon row.
        TODO:
            options to filter by year
            options to filter by month
        """
        # create polygon row
        date_match = re.search(
            r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})', filename)

        # adding site name
        if os.path.basename(os.path.dirname(filename)) == 'M1BS':
            study_area = os.path.basename(
                os.path.dirname(os.path.dirname(filename)))
        else:
            study_area = os.path.basename(os.path.dirname(filename))

        # open raster and create gdb object
        raster = rasterio.open(filename)
        d = {
            'study_area': study_area,
            'scene_id': filename,
            'wv_year': int(date_match['year']),
            'wv_month': int(date_match['month']),
            'geometry': box(*raster.bounds)
        }
        polygon_row = gpd.GeoDataFrame(d, crs=raster.crs, index=[1])
        atl08_gdf = atl08_gdf.to_crs(raster.crs)

        # intersect both datasets
        intersection = gpd.overlay(atl08_gdf, polygon_row, how='intersection')

        # filter by year
        intersection = intersection[
            intersection['y'] == intersection['wv_year']]

        # set output_dir and create directory
        output_dir = os.path.join(output_dir, study_area)
        os.makedirs(output_dir, exist_ok=True)

        # save geopackage file within the output_dir
        try:
            if intersection.shape[0] > 0:
                intersection.to_file(
                    os.path.join(
                        output_dir, f"{Path(filename).stem}.gpkg"),
                    driver='GPKG', layer='intersection'
                )
        except ValueError:
            return

        return

    def filter_polygon_in_raster_parallel(
                self,
                wv_evhr_gdf,
                atl08_gdf,
                output_dir: str,
                n_processes: int = cpu_count()
            ) -> None:
        """
        Return overalapping points
        """
        logging.info(f"Iterating over {len(wv_evhr_gdf)} rasters")
        logging.info(f"Iterating over {atl08_gdf.shape[0]} points")

        # multiprocessing pool
        p = Pool(processes=n_processes)
        list(
            tqdm(
                p.starmap(
                    self.filter_polygon_in_raster,
                    zip(
                        wv_evhr_gdf,
                        repeat(atl08_gdf),
                        repeat(output_dir)
                    )
                ),
                total=len(wv_evhr_gdf)
            )
        )
        p.close()
        p.join()
        return

    def filter_gdf_by_list(
                self,
                gdf,
                gdf_key: str = 'acq_year',
                isin_list: list = [],
            ):
        """
        Filter GDF by year range.
        """
        return gdf[gdf[gdf_key].isin(isin_list)].reset_index(drop=True)
