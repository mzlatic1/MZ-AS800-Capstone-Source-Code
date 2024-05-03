# Title: Wildfire Prediction Methodology Part B
# Author: Marko Zlatic
# Date: 2024-05-03 (yyyy-mm-dd)
# University: Johns Hopkins University
# Program: Master of Science in Geographic Information Systems (GIS)
# Purpose: This code was written to uphold the requirements for the AS800 GIS Capstone

from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from sqlalchemy import create_engine
from shapely.geometry import Point
from sklearn.cluster import KMeans
import pyspark.sql.functions as F
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
import psycopg2
import sys
import os
import gc


def part_b(master_folder, postgres_database_name, postgres_host, postgres_port, postgres_username, postgres_password):
    # Script Objective: Engineer the WFIL dataset for the Part C machine learning script

    def mercator_proj(coord_value, field_type: str):
        # Convert longitude and latitude values to their meters East (mE) and meters North (mN) values respectively.
        # Formula for web Mercator can be found here: https://github.com/mraad/WebMercator/blob/master/src/main/python/webmercator/__init__.py
        if field_type == 'x':
            x = coord_value * 111319.490793274
            return float(x)
        elif field_type == 'y':
            sin = np.sin(coord_value * np.pi / 180)
            y = np.log((1 + sin) / (1 - sin)) * 3189068.5
            return float(y)
        else:
            print('Invalid field type, please try again.')
            return None

    input_gdb = os.path.join(master_folder, "input_datasets.gdb")
    # Prep feature class dictionary
    # The format is as follows: {path_to_feature_class: [{original_field_name: new_field_name}, field_to_keep]}
    fcs = {
        # The original shapefile for the US Boundaries dataset was first converted into a feature class dataset using
        # the Arcpy Export Features Function
        os.path.join(input_gdb, 'us_boundaries'): ['STATE'],
        os.path.join(input_gdb, 'OUTPUT_gpw_v4_land_water_area_rev11_landareakm_2pt5_min'): [{"gridcode": "land_area"}],
        os.path.join(input_gdb, 'OUTPUT_gpw_v4_land_water_area_rev11_waterareakm_2pt5_min'): [{"gridcode": "water_area"}],
        os.path.join(input_gdb, 'GDEM_10km_BW'): [{"gridcode": "elevation"}],
        os.path.join(input_gdb, 'hwsd'): [{"gridcode": "soil_type"}],
        os.path.join(input_gdb, 'population_count'): [{"gridcode": "population_count"}, 'year'],
        os.path.join(input_gdb, 'population_density'): [{"gridcode": "population_density"}, 'year'],
        os.path.join(input_gdb, 'tba'): [{"gridcode": "tba"}],  # total basal area
        os.path.join(input_gdb, 'sdi'): [{"gridcode": "sdi"}],  # total stand density index
        # The original feature layer for the USFS and BLM Campsite dataset was first converted into a feature class
        # dataset using the Arcpy Export Features Function
        os.path.join(input_gdb, 'usfs_blm_campsites'): [],
        os.path.join(input_gdb, 'Incidents'): ['FireDiscoveryDateTime'], # WFIL dataset
    }

    # Establish connection with PostgreSQL (Postgres) database
    params = {
        'database': postgres_database_name,
        'host': postgres_host,
        'password': postgres_password,
        'port': postgres_port,
        'user': postgres_username
    }

    connection = psycopg2.connect(dbname=params['database'],
                                  user=params['user'],
                                  password=params['password'],
                                  host=params['host'],
                                  port=params['port'])
    cursor = connection.cursor()

    postgres = create_engine(
        f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}")

    # Convert each feature classes to Postgres tables
    cursor.execute("create schema if not exists input_data;")
    connection.commit()

    pg_tables = {}
    total_fcs = len(list(fcs.keys()))

    wgs84_epsg = 4326
    wfil_dataset = ''
    for i, fc in enumerate(list(fcs.keys())):
        print(f"Processing {i+1} out of {total_fcs}")
        print(f'\tFeature class: {fc}\n')

        # Load feature class as a geodataframe
        gdb = os.path.split(fc)[0]
        name = os.path.split(fc)[1]
        gdf = gpd.read_file(gdb, layer=name, engine='pyogrio', use_arrow=True)

        if gdf.crs.to_epsg() != wgs84_epsg:
            gdf = gdf.to_crs(epsg=wgs84_epsg)

        if 'sdi' in name or 'tba' in name:
            print(
                '\nWARNING, BOTH THE TBA AND SDI DATASETS WILL CONSUME AROUND 120GB OF RAM WHEN EXPORTING TO POSTGRES.\n')

        # Iterate through the feature class dictionary to keep the fields that are required
        keep_fields = []
        for f in fcs[fc]:
            if type(f) == dict:
                for key in list(f.keys()):
                    gdf.rename(columns=f, inplace=True)
                    keep_fields.append(f[key])
            else:
                keep_fields.append(f)

        if len(name) > 50:
            name = name[25:] if not name[25:].startswith('_') else name[26:]

        # Execute field calculations
        pg_table_name = name.lower()
        if 'incidents' in pg_table_name:
            gdf['year_num'] = gdf['FireDiscoveryDateTime'].dt.year
            keep_fields.append('year_num')
            fcs[fc].append('year_num')
            wfil_dataset = fc
        if i > 8:
            gdf['longitude'] = gdf.geometry.x
            gdf['mE'] = gdf['longitude'].apply(lambda row: mercator_proj(row, 'x'))
            gdf['latitude'] = gdf.geometry.y
            gdf['mN'] = gdf['latitude'].apply(lambda row: mercator_proj(row, 'y'))
            for geo_c in ['longitude', 'latitude', 'mE', 'mN']:
                keep_fields.append(geo_c)
                fcs[fc].append(geo_c)
        else:
            # Populate a dictionary with the output Postgres table names, the format is as follows: {table_name: [fields]}
            # Only applies to polygon feature classes
            pg_tables[pg_table_name] = fcs[fc]

        # Write to Postgres
        keep_fields.append('geometry')
        gdf = gdf[keep_fields]
        gdf.reset_index(names=['orig_oid']).to_postgis(pg_table_name, postgres, schema='input_data', if_exists='replace')

        # Clear memory
        del gdf
        gc.collect()

        # Generate indexes for each field
        in_table = f'{postgres_database_name}.input_data.{name.lower()}'
        for idx, field in enumerate(keep_fields + ['orig_oid']):
            is_uppercase = False
            for letter in field:
                if letter.isupper():
                    is_uppercase = True
                    break
            if field != 'geometry': # geometry is already indexed via geopandas
                if is_uppercase:
                    cursor.execute(f"""create index "{pg_table_name}_{field.lower()}_{idx}" ON {in_table}("{field}");""")
                else:
                    cursor.execute(f"create index {pg_table_name}_{field}_{idx} ON {in_table}({field});")
                connection.commit()

    # Prep spatial join using the PostGIS Postgres extension

    def generate_col_str(left_cols: list, right_cols: list) -> str:
        # Create a string that can be used as input for the main spatial join function. The left tables respected columns
        # are referred to as 'fire_points', since the spatial join function will be used exclusively for the WFIL dataset.
        # The right tables respected columns are referred to as 'poly_table' as these fields are needed to be derived from
        # the raster-converted tables (all tables having polygon geometry).
        def populate_col_str(col_list: list):
            # Ensure the column string is properly formatted and is SQL compliant for Postgres databases
            output_col_str = ''
            for column in col_list:
                add_double_quote = False
                for letter in column:
                    if letter.isupper():
                        add_double_quote = True
                        break
                if add_double_quote:
                    col_split = column.split('.')
                    output_col_str += f'{col_split[0]}."{col_split[1]}", '
                else:
                    output_col_str += f'{column}, '
            return output_col_str
        right_cols_str = populate_col_str(['poly_table.' + rc for rc in right_cols if rc not in ['geometry', 'year']])
        left_cols_str = populate_col_str(['fire_points.' + lc for lc in left_cols])
        if len(right_cols_str) > 1 and right_cols_str.split(', ')[-1] != 'poly_table., ':
            return left_cols_str + right_cols_str[:-2]
        else:
            return left_cols_str[:-2]

    # Prep the preliminary variables for the spatial join iteration where the WFIL dataset is being declared as the
    # first left table of the iteration
    left_columns = []
    for col in fcs[wfil_dataset]:
        left_columns.append(col)
    left_columns.append('geometry')
    left_columns = ['orig_oid'] + left_columns
    prev_table = f'{postgres_database_name}.input_data.incidents'

    def spat_join(table1: str, table2: str, generated_col_str_input: str, out_table_name: str) -> str:
        # An inner join spatial join function where the 'fire_points' table is a point-geometry table and the
        # 'poly_table' is a polygon-geometry table. The generated_col_str_input variable can be derived using
        # the generate_col_str() function.
        return f"""
        create table {out_table_name} as
            select {generated_col_str_input} from {table1} as fire_points
        inner join {table2} as poly_table
        on st_intersects(fire_points.geometry, poly_table.geometry);
        """

    cursor.execute("create schema if not exists out_tables;")
    connection.commit()

    # Execute spatial joins
    finalized_table = ''
    pg_keys = list(pg_tables.keys())
    idx = 0

    while idx != len(pg_keys):
        name = pg_keys[idx]
        table = pg_tables[name]

        print(f'\nProcessing table {idx + 1} of {len(list(pg_tables.keys()))}')
        print(f'\tTable: {pg_keys[idx]}')

        # Get the columns needed from polygon table
        right_columns = []
        for i, col in enumerate(table):
            if type(col) != dict and 'year' not in col and col != 'orig_oid':
                right_columns.append(col)
            elif type(col) == dict:
                for pg_c in list(col.keys()):
                    if table[i][pg_c] != 'year':
                        right_columns.append(table[i][pg_c])

        # Prep variables
        out_table = f"{postgres_database_name}.out_tables.out_table_{idx + 1}"
        input_table = f"{postgres_database_name}.input_data.{name}"

        generated_col_str = generate_col_str(left_columns, right_columns)
        spat_join_str = spat_join(prev_table, input_table, generated_col_str, out_table)
        adjusted_spat_join_str = spat_join_str.replace(';', '')

        # Run spatial join function
        print('\tExecuting computation...')
        cursor.execute(f"drop table if exists {out_table};")
        connection.commit()
        if 'population' in name: # Join by year and location for population datasets
            cursor.execute(adjusted_spat_join_str + """ and (
              (fire_points.year_num <= 2000 and poly_table.year  = 2000) or
              (fire_points.year_num >= 2001 and fire_points.year_num <= 2005 and poly_table.year = 2005) or
              (fire_points.year_num >= 2006 and fire_points.year_num <= 2010 and poly_table.year = 2010) or
              (fire_points.year_num >= 2011 and fire_points.year_num <= 2019 and poly_table.year = 2015) or
              (fire_points.year_num >= 2020 and poly_table.year = 2020)
            );""")
        else:
            cursor.execute(spat_join_str)
        connection.commit()

        # Generate indexes for each field
        print('\tGenerate indexes...')
        for lr_col in generated_col_str.split(', '):
            actual_col = lr_col.replace('fire_points.', '').replace('poly_table.', '')
            if actual_col != 'geometry':
                if '"' in lr_col:
                    cursor.execute(f"""create index "{actual_col.replace('"', '')}_{idx}" ON {out_table}({actual_col});""")
                else:
                    cursor.execute(f"create index {actual_col}_{idx} ON {out_table}({actual_col});")
            else:
                cursor.execute(f"CREATE INDEX spat_index_{idx + 1} ON {out_table} USING gist (geometry);")
            connection.commit()

        # Populate the left_columns array with the newly joined fields
        for r_col in right_columns:
            left_columns.append(r_col)

        # Before the iteration completes, export the last output table and order by the FireDiscoveryDateTime field
        if idx == len(pg_keys) - 1:
            print('Ordering last out table by date and exporting...')
            finalized_table = out_table.replace(f"out_table_{idx + 1}", "almost_finalized_parent_table_ordered")

            cursor.execute(f"drop table if exists {finalized_table};")
            connection.commit()

            cursor.execute(f"""create table {finalized_table} as select * from {out_table} order by "FireDiscoveryDateTime";""")
            connection.commit()

        prev_table = out_table
        idx += 1

    # Finish preprocessing calculations
    crs = f'EPSG:{wgs84_epsg}'
    print('Loading preprocessed WFIL dataset...')
    ordered_parent_df = gpd\
        .read_postgis(f'select * from {finalized_table}', postgres, geom_col='geometry', crs=crs)\
        .drop_duplicates(subset=['orig_oid'])\
        .reset_index(drop=True)

    # Longitude/Latitude cluster calculations
    print('Calculating longitude/latitude clusters...')
    kmeans = KMeans(random_state=42, n_init='auto').fit(ordered_parent_df[['longitude', 'latitude']])
    long_lat_clusters = [[mercator_proj(coords[0], 'x'), mercator_proj(coords[1], 'y')] for coords in kmeans.cluster_centers_]

    ordered_parent_df['cluster_id'] = kmeans.labels_

    def calc_dist_from_centers(cluster_array, cluster_id, x2, y2):
        # A function to calculate the 2D Euclidean distance between an input point and the respected cluster center
        x1, y1 = cluster_array[int(cluster_id)]

        x_sqrd = (x2 - x1) ** 2
        y_sqrd = (y2 - y1) ** 2

        return np.sqrt(x_sqrd + y_sqrd)

    ordered_parent_df['dist_from_center'] = ordered_parent_df[['cluster_id', 'mE', 'mN']]\
        .apply(lambda row: calc_dist_from_centers(long_lat_clusters, row['cluster_id'], row['mE'], row['mN']), axis=1)

    # Load KMeans function for non-coordinate-based clustering

    def run_kmeans_calculations(input_df, field1, field2, out_field_prefix):
        # Execute KMeans clustering for non-coordinate-based clustering, then populate the input dataframe with the
        # respected cluster IDs and distances to cluster centers.
        kmeans_results = KMeans(random_state=42, n_init='auto').fit(input_df[[field1, field2]])
        cluster_id_field = out_field_prefix + '_cluster_id'
        input_df[cluster_id_field] = kmeans_results.labels_
        clusters = []

        for cluster_id in sorted(list(input_df[cluster_id_field].unique())):
            query_by_id = input_df.query(f"{cluster_id_field} == {cluster_id}")
            clusters.append([np.median(query_by_id['mE']), np.median(query_by_id['mN'])])

        distance_field_name = out_field_prefix + '_dist_from_center'
        input_df[distance_field_name] = input_df[[cluster_id_field, 'mE', 'mN']].apply(
            lambda row: calc_dist_from_centers(clusters, row[cluster_id_field], row['mE'], row['mN']), axis=1)

    # Population count and population density cluster calculations
    print('Calculating population clusters...')
    run_kmeans_calculations(ordered_parent_df, 'population_count', 'population_density', 'pop')

    # Tree density and thickness cluster calculations
    print('Calculating SDI and TBA clusters...')
    run_kmeans_calculations(ordered_parent_df, 'sdi', 'tba', 'sdi_tba')

    # Start spark session
    print('Starting Spark session...')
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession \
        .builder \
        .appName('WFIL_Preprocessing') \
        .master(f'local[{os.cpu_count() - 4}]') \
        .config("spark.driver.memory", "110G") \
        .config("spark.executor.memory", "110G") \
        .config("spark.driver.maxResultSize", "20G") \
        .config("spark.hadoop.fs.s3a.path.style.access", True) \
        .config('spark.network.timeout', 10000000) \
        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
        .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
        .getOrCreate()

    # Fire count by state calculation
    print('Calculating total number of fires by State...')
    output_folder = os.path.join(master_folder, 'output_datasets')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fire_count_by_state = ordered_parent_df['STATE'].value_counts().sort_values(ascending=True).reset_index(
        name='state').reset_index(names=['label_val'])
    fire_count_by_state.to_csv(os.path.join(output_folder, 'state_ordinal_encoder.csv'), index=False)

    b_fire_count_by_state = spark.sparkContext.broadcast(fire_count_by_state)

    @udf(IntegerType())
    def populate_fire_count_by_state(state_col: str) -> int:
        # Get the encoded State value and replace it with the original value
        return int(b_fire_count_by_state.value.query(f"STATE == '{state_col}'")['label_val'].values.tolist()[0])

    fire_count_by_state_sdf = spark\
        .createDataFrame(ordered_parent_df[['STATE']])\
        .withColumn('encoded_state', populate_fire_count_by_state(F.col('STATE')))\
        .toPandas()
    ordered_parent_df['STATE'] = fire_count_by_state_sdf['encoded_state']

    # Campground cluster id calculation
    print('Calculating campsite clusters...')
    campgrounds = gpd.read_postgis(f'select * from {postgres_database_name}.input_data.campgrounds_exportfeatures;',
                                   postgres, geom_col='geometry', crs=crs)
    kmeans = KMeans(random_state=42, n_init='auto').fit(campgrounds[['longitude', 'latitude']])
    campgrounds['campground_centers'] = kmeans.labels_

    campground_cluster_id_w_coords = {}
    for label in sorted(list(campgrounds['campground_centers'].unique())):
        query = campgrounds.query(f"campground_centers == {label}")
        median_long = np.median(query['mE'])
        median_lat = np.median(query['mN'])
        campground_cluster_id_w_coords[label] = [median_long, median_lat]
    b_camp_clusters = spark.sparkContext.broadcast(campground_cluster_id_w_coords)

    @udf(IntegerType())
    def populate_campground_clusterid(mE: float, mN: float) -> float:
        # Populate each input points respected campsite cluster id by calculating the 2D Euclidean distance and taking
        # the cluster id that had the shortest distance
        dist_formula = lambda x2, y2, x1, y1: np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        camp_cluster_id = -1
        current_min_dist = 0
        for idx, c_id in enumerate(list(b_camp_clusters.value.keys())):
            coords = b_camp_clusters.value[c_id]
            curr_dist = float(np.min(dist_formula(coords[0], coords[1], mE, mN)))
            if idx == 0 or curr_dist < current_min_dist:
                current_min_dist = curr_dist
                camp_cluster_id = c_id

        return int(camp_cluster_id)

    camp_cluster_ids = spark\
        .createDataFrame(ordered_parent_df[['mE', 'mN']])\
        .withColumn('camp_cluster_id', populate_campground_clusterid(F.col('mE'), F.col('mN')))\
        .toPandas()
    ordered_parent_df['camp_cluster_id'] = camp_cluster_ids['camp_cluster_id'].values

    # Distance to closest state boundary calculation
    print('Calculating distance to closest State boundary...')
    us_boundaries = gpd.read_postgis(f'select * from {postgres_database_name}.input_data.us_boundaries;',
                                     postgres, geom_col='geometry', crs=crs)

    boundaries = gpd.GeoSeries([b.boundary for b in us_boundaries.geometry], crs=crs).to_crs(epsg=3857)
    b_boundaries = spark.sparkContext.broadcast(boundaries)

    @udf(FloatType())
    def calc_dist_to_closest_state_boundary(mN: float, mE: float) -> float:
        # Calculates the distance between an input point and line geometry and take the smallest value
        geom = gpd.GeoSeries([Point(mE, mN)], crs='EPSG:3857')
        return float(np.min([dist.distance(geom) for dist in b_boundaries.value]))

    state_distances = spark\
        .createDataFrame(ordered_parent_df[['mN', 'mE']])\
        .withColumn('dist_to_closest_state_boundary',
                    calc_dist_to_closest_state_boundary(F.col('mN'), F.col('mE')))\
        .toPandas()
    ordered_parent_df['dist_to_closest_state_boundary'] = state_distances['dist_to_closest_state_boundary'].values

    # Counting nearest neighbor calculations of all time
    print('Calculating nearest neighbors...')
    geom_ordered_parent_df = ordered_parent_df[['mN', 'mE']]
    b_geom_ordered_parent_df_points = spark.sparkContext.broadcast(geom_ordered_parent_df)

    @udf(IntegerType())
    def calc_num_of_closest_points(mN: float, mE: float, radius_m: int) -> int:
        # Count the number of points that fall within the input radius (in meters)
        query = f"mN <= {mN + radius_m} and mN >= {mN - radius_m} and mE <= {mE + radius_m} and mE >= {mE - radius_m}"
        return len(b_geom_ordered_parent_df_points.value.query(query).index)

    starting_meter = 10_000
    while starting_meter < 1_000_001:
        print(starting_meter, 'meters')

        num_points_df = spark\
            .createDataFrame(ordered_parent_df[['mN', 'mE']])\
            .withColumn(
            f'num_points_within_meters_{starting_meter}',
                    calc_num_of_closest_points(F.col('mN'), F.col('mE'), F.lit(starting_meter))
            )\
            .toPandas()
        ordered_parent_df[f'num_points_within_meters_{starting_meter}'] = \
            num_points_df[f'num_points_within_meters_{starting_meter}'].values

        starting_meter *= 10

    # Remaining cluster calculations
    print('Calculate remaining cluster centers and distances...')

    for fields in [['pop_dist_from_center', 'num_points_within_meters_1000000'],
                   ['pop_dist_from_center', 'sdi_tba_dist_from_center']]:
        print(fields)
        run_kmeans_calculations(ordered_parent_df, fields[0], fields[1], f'cluster_{fields[0]}_{fields[1]}')

    # Export finalized dataframe
    print('Exporting finalized WFIL dataset...')
    ordered_parent_df.to_parquet(os.path.join(output_folder, "finalized_preprocessed_parent_df.parquet"))
    spark.stop()

    return output_folder


if __name__ == '__main__':
    error = False
    for i in range(1, len(sys.argv)):
        if sys.argv[i] is None:
            error = True
            break
    if not error:
        start_time = datetime.datetime.now()
        part_b(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        end_time = datetime.datetime.now()
        print('Script complete! Duration:', (end_time - start_time).total_seconds() / 60, 'minutes.')
        sys.exit(0)
    else:
        print('ERROR: Invalid/absent arguments.')
        sys.exit(1)
