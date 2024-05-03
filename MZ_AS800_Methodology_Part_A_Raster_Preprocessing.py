# Title: Wildfire Prediction Methodology Part A
# Author: Marko Zlatic
# Date: 2024-05-03 (yyyy-mm-dd)
# University: Johns Hopkins University
# Program: Master of Science in Geographic Information Systems (GIS)
# Purpose: This code was written to uphold the requirements for the AS800 GIS Capstone

import datetime
import arcpy
import sys
import os


def part_a(master_folder, wfil_dataset):
    # Script Objective: converts raster datasets to feature class datasets.

    arcpy.env.overwriteOutput = True

    out_gdb = os.path.join(master_folder, 'input_datasets.gdb')
    if not arcpy.Exists(out_gdb):
        arcpy.management.CreateFileGDB(
            out_folder_path=os.path.split(out_gdb)[0],
            out_name=os.path.split(out_gdb)[1]
        )

    # There are two additional vector datasets to process, as well as the primary WFIL dataset
    vector_datasets = {
        'us_boundaries': os.path.join(master_folder, 's_05mr24.shp'),
        'usfs_blm_campsites':
            'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Campgrounds/FeatureServer/0',
        'Incidents': wfil_dataset
    }
    print('Importing vector datasets...')
    for dataset in list(vector_datasets.keys()):
        arcpy.conversion.ExportFeatures(
            in_features=vector_datasets[dataset],
            out_features=os.path.join(out_gdb, dataset)
        )

    # Import datasets that do not require raster calculations
    print('Importing raster datasets without additional calculations...')
    raster_to_polys = [
        os.path.join(master_folder, 'GDEM-10km-BW.tif'), # Elevation dataset
        os.path.join(master_folder, 'hwsd.bil'), # Soil type dataset
        os.path.join(master_folder, 'L48_Totals.gdb', 'sdi'), # SDI dataset
        os.path.join(master_folder, 'L48_Totals.gdb', 'tba') # TBA dataset
    ]

    for raster in raster_to_polys:
        print('Converting:', raster)
        name_output = arcpy.ValidateTableName(os.path.split(raster)[1].split('.')[0], out_gdb)
        arcpy.RasterToPolygon_conversion(
            in_raster=raster,
            out_polygon_features=os.path.join(out_gdb, name_output),
            simplify="SIMPLIFY",
            raster_field="Value",
            create_multipart_features="SINGLE_OUTER_PART"
        )

    # Import datasets that require raster calculations
    scratch_part_a_folder = os.path.join(master_folder, 'scratch_part_a_outputs')
    if not os.path.exists(scratch_part_a_folder):
        os.makedirs(scratch_part_a_folder)

    print('Importing raster datasets with additional calculations...')
    rasters_with_additional_calcs = [
        'gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_2020_2pt5_min.tif', # 2020 pop dense
        'gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_2015_2pt5_min.tif', # 2015 pop dense
        'gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_2010_2pt5_min.tif', # 2010 pop dense
        'gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_2005_2pt5_min.tif', # 2005 pop dense
        'gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_2000_2pt5_min.tif', # 2000 pop dense

        "gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2000_2pt5_min.tif", # 2000 pop count
        "gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2005_2pt5_min.tif", # 2005 pop count
        "gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2010_2pt5_min.tif", # 2010 pop count
        "gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2015_2pt5_min.tif", # 2015 pop count
        "gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_2pt5_min.tif", # 2020 pop count

        'gpw_v4_land_water_area_rev11_landareakm_2pt5_min.tif', # land area dataset
        'gpw_v4_land_water_area_rev11_waterareakm_2pt5_min.tif' # water area dataset
    ]

    population_count_fcs = []
    population_density_fcs = []
    for index, file in enumerate(rasters_with_additional_calcs):
        print(index + 1, 'out of ', len(rasters_with_additional_calcs))
        if os.path.isfile(file):
            raster_file = file
        else:
            raster_file = os.path.join(master_folder, file)

        # Convert raster to a binary float file
        r_float = arcpy.RasterToFloat_conversion(
            in_raster=raster_file,
            out_float_file=os.path.join(scratch_part_a_folder, file.replace('.tif', '.FLT'))
        )

        calculated_raster = os.path.join(out_gdb, 'calculated_' + os.path.split(file)[-1].replace('.tif', ''))
        r_int = os.path.join(out_gdb, 'int_' + os.path.split(file)[-1].replace('.tif', ''))

        # Multiply raster values by 10
        with arcpy.EnvManager(scratchWorkspace=out_gdb):
            output_raster = arcpy.sa.RasterCalculator(
                rasters=[r_float[0]],
                input_names=['x'],
                expression='x * 10'
            )
            output_raster.save(calculated_raster)

            out_raster = arcpy.sa.Int(
                in_raster_or_constant=calculated_raster
            )
            out_raster.save(r_int)

        # Convert to polygon feature class dataset
        out_fc = os.path.join(out_gdb,
                              arcpy.ValidateTableName('OUTPUT_' + os.path.split(file)[-1].replace('.tif', ''), out_gdb))
        with arcpy.EnvManager(outputZFlag="Disabled", outputMFlag="Disabled"):
            arcpy.conversion.RasterToPolygon(
                in_raster=r_int,
                out_polygon_features=out_fc,
                simplify="SIMPLIFY",
                raster_field="Value",
                create_multipart_features="SINGLE_OUTER_PART"
            )
        if 'population' in out_fc: # Add a year column if its a population dataset
            year = int(os.path.split(out_fc)[-1].split('_')[-3])
            arcpy.management.CalculateField(
                in_table=out_fc,
                field="year",
                expression=f"{year}",
                expression_type="PYTHON3",
                field_type="SHORT"
            )
        if index <= 4:
            population_density_fcs.append(out_fc)
        elif 5 < index <= 9:
            population_count_fcs.append(out_fc)

    # Merge population datasets
    print('Merging population density datasets...')
    arcpy.Merge_management(population_density_fcs, os.path.join(out_gdb, 'population_density'))
    print('Merging population count datasets...')
    arcpy.Merge_management(population_count_fcs, os.path.join(out_gdb, 'population_count'))

    return out_gdb


if __name__ == '__main__':
    if sys.argv[1] is not None and sys.argv[2] is not None:
        start_time = datetime.datetime.now()
        part_a(sys.argv[1], sys.argv[2])
        end_time = datetime.datetime.now()
        print('Script complete! Duration:', (end_time - start_time).total_seconds() / 60, 'minutes.')
        sys.exit(0)
    else:
        print('ERROR: Invalid/absent arguments.')
        sys.exit(1)
