# Title: Wildfire Prediction Methodology Part D
# Author: Marko Zlatic
# Date: 2024-05-03 (yyyy-mm-dd)
# University: Johns Hopkins University
# Program: Master of Science in Geographic Information Systems (GIS)
# Purpose: This code was written to uphold the requirements for the AS800 GIS Capstone

import datetime
import arcpy
import sys
import os


def part_d(master_folder):
    # Script Objective: Calculate summary statistics based on root mean squared error (RMSE) and total number of fires
    # by State

    arcpy.env.overwriteOutput = True

    output_datasets = os.path.join(master_folder, 'output_datasets')
    input_gdb = os.path.join(master_folder, 'input_datasets.gdb')
    output_gdb = os.path.join(output_datasets, 'output_datasets.gdb')

    if not arcpy.Exists(output_gdb):
        arcpy.management.CreateFileGDB(
            out_folder_path=os.path.split(output_gdb)[0],
            out_name=os.path.split(output_gdb)[1]
        )

    # Prediction output derived from the Part C machine learning script
    pred_fc = os.path.join(output_datasets, "finalized_predictions.csv")

    # Determine start and end date, as well as total number of days
    dates = [r[0] for r in arcpy.da.SearchCursor(in_table=pred_fc, field_names=["FireDiscoveryDateTime"])]
    start_date = dates[0]
    end_date = dates[-1]
    print('Start date:', start_date,'\nEnd date:', end_date, '\nTotal Number of Days:', (end_date - start_date).days)

    print('Generating summary statistics...')
    # Load csv export as a feature class dataset
    pred_fc = arcpy.management.XYTableToPoint(
        in_table=pred_fc,
        out_feature_class=os.path.join(output_gdb, "finalized_predictions"),
        x_field="pred_mE",
        y_field="pred_mN",
        coordinate_system=arcpy.SpatialReference(3857) # web mercator projection
    )

    # Calculate squared error
    arcpy.management.CalculateField(
        in_table=pred_fc,
        field="SQUARED_ERROR",
        expression="(!actual_mE! - !pred_mE!)**2 + (!actual_mN! - !actual_mN!)**2",
        expression_type="PYTHON3",
        field_type="DOUBLE"
    )

    # Summarize the squared error and row count (ie number of wildfires) by State
    rmse_by_state_table = arcpy.analysis.Statistics(
        in_table=pred_fc,
        out_table=os.path.join(output_gdb, "rmse_by_state_table"),
        statistics_fields="SQUARED_ERROR SUM;OBJECTID COUNT",
        case_field="STATE",
    )

    # Calculate RMSE
    arcpy.management.CalculateField(
        in_table=rmse_by_state_table,
        field="RMSE",
        expression="math.sqrt((!SUM_SQUARED_ERROR! / !FREQUENCY!))",
        expression_type="PYTHON3",
        field_type="DOUBLE"
    )

    # Create copy of the original US States and Territories dataset that was converted to a feature class dataset
    rmse_by_state_fc = arcpy.conversion.ExportFeatures(
        in_features=os.path.join(input_gdb, "us_boundaries"),
        out_features=os.path.join(output_gdb, "rmse_by_state_fc")
    )

    # Join RMSE information and delete null joins
    arcpy.management.JoinField(
        in_data=rmse_by_state_fc,
        in_field="STATE",
        join_table=rmse_by_state_table,
        join_field="STATE",
        fields="FREQUENCY;SUM_SQUARED_ERROR;RMSE"
    )

    # Delete tabular fields that are not needed
    fields_to_keep = ["STATE", "NAME", "FREQUENCY", "SUM_SQUARED_ERROR", "RMSE"]
    arcpy.management.DeleteField(
        in_table=rmse_by_state_fc,
        drop_field=fields_to_keep,
        method="KEEP_FIELDS"
    )

    # Delete rows that don't have an RMSE present
    null_rmse = arcpy.management.SelectLayerByAttribute(
        in_layer_or_view=rmse_by_state_fc,
        selection_type="NEW_SELECTION",
        where_clause="RMSE IS NULL"
    )
    arcpy.management.DeleteRows(null_rmse)

    return None


if __name__ == '__main__':
    if sys.argv[1] is not None:
        start_time = datetime.datetime.now()
        part_d(sys.argv[1])
        end_time = datetime.datetime.now()
        print('Script complete! Duration:', (end_time - start_time).total_seconds() / 60, 'minutes.')
        sys.exit(0)
    else:
        print('ERROR: Invalid/absent arguments.')
        sys.exit(1)
