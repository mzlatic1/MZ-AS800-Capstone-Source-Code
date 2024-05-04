# RUN IN A CONDA ENABLED POWERSHELL TERMINAL #

# INPUT FOLDER PATH THAT CONTAINS ALL PY FILES
cd "C:\Users\marko\Downloads\PROD_READY_PY_FILES"

# INPUT FOLDERPATH THAT CONTAINS ALL REQUIRED DATASETS
$masterFolder = "C:\Users\marko\Downloads\PROD_INPUTS_OUTPUTS"

# INPUT FULL FEATURE CLASS PATH FOR THE WFIL DATASET
$wfilDataset = "C:\Users\marko\Downloads\f022c91d-dcdf-4676-9d17-3f54b1ec303b.gdb\Incidents"

# INPUT ARCPY PYTHON EXECUTABLE PATH
$ArcPyPath = "C:\Users\marko\AppData\Local\ESRI\conda\envs\arcgispro-py3-clone-1\python.exe"

# INPUT NON-ARCPY PYTHON EXECUTABLE PATH
$nonArcPyPath = "C:\Users\marko\.conda\envs\win_conda_env_v2\python.exe"

# INPUT POSTGRESQL (POSTGRES) DATABASE INFORMATION
$postgresDatabase = "Name of Postgres database"
$postgresHost = "Host path for the Postgres database"
$postgresPort = "Port number for the Postgres database"
$postgresUsername = "Username for the Postgres database"
$postgresPassword = "Password for the Postgres database"

# INDICATE THE NUMBER OF ITERATIONS TO RUN ALL SCRIPTS
$numIterations = 1

for ($i=0; $i -lt $numIterations; $i++){
    # Part A Script
    & $ArcPyPath "MZ_AS800_Methodology_Part_A_Raster_Preprocessing.py" $masterFolder $wfilDataset

    # Part B Script (the $arcPyPath variable can be substituted if the required packages are installed)
    & $nonArcPyPath "MZ_AS800_Methodology_Part_B_WFIL_Preprocessing.py" $masterFolder $postgresDatabase $postgresHost $postgresPort $postgresUsername $postgresPassword

    # Part C Script (the $arcPyPath variable can be substituted if the required packages are installed)
    & $nonArcPyPath "MZ_AS800_Methodology_Part_C_Machine_Learning.py" $masterFolder

    # Part D Script
    & $ArcPyPath "MZ_AS800_Methodology_Part_D_RMSE_Summary_Statistics.py" $masterFolder
}
function exitScript{$null = Read-Host 'Press Enter to exit.'}
exitScript
