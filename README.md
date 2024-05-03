# MZ AS800 Capstone Source Code
This repository was created to uphold the requirements for the Johns Hopkins University AS800 GIS Capstone. The source code contained in this repository was designed to ensure replicable and accurate results as indicated in the Results Chapter of the capstone research paper.

## Abstract
Wildfires can be interpreted as both damaging and beneficial for forest ecosystems and a country’s respected economic health. As a result, this capstone was constructed to research the potential in predicting wildfire ignition sources within a 5-kilometer (km) radius of the actual location. The mainland United States (US) was used as the study location for this research. The results indicate an overall root mean squared error (RMSE) of 4,868 meters (indicating an accuracy of 4.868 km), where 30% of the US States (including the District of Columbia) used in the sample had an RMSE that was less than 2kms. Additionally, 4 US States had an RMSE greater than 15km; most likely causing the overall RMSE to increase due to the outliers present in those respected States. A map deliverable illustrating the relationship between RMSE and total number of fires by US State was visualized using bivariate symbology; this map can be used for US agencies and private enterprises to understand which components are needed to further increase the accuracy of the results, for the States that received a large RMSE. The research done for this capstone was able to answer the original research question and provided a replicable approach to predicting wildfire ignition sources across the US mainland. 

### Source Code
There are a total of four Python scripts that are intended to be run in sequential order, starting with the script that contains 'Part A' in the file name and ending with the script that contains 'Part D' in the file name. Additionally, a PowerShell script was created to allow a streamlined process when running all of the required Python scripts. Below is a brief summary for each of the respected scripts in this repository:

***MZ_AS800_RUN_ALL_SCRIPTS.ps1***
***Objective:*** Run all of the required Python scripts in the necessary order. Additional parameters are required to be filled in by the user. This will include the folder path that contains all of the necessary datasets, as well as, parameters to access a PostgreSQL database.

***MZ_AS800_Methodology_Part_A_Raster_Preprocessing.py***
***Objective:*** Convert all of the necessary raster datasets to feature class datasets. No additional parameters are required if the MZ_AS800_RUN_ALL_SCRIPTS.ps1 is executed, otherwise the user will need to indicate the folder path that contains all of the necessary datasets, as well as parameters to access a PostgreSQL database.

***MZ_AS800_Methodology_Part_B_WFIL_Preprocessing.py***
***Objective:*** Process the primary wildfire dataset (WFIL) to be adequate for the Part C machine learning script. No additional parameters are required if the MZ_AS800_RUN_ALL_SCRIPTS.ps1 is executed, otherwise the user will need to indicate the folder path that contains all of the necessary datasets, as well as parameters to access a PostgreSQL database.

***MZ_AS800_Methodology_Part_C_Machine_Learning.py***
***Objective:*** Create a machine learning model that is able to predict wildfire points within a 5km radius of the original locations. No additional parameters are required if the MZ_AS800_RUN_ALL_SCRIPTS.ps1 is executed, otherwise the user will need to indicate the folder path that contains all of the necessary datasets.

***MZ_AS800_Methodology_Part_D_RMSE_Summary_Statistics.py***
***Objective:*** Generate the root mean square error summary statistics by State. No additional parameters are required if the MZ_AS800_RUN_ALL_SCRIPTS.ps1 is executed, otherwise the user will need to indicate the folder path that contains all of the necessary datasets.

### Datasets
Below are the datasets that are required to be downloaded into a single folder location:<br/>
- ASTER Global Digital Elevation Map. (n.d.). Retrieved January 20, 2024, from https://asterweb.jpl.nasa.gov/gdem.asp<br/>
	The primary raster dataset used to gather elevation information.<br/>
<br/><br/>

- Center For International Earth Science Information Network-CIESIN-Columbia University. (2018a). Gridded Population of the World, Version 4 (GPWv4): Land and Water Area, Revision 11 [dataset]. Palisades, NY: NASA Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/H4Z60M4Z<br/>
	The primary raster dataset used to gather land and water area information.
<br/><br/>

- Center For International Earth Science Information Network-CIESIN-Columbia University. (2018b). Gridded Population of the World, Version 4 (GPWv4): Population Count Adjusted to Match 2015 Revision of UN WPP Country Totals, Revision 11 [dataset]. Palisades, NY: NASA Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/H4PN93PB<br/>
	The primary raster dataset used to gather human population count information.
<br/><br/>

- Center For International Earth Science Information Network-CIESIN-Columbia University. (2018c). Gridded Population of the World, Version 4 (GPWv4): Population Density Adjusted to Match 2015 Revision UN WPP Country Totals, Revision 11 [dataset]. Palisades, NY: NASA Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/H4F47M65<br/>
	The primary raster dataset used to gather human population density information.
<br/><br/>

- Harmonized world soil database v1.2 | FAO SOILS PORTAL | Food and Agriculture Organization of the United Nations. (n.d.). Retrieved February 24, 2024, from https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v12/en/<br/>
	The primary raster dataset used to gather soil type information.

<br/><br/>
- Individual Tree Species Parameter Maps. (n.d.). Retrieved February 23, 2024, from https://www.fs.usda.gov/foresthealth/applied-sciences/mapping-reporting/indiv-tree-parameter-maps.shtml<br/>
	The primary raster dataset used to gather total basal area (TBA) and stand density index (SDI) information.

<br/><br/>
- US Department of Commerce, N. (n.d.). U.S. States and Territories. NOAA’s National Weather Service. Retrieved January 24, 2024, from https://www.weather.gov/gis/USStates<br/>
	The primary ShapeFile dataset used to gather the US State boundary information.
<br/><br/>
- USFS and BLM Campgrounds—Overview. (n.d.). Retrieved March 25, 2024, from https://www.arcgis.com/home/item.html?id=e614311615f3403c88974a89c7e18f72<br/>
	The primary Feature Layer dataset used to gather campsite location information.
<br/><br/>
- Wildland Fire Incident Locations. (n.d.). Retrieved February 26, 2024, from https://data-nifc.opendata.arcgis.com/datasets/b4402f7887ca4ea9a6189443f220ef28_0/explore?location=34.515321,-92.437644,4.93<br/>
	The primary feature class dataset that will be used to predict wildfire point locations. Otherwise abbreviated as the WFIL dataset.
<br/><br/>

### Contact Info
Email: mzlatic1@jhu.edu<br />
Name: Marko Zlatic<br />
Status: Graduate Student<br />
Degree: Master of Science in Geographic Information Systems<br />
University: Johns Hopkins University<br />
Graduatation Date: May 2024<br />

## Python, Software, and Hardware Requirements
Below is the Python, software, and hardware requirements that were used when developing the scripts for this capstone research. There is also a YML file (mz_as800_conda_env) that can be used to replicate the non-ArcPy conda environment (however not all packages installed will be used).<br /><br />
Operating System -> Windows 10 Pro version 22H2<br />
Random Access Memory (RAM) -> Corsair Vengeance DDR5 4 x 32 GB (128 GB total)<br />
Central Processing Unit (CPU) -> AMD Ryzen 9 7900X 12-core<br />
Storage -> Samsung 990 Pro 1TB SSD<br /><br />
Python Version (for non-ArcPy scripts) -> 3.10.13<br />
	Conda Version -> 23.7.4<br />
PIP Version -> 24.0<br />
Packages (all were installed using Conda (via the conda-forge channel), except for Psycopg2, XGBoost, and Optuna, these were installed using PIP):<br />
-	Psycopg2 (used for reading/writing to PostgreSQL database) -> 2.9.9<br />
-	Shapely (used for geometric computations) -> 2.0.3<br />
-	Numpy (used for mathematical computations) -> 1.26.4<br />
-	SQLalchemy (used for reading/writing to PostgreSQL database) -> 2.0.28<br />
-	Pandas (used for dataframe manipulation) -> 2.2.0<br />
-	GeoPandas (used for geometric dataframe manipulation) -> 0.14.3<br />
-	PySpark (used for interacting with Spark installation) -> 3.5.0<br />
-	SciKit-Learn (used for machine learning computation) -> 1.4.1.post1<br />
-	XGBoost (used for machine learning computation) -> 2.0.3
-	Matplotlib (used to create graphics) -> 3.4.3<br />
-	Optuna (used for hyperparameter tuning) -> 3.6.1<br />
-	Datetime*<br />
-	Sys*<br />
-	OS*<br />
-	GC*<br />
	\* Python packages that came with the base installation of Python 3.10.13<br /><br />

Spark Version -> 3.5.0<br />
Hadoop Version -> 3.3.4<br />
Java Version -> 21.0.2<br />
Scala Version -> 2.12.18<br /><br />

ArcGIS Pro Version -> 3.1.0<br />
-	Python Version -> 3.9.16<br />
-	ArcPy (used for raster conversions and RMSE summary statistics) -> 3.1<br />
-	Conda Version -> 4.14.0<br /><br />
 
PostgreSQL Version -> 16.2<br />
	Extensions:<br />
-	PostGIS (used to apply spatial joins) -> 3.4.1<br />
