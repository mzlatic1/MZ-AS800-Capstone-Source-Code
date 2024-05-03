# MZ AS800 Capstone Source Code
This repository was created to uphold the requirements for the Johns Hopkins University AS800 GIS Capstone. The source code contained in this repository was designed to ensure replicable and accurate results as indicated in the Results Chapter of the capstone research paper.

## Abstract
Wildfires can be interpreted as both damaging and beneficial for forest ecosystems and a country’s respected economic health. As a result, this capstone was constructed to research the potential in predicting wildfire ignition sources within a 5-kilometer (km) radius of the actual location. The mainland United States (US) was used as the study location for this research. The results indicate an overall root mean squared error (RMSE) of 4,868 meters (indicating an accuracy of 4.868 km), where 30% of the US States (including the District of Columbia) used in the sample had an RMSE that was less than 2kms. Additionally, 4 US States had an RMSE greater than 15km; most likely causing the overall RMSE to increase due to the outliers present in those respected States. A map deliverable illustrating the relationship between RMSE and total number of fires by US State was visualized using bivariate symbology; this map can be used for US agencies and private enterprises to understand which components are needed to further increase the accuracy of the results, for the States that received a large RMSE. The research done for this capstone was able to answer the original research question and provided a replicable approach to predicting wildfire ignition sources across the US mainland. 

### Source Code
There are a total of four Python scripts that are intended to be ran in sequential order, starting with the script that contains 'Part A' in the file name and ending with the script contains 'Part D' in the file name. Additionally, a Powershell script was created to allow a streamline process when running all of the required Python scripts. Below is a brief objective for each of the respected scripts in this repository:

***MZ_AS800_RUN_ALL_SCRIPTS.ps1***
***Objective:*** Run all of the required Python scripts in the necessary order. Additional parameters are required to be filled in by the user. This will include the folder path that contains all of the necessary datasets, as well as, parameters to access a PostgreSQL database.

***MZ_AS800_Methodology_Part_A_Raster_Preprocessing.py***
***Objective:*** Convert all of the necessary raster datasets to feature class datasets. No additional parameters are required if the MZ_AS800_RUN_ALL_SCRIPTS.ps1 is ran, otherwise the user will need to incdicate the folder path that contains all of the necessary datasets, as well as parameters to access a PostgreSQL database.

***MZ_AS800_Methodology_Part_B_WFIL_Preprocessing.py***
***Objective:*** Process the primary wildfire dataset (WFIL) to be adequate for the Part C machine learning script. No additional parameters are required if the MZ_AS800_RUN_ALL_SCRIPTS.ps1 is ran, otherwise the user will need to incdicate the folder path that contains all of the necessary datasets, as well as parameters to access a PostgreSQL database.

***MZ_AS800_Methodology_Part_C_Machine_Learning.py***
***Objective:*** Create a machine learning model that is able to predict wildfire points within a 5km radius of the original locations. No additional parameters are required if the MZ_AS800_RUN_ALL_SCRIPTS.ps1 is ran, otherwise the user will need to incdicate the folder path that contains all of the necessary datasets.

***MZ_AS800_Methodology_Part_D_RMSE_Summary_Statistics.py***
***Objective:*** Generate the root mean square error summary statistics by State. No additional parameters are required if the MZ_AS800_RUN_ALL_SCRIPTS.ps1 is ran, otherwise the user will need to incdicate the folder path that contains all of the necessary datasets.

### Contact Info
Email: mzlatic1@jhu.edu<br />
Name: Marko Zlatic<br />
Status: Graduate Student<br />
Degree: Master of Science in Geographic Information Systems<br />
University: Johns Hopkins University<br />
Graduatation Date: May 2024<br />
