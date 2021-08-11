# Multiple Wind Velocity Fields Visualization

This reposity includes the files developed in article https://doi.org/10.1016/j.apenergy.2021.116656. This article introduces an algorithm for the visualization of the wind velocity in which cloud are flowing. This algorithms uses infrared sky images. See article https://doi.org/10.1016/j.dib.2021.106914 for more information about the sky imager.

## Velocity Vectors Processing

The velocity vectors are processed to reduce the computational burden of the algorithm using the functions implemented in file cloud_velocity_vector_utils.py. The implementation of the Lucas-Kanade algorithm is in this repository https://github.com/gterren/multiple_cloud_dection_and_segmentation, and the segmentation of clouds is in this repository https://github.com/gterren/cloud_segmentation.

## Multi-output Support Vector Regression with Flow Constrains

The different implementation of the SVR in the experiments of the articles are in this file wind_velocity_field_utils.py. The function that are to validate the SVM and GPR for one wind velocity field are in double_layer_wind_velocity_field_SVM_validation.py and double_layer_wind_velocity_field_GPR_validation.py respectivaly. The files make usage of MPI to run in parallel the experiments.
