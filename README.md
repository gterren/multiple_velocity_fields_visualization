# Multiple Wind Velocity Fields Visualization

This reposity includes the files developed in article https://doi.org/10.1016/j.apenergy.2021.116656 and https://arxiv.org/pdf/2103.02556.pdf. This article introduces an algorithm for the visualization of the wind velocity in which cloud are flowing. This algorithms uses infrared sky images. See article https://doi.org/10.1016/j.dib.2021.106914 for more information about the sky imager.

## Velocity Vectors Processing

The velocity vectors are processed to reduce the computational burden of the algorithm using the functions implemented in file cloud_velocity_vector_utils.py. The implementation of the Lucas-Kanade algorithm is in this repository https://github.com/gterren/multiple_cloud_dection_and_segmentation, and the segmentation of clouds is in this repository https://github.com/gterren/cloud_segmentation.

## Multi-output Support Vector Regression with Flow Constrains

The different implementation of the SVR in the experiments of the articles are in this file wind_velocity_field_utils.py. The function that are used for running the cross-validation of the SVM and GPR for two wind velocity fields are in double_layer_wind_velocity_field_SVM_validation.py and double_layer_wind_velocity_field_GPR_validation.py respectivaly. The files make usage of MPI to run in parallel the experiments. There is an implementation of the validatation when there is only one velocity field in the images in file single_layer_wind_velocity_field_validation.py.

## Dataset

A sample dataset is publicaly available in DRYAD repository: https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.zcrjdfn9m
