This is the repository for the paper Dose distribution of proton therapy plans are robust against lowering the resolution of CTs in combination with increasing noise.

The folder data contains first the CT dicom image files for the different filters and pixelsizes used and second the gamma CT data from the FDG and Ga experiment for all gl
obal iterations done.
The code used for evaluation is in the folder Experiment_Evaluation. Modify the main.py for getting the data wished.

Artificial CT data creation for the different patients was done with the code in the folder Artificial_Data_Generation. Therefore a patients CT dataset is needed which is not provided here. 
Look in main_spatialresolution_example.py for creating data with inferior spatial resolution. Select the mods gaussian rectangle and the filterwidths.

An example with also modifies noise is in main_spatialresolution_and_noise_example.py. Here the absolut value for the Std must be set. 
After getting the gamma test arrays as np out of RayStation they are evaluated with the code in Artificial_Data_Evaluation. Overlay pictures can be created which lay the gamma test failure pixels visual over the CT images and exports png files for each layer.
