# GraphComm: A Graph-based Deep Learning Method to Predict Cell-Cell Communication in single-cell RNA data

This repository will allow you to seamlessly reproduce all figures from the manuscript GraphComm: A Graph-based Deep Learning Method to Predict Cell-Cell Communication in single-cell RNA data, as well as generate new CCC predictions on your own data.

## Reproducing Original Figures 
## Working in Code Ocean 
- If you would like to use the **exact** results used in the publication, those can be found in  `/data/GraphComm_Output`
- To use the pre-saved models on the same datasets, simply clik the `Reproducible Run` button and CCI predictions will be generated in the results folder. 
- benchmarking figures can be generated using `LIANA.iypnb` and `CCI.ipynb` in the folder `/code/benchmarking`. 
## Working outside of Code Ocean
- Please navigate to the Code Ocean Capsule and download the `data` folder 
- navigate to the code directory
- execute the command `bash ./run`

# Generating new Predictions on new Datasets
- Download dataset of choice into the folder `/data/`
- To generate directed cell graphs (required input for GraphComm), use the notebook `/code/preprocessing/make_graphs.ipynb`. Examples are provided for importing multiple matrices, one matrix and h5ad object. 
- Save your directed Graph files to the directory `/data/GraphComm_Input` with a unique dataset name. 
- Execute the python file `train.py`, specifying the appropriate flags. 

Should you have any inquiries or questions, pelase contact emily.so@mail.utoronto.ca


