# GraphComm: A Graph-based Deep Learning Method to Predict Cell-Cell Communication in single-cell RNA data

This repository will allow you to seamlessly reproduce all figures from the manuscript GraphComm: A Graph-based Deep Learning Method to Predict Cell-Cell Communication in single-cell RNA data, as well as generate new CCC predictions on your own data.

## Reproducing Original Figures 
## Working in Code Ocean 
- If you would like to use the **exact** results used in the publication, those can be found in  `/data/GraphComm_Output`
- To use the pre-saved models on the same datasets, simply clik the `Reproducible Run` button and CCI predictions will be generated in the results folder. 
- benchmarking figures can be generated using `LIANA.iypnb` and `CCI.ipynb` in the folder `/code/benchmarking`. 
## Working outside of Code Ocean
- Please navigate to the [Code Ocean Capsule](https://codeocean.com/capsule/8269062/tree/v2) and download the `data` folder 
- navigate to the code directory
- execute the command `bash ./run` (this may require the creation of certain folders/changing of paths. the files `/code/predictions/train.py` and `/code/predictions/utils.py` will require the changing of paths. )

# Generating new Predictions on new Datasets
- Download dataset of choice into the folder `/data/`
- To generate directed cell graphs (required input for GraphComm), use the notebook `/code/preprocessing/make_graphs.ipynb`. Examples are provided for importing multiple matrices, one matrix and h5ad object. 
- Save your directed Graph files to the directory `/data/GraphComm_Input` with a unique dataset name (you will have to create the directory first). 
- Execute the python file `train.py`, specifying the appropriate flags. 

# Navigating the code folder 
- `/code/benchmarking`:notebooks that will allow you to regenerate figures from the original experiments, benchmarking against other methods and randomization interactions
- `/code/preprocessing`:notebook that will lead you through how to create the required input for GraphComm inference
- `/code/randomization`:an example script detailing how to run randomization expriments that were conducted for the manuscript
- `/code/predictions`:all utilies and scripts necessary to conduct Graphcomm training and inference
 - `/code/predictions/utils.py`: all functions to conduct training processes for GraphComm (Representation Learning from the ground truth, computing probabilites using GAT
 - `/code/predictions/model.py`: architecture for both the ground truth Node2Vec and the GAT 
 - `/code/predictions/train.py`: script that will, with  input created from `/code/preprocessing` will perform the two step training and inference of CCC by GraphComm

Should you have any inquiries or questions, pelase contact emily.so@mail.utoronto.ca


