# GraphComm: A Graph-based Deep Learning Method to Predict Cell-Cell Communication in single-cell RNAseq data

Emily So<sup>1,2,5</sup>, Sikander Hayat<sup>3</sup>, Sisira Kadambat Nair<sup>1</sup>, Bo Wang<sup>4,5,6,7,\$</sup>,Benjamin Haibe-Kains<sup>1,2,4,6,8,9,\$</sup>

<sup>1</sup>Princess Margaret Cancer Centre, University Health Network, Toronto, Canada

<sup>2</sup>Department of Medical Biophysics, University of Toronto, Toronto, Canada

<sup>3</sup>Institute of Experimental Medicine and Systems Biology, UniKlinik RWTH Aachen, Aachen, Germany

<sup>4</sup>Vector Institute for Artificial Intelligence, Toronto, Canada

<sup>5</sup>Peter Munk Cardiac Centre, Toronto, Canada

<sup>6</sup>Department of Computer Science, University of Toronto, Toronto, Canada

<sup>7</sup>Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Canada

<sup>8</sup>Ontario Institute for Cancer Research, Toronto, Canada

<sup>9</sup>Department of Biostatistics, Dalla Lana School of Public Health, Toronto, Canada


<sup>$</sup> These authors contributed equally to the present work\
<sup>#</sup> Corresponding author: Benjamin Haibe-Kains, Princess Margaret Cancer Centre, University Health Network, Toronto, Ontario M5G 2C4 Canada

# Abstract 
Cell-cell interactions coordinate various functions across cell-types in health and disease. Novel single-cell techniques allow us to investigate cellular crosstalk at single-cell resolution. Cell-cell communication (CCC) is mediated by underlying gene-gene networks, however most current methods are unable to account for complex inter-connections within the cell as well as incorporate the effect of pathway and protein complexes on interactions. This results in the inability to infer overarching signalling patterns within a dataset as well as limit the ability to successfully explore other data types such as spatial cell dimension. Therefore, to represent transcriptomic data as intricate networks connecting cells to ligands and receptors for relevant cell-cell communication inference as well as incorporating descriptive information independent of gene expression, we present GraphComm - a new graph-based deep learning method for predicting cell-cell communication in single-cell RNAseq datasets. GraphComm improves CCC inference by capturing detailed information such as cell location and intracellular signalling patterns from a database of more than 30,000 protein interaction pairs. With this framework, GraphComm is able to predict biologically relevant results in datasets previously validated for CCC,datasets that have undergone chemical or genetic perturbations and datasets with spatial cell information.

This repository will allow you to seamlessly reproduce all figures from the manuscript GraphComm: A Graph-based Deep Learning Method to Predict Cell-Cell Communication in single-cell RNA data, as well as generate new CCC predictions on your own data.

# Navigating the repository
## code folder 
- `/code/benchmarking`:notebooks that will allow you to regenerate figures from the original experiments, benchmarking against other methods and randomization interactions
- `/code/preprocessing`:notebook that will lead you through how to create the required input for GraphComm inference
- `/code/randomization`:an example script detailing how to run randomization expriments that were conducted for the manuscript
- `/code/predictions`:all utilies and scripts necessary to conduct Graphcomm training and inference
 - `/code/predictions/utils.py`: all functions to conduct training processes for GraphComm (extracting embeddings from the ground truth, computing probabilites using GAT
 - `/code/predictions/model.py`: architecture for both the ground truth Node2Vec and the GAT 
 - `/code/predictions/train.py`: script that will, with  input created from `/code/preprocessing` will perform the two step training and inference of CCC by GraphComm
## data folder
- `/data/GraphComm_Input`: input directed graphs used in the original manuscript to generate GraphComm results 
- `/data/GraphComm_Output`: results from GraphComm inference used in the original publication
- `/data/models`: saved models for each dataset used in the original publication
- `/data/LR_database`: all information used from the Omnipath Database used for CCC predictions
- `/data/random_data`: results from randomization expreiments on each dataset for benchmarking
- `/data/raw_data`: original count matrices from each dataset

# Reproducing Original Figures 
## Working in Code Ocean 
- If you would like to use the **exact** results used in the publication, those can be found in  `/data/GraphComm_Output`
- Navigate to the App Panel in the left Tab viewer, and select your parameters for running
## Working outside of Code Ocean
- Please navigate to the Code Ocean Capsule and download the `data` folder 
- navigate to the code directory
- execute the command `bash ./run <dataset to run> <True/False for using presaved models/>` (this may require the creation of certain folders/changing of paths. the files `/code/predictions/train.py` and `/code/predictions/utils.py` will require the changing of paths. )

# Generating new Predictions on new Datasets
- Download dataset of choice into the folder `/data/`
- To generate directed cell graphs (required input for GraphComm), use the notebook `/code/preprocessing/make_graphs.ipynb`. Examples are provided for importing multiple matrices, one matrix and h5ad object. 
- Save your directed Graph files to the directory `/data/GraphComm_Input` with a unique dataset name (you will have to create the directory first). 
- Execute the python file `train.py`, specifying the appropriate flags. 



Should you have any inquiries or questions, please contact the developer at emily.so@mail.utoronto.ca


