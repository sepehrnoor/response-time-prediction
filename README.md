# Applying Knowledge Tracing to Predict Exercise Response Time
## By Shamiran Jaf and Sepehr Noorzadeh

## Overview
This repository contains the master's thesis written by Shamiran Jaf and Sepehr Noorzadeh at Akribian AB while studying at the Faculty of Engineering (LTH) at Lund University.

It is a machine learning pipeline for knowledge tracing. It supports several public datasets, and one private dataset. It was built with PyTorch, NumPy and Pandas.

## Dependencies
* PyTorch (1.7.1)
* NumPy (1.19.2)
* Pandas (1.1.3)
* SciPy (1.5.2)
* scikit-learn (0.23.2)


## Datasets
Parameters for supported datasets can be found in `config/dataset_parameters.py`. To add support for a new dataset, add new key-value pairs to each parameter dictionary.

### Supported public datasets
* ASSISTments 2012 (https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect)
* Junyi Academy(https://www.kaggle.com/junyiacademy/learning-activity-public-dataset-by-junyi-academy)
* Ednet (https://www.kaggle.com/c/riiid-test-answer-prediction/data, https://github.com/riiid/ednet)

## GPU support

If you want to run the notebooks on a local GPU, you will need the appropriate CUDA (11.0) and cuDNN (8.0.4) version for the version of PyTorch that you are using.

CUDA and cuDNN are already installed on Google Colab, but the notebook must be set to use the GPU in the notebook settings on Colab.

## How to run a notebook

### Alternative 1: Run on Google Colab
1. Upload the project to Google Drive
2. Navigate to https://colab.research.google.com
3. Select the notebook that you wish to run

### Alternative 2: Run locally
To run locally, simply start a Jupyter server in the root directory and connect to it through a web browser. Please see the [Jupyter notebook documentation](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html) for instructions.

## Usage
### Preprocessing
First, you will have to preprocess the dataset you want to use. If you already have preprocessed data you can skip the following steps:

1. Open `thesis/preprocessing/sort_dataset.ipynb`
2. Set the `DATASET` parameter to the name of the dataset you want to preprocess.
3. Run all cells in the notebook
4. Open `thesis/preprocessing/transform_dataset.ipynb`
5. Set the `DATASET` parameter to the name of the dataset you want to preprocess.
6. Run all cells in the notebook
7. Open `thesis/preprocessing/split_and_window.ipynb`
8. Set the `DATASET` parameter to the name of the dataset you want to preprocess.
9. Run all cells in the notebook

The preprocessed data should now be located at `thesis/data/[DATASET_NAME]/processed/processed.h5`

### Training

In order to train the model 
1. Open `train_evaluate.ipynb`
2. Make sure that the `DATASET` parameter is set to the dataset you want to preprocess.
3. Set the `DATASET` parameter to the name of the dataset you want to use.
4. Set the `MODEL` parameter to the name of the model you want to train.
5. Set the `MODE` parameter to either correctness or latency.
6. Run all cells in the notebook

The model will be trained and continually saved to `/thesis/models/checkpoints/[MODE + '_' + MODEL + '_' + DATASET].torch`. When the threshold for early stopping has been reached the training will stop and results will be displayed. 

### Run predictions and show results
If you already have a trained model in `/thesis/models/checkpoints/[MODE + '_' + MODEL + '_' + DATASET].torch` you can set the `SKIP_TRAINING` switch to true in the `train_evaluate.ipynb` notebook to `True` to skip the training and just load the checkpoint before doing prediction and evaluation.

### Add support for a new dataset
You can edit the file `config/dataset_parameters.py` and add parameters for your own dataset.

### Folder structure
`data` contains data separated into dataset > raw/processed > model.

`models/model.py` contains the models used.

`preprocessing` contains preprocessing notebooks.

`lib` contains common functions across notebooks.

`config` contains parameters for the datasets.

`history` contains deprecated files that are kept for archival purposes.
