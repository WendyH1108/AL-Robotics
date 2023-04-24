import os, re
from typing import List, Dict
from ast import literal_eval
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

folder = './data/experiment'
filename_fields = ['vehicle', 'trajectory', 'method', 'condition']

def extract_features(rawdata, features):
    """extract features from all sources tasks, which is x in algorithm

    Args:
        rawdata (dictionary): _description_
        features (list): the list of names of features that are shared around all sources tasks

    Returns:
        list: the list of data that only contians the selected shared features 
    """
    feature_data = []
    hover_pwm_ratio = 1.
    for feature in features:
      if isinstance(rawdata[feature], str):
        condition_list = re.findall(r'\d+', rawdata[feature]) 
        condition = 0 if condition_list == [] else float(condition_list[0])
        feature_data.append(np.tile(condition,(len(rawdata['v']),1)))
        continue
      feature_len = rawdata[feature].shape[1] if len(rawdata[feature].shape)>1 else 1
      if feature == 'pwm':
          feature_data.append(rawdata[feature] / 1000 * hover_pwm_ratio)
      else:
          feature_data.append(rawdata[feature].reshape(rawdata[feature].shape[0],feature_len))
    feature_data = np.hstack(feature_data)
    return feature_data

def load_data(folder : str, expnames = None) -> List[dict]:
    ''' Loads csv files from {folder} and return as list of dictionaries of ndarrays '''
    Data = []

    if expnames is None:
        filenames = os.listdir(folder)
    elif isinstance(expnames, str): # if expnames is a string treat it as a regex expression
        filenames = []
        for filename in os.listdir(folder):
            if re.search(expnames, filename) is not None:
                filenames.append(filename)
    elif isinstance(expnames, list):
        filenames = (expname + '.csv' for expname in expnames)
    else:
        raise NotImplementedError()
    for filename in filenames:
        # Ingore not csv files, assume csv files are in the right format
        if not filename.endswith('.csv'):
            continue

        # Load the csv using a pandas.DataFrame
        df = pd.read_csv(folder + '/' + filename)

        # Lists are loaded as strings by default, convert them back to lists
        for field in df.columns[1:]:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        # Copy all the data to a dictionary, and make things np.ndarrays
        Data.append({})
        for field in df.columns[1:]:
            Data[-1][field] = np.array(df[field].tolist(), dtype=float)

        # Add in some metadata from the filename
        namesplit = filename.split('.')[0].split('_')
        for i, field in enumerate(filename_fields):
            Data[-1][field] = namesplit[i]
        # Data[-1]['method'] = namesplit[0]
        # Data[-1]['condition'] = namesplit[1]

    return Data

def load_and_process_data(dataset_folder, features):
    ''' 
    Loads data from {dataset_folder} and extracts the features {features}
    :param str dataset_folder: the name of the folder containing the data
    :param list features: the list of features to extract
    :return: the extracted features formated in a numpy array (n_samples, n_features)
    '''
    rawdata = load_data(dataset_folder)[0]
    feature_data = extract_features(rawdata, features)
    print("Data has shape: ", feature_data.shape)
    return feature_data

def generate_orth(shape, seed=None):
    assert len(shape) == 2, "Shape must be a 2-tuple."
    if seed is not None:
        np.random.seed(seed)
    gaus = np.random.normal(0, 1, shape)
    if shape[0] < shape[1]:
        _, _, orth = np.linalg.svd(gaus, full_matrices=False)
        print(f"{orth[[0]]@orth[[1]].T}")
    else:
        orth, _, _,  = np.linalg.svd(gaus, full_matrices=False)
        print(f"{orth[:,[0]].T@orth[:,[1]]}")
    print(f" orth shape: {orth.shape}")
    return orth
