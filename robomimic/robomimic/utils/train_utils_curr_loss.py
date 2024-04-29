"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
from torch.utils.data import ConcatDataset

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils

from robomimic.utils.dataset import SequenceDataset
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from robomimic.utils.train_utils import dataset_factory


def load_data_for_training_curriculum_loss(config, obs_keys, dataset_path=None):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """
    if dataset_path is None:
        dataset_path = config.train.data

    # config can contain an attribute to filter on
    filter_by_attribute = config.train.hdf5_filter_key
    assert filter_by_attribute is not None
    attributes = filter_by_attribute.split('_')
    print("The data set contains {} subsets".format(len(attributes)))

    # load the dataset into memory
    if config.experiment.validate:
        # construct validation set
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        train_datasets, valid_datasets = [], []

        for attribute in attributes:
            real_dataset_path = dataset_path
            real_attribute = attribute
            if attribute == 'proficient':
                real_dataset_path = real_dataset_path.replace('/mh/', '/ph/')
                real_attribute = '20_percent'
            cur_train_filter_attribute = "{}_{}".format(real_attribute, 'train')
            cur_valid_filter_attribute = "{}_{}".format(real_attribute, 'valid')
            #print('Loading dataset {}, real attribute {}/{} from {}'.format(attribute, cur_train_filter_attribute, cur_valid_filter_attribute, real_dataset_path))
            cur_train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=cur_train_filter_attribute, dataset_path=real_dataset_path)
            cur_valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=cur_valid_filter_attribute, dataset_path=real_dataset_path)
            train_datasets.append(cur_train_dataset)
            valid_datasets.append(cur_valid_dataset)
    else:
        train_datasets, valid_datasets = [], None
        for attribute in attributes:
            real_dataset_path = dataset_path
            real_attribute = attribute
            if attribute == 'proficient':
                real_dataset_path = real_dataset_path.replace('/mh/', '/ph/')
                real_attribute = '20_percent'
            cur_train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=real_attribute, dataset_path=real_dataset_path)
            train_datasets.append(cur_train_dataset)

    return train_datasets, valid_datasets
                
                
def concate_datasets(datasets):
    '''
    Make multiple loaded dataset into one

    Args:
        datasets: A list of datasets

    Return:
        dataset (SequenceDataset instance): dataset object
    '''
    if datasets is None:
        dataset = None
    else:
        dataset = ConcatDataset(datasets)
    return dataset
    

