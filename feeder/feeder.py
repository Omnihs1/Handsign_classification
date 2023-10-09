import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pickle 
# Class read npy and pickle file to make data and label in couple
class FeederINCLUDE(Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
    """
    def __init__(self, data_path: Path, label_path: Path):
        super(FeederINCLUDE, self).__init__
        self.data_path = data_path
        self.label_path = label_path
        self.load_data()
    
    def load_data(self):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        self.data = np.load(self.data_path)     
        self.N, self.C, self.T, self.V, self.M = self.data.shape
    def __getitem__(self, index):
        """
        Input shape (N, C, V, T, M)
        N : batch size
        C : numbers of features
        V : numbers of joints (as nodes)
        T : numbers of frames
        M : numbers of people (should delete)
        
        Output shape (C, V, T, M)
        C : numbers of features
        V : numbers of joints (as nodes)
        T : numbers of frames
        label : label of videos
        """
        data = np.squeeze(np.array(self.data_path[index]))
        label = self.label[index]
        return data, label
    def __len__(self):
        return len(self.label_path)