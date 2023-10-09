import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

# Class read npy and pickle file to make data and label in couple
class FeederINCLUDE(Dataset):
    def __init__(self, data_path: Path, label_path: Path):
        super(FeederINCLUDE, self).__init__
        self.data_path = data_path
        self.label_path = label_path
        self.data = np.load(self.data_path)
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
        M : numbers of people (should delete)
        label : label of videos
        """
        data = self.data_path[index]
        label = self.self.label_path[index]
        return data, label
    def __len__(self):
        return len(self.label_path)