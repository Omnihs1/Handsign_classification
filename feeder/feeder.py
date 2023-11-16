import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import pickle 
import random

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
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        return data_numpy, label
    
    def __len__(self):
        return len(self.label)
    
if __name__ == '__main__':
    data = FeederINCLUDE(data_path="data/npy_train.npy", label_path="data/label_train.pickle")
    print(data.N, data.C, data.T, data.V, data.M)
    print(data.data.shape)
    print(data.__len__())
    train_dataloader = DataLoader(data, batch_size=4, shuffle=True)

    # rd_number = random.randint(0, 2860)
    # a, label = data.__getitem__(rd_number)
    # print(a.shape)
