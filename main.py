import yaml
# from net2.st_gcn import Model
# from net2.st_gcn_attent import Model
from net3.decouple_gcn import Model
from trainer.trainer import Trainer
from feeder.feeder import FeederINCLUDE
from torch.utils.data import DataLoader
import torch
from torchinfo import summary
from dataclasses import dataclass
from typing import Type
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

@dataclass
class TrainConfig:
    experiment_name: str = "INCLUDE_CLASSIFICATION"
    model_name: str = "ST_GCN"
    model: Type["Model"] = None
    loss_name: str = "cross_entropy"
    optimizer_name: str = "adam"
    lr_rate: int = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 50
    save_path: str = "models/model7.pth"
    patience: int = 50

if __name__ == '__main__':
    # seed = 0
    # torch.manual_seed(seed)
    # # GPU operations have a separate seed we also want to set
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(42)
    #     torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open('config/model.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # model = Model(2, 263, graph_args = {"layout": "mediapipe", "strategy": "spatial"}, edge_importance_weighting=True, dropout = 0.2).to(device)
    # model = Model(config["in_channels"], config["classes"], graph_args = config["graph_args"], edge_importance_weighting=True).to(device)
    # model = Model(2, 263, graph_args = {"layout" :"mediapipe"}, edge_importance_weighting=False).to(device)
    model = Model(num_class=263, num_point=25, num_person=1, groups=8, block_size=41, \
                  in_channels = 2, graph_args={"labeling_mode": "spatial"}).to(device)
    config_train = TrainConfig(config["experiment_name"], config["model_name"], \
                        model, config["loss_name"], config["optimizer_name"], \
                        config["lr_rate"], config["weight_decay"], config["batch_size"], \
                        config["epochs"], config["save_path"], config["patience"])
    
    train_dataset = FeederINCLUDE(data_path="data/npy_train.npy", label_path="data/label_train.pickle")
    test_dataset = FeederINCLUDE(data_path="data/npy_test.npy", label_path="data/label_test.pickle")
    val_dataset = FeederINCLUDE(data_path="data/npy_val.npy", label_path="data/label_val.pickle")
    train_dataloader = DataLoader(train_dataset, batch_size=config_train.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config_train.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config_train.batch_size, shuffle=False)
    #pretrain 
    model.load_state_dict(torch.load(config_train.save_path))
    #train
    train = Trainer(config_train)
    summary(model)
    results = train.train(train_dataloader = train_dataloader, test_dataloader = val_dataloader)
    # Specify the file path to save the model
    train.save_model(config_train.save_path)
    print("Done")