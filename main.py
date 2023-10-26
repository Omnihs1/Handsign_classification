from net.dg_sta import DG_STA
from trainer.trainer import Trainer
from feeder.feeder import FeederINCLUDE
from torch.utils.data import DataLoader
import torch
from torchinfo import summary
class Arg():
    def __init__(self, args):
        self.model = args["model"]
        self.loss_name = args["loss_name"]
        self.optimizer_name = args["optimizer_name"]
        self.lr_rate = args["lr_rate"]
        self.experiment_name = args["experiment_name"]
        self.model_name = args["model_name"]
        self.weight_decay = args["weight_decay"]
        self.batch_size = args["batch_size"]
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = DG_STA(num_channels= 2, num_classes = 263, dp_rate = 0.2,
                   time_len=80, joint_num=25).to(device)
    # model = Model(2, 263, graph_args = {"layout" :"mediapipe"}, edge_importance_weighting=False).to(device)
    args = {"experiment_name" : "INCLUDE_CLASSIFICATION",
            "model_name" : "DG_STA",
            "model" : model,
            "loss_name" : "cross_entropy",
            "optimizer_name" : "adam",
            "lr_rate" : 0.0001,
            "weight_decay" : 0.98, 
            "batch_size" : 1}
    a = Arg(args)
    train_dataset = FeederINCLUDE(data_path="data/npy_train.npy", label_path="data/label_train.pickle")
    test_dataset = FeederINCLUDE(data_path="data/npy_test.npy", label_path="data/label_test.pickle")
    val_dataset = FeederINCLUDE(data_path="data/npy_val.npy", label_path="data/label_val.pickle")
    train_dataloader = DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=a.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=a.batch_size, shuffle=False)
    train = Trainer(a)
    summary(model)
    results = train.train(epochs=20, train_dataloader = train_dataloader, test_dataloader = val_dataloader)
    print("Done")