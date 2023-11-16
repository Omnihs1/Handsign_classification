from net2.st_gcn import Model
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
        self.epochs = args["epochs"]
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(2, 263, graph_args = {"layout": "mediapipe"}, edge_importance_weighting=False).to(device)
    # model = Model(2, 263, graph_args = {"layout" :"mediapipe"}, edge_importance_weighting=False).to(device)
    args = {"experiment_name" : "INCLUDE_CLASSIFICATION",
            "model_name" : "ST_GCN",
            "model" : model,
            "loss_name" : "cross_entropy",
            "optimizer_name" : "adam",
            "lr_rate" : 0.0001,
            "weight_decay" : 0, 
            "batch_size" : 4,
            "epochs": 50}
    a = Arg(args)
    train_dataset = FeederINCLUDE(data_path="data/npy_train.npy", label_path="data/label_train.pickle")
    test_dataset = FeederINCLUDE(data_path="data/npy_test.npy", label_path="data/label_test.pickle")
    val_dataset = FeederINCLUDE(data_path="data/npy_val.npy", label_path="data/label_val.pickle")
    train_dataloader = DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=a.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=a.batch_size, shuffle=False)
    train = Trainer(a)
    # summary(model, input_size = (4, 2, 80, 25, 1), col_names = ["input_size", "output_size", "num_params"], device = device)
    results = train.train(train_dataloader = train_dataloader, test_dataloader = val_dataloader)
    # Specify the file path to save the model
    train.save_model("models/model1.pth")
    print("Done")