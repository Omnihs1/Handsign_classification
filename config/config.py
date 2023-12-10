from dataclasses import dataclass

@dataclass
class TrainConfig:
    output_dir: str = "checkpoints"
    num_train_epochs: int = 10
    save_steps: int = 1000
    logging_steps: int = 50
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    learning_rate: float = 1e-4
    batch_size: int = 8
    seed: int = 42

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

# args = {"experiment_name" : "INCLUDE_CLASSIFICATION",
    #         "model_name" : "ST_GCN",
    #         "model" : model,
    #         "loss_name" : "cross_entropy",
    #         "optimizer_name" : "adam",
    #         "lr_rate" : 0.0001,
    #         "weight_decay" : 0.001, 
    #         "batch_size" : 4,
    #         "epochs": 50}
    # a = Arg(args)