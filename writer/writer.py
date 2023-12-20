from datetime import datetime
import os
import numpy as np
import wandb
def init_wandb(args):
    wandb.init(
        project="handsign_classification",
        name = args.model_name,
        config={
        "learning_rate": args.lr_rate,
        "architecture": args.model_name,
        "dataset": "INCLUDE",
        "epochs": args.epochs,
        "weight_decay" : args.weight_decay, 
        }
    )
    wandb.watch(args.model)

def writer_to_wandb(results, epoch, writer):
    wandb.log({"train_loss": results["train_loss"][-1], 
               "val_loss": results["val_loss"][-1],
               "train_acc": results["train_acc"][-1],
               "val_acc": results["val_acc"][-1]})
def end():
    wandb.finish()