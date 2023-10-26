from datetime import datetime
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
def create_writer(args):
    
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format
    log_dir = os.path.join("runs", timestamp, args.experiment_name, args.model_name)
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def writer_to_tensorboard(results, epoch, writer):
    # Add results to SummaryWriter
    writer.add_scalars(main_tag="Loss", 
                    tag_scalar_dict={"train_loss": results["train_loss"][-1],
                                        "test_loss": results["val_loss"][-1]},
                    global_step=epoch)
    writer.add_scalars(main_tag="Accuracy", 
                    tag_scalar_dict={"train_acc": results["train_acc"][-1],
                                        "test_acc": results["val_acc"][-1]}, 
                    global_step=epoch)

    # Close the writer
    writer.close()