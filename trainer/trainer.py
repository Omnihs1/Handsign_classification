from utils import plot, early
from writer import writer
from metrics.accuracy import Accuracy
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm.auto import tqdm

class Trainer():
    def __init__(self, args):
        self.model = args.model
        self.loss_name = args.loss_name
        self.optimizer_name = args.optimizer_name
        self.weight_decay = args.weight_decay
        self.lr_rate = args.lr_rate
        self.epochs = args.epochs
        self.early_stopping = early.EarlyStopping(path = "models/model1.pth")
        self.writer = writer.init_wandb(args)
        self.init_loss()
        self.init_optimizer()
        self.init_device()
    def init_loss(self):
        if (self.loss_name == "cross_entropy"):
            self.loss = nn.CrossEntropyLoss() 
    def init_optimizer(self):
        if (self.optimizer_name == "adam"):
            self.optimizer = optim.Adam(params = self.model.parameters(), 
                                        lr=self.lr_rate,
                                        weight_decay = self.weight_decay)
    def init_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
    def train_epoch(self, dataloader, device):
        self.model.train()
        train_loss, train_acc = 0, 0
        # print(len(dataloader))
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            y_pred = self.model(X)
            # print("truth", y)

            # Calculate loss
            loss = self.loss(y_pred, y)
            train_loss += loss.item()

            # Optimizer zero grad
            self.optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Calculate accuracy by select the biggest probability in each row
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            # print("pred", y_pred_class)
            train_acc += Accuracy(y_pred_class, y).item()
            # print(batch)
    
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        
        return train_loss, train_acc

    def test_epoch(self, dataloader, device):
        # Put model in eval mode
        self.model.eval() 
        
        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        
        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)
        
                # 1. Forward pass
                test_pred_logits = self.model(X)

                # 2. Calculate and accumulate loss
                loss = self.loss(test_pred_logits, y)
                test_loss += loss.item()
                
                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += Accuracy(test_pred_labels, y).item()
                
        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc
    
    def train(self, train_dataloader, test_dataloader):
        # 2. Create empty results dictionary
        results = {"train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        # print(len(train_dataloader))
        # 3. Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            train_loss, train_acc = self.train_epoch(dataloader = train_dataloader, device = self.device)
            val_loss, val_acc = self.test_epoch(dataloader=test_dataloader, device=self.device)
            # 4. Print out what's happening
            print(
                f"\nEpoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {val_loss:.4f} | "
                f"test_acc: {val_acc:.4f}"
            )
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

            # Writer to tensorboard
            writer.writer_to_wandb(results, epoch, self.writer)
            
        # 6. Return the filled results at the end of the epochs
        writer.end()
        return results
    
    def save_model(self, path):
        # Save the model's state dictionary
        torch.save(self.model.state_dict(), path)

    def plot_loss_accuracy(self, results):
        plot.plot_curve(results)

    def plot_confusion_matrix(self, test_data, class_names, device):
        # class_names is a list of class names
        # 1. Make predictions with trained model
        y_preds = []
        self.eval()
        with torch.inference_mode():
            for X, y in tqdm(test_data, desc="Making predictions"):
                # Send data and targets to target device
                X, y = X.to(device), y.to(device)
                # Do the forward pass
                y_logit = self.model(X)
                # Turn predictions from logits -> prediction probabilities -> predictions labels
                y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())
        # Concatenate list of predictions into a tensor
        y_pred_tensor = torch.cat(y_preds)
        plot.plot_confusion(y_pred_tensor, test_data, class_names)

    

