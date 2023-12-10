from net2.st_gcn import Model
from trainer.trainer import Trainer
from feeder.feeder import FeederINCLUDE
from torch.utils.data import DataLoader
from metrics.accuracy import Accuracy
import torch

def test(model, dataloader, device):
        # Put model in eval mode
        model.eval() 
        
        # Setup test accuracy values
        test_acc = 0
        
        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)
        
                # 1. Forward pass
                test_pred_logits = model(X)
                
                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                
                test_acc += Accuracy(test_pred_labels, y).item()
                print(batch, y, test_pred_labels, test_acc)
                
        # Adjust metrics to get average accuracy per batch 
        test_acc = test_acc / len(dataloader)
        return test_acc
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(2, 263, graph_args = {"layout": "mediapipe", "strategy": "spatial"}, 
                  edge_importance_weighting = True).to(device)
    model.load_state_dict(torch.load('models/model2.pth'))
    test_dataset = FeederINCLUDE(data_path="data/npy_test.npy", label_path="data/label_test.pickle")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    acc = test(model = model, dataloader = test_dataloader, device = device)
    print(acc)
    print("Done")

