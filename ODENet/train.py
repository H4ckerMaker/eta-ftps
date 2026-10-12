import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from csv_parser_v2 import extract_features
from data import make_label_map, split_by_flow, build_blocks ,undersample
from models import LSTMBaseline, ODELSTM

def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))

@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    preds, ys = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu())
        ys.append(yb)
    p, y = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return float((p == y).mean()), macro_f1(y, p, n_classes)


def main(n=20,seed=3):
    flows = extract_features('csv_file.csv') #TODO: import all the flows for data training
    label_map = make_label_map(flows)
    n_classes = len(label_map)
    print("classes:", label_map)
    train_flows, test_flows = split_by_flow(flows, 0.2, seed) #splitting training set and test set from the csv file
    Xtr, ytr = undersample(*build_blocks(train_flows, n, label_map=label_map), seed=seed) #balanced sets for training
    Xte, yte = build_blocks(train_flows, n, label_map=label_map) #unblanced sets for testing
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=True) #Dataset for training with shuffling for each batch
    te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=256) #Dataset for testing without shuffling

    model = {"ode": lambda: ODELSTM(n_classes),
             "lstm_size": lambda: LSTMBaseline(n_classes, "size"),
             "lstm_both": lambda: LSTMBaseline(n_classes, "both")}[model_name]().to(device) #This creates a model at runtime depending on the name we pass

    opt = torch.optim.Adam(model.parameters(), lr=lr) #We use Adam to update the weights on the training phase with a learning rate lr
    loss_fn = nn.CrossEntropyLoss() #We use a softmax to calculate the losses

    best = 0.0
    for ep in range(epochs):
        model.train() 
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad() #we clean the gradient
            loss = loss_fn(model(xb), yb) #we calculate the loss
            loss.backward() #we backpropagate the error
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc, f1 = evaluate(model, te, device, n_classes)
        if f1 > best:
            best = f1
            torch.save(model.state.dict(), f"best_{model_name}_n{n}.pt")
        print("ep {ep + 1:02d}  loss {loss.item():.4f}  acc {acc:.4f}  macroF1 {f1:.4f}")

if __name__ == "__main__":
    main()
