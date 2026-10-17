import collections
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from Parser.csv_parser import extract_features
from data import make_label_map, split_by_flow, build_blocks, undersample
from models import LSTMBaseline, ODELSTM

FILES = sorted(glob.glob('./classes_csvs/*/reg/my_*.raw.csv'))

def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu())
        ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def main(n=20, epochs=100, batch_size=64, lr=1e-3, model_name="ode", seed=5, time_scale=1.0):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    flows = []
    for path in FILES:
        flows += extract_features(path)

    label_map = make_label_map(flows)
    n_classes = len(label_map)
    names = list(label_map)
    print("classes:", label_map)
    print("flow per class:", dict(collections.Counter(f["label"] for f in flows)))

    train_flows, test_flows = split_by_flow(flows, 0.2, seed)
    Xall, yall = build_blocks(flows, n, label_map=label_map)
    print("block per class:", dict(zip(names, np.bincount(yall, minlength=n_classes).tolist())))
    print("mean dt between packets (s):", float(np.median(np.diff(Xall[:, :, 1], axis=1))))

    Xtr, ytr = undersample(*build_blocks(train_flows, n, label_map=label_map), seed=seed)
    Xte, yte = build_blocks(test_flows, n, label_map=label_map)
    print("train after undersample:", len(Xtr), np.bincount(ytr, minlength=n_classes).tolist(),
          "| test:", len(Xte), np.bincount(yte, minlength=n_classes).tolist())

    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=True)
    te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=256)

    model = {"ode": lambda: ODELSTM(n_classes, time_scale=time_scale),
             "lstm_size": lambda: LSTMBaseline(n_classes, "size"),
             "lstm_both": lambda: LSTMBaseline(n_classes, "both")}[model_name]().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best, f1_hist, ckpt = 0.0, [], f"best_{model_name}_n{n}.pt"
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        y, p = predict(model, te, device)
        acc, f1 = float((p == y).mean()), macro_f1(y, p, n_classes)
        f1_hist.append(f1)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), ckpt)
        print(f"ep {ep + 1:03d}  loss {loss.item():.4f}  acc {acc:.4f}  macroF1 {f1:.4f}")

    print(f"\nmean macroF1 of last 10 epochs: {np.mean(f1_hist[-10:]):.4f}  (best: {best:.4f})")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    y, p = predict(model, te, device)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    print("\nconfusion matrix (rows = true class, columns = predicted):")
    print("classes:", names)
    print(cm)
    print("recall for each class:", {names[i]: round(float(cm[i, i] / max(cm[i].sum(), 1)), 3) for i in range(n_classes)})


if __name__ == "__main__":
    main()