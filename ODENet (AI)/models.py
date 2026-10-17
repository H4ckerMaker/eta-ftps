import math
import torch
import torch.nn as nn


def log_time(dt):
    return (torch.log1p(dt.clamp(min=0.0) * 1e6) / math.log1p(1e7)).clamp(max=1.0)


class ODEFunc(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))

    def forward(self, h):
        return self.net(h)


class ODEBlock(nn.Module):
    def __init__(self, func, steps=2):
        super().__init__()
        self.func = func
        self.steps = steps

    def forward(self, h, dt):  
        step = dt / self.steps
        f = self.func
        for _ in range(self.steps):
            k1 = f(h)
            k2 = f(h + 0.5 * step * k1)
            k3 = f(h + 0.5 * step * k2)
            k4 = f(h + step * k3)
            h = h + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


class ODEEncoder(nn.Module):

    def __init__(self, ode_dim=16, ode_hidden=64, steps=2, time_scale=1.0):
        super().__init__()
        self.ode_dim = ode_dim
        self.time_scale = time_scale
        self.ode = ODEBlock(ODEFunc(ode_dim, ode_hidden), steps)
        self.cell = nn.GRUCell(2, ode_dim)  

    def forward(self, x):
        B, n, _ = x.shape
        size, t = x[..., 0], x[..., 1]
        h = x.new_zeros(B, self.ode_dim)
        outs = []
        prev_t = t[:, 0]
        for i in range(n):
            dt = log_time(t[:, i] - prev_t)  
            h = self.ode(h, (dt * self.time_scale).unsqueeze(1))
            h = self.cell(torch.stack([size[:, i], dt], dim=1), h)
            outs.append(h)
            prev_t = t[:, i]
        return torch.stack(outs, dim=1) 


class Head(nn.Module):
    def __init__(self, in_dim, n_classes, lstm_hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, lstm_hidden, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1])  


class ODELSTM(nn.Module):
    def __init__(self, n_classes, ode_dim=16, **ode_kwargs):
        super().__init__()
        self.encoder = ODEEncoder(ode_dim=ode_dim, **ode_kwargs)
        self.head = Head(ode_dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


class LSTMBaseline(nn.Module):

    def __init__(self, n_classes, features="both"):
        super().__init__()
        self.features = features
        self.head = Head(1 if features == "size" else 2, n_classes)

    def forward(self, x):
        size, t = x[..., 0:1], x[..., 1]
        if self.features == "size":
            return self.head(size)
        iat = log_time(torch.diff(t, dim=1, prepend=t[:, :1])).unsqueeze(-1)
        return self.head(torch.cat([size, iat], dim=-1))