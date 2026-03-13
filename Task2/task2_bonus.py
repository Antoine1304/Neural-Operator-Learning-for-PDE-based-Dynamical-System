import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --------- Hyperparameters ----------
DATA_DIR = "FNO - Data 2025-2026"
FINETUNE_TRAIN_UNKNOWN = os.path.join(DATA_DIR, "data_finetune_train_unknown_128.npy")
TEST_UNKNOWN = os.path.join(DATA_DIR, "data_test_unknown_128.npy")

DEVICE = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------- utilities ----------
def relative_l2_error(pred, true):
    num = np.linalg.norm(pred - true, axis=1)
    den = np.linalg.norm(true, axis=1)
    den = np.where(den == 0, 1e-12, den)
    return (num / den).mean()

# --------- dataset (32 traj) ----------
class SmallAll2AllDataset(Dataset):
    def __init__(self, npy_path, first_k=32, times=None):
        data = np.load(npy_path)   # (N,5,n)
        if times is None:
            times = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        N, T, n = data.shape
        k = min(first_k, N)
        u0 = data[:k, 0, :].astype(np.float32)
        X_u0, X_t, Y = [], [], []
        for i in range(k):
            for ti in range(T):
                X_u0.append(u0[i])
                X_t.append(times[ti])
                Y.append(data[i, ti, :].astype(np.float32))
        self.X_u0 = np.stack(X_u0, axis=0)
        self.X_t  = np.array(X_t, dtype=np.float32)
        self.Y    = np.stack(Y, axis=0)

    def __len__(self):
        return self.X_u0.shape[0]

    def __getitem__(self, idx):
        return self.X_u0[idx], self.X_t[idx], self.Y[idx]

# --------- Spectral conv and FNO (same design as before) ----------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def compl_mul1d(self, input_fft, w_r, w_i):
        real = input_fft[..., 0]
        imag = input_fft[..., 1]
        w_r = w_r.permute(0, 2, 1)
        w_i = w_i.permute(0, 2, 1)
        out_real = torch.einsum("bim,imj->bjm", real, w_r) - torch.einsum("bim,imj->bjm", imag, w_i)
        out_imag = torch.einsum("bim,imj->bjm", real, w_i) + torch.einsum("bim,imj->bjm", imag, w_r)
        return torch.stack([out_real, out_imag], dim=-1)

    def forward(self, x):
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        x_ft_stack = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        n_freq = x_ft.shape[-1]
        modes = min(self.modes, n_freq)
        out_ft = torch.zeros(x.shape[0], self.out_channels, n_freq, 2, device=x.device, dtype=x_ft_stack.dtype)
        w_r = self.weights_real[:, :, :modes]
        w_i = self.weights_imag[:, :, :modes]
        out_ft[:, :, :modes, :] = self.compl_mul1d(x_ft_stack[:, :, :modes, :], w_r, w_i)
        out_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x_out = torch.fft.irfft(out_complex, n=n, dim=-1)
        return x_out

class FNO1d_time(nn.Module):
    def __init__(self, modes, width, in_channels=2):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, 1)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.w2 = nn.Conv1d(width, width, 1)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = nn.GELU()

    def forward(self, u0, t_scalar):
        if t_scalar.dim() == 1:
            t_scalar = t_scalar.unsqueeze(-1)
        n = u0.shape[1]
        t_chan = t_scalar.unsqueeze(1).repeat(1, n, 1)
        u0_chan = u0.unsqueeze(-1)
        x = torch.cat([u0_chan, t_chan], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv0(x); x2 = self.w0(x); x = self.act(x1 + x2)
        x1 = self.conv1(x); x2 = self.w1(x); x = self.act(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = self.act(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = self.act(x1 + x2)
        x = x.permute(0, 2, 1)
        x = self.fc1(x); x = self.act(x)
        x = self.fc2(x).squeeze(-1)
        return x

def evaluate_all2all(model, data_path, times_to_eval=[1.0], batch_size=32):
    data = np.load(data_path)
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    N, T, n = data.shape
    results = {}
    model.eval()
    for t_eval in times_to_eval:
        idx = int(np.argmin(np.abs(times - t_eval)))
        u0 = data[:, 0, :].astype(np.float32)
        y_true = data[:, idx, :].astype(np.float32)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(u0), torch.from_numpy(np.full((N,), t_eval, dtype=np.float32)), torch.from_numpy(y_true))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds, trues = [], []
        with torch.no_grad():
            for u0b, tb, yb in loader:
                u0b = u0b.to(DEVICE).float()
                tb  = tb.to(DEVICE).float()
                pred = model(u0b, tb)
                preds.append(pred.cpu().numpy()); trues.append(yb.numpy())
        preds = np.concatenate(preds, axis=0); trues = np.concatenate(trues, axis=0)
        results[t_eval] = relative_l2_error(preds, trues)
        print(f"t={t_eval:.2f}  rel L2 = {results[t_eval]:.6e}")
    return results

# --------- training from-scratch on 32 traj ----------
def train_from_scratch_on_unknown(finetune_train_path, test_unknown_path,
                                  first_k=32, modes=20, width=64,
                                  batch_size=8, epochs=40, lr=1e-3,
                                  save_path="fno_all2all_scratch_unknown.pth"):
    train_ds = SmallAll2AllDataset(finetune_train_path, first_k=first_k)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = FNO1d_time(modes=modes, width=width, in_channels=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for u0b, tb, yb in train_loader:
            u0b = u0b.to(DEVICE).float(); tb = tb.to(DEVICE).float(); yb = yb.to(DEVICE).float()
            opt.zero_grad()
            pred = model(u0b, tb)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            epoch_loss += loss.item() * u0b.size(0)
        epoch_loss /= len(train_ds)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:3d}  train loss {epoch_loss:.6e}")
    torch.save(model.state_dict(), save_path)
    print(f"Saved scratch model to {save_path}")
    results = evaluate_all2all(model, test_unknown_path, times_to_eval=[1.0], batch_size=batch_size)
    return model, results.get(1.0)


def visualize_prediction_128(model, data_path, save_path="prediction_example_128.png"):
    """
    Visualise une prédiction vs la vérité pour un exemple du dataset test 128.
    """
    data = np.load(data_path)  # shape (128, 5, 128)
    x = data[:, 0, :].astype(np.float32)
    y = data[:, 4, :].astype(np.float32)

    idx = 0
    x0 = torch.from_numpy(x[idx]).unsqueeze(0).to(DEVICE)  # (1,128)
    y_true = y[idx]

    model.eval()
    with torch.no_grad():
        try:
            y_pred = model(x0).cpu().numpy()[0]
        except TypeError:
            t_tensor = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
            y_pred = model(x0, t_tensor).cpu().numpy()[0]


    # plot
    plt.figure(figsize=(8,4))
    plt.plot(y_true, label="True", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.xlabel("Spatial coordinate")
    plt.ylabel("u(x, t=1)")
    plt.title("Prediction vs Ground Truth (resolution 128)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved prediction plot to {save_path}")

model_scratch, scratch_err = train_from_scratch_on_unknown(
    finetune_train_path = FINETUNE_TRAIN_UNKNOWN,
    test_unknown_path   = TEST_UNKNOWN,
    first_k = 32,
    modes = 20,
    width = 64,
    batch_size = 8,
    epochs = 50,
    lr = 0.001,
    save_path = "fno_all2all_scratch_unknown.pth"
)

visualize_prediction_128(model_scratch, TEST_UNKNOWN, save_path="prediction_model_scratch_128.png")
