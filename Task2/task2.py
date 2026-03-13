import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# Config & Hyperparameters
# ==========================================
MODES      = 20
WIDTH      = 64
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42

DATA_DIR   = "FNO - Data 2025-2026"
TEST_FILE  = os.path.join(DATA_DIR, "data_test_128.npy")
TEST_UNKNOWN = os.path.join(DATA_DIR, "data_test_unknown_128.npy")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# Utilities
# ==========================================
def relative_l2_error(pred, true):
    """Computes the average relative L2 error over a batch"""
    num = np.linalg.norm(pred - true, axis=1)
    den = np.linalg.norm(true, axis=1)
    den = np.where(den == 0, 1e-12, den)
    return (num / den).mean()

def visualize_prediction_128(model, data_path, save_path):
    """
    Visualize a prediction vs ground thruth for a sample of the dataset 128
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

# ==========================================
# Datasets
# ==========================================
class TrajDataset(Dataset):
    """Dataset for Task 1 (One2One): Input t=0 -> Target t=1"""
    def __init__(self, npy_path):
        data = np.load(npy_path) # (N, 5, 128)
        self.x = data[:, 0, :].astype(np.float32)
        self.y = data[:, 4, :].astype(np.float32)

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class All2AllDataset(Dataset):
    """Dataset for Task 3 & 4 (All2All): (u0, t) -> ut, also handles finetuning via the 'limit' parameter"""
    def __init__(self, npy_path, times=None, limit=None):
        data = np.load(npy_path) # (N, 5, n)
        if times is None: times = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Option to deacrease the dataset size (Finetuning Task 4)
        if limit: data = data[:limit]

        self.times = np.array(times, dtype=np.float32)
        N, T, n = data.shape

        self.u0_list, self.t_list, self.y_list = [], [], []
        for i in range(N):
            u0_val = data[i, 0, :].astype(np.float32)
            for t_idx, t_val in enumerate(self.times):
                self.u0_list.append(u0_val)
                self.t_list.append(t_val)
                self.y_list.append(data[i, t_idx, :].astype(np.float32))

        self.u0 = np.stack(self.u0_list)
        self.t  = np.array(self.t_list)
        self.y  = np.stack(self.y_list)

    def __len__(self): return self.u0.shape[0]
    def __getitem__(self, idx): return self.u0[idx], self.t[idx], self.y[idx]

# ==========================================
# Models (FNO)
# ==========================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes))

    def compl_mul1d(self, input_fft, w_real, w_imag):
        # Manual complex multiplication : (a+ib)(c+id) = (ac-bd) + i(ad+bc)
        real, imag = input_fft[..., 0], input_fft[..., 1]
        w_r, w_i = w_real.permute(0, 2, 1), w_imag.permute(0, 2, 1)
        out_r = torch.einsum("bim,imj->bjm", real, w_r) - torch.einsum("bim,imj->bjm", imag, w_i)
        out_i = torch.einsum("bim,imj->bjm", real, w_i) + torch.einsum("bim,imj->bjm", imag, w_r)
        return torch.stack([out_r, out_i], dim=-1)

    def forward(self, x):
        batchsize, n = x.shape[0], x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        modes = min(self.modes, x_ft.shape[-2])
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-2], 2, device=x.device)
        
        out_ft[:, :, :modes, :] = self.compl_mul1d(
            x_ft[:, :, :modes, :], 
            self.weights_real[:, :, :modes], 
            self.weights_imag[:, :, :modes]
        )

        x = torch.fft.irfft(torch.complex(out_ft[..., 0], out_ft[..., 1]), n=n, dim=-1)
        return x

class FNO1d(nn.Module):
    """Standard architecture for Task 1 & 2"""
    def __init__(self, modes, width):
        super().__init__()
        self.name = "One2One"
        self.width = width
        self.fc0 = nn.Linear(1, width) # Lifting

        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1) # Projection

    def forward(self, x):
        x = self.fc0(x.unsqueeze(-1)).permute(0, 2, 1)
        for conv, w in zip([self.conv0, self.conv1, self.conv2, self.conv3], 
                           [self.w0, self.w1, self.w2, self.w3]):
            x = nn.functional.gelu(conv(x) + w(x))
        
        x = self.fc2(nn.functional.gelu(self.fc1(x.permute(0, 2, 1))))
        return x.squeeze(-1)

class FNO1d_time(nn.Module):
    """Time-Conditioned Architecture for Task 3 & 4 (Concatenation)"""
    def __init__(self, modes, width):
        super().__init__()
        self.name = "All2All"
        # Input channels = 2 (u0 + time)
        self.fc0 = nn.Linear(2, width) 

        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, u0, t):
        n = u0.shape[1]
        # Time concatenation: (Batch, n, 2)
        t_expand = t.view(-1, 1, 1).repeat(1, n, 1)
        x = torch.cat([u0.unsqueeze(-1), t_expand], dim=-1)
        
        x = self.fc0(x).permute(0, 2, 1)
        for conv, w in zip(self.convs, self.ws):
            x = nn.functional.gelu(conv(x) + w(x))

        x = self.fc2(nn.functional.gelu(self.fc1(x.permute(0, 2, 1))))
        return x.squeeze(-1)

# ==========================================
# Training & Testing Loops
# ==========================================
def train_model(model, train_loader, val_loader, epochs, lr, save_path, plot_path, is_all2all=False):
    """Generic training function"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"Training {save_path}...")
    for ep in range(epochs):
        model.train()
        t_loss = 0
        for batch in train_loader:
            if is_all2all:
                x, t, y = [b.to(DEVICE) for b in batch]
                pred = model(x, t)
            else:
                x, y = [b.to(DEVICE) for b in batch]
                pred = model(x)
            
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
        
        t_loss /= len(train_loader.dataset)
        train_losses.append(t_loss)

        # Validation
        if val_loader:
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if is_all2all:
                        x, t, y = [b.to(DEVICE) for b in batch]
                        pred = model(x, t)
                    else:
                        x, y = [b.to(DEVICE) for b in batch]
                        pred = model(x)
                    v_loss += loss_fn(pred, y).item() * x.size(0)
            v_loss /= len(val_loader.dataset)
            val_losses.append(v_loss)
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), save_path)
        
        scheduler.step()

        print(f"[{model.name}] Epoch {ep:3d}  Train loss {t_loss:.4e}  Val loss {v_loss:.4e}")

    # Plot Loss
    plt.figure(figsize=(8,5))
    plt.semilogy(train_losses, label='Train loss')
    if val_loader: plt.semilogy(val_losses, label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(); plt.title(f"{model.name} Training Loss (log scale)")
    plt.tight_layout()
    plt.savefig(plot_path); plt.close()
    
    # Reload best model
    if os.path.exists(save_path): model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model

def test_resolution(model, data_dir, resolutions=[32, 64, 96, 128]):
    """Task 2: Testing on different spatial resolutions"""
    print("\n--- Testing Resolutions ---")
    model.eval()
    for s in resolutions:
        path = os.path.join(data_dir, f"data_test_{s}.npy")
        if not os.path.exists(path): continue
        
        data = np.load(path)
        x = torch.from_numpy(data[:, 0, :]).float().to(DEVICE)
        y = data[:, 4, :]
        
        with torch.no_grad(): pred = model(x).cpu().numpy()
        print(f"Res {s}: Rel L2 Error = {relative_l2_error(pred, y):.4e}")

def evaluate_all2all(model, path, times=[0.25, 0.5, 0.75, 1.0]):
    """Task 3/4: Evaluation at different time steps"""
    print(f"\n--- Eval All2All on {os.path.basename(path)} ---")
    data = np.load(path) # (N, 5, n)
    dataset_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    model.eval()
    
    for t_target in times:
        t_idx = np.argmin(np.abs(dataset_times - t_target))
        x = torch.from_numpy(data[:, 0, :]).float().to(DEVICE)
        t = torch.full((x.shape[0],), t_target, device=DEVICE)
        y = data[:, t_idx, :]
        
        with torch.no_grad(): pred = model(x, t).cpu().numpy()
        print(f"Time {t_target}: Rel L2 Error = {relative_l2_error(pred, y):.4e}")
    return relative_l2_error(pred, y) # Return error at last time step (t=1.0)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Paths
    train_f = os.path.join(DATA_DIR, "data_train_128.npy")
    val_f   = os.path.join(DATA_DIR, "data_val_128.npy")
    test_f  = os.path.join(DATA_DIR, "data_test_128.npy")
    
    # --- TASK 1: One-to-One ---
    print("\n=== TASK 1: One-to-One Training ===")
    train_ds = TrajDataset(train_f)
    val_ds   = TrajDataset(val_f)
    
    model_1 = FNO1d(MODES, WIDTH).to(DEVICE)
    model_1 = train_model(model_1, 
                          DataLoader(train_ds, BATCH_SIZE, shuffle=True),
                          DataLoader(val_ds, BATCH_SIZE),
                          EPOCHS, LR, "model_task1.pth", "loss_task1.png")
    
    visualize_prediction_128(model_1, TEST_FILE, save_path="prediction_one2one_128.png")

    # --- TASK 2: Resolutions ---
    print("\n=== TASK 2: Resolutions ===")
    test_resolution(model_1, DATA_DIR)

    # --- TASK 3: All-to-All ---
    print("\n=== TASK 3: All-to-All Training ===")

    train_ds_all = All2AllDataset(train_f)
    val_ds_all   = All2AllDataset(val_f)
    
    model_all = FNO1d_time(MODES, WIDTH).to(DEVICE)
    model_all = train_model(model_all,
                            DataLoader(train_ds_all, BATCH_SIZE, shuffle=True),
                            DataLoader(val_ds_all, BATCH_SIZE),
                            EPOCHS, LR, "model_all2all.pth", "loss_all2all.png", is_all2all=True)
    
    evaluate_all2all(model_all, test_f)
    visualize_prediction_128(model_all, TEST_FILE, save_path="prediction_all2all_128.png")

    # --- TASK 4: Finetuning ---
    print("\n=== TASK 4: Unknown Distribution & Finetuning ===")
    ft_train = os.path.join(DATA_DIR, "data_finetune_train_unknown_128.npy")
    ft_val   = os.path.join(DATA_DIR, "data_finetune_val_unknown_128.npy")
    test_unk = os.path.join(DATA_DIR, "data_test_unknown_128.npy")

    # 1. Zero-shot
    print("Zero-shot evaluation:")
    evaluate_all2all(model_all, test_unk, times=[1.0])
    visualize_prediction_128(model_all, TEST_UNKNOWN, save_path="prediction_unknow_zero_shot_128.png")

    # 2. Finetuning (limit=32 trajectories)
    print("\nFinetuning on 32 trajectories...")
    ft_ds = All2AllDataset(ft_train, limit=32)
    ft_val_ds = All2AllDataset(ft_val)
    
    model_ft = FNO1d_time(MODES, WIDTH).to(DEVICE)
    model_ft.load_state_dict(model_all.state_dict())
    
    model_ft = train_model(model_ft,
                           DataLoader(ft_ds, batch_size=8, shuffle=True), # small batch size
                           DataLoader(ft_val_ds, batch_size=32),
                           epochs=20, lr=1e-4, # reduced learning rate
                           save_path="model_finetuned.pth", 
                           plot_path="loss_finetune.png", 
                           is_all2all=True)
    
    print("Finetuned evaluation:")
    evaluate_all2all(model_ft, test_unk, times=[1.0])
    visualize_prediction_128(model_ft, TEST_UNKNOWN, save_path="prediction_unknow_finetuned_128.png")