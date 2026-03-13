import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import AdamW
from dataclasses import dataclass
import time
from scipy.spatial import cKDTree

from model.gaot import GAOT
from model.layers.magno import MAGNOConfig
from model.layers.attn import TransformerConfig, AttentionConfig

# ==========================================
# 1. Configuration
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Task 2 Settings
N_latents = 256
resample_seed_base = 42  
use_dataset_bbox = True  

# Training yperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 0.001

# ==========================================
# 2. Dataset
# ==========================================
class ElasticityDataset(Dataset):
    def __init__(self, inputs, targets, coords):
        self.inputs = inputs.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.coords = coords.astype(np.float32) if coords is not None else None
        
        self.in_min = self.inputs.min(axis=(0,1))
        self.in_max = self.inputs.max(axis=(0,1))
        self.out_min = self.targets.min(axis=(0,1))
        self.out_max = self.targets.max(axis=(0,1))
        
        self.in_range = np.maximum(self.in_max - self.in_min, 1e-8)
        self.out_range = np.maximum(self.out_max - self.out_min, 1e-8)

    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self, idx):
        inp = (self.inputs[idx] - self.in_min) / self.in_range
        tgt = (self.targets[idx] - self.out_min) / self.out_range
        inp_t = torch.from_numpy(inp).float()
        tgt_t = torch.from_numpy(tgt).float()
        coords_t = torch.from_numpy(self.coords[idx]).float() if self.coords is not None else None
        return {"input": inp_t, "target": tgt_t, "coords": coords_t}

def create_datasets(path="Elasticity.nc", test_frac=0.2, seed=0):
    with h5py.File(path, "r") as f:
        u = np.array(f["u"]).squeeze(axis=1)
        c = np.array(f["c"]).squeeze(axis=1)
        x = np.array(f["x"]).squeeze(axis=1)
    
    N = c.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(N * (1 - test_frac))
    
    return (ElasticityDataset(c[idx[:split]], u[idx[:split]], x[idx[:split]]),
            ElasticityDataset(c[idx[split:]], u[idx[split:]], x[idx[split:]]))

train_set, test_set = create_datasets("Elasticity.nc", test_frac=0.2, seed=42)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

in_ch = train_set.inputs.shape[2]
out_ch = train_set.targets.shape[2]
print(f"Data loaded. Input: {in_ch}, Output: {out_ch}")

# =========================================
# 3. Utilities
# =========================================
def sample_latent_tokens(N, coords_all=None, seed=0, device='cpu', use_bbox=True):
    rng = np.random.RandomState(seed)
    mins, maxs = np.array([0.,0.]), np.array([1.,1.])
    if use_bbox and coords_all is not None:
        coords_sub = coords_all[:100] if len(coords_all)>100 else coords_all
        if isinstance(coords_sub, torch.Tensor): coords_sub=coords_sub.cpu().numpy()
        flat = coords_sub.reshape(-1, 2)
        mins, maxs = flat.min(axis=0), flat.max(axis=0)
    samples = rng.rand(N, 2) * (maxs - mins)[None, :] + mins[None, :]
    return torch.from_numpy(samples.astype(np.float32)).to(device)

def list_of_arrays_to_csr_torch(list_of_arrays, device):
    lens = np.array([len(arr) for arr in list_of_arrays], dtype=np.int64)
    row_splits = np.concatenate(([0], np.cumsum(lens)))
    indices = np.concatenate(list_of_arrays).astype(np.int64) if row_splits[-1]>0 else np.empty(0, dtype=np.int64)
    return {"neighbors_index": torch.from_numpy(indices).long().to(device),
            "neighbors_row_splits": torch.from_numpy(row_splits).long().to(device)}


# ==========================================
# 4. Model Config & Init
# ==========================================
@dataclass
class ModelArgs:
    magno: MAGNOConfig
    transformer: TransformerConfig

@dataclass
class GAOTConfig:
    latent_tokens_size: list
    args: ModelArgs

magno_conf = MAGNOConfig(
    coord_dim=2, 
    radius=0.033,
    hidden_size=32,
    mlp_layers=3,
    lifting_channels=32,
)

transformer_config = TransformerConfig(
    patch_size=2, 
    hidden_size=128, 
)

model_args = ModelArgs(
    magno=magno_conf,
    transformer=transformer_config)

config = GAOTConfig(
    latent_tokens_size=[16, 16],
    args=model_args)

model = GAOT(
    input_size=in_ch,
    output_size=out_ch, 
    config=config
).to(device)

print(f"Model initialized: GAOT")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, eta_min=1e-6)

# ==========================================
# 5. Training Loop (with KDTree)
# ==========================================
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # --- Resample Tokens (Dynamic) ---
    seed_epoch = resample_seed_base + epoch
    latent_tokens_coord = sample_latent_tokens(N_latents, coords_all=train_set.coords, 
                                             seed=seed_epoch, device=device, use_bbox=use_dataset_bbox)
    

    # --- Training ---
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        coords = batch["coords"].to(device)

        output_pred = model(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coords,
            pndata=inp,
            query_coord=coords,
            encoder_nbrs=None,
            decoder_nbrs=None,
            condition=None)

        loss = torch.mean(torch.abs(output_pred - tgt)) / (torch.mean(torch.abs(tgt)) + 1e-8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()

    # --- Validation ---
    model.eval()
    test_rel_l1 = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            inp, tgt, coords = batch["input"].to(device), batch["target"].to(device), batch["coords"].to(device)
            coords_np = coords.cpu().numpy()
            
            output_pred = model(
                latent_tokens_coord=latent_tokens_coord,
                xcoord=coords,
                pndata=inp,
                query_coord=coords,
                encoder_nbrs=None,
                decoder_nbrs=None,
                condition=None)
            
            err = torch.mean(torch.abs(output_pred - tgt)) / (torch.mean(torch.abs(tgt)) + 1e-8)
            test_rel_l1 += err.item()
    
    test_rel_l1 /= len(test_loader)
    
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Rel L1: {test_rel_l1*100:.2f}% | Time: {time.time()-epoch_start:.1f}s")

end_time = time.time()
print(f"Total Training Time: {end_time - start_time:.2f} sec")

# ==========================================
# 6. Evaluation & Plotting
# ==========================================
print("Evaluation complete. Generating plots...")
model.eval()

with torch.no_grad():
    batch = next(iter(test_loader))
    inp = batch["input"].to(device)
    tgt = batch["target"].to(device)
    coords = batch["coords"].to(device)

    output_points = model(
        latent_tokens_coord=latent_tokens_coord,
        xcoord=coords,
        pndata=inp,
        query_coord=coords,
        encoder_nbrs=None,
        decoder_nbrs=None,
        condition=None
    )

    # choose sample to display (0..B-1)
    sample_idx = 4
    B = output_points.shape[0]
    assert 0 <= sample_idx < B

    # prepare arrays (CPU numpy)
    coords_np = coords.cpu().numpy() if coords.dim() == 2 else coords[sample_idx].cpu().numpy()  # [S,2] or [S,2]
    inp_np = inp[sample_idx].cpu().numpy()    # [S, in_ch]
    tgt_np = tgt[sample_idx].cpu().numpy()    # [S, out_ch]
    pred_np = output_points[sample_idx].cpu().numpy()  # [S, out_ch]

    # color ranges: use same vmin/vmax for GT and Pred for fair comparison
    vmin = float(min(tgt_np[:, 0].min(), pred_np[:, 0].min()))
    vmax = float(max(tgt_np[:, 0].max(), pred_np[:, 0].max()))

    # scatter plot on original points
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sc0 = axes[0].scatter(coords_np[:, 0], coords_np[:, 1], c=inp_np[:, 0], cmap="viridis", s=12)
    axes[0].set_title("Input (point values)")
    axes[0].axis("equal")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(coords_np[:, 0], coords_np[:, 1], c=tgt_np[:, 0], cmap="viridis", vmin=vmin, vmax=vmax, s=12)
    axes[1].set_title("Ground truth (points)")
    axes[1].axis("equal")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    sc2 = axes[2].scatter(coords_np[:, 0], coords_np[:, 1], c=pred_np[:, 0], cmap="viridis", vmin=vmin, vmax=vmax, s=12)
    axes[2].set_title("Prediction (points)")
    axes[2].axis("equal")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    # compute pointwise relative L1 (percent) for the sample
    eps = 1e-8
    rel_l1_point = (np.mean(np.abs(pred_np - tgt_np)) / (np.mean(np.abs(tgt_np)) + eps)) * 100.0
    axes[2].text(0.02, 0.95, f"Rel L1 (points): {rel_l1_point:.3f}%", color="white", fontsize=12,
                 transform=axes[2].transAxes, bbox=dict(facecolor="black", alpha=0.6))

    plt.tight_layout()
    plt.savefig("result_task2_1.png", dpi=200)
    plt.close(fig)
    
    print(f"Saved pointwise plots and latent comparison for sample {sample_idx}. Rel L1 (points): {rel_l1_point:.3f}%")