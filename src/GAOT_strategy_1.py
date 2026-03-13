import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.optim import AdamW
from dataclasses import dataclass
import time

from model.gaot import GAOT
from model.layers.magno import MAGNOConfig
from model.layers.attn import TransformerConfig

# ==========================================
# 1. Configuration
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
g = torch.Generator()
g.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 0.001

# ==========================================
# 2. Dataset
# ==========================================
class ElasticityDataset(Dataset):
    def __init__(self, inputs, targets, coords):
        # inputs: np.array [N, S, in_ch]
        # targets: np.array [N, S, out_ch]
        # coords: np.array [N, S, 2]  (ou None)
        self.inputs = inputs.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.coords = coords.astype(np.float32) if coords is not None else None

        # Normalisation par canal (min/max) pour inputs et targets
        # inputs: [N, S, C_in] -> min/max shape [C_in]
        self.in_min = self.inputs.min(axis=(0,1))
        self.in_max = self.inputs.max(axis=(0,1))
        self.out_min = self.targets.min(axis=(0,1))
        self.out_max = self.targets.max(axis=(0,1))

        # éviter division par zéro
        self.in_range = np.maximum(self.in_max - self.in_min, 1e-8)
        self.out_range = np.maximum(self.out_max - self.out_min, 1e-8)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        inp = (self.inputs[idx] - self.in_min) / self.in_range    # [S, in_ch]
        tgt = (self.targets[idx] - self.out_min) / self.out_range # [S, out_ch]

        inp_t = torch.from_numpy(inp).float()
        tgt_t = torch.from_numpy(tgt).float()

        coords_t = None
        if self.coords is not None:
            coords_t = torch.from_numpy(self.coords[idx]).float()  # [S,2]

        return {"input": inp_t, "target": tgt_t, "coords": coords_t}

def create_datasets(path="Elasticity.nc", test_frac=0.2, seed=0, max_samples=None):
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

train_set, test_set = create_datasets("Elasticity.nc", test_frac=0.2, seed=42, max_samples=None)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g)
test_loader  = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

in_ch = train_set.inputs.shape[2]
out_ch = train_set.targets.shape[2]

print(f"Data loaded. Input: {in_ch}, Output: {out_ch}")

# =========================================
# 3. Utilities
# =========================================
def create_coordinate_grid(height=32, width=32):
    """
    Create a regular 2D coordinate grid for 32x32 regular grid
    """
    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

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

magno_config = MAGNOConfig(
    coord_dim=2,
    radius=0.033,
    hidden_size=32,
    mlp_layers=3,
    lifting_channels=32
)

transformer_config = TransformerConfig(
    patch_size=2,
    hidden_size=128
)

model_args = ModelArgs(
    magno=magno_config,
    transformer=transformer_config
)

config = GAOTConfig(
    latent_tokens_size=[32, 32],
    args=model_args
)

# Initialize GAOT model
model = GAOT(
    input_size=in_ch, 
    output_size=out_ch,
    config=config
).to(device)

print(f"Model initialized: GAOT")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


# Create coordinate grids (fixed for all samples)
latent_tokens_coord = create_coordinate_grid(32, 32).to(device)  # Shape: [4096, 2]

# --- Training loop  ---
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, eta_min=1e-6)
loss_l1 = torch.nn.L1Loss(reduction='mean')

# ==========================================
# 5. Training Loop
# ==========================================
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    model.train()
    train_loss = 0.0

    for step, batch in enumerate(train_loader):
        inp = batch["input"].to(device)   # [B, S, in_ch]
        tgt = batch["target"].to(device)  # [B, S, out_ch]
        coords = batch["coords"].to(device)  # [B, S, 2] or None

        output_pred = model(
            latent_tokens_coord=latent_tokens_coord,  # [G,2]
            xcoord=coords,                            # [B,S,2] or [S,2]
            pndata=inp,                               # [B,S,in_ch]
            query_coord=coords,                       # predict on original points
            encoder_nbrs=None,
            decoder_nbrs=None,
            condition=None
        ) 
        loss = torch.mean(torch.abs(output_pred - tgt)) / (torch.mean(torch.abs(tgt)) + 1e-8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()

    # Evaluation on test set
    model.eval()
    test_rel_l1 = 0.0
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            inp = batch["input"].to(device)   # [B, S, in_ch]
            tgt = batch["target"].to(device)  # [B, S, out_ch]
            coords = batch["coords"].to(device)  # [B, S, 2] or None

            output_pred = model(
                latent_tokens_coord=latent_tokens_coord,  # [G,2]
                xcoord=coords,                            # [B,S,2] or [S,2]
                pndata=inp,                               # [B,S,in_ch]
                query_coord=coords,                       # predict on original points
                encoder_nbrs=None,
                decoder_nbrs=None,
                condition=None
            ) 

            rel_l1 = torch.mean(torch.abs(output_pred - tgt)) / (torch.mean(torch.abs(tgt)) + 1e-8)
            test_rel_l1 += rel_l1.item()

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

    # choose channel to visualize (0 by default)
    ch_in = 0
    ch_out = 0

    # color ranges: use same vmin/vmax for GT and Pred for fair comparison
    vmin = float(min(tgt_np[:, ch_out].min(), pred_np[:, ch_out].min()))
    vmax = float(max(tgt_np[:, ch_out].max(), pred_np[:, ch_out].max()))
    # if vmin==vmax, expand a bit to avoid degenerate colormap
    if abs(vmax - vmin) < 1e-8:
        vmin -= 1e-3
        vmax += 1e-3

    # scatter plot on original points
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sc0 = axes[0].scatter(coords_np[:, 0], coords_np[:, 1], c=inp_np[:, ch_in], cmap="viridis", s=12)
    axes[0].set_title("Input (point values)")
    axes[0].axis("equal")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(coords_np[:, 0], coords_np[:, 1], c=tgt_np[:, ch_out], cmap="viridis", vmin=vmin, vmax=vmax, s=12)
    axes[1].set_title("Ground truth (points)")
    axes[1].axis("equal")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    sc2 = axes[2].scatter(coords_np[:, 0], coords_np[:, 1], c=pred_np[:, ch_out], cmap="viridis", vmin=vmin, vmax=vmax, s=12)
    axes[2].set_title("Prediction (points)")
    axes[2].axis("equal")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    # compute pointwise relative L1 (percent) for the sample
    eps = 1e-8
    rel_l1_point = (np.mean(np.abs(pred_np - tgt_np)) / (np.mean(np.abs(tgt_np)) + eps)) * 100.0
    axes[2].text(0.02, 0.95, f"Rel L1 (points): {rel_l1_point:.3f}%", color="white", fontsize=12,
                 transform=axes[2].transAxes, bbox=dict(facecolor="black", alpha=0.6))

    plt.tight_layout()
    plt.savefig("result_task1", dpi=200)
    plt.close(fig)

    print(f"Saved pointwise plots and latent comparison for sample {sample_idx}. Rel L1 (points): {rel_l1_point:.3f}%")