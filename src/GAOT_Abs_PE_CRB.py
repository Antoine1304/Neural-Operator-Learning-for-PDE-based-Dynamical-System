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

import model.layers.attn as attn
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

# Parameters Strategy II
N_latents = 1024         # Latent Tokens Number
S_seeds = 256            # Number of Perceiver Seeds
resample_seed_base = 42  # Seed to resample tokens each epoch
use_dataset_bbox = True 

# Training yperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 0.001

# ==========================================
# 2. Dataset & Dataloaders
# ==========================================
class ElasticityDataset(Dataset):
    def __init__(self, inputs, targets, coords):
        self.inputs = inputs.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.coords = coords.astype(np.float32) if coords is not None else None
        
        # Normalization Min-Max
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
    print(f"Loading {path}...")
    with h5py.File(path, "r") as f:
        u = np.array(f["u"]).squeeze(axis=1) # [B, S, u_vars]
        c = np.array(f["c"]).squeeze(axis=1) # [B, S, c_vars]
        x = np.array(f["x"]).squeeze(axis=1) # [B, S, 2]
    
    N = c.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(N * (1 - test_frac))
    
    train_ds = ElasticityDataset(c[idx[:split]], u[idx[:split]], x[idx[:split]])
    test_ds  = ElasticityDataset(c[idx[split:]], u[idx[split:]], x[idx[split:]])
    return train_ds, test_ds

train_set, test_set = create_datasets("Elasticity.nc", test_frac=0.2, seed=42)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

in_ch = train_set.inputs.shape[2]
out_ch = train_set.targets.shape[2]
print(f"Data loaded. Input ch: {in_ch}, Output ch: {out_ch}")

# ==========================================
# 3. Utilities
# ==========================================
def sample_latent_tokens(N, coords_all=None, seed=0, device='cpu', use_bbox=True):
    rng = np.random.RandomState(seed)
    mins, maxs = np.array([0.,0.]), np.array([1.,1.])
    if use_bbox and coords_all is not None:
        # On utilise une petite partie pour estimer la bbox pour aller plus vite
        coords_sub = coords_all[:100] if len(coords_all) > 100 else coords_all
        if isinstance(coords_sub, torch.Tensor): coords_sub = coords_sub.cpu().numpy()
        flat = coords_sub.reshape(-1, 2)
        mins, maxs = flat.min(axis=0), flat.max(axis=0)
    samples = rng.rand(N, 2) * (maxs - mins)[None, :] + mins[None, :]
    return torch.from_numpy(samples.astype(np.float32)).to(device)

def list_of_arrays_to_csr_torch(list_of_arrays, device):
    """Convertit une liste de listes numpy en tenseurs CSR Torch"""
    lens = np.array([len(arr) for arr in list_of_arrays], dtype=np.int64)
    row_splits = np.concatenate(([0], np.cumsum(lens)))
    if row_splits[-1] == 0:
        indices = np.empty(0, dtype=np.int64)
    else:
        indices = np.concatenate(list_of_arrays).astype(np.int64)
    return {
        "neighbors_index": torch.from_numpy(indices).long().to(device),
        "neighbors_row_splits": torch.from_numpy(row_splits).long().to(device)
    }

class AbsolutePE(torch.nn.Module):
    def __init__(self, coord_dim, out_dim, hidden=64, n_layers=2):
        super().__init__()
        layers = []
        in_dim = coord_dim
        for i in range(n_layers-1):
            layers.append(torch.nn.Linear(in_dim, hidden))
            layers.append(torch.nn.GELU())
            in_dim = hidden
        layers.append(torch.nn.Linear(in_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)
        self.norm = torch.nn.LayerNorm(out_dim)

    def forward(self, coords):
        return self.norm(self.mlp(coords))

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
    precompute_edges=True
)

attn_conf = AttentionConfig(
    use_continuous_relative_bias=True # CRB option activated
)

trans_conf = TransformerConfig(
    patch_size=2,
    hidden_size=128,
    use_perceiver_seeds=True, # Perceiver option activated
    num_seeds=S_seeds,
    attn_config=attn_conf
)

model_args = ModelArgs(magno=magno_conf, transformer=trans_conf)
config = GAOTConfig(latent_tokens_size=[32, 32], args=model_args)

model = GAOT(in_ch, out_ch, config=config).to(device)
print(f"Model initialized: GAOT + CRB + Perceiver. Params: {sum(p.numel() for p in model.parameters())}")

# Module Positional Encoding Externe
pe = AbsolutePE(coord_dim=2, out_dim=32, hidden=64, n_layers=3).to(device)

optimizer = AdamW(list(model.parameters()) + list(pe.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, eta_min=1e-6)

# ==========================================
# 5. Training Loop
# ==========================================
print("\n--- Starting Training ---")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # --- A. Tokens Update (Strategy II) ---
    seed_epoch = resample_seed_base + epoch
    latent_tokens_coord = sample_latent_tokens(N_latents, coords_all=train_set.coords, 
                                             seed=seed_epoch, device=device, use_bbox=use_dataset_bbox)
    
    # --- B. Construction KDTree Tokens ---
    tokens_np = latent_tokens_coord.detach().cpu().numpy()
    tree_tokens = cKDTree(tokens_np)
    
    # Dynamic Radius
    d_k, _ = tree_tokens.query(tokens_np, k=9)
    radii = 1.0 * d_k[:, -1]
    max_r = np.max(radii)

    # --- C. Training Loop ---
    model.train()
    pe.train()
    train_loss = 0.0
    
    for batch in train_loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        coords = batch["coords"].to(device)
        
        # 1. Build Neighbors (Vectorized KDTree)
        coords_np_batch = coords.cpu().numpy()
        B_curr = coords.shape[0]
        encoder_nbrs, decoder_nbrs = [], []
        
        for b in range(B_curr):
            pts_np = coords_np_batch[b]
            tree_pts = cKDTree(pts_np)
            
            # Encoder: Tokens -> Points (exact radius per token)
            token_to_points_idxs = tree_pts.query_ball_point(tokens_np, radii)
            
            # Decoder: Points -> Tokens (approx with max_radius)
            point_to_tokens_idxs = tree_tokens.query_ball_point(pts_np, max_r)
            
            # Conversion en numpy array pour le CSR builder
            enc_list = [np.array(x, dtype=np.int64) for x in token_to_points_idxs]
            dec_list = [np.array(x, dtype=np.int64) for x in point_to_tokens_idxs]
            
            encoder_nbrs.append([list_of_arrays_to_csr_torch(enc_list, device)])
            decoder_nbrs.append([list_of_arrays_to_csr_torch(dec_list, device)])

        optimizer.zero_grad()

        # 2. Pipeline (Encode -> PE -> Process -> Decode)
        rndata = model.encode(coords, inp, latent_tokens_coord, encoder_nbrs)
        rndata = rndata + pe(latent_tokens_coord) # We add Absolute PE
        rndata = model.process(rndata, condition=None, positions=latent_tokens_coord)
        output_pred = model.decode(latent_tokens_coord, rndata, coords, decoder_nbrs)

        loss = torch.mean(torch.abs(output_pred - tgt)) / (torch.mean(torch.abs(tgt)) + 1e-8)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()

    # --- D. Validation / Test Loop  ---
    model.eval()
    pe.eval()
    test_rel_l1 = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            coords = batch["coords"].to(device)
            
            # Re-build neighbors FAST for test set
            coords_np_batch = coords.cpu().numpy()
            B_curr = coords.shape[0]
            encoder_nbrs, decoder_nbrs = [], []
            
            for b in range(B_curr):
                pts_np = coords_np_batch[b]
                tree_pts = cKDTree(pts_np)
                
                token_to_points_idxs = tree_pts.query_ball_point(tokens_np, radii)
                point_to_tokens_idxs = tree_tokens.query_ball_point(pts_np, max_r)
                
                enc_list = [np.array(x, dtype=np.int64) for x in token_to_points_idxs]
                dec_list = [np.array(x, dtype=np.int64) for x in point_to_tokens_idxs]
                
                encoder_nbrs.append([list_of_arrays_to_csr_torch(enc_list, device)])
                decoder_nbrs.append([list_of_arrays_to_csr_torch(dec_list, device)])

            # Forward pass
            rndata = model.encode(coords, inp, latent_tokens_coord, encoder_nbrs)
            rndata = rndata + pe(latent_tokens_coord)
            rndata = model.process(rndata, condition=None, positions=latent_tokens_coord)
            output_pred = model.decode(latent_tokens_coord, rndata, coords, decoder_nbrs)
            
            # Metric
            rel_l1 = torch.mean(torch.abs(output_pred - tgt)) / (torch.mean(torch.abs(tgt)) + 1e-8)
            test_rel_l1 += rel_l1.item()

    test_rel_l1 /= len(test_loader)
    
    epoch_duration = time.time() - epoch_start
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Rel L1: {test_rel_l1*100:.2f}% | Time: {epoch_duration:.1f}s")

end_time = time.time()
print(f"Total training time: {end_time-start_time:.2f}s")

# ==========================================
# 6. Evaluation & Plotting
# ==========================================
print("Training complete.")
print("Evaluating...")
model.eval()
pe.eval()
test_rel_l1 = 0.0
with torch.no_grad():
    batch = next(iter(test_loader))
    inp, tgt, coords = batch["input"].to(device), batch["target"].to(device), batch["coords"].to(device)
    
    # Re-build neighbors for one batch
    coords_np = coords.cpu().numpy()
    enc_nbrs, dec_nbrs = [], []
    max_r = np.max(radii)
    
    for b in range(coords.shape[0]):
        pts = coords_np[b]
        tree_pts = cKDTree(pts)
        enc_l = [np.array(x) for x in tree_pts.query_ball_point(tokens_np, radii)]
        dec_l = [np.array(x) for x in tree_tokens.query_ball_point(pts, max_r)]
        enc_nbrs.append([list_of_arrays_to_csr_torch(enc_l, device)])
        dec_nbrs.append([list_of_arrays_to_csr_torch(dec_l, device)])

    rndata = model.encode(coords, inp, latent_tokens_coord, enc_nbrs)
    rndata = rndata + pe(latent_tokens_coord)
    rndata = model.process(rndata, positions=latent_tokens_coord)
    pred = model.decode(latent_tokens_coord, rndata, coords, dec_nbrs)

    # choose sample to display (0..B-1)
    sample_idx = 4
    B = pred.shape[0]
    assert 0 <= sample_idx < B

    coords_np = coords.cpu().numpy() if coords.dim() == 2 else coords[sample_idx].cpu().numpy()
    inp_np = inp[sample_idx].cpu().numpy()
    tgt_np = tgt[sample_idx].cpu().numpy()
    pred_np = pred[sample_idx].cpu().numpy()

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
    plt.savefig("result_bonus.png", dpi=200)
    plt.close(fig)
    
    print(f"Saved pointwise plots and latent comparison for sample {sample_idx}. Rel L1 (points): {rel_l1_point:.3f}%")

    # compute token->points for this sample (reuse tokens_np and radii from epoch)
    if coords.dim() == 3:
        coords_np_sample = coords[sample_idx].cpu().numpy()  # [S,2]
    else:
        coords_np_sample = coords.cpu().numpy()  # fx

    tree_pts = cKDTree(coords_np_sample)
    token_to_points_sample = [np.array(x, dtype=np.int32) for x in tree_pts.query_ball_point(tokens_np, radii)]
    # coverage count per point
    S = coords_np_sample.shape[0]
    coverage_counts = np.zeros(S, dtype=np.int32)
    for t_idx, pts in enumerate(token_to_points_sample):
        coverage_counts[pts] += 1

    # fraction covered
    frac_covered = np.count_nonzero(coverage_counts > 0) / float(S)
    print(f"Coverage for sample {sample_idx}: {frac_covered*100:.2f}% (mean tokens per point: {coverage_counts.mean():.2f})")

    # scatter plot coverage
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    sc = ax.scatter(coords_np_sample[:,0], coords_np_sample[:,1], c=coverage_counts, cmap='magma', s=18)
    ax.set_title(f"Coverage map sample {sample_idx} (frac covered {frac_covered*100:.1f}%)")
    ax.axis('equal'); ax.axis('off')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='# tokens covering point')
    plt.tight_layout()
    plt.savefig("tokens_per_point_bonus.png", dpi=200, bbox_inches='tight')
    plt.close(fig)