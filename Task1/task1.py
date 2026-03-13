import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import glob, os

# -------------------------
#  Parameters / Grid
# -------------------------
N = 64
K_list = [1, 4, 16]
K_values = [1, 4, 8, 16]  # complexity
r = 0.5
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

device = torch.device("cpu")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# -------------------------
#  Data generation
# -------------------------
def generate_sample(K, N=64, r=0.5, a=None):
    f = np.zeros((N, N))
    u = np.zeros((N, N))

    if a is None:
        a = np.random.randn(K, K)
    for i in range(1, K+1):
        for j in range(1, K+1):
            coeff = a[i-1, j-1]
            f += coeff * (i**2 + j**2)**r * np.sin(i*np.pi*X) * np.sin(j*np.pi*Y)
            u += coeff * (i**2 + j**2)**(r-1) * np.sin(i*np.pi*X) * np.sin(j*np.pi*Y)

    f = f * np.pi / (K**2)
    u = u / (K**2 * np.pi)
    return f, u, a

# Generation et visualisation (exemples)
num_samples = 3
fig, axes = plt.subplots(len(K_values), num_samples*2, figsize=(14, 12))

for row_idx, K in enumerate(K_values):
    for sample_idx in range(num_samples):
        f, u, a_ = generate_sample(K, N, r)

        # Source f
        im1 = axes[row_idx, 2*sample_idx].imshow(f, cmap='viridis', origin='lower')
        axes[row_idx, 2*sample_idx].set_title(f"f, K={K}, sample {sample_idx+1}")
        plt.colorbar(im1, ax=axes[row_idx, 2*sample_idx])

        # Solution u
        im2 = axes[row_idx, 2*sample_idx+1].imshow(u, cmap='viridis', origin='lower')
        axes[row_idx, 2*sample_idx+1].set_title(f"u, K={K}, sample {sample_idx+1}")
        plt.colorbar(im2, ax=axes[row_idx, 2*sample_idx+1])

plt.tight_layout()
plt.savefig("samples.png")
plt.close()


# -------------------------
#  MLP
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=128, n_hidden=4, out_dim=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_hidden-1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

# -------------------------
#  Utilities
# -------------------------
def to_tensor(x_np, dtype=torch.float32):
    return torch.tensor(x_np, dtype=dtype, device=device)

def compute_l2_rel_error(u_pred, u_true):
    num = torch.norm(u_pred - u_true)
    den = torch.norm(u_true)
    return (num / den).item()

# -------------------------
#  Laplacian (uses create_graph=True)
# -------------------------
def laplacian(u, x):
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # (Npts,2)
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_xx + u_yy  # (Npts,1)

# -------------------------
#  Parameter flat helpers
# -------------------------
def get_flat_params(model):
    params = []
    for p in model.parameters():
        params.append(p.detach().cpu().view(-1))
    return torch.cat(params)

def set_flat_params(model, flat_params):
    flat = flat_params.detach().cpu()
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = flat[idx: idx + numel].view(p.shape)
        p.data.copy_(chunk.to(p.device))
        idx += numel

def raw_random_directions(model, seed=None, scale=1.0):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    theta = get_flat_params(model)
    d = torch.randn_like(theta) * scale
    n = torch.randn_like(theta) * scale
    return d, n

# -------------------------
#  Loss evaluation (tensor for training, float for visualization)
# -------------------------
def compute_model_loss(model, mode,
                       xy_interior, u_interior, xy_boundary, f_interior,
                       u_mean, u_std, f_mean, f_std,
                       lambda_bd=1.0, device=torch.device("cpu"),
                       return_scalar=False):
    model.to(device)
    # we want xy_interior to require grad for PINN laplacian
    xy = xy_interior.to(device)
    u_in = u_interior.to(device)
    f_in = f_interior.to(device)

    u_pred_norm = model(xy)
    if mode == 'data':
        loss_tensor = torch.mean((u_pred_norm - u_in)**2)
    elif mode == 'pinn':
        # laplacian expects xy.requires_grad_(True)
        lap_norm = laplacian(u_pred_norm, xy)   # Δ u_norm
        lap_original = u_std.to(device) * lap_norm
        f_original = f_std.to(device) * f_in + f_mean.to(device)
        residual = - lap_original - f_original
        interior_loss = torch.mean(residual**2)
        u_b_norm = model(xy_boundary.to(device))
        u_b_target = (0.0 - u_mean.to(device)) / u_std.to(device)
        boundary_loss = torch.mean((u_b_norm - u_b_target)**2)
        loss_tensor = interior_loss + lambda_bd * boundary_loss
    else:
        raise ValueError("mode must be 'data' or 'pinn'")

    if return_scalar:
        return loss_tensor.item()
    return loss_tensor

# -------------------------
#  Loss landscape computation & plotting
# -------------------------
def compute_loss_landscape(model, mode,
                           xy_interior, u_interior, xy_boundary, f_interior,
                           u_mean, u_std, f_mean, f_std,
                           d_flat, n_flat,
                           alpha_range=(-1.0, 1.0), beta_range=(-1.0, 1.0),
                           n_alpha=21, n_beta=21,
                           out_prefix="landscape",
                           lambda_bd=1.0,
                           device=torch.device("cpu")):
    theta_star = get_flat_params(model)
    d = d_flat.clone()
    n = n_flat.clone()
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    betas = np.linspace(beta_range[0], beta_range[1], n_beta)
    loss_grid = np.zeros((n_alpha, n_beta), dtype=float)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            new_theta = theta_star + a * d + b * n
            set_flat_params(model, new_theta)

            # --- make sure xy_interior requires grad for PINN laplacian ---
            if mode == 'pinn':
                xy_local = xy_interior.clone().detach().to(device).requires_grad_(True)
            else:
                # no grad needed for data mode
                xy_local = xy_interior.clone().detach().to(device)

            # compute loss as a scalar float
            loss_val = compute_model_loss(
                model, mode,
                xy_local, u_interior, xy_boundary, f_interior,
                u_mean, u_std, f_mean, f_std,
                lambda_bd=lambda_bd, device=device,
                return_scalar=True
            )

            loss_grid[i, j] = float(loss_val)

    set_flat_params(model, theta_star)
    # contour
    eps = 1e-12
    loss_log = np.log10(loss_grid + eps) 
    fig, ax = plt.subplots(figsize=(6,5))
    CS = ax.contourf(
        betas, alphas, loss_log,
        levels=50,
        cmap='viridis'
    )
    ax.set_xlabel('beta')
    ax.set_ylabel('alpha')
    ax.set_title(f'Log10 Loss landscape ({mode})')
    cbar = fig.colorbar(CS, ax=ax)
    cbar.set_label('log10(loss + eps)')
    contour_path = f"{out_prefix}_contour_{mode}.png"
    plt.tight_layout()
    plt.savefig(contour_path, dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(7,6))
    ax3 = fig.add_subplot(111, projection='3d')
    B, A = np.meshgrid(betas, alphas)

    ax3.plot_surface(B, A, loss_log,cmap='viridis',edgecolor='none')
    ax3.set_xlabel('beta')
    ax3.set_ylabel('alpha')
    ax3.set_zlabel('log10(loss + eps)')
    ax3.set_title(f'Log10 Loss surface ({mode})')
    surface_path = f"{out_prefix}_surface_{mode}.png"
    plt.tight_layout()
    plt.savefig(surface_path, dpi=200)
    plt.close(fig)

    return loss_grid, contour_path, surface_path

# -------------------------
#  Curriculum helpers (resize a)
# -------------------------
def resize_a(a_in, new_size):
    size = a_in.shape[1]
    row_coords = (np.arange(new_size) + 0.5) * (size / new_size) - 0.5
    col_coords = (np.arange(new_size) + 0.5) * (size / new_size) - 0.5
    row_idx = np.round(row_coords).astype(int)
    col_idx = np.round(col_coords).astype(int)
    row_idx = np.clip(row_idx, 0, size - 1)
    col_idx = np.clip(col_idx, 0, size - 1)
    rr, cc = np.meshgrid(row_idx, col_idx, indexing='ij')
    out = a_in[rr, cc]
    return out

def curriculum_train(model, K_target, XY_grid, N, r,
                     curriculum_Ks=None,
                     adam_steps_schedule=None, lbfgs_maxiter_schedule=None,
                     lr=1e-3, lambda_schedule=None,
                     device=torch.device("cpu"),
                     seed=42,
                     a_np=None):
    np.random.seed(seed); torch.manual_seed(seed)
    if a_np is None:
        raise ValueError("a_np must be provided for curriculum training")
    if curriculum_Ks is None:
        ks = []; k = 1
        while k < K_target:
            ks.append(k); k *= 2
        if ks[-1] != K_target: ks.append(K_target)
        curriculum_Ks = ks
    else:
        curriculum_Ks = sorted(curriculum_Ks)
    n_stages = len(curriculum_Ks)
    if lambda_schedule is None:
        base = 0.1; lambda_schedule = []
        for i in range(n_stages):
            lambda_schedule.append(float(base * (10 ** (i * (2.0/(n_stages-1) if n_stages>1 else 0)))))
    history_by_stage = {}
    for idx, K_stage in enumerate(curriculum_Ks):
        print(f"\n--- Curriculum stage {idx+1}/{n_stages}: K = {K_stage}, lambda_bd = {lambda_schedule[idx]:.3g} ---")
        if K_stage == K_target:
            f_np, u_np, _ = generate_sample(K_stage, N=N, r=r, a=a_np)
        else:
            a_small = resize_a(a_np, K_stage)
            f_np, u_np, _ = generate_sample(K_stage, N=N, r=r, a=a_small)
        XY = np.stack([X.ravel(), Y.ravel()], axis=1)
        f_vec = f_np.ravel()[:, None]; u_vec = u_np.ravel()[:, None]
        mask_boundary = (np.isclose(X.ravel(), 0) | np.isclose(X.ravel(), 1) |
                         np.isclose(Y.ravel(), 0) | np.isclose(Y.ravel(), 1))
        mask_interior = ~mask_boundary
        xy_interior = to_tensor(XY[mask_interior]).float().to(device)
        f_interior = to_tensor(f_vec[mask_interior]).float().to(device)
        u_interior = to_tensor(u_vec[mask_interior]).float().to(device)
        xy_boundary = to_tensor(XY[mask_boundary]).float().to(device)
        u_mean = u_interior.mean(); u_std = u_interior.std()
        if u_std.item() == 0: u_std = torch.tensor(1.0, device=device)
        u_interior_norm = (u_interior - u_mean) / u_std
        f_mean = f_interior.mean(); f_std = f_interior.std()
        if f_std.item() == 0: f_std = torch.tensor(1.0, device=device)
        f_interior_norm = (f_interior - f_mean) / f_std

        model, hist = train_model(model, mode='pinn',
                                  xy_interior=xy_interior,
                                  u_interior=u_interior_norm,
                                  xy_boundary=xy_boundary,
                                  f_interior=f_interior_norm,
                                  u_mean=u_mean, u_std=u_std, f_mean=f_mean, f_std=f_std,
                                  adam_steps=adam_steps_schedule[idx] if adam_steps_schedule is not None else 500,
                                  lbfgs_maxiter=lbfgs_maxiter_schedule[idx] if lbfgs_maxiter_schedule is not None else 50,
                                  lr=lr,
                                  lambda_bd=lambda_schedule[idx])
        
        history_by_stage[K_stage] = {'hist': hist, 'u_mean': u_mean.cpu().item(), 'u_std': u_std.cpu().item(),
                                    'f_mean': f_mean.cpu().item(), 'f_std': f_std.cpu().item(),
                                    'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
    return model, hist

# -------------------------
#  Training wrapper 
# -------------------------
def train_model(model, mode, xy_interior, u_interior, xy_boundary, f_interior,
                u_mean, u_std, f_mean, f_std,
                adam_steps=2000, lbfgs_maxiter=200, lr=0.001, lambda_bd=1.0):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'loss': []}
    for step in range(adam_steps):
        optimizer.zero_grad()
        xy_interior_local = xy_interior.clone().detach().requires_grad_(True)
        loss_tensor = compute_model_loss(model, mode,
                                               xy_interior_local, u_interior, xy_boundary, f_interior,
                                               u_mean, u_std, f_mean, f_std,
                                               lambda_bd=lambda_bd, device=device,
                                               return_scalar=False)
        loss_tensor.backward()
        optimizer.step()
        if step % 50 == 0:
            history['loss'].append(loss_tensor.item())
    lbfgs_optimizer = optim.LBFGS(model.parameters(), max_iter=lbfgs_maxiter,
                                 tolerance_grad=1e-9, tolerance_change=1e-12)
    def closure():
        lbfgs_optimizer.zero_grad()
        xy_interior_local = xy_interior.clone().detach().requires_grad_(True)
        loss_tensor = compute_model_loss(model, mode,
                                               xy_interior_local, u_interior, xy_boundary, f_interior,
                                               u_mean, u_std, f_mean, f_std,
                                               lambda_bd=lambda_bd, device=device,
                                               return_scalar=False)
        loss_tensor.backward()
        return loss_tensor
    lbfgs_optimizer.step(closure)
    xy_interior_eval = xy_interior.clone().detach().requires_grad_(True)
    final_loss = compute_model_loss(model, mode,
                                   xy_interior_eval, u_interior, xy_boundary, f_interior,
                                   u_mean, u_std, f_mean, f_std,
                                   lambda_bd=lambda_bd, device=device,
                                   return_scalar=True)
    history['loss'].append(final_loss)
    return model, history

# -------------------------
#  Compose landscapes into single image
# -------------------------
def compose_pairs_grid(pairs_list, out_file="composed.png", thumb_size=(600,450), padding=8, bg_color=(255,255,255)):
    rows = len(pairs_list)
    if rows == 0:
        print("No images to compose for", out_file)
        return
    cols = 2
    w, h = thumb_size
    total_w = cols * w + (cols + 1) * padding
    total_h = rows * h + (rows + 1) * padding
    canvas = Image.new("RGB", (total_w, total_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (left_path, right_path) in enumerate(pairs_list):
        # ouvrir ou vignette grise
        if left_path and os.path.exists(left_path):
            im_l = Image.open(left_path).convert("RGB")
        else:
            im_l = Image.new("RGB", thumb_size, color=(230,230,230))
        if right_path and os.path.exists(right_path):
            im_r = Image.open(right_path).convert("RGB")
        else:
            im_r = Image.new("RGB", thumb_size, color=(230,230,230))

        im_l = im_l.resize(thumb_size, Image.LANCZOS)
        im_r = im_r.resize(thumb_size, Image.LANCZOS)

        y = padding + i * (h + padding)
        x_l = padding
        x_r = padding + w + padding
        canvas.paste(im_l, (x_l, y))
        canvas.paste(im_r, (x_r, y))

        label_l = f"Row {i+1} Left"
        label_r = f"Row {i+1} Right"
        if font:
            draw.text((x_l + 6, y + 6), label_l, fill=(0,0,0), font=font)
            draw.text((x_r + 6, y + 6), label_r, fill=(0,0,0), font=font)
        else:
            draw.text((x_l + 6, y + 6), label_l, fill=(0,0,0))
            draw.text((x_r + 6, y + 6), label_r, fill=(0,0,0))

    canvas.save(out_file, dpi=(150,150))
    print(f"Saved composed image: {out_file}")

# -------------------------
#  Main loop
# -------------------------
results = {}
contour_pairs = []
surface_pairs = []

for K in K_list:
    print(f"\n=== K = {K} ===")

    f_np, u_np, a_np = generate_sample(K, N=N, r=r)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    f_vec = f_np.ravel()[:, None]; u_vec = u_np.ravel()[:, None]

    mask_boundary = (np.isclose(X.ravel(), 0) | np.isclose(X.ravel(), 1) |
                     np.isclose(Y.ravel(), 0) | np.isclose(Y.ravel(), 1))
    mask_interior = ~mask_boundary

    xy_interior = to_tensor(XY[mask_interior]).float().to(device)
    f_interior = to_tensor(f_vec[mask_interior]).float().to(device)
    u_interior = to_tensor(u_vec[mask_interior]).float().to(device)
    xy_boundary = to_tensor(XY[mask_boundary]).float().to(device)

    u_mean = u_interior.mean(); u_std = u_interior.std()
    if u_std.item() == 0: u_std = torch.tensor(1.0, device=device)
    u_interior_norm = (u_interior - u_mean) / u_std
    f_mean = f_interior.mean(); f_std = f_interior.std()
    if f_std.item() == 0: f_std = torch.tensor(1.0, device=device)
    f_interior_norm = (f_interior - f_mean) / f_std

    model_data = MLP(in_dim=2, hidden=64, n_hidden=4, out_dim=1).to(device)
    model_pinn = MLP(in_dim=2, hidden=64, n_hidden=4, out_dim=1).to(device)

    print("Training Data-Driven model...")
    model_data, hist_data = train_model(model_data, mode='data',
                                       xy_interior=xy_interior,
                                       u_interior=u_interior_norm,
                                       xy_boundary=xy_boundary,
                                       f_interior=f_interior_norm,
                                       u_mean=u_mean, u_std=u_std, f_mean=f_mean, f_std=f_std,
                                       adam_steps=2000, lbfgs_maxiter=200, lr=0.001)
    
    if K < 10:
        print("Training PINN model...")
        model_pinn, hist_pinn = train_model(model_pinn, mode='pinn',
                                            xy_interior=xy_interior,
                                            u_interior=u_interior_norm,
                                            xy_boundary=xy_boundary,
                                            f_interior=f_interior_norm,
                                            u_mean=u_mean, u_std=u_std, f_mean=f_mean, f_std=f_std,
                                            adam_steps=2000, lbfgs_maxiter=200, lr=0.001, lambda_bd=1.0)
    else:
        print("Training PINN model with curriculum regularization...")
        curriculum_Ks = np.linspace(max(1, K//4), K, 4).astype(int)
        curriculum_Ks = sorted(list(dict.fromkeys(curriculum_Ks)))
        lambda_schedule = np.ones(len(curriculum_Ks))
        adam_steps_schedule = [500, 500, 1000, 2000][:len(curriculum_Ks)]
        lbgfs_steps_schedule = [50, 50, 100, 200][:len(curriculum_Ks)]
        model_pinn, hist_pinn = curriculum_train(model_pinn, K_target=K,
                                                 XY_grid=XY, N=N, r=r,
                                                 curriculum_Ks=curriculum_Ks,
                                                 adam_steps_schedule=adam_steps_schedule,
                                                 lbfgs_maxiter_schedule=lbgfs_steps_schedule,
                                                 lr=0.001,
                                                 lambda_schedule=lambda_schedule,
                                                 device=device,
                                                 seed=seed,
                                                 a_np=resize_a(a_np, K))
        
    with torch.no_grad():
        xy_all_t = to_tensor(XY).float().to(device)
        u_pred_data_norm = model_data(xy_all_t).cpu()
        u_pred_pinn_norm = model_pinn(xy_all_t).cpu()
        u_pred_data = u_pred_data_norm * u_std.cpu() + u_mean.cpu()
        u_pred_pinn = u_pred_pinn_norm * u_std.cpu() + u_mean.cpu()

    u_true_all = to_tensor(u_vec).float()
    u_pred_data_interior = u_pred_data[mask_interior]
    u_pred_pinn_interior = u_pred_pinn[mask_interior]
    u_true_interior = u_interior.cpu()

    l2_data = compute_l2_rel_error(u_pred_data_interior, u_true_interior)
    l2_pinn = compute_l2_rel_error(u_pred_pinn_interior, u_true_interior)

    print(f"K={K}  L2 rel error Data: {l2_data:.4e}  PINN: {l2_pinn:.4e}")

    results[K] = {'f': f_np, 'u': u_np, 'u_pred_data': u_pred_data.numpy().reshape(N, N),
                  'u_pred_pinn': u_pred_pinn.numpy().reshape(N, N),
                  'hist_data': hist_data, 'hist_pinn': hist_pinn,
                  'l2_data': l2_data, 'l2_pinn': l2_pinn}
    # Loss landscapes (coarse grid to limit cost)

    try:
        # PINN landscape
        d_flat_p, n_flat_p = raw_random_directions(model_pinn, seed=seed, scale=0.5)
        loss_grid_pinn, contour_pinn, surface_pinn = compute_loss_landscape(
            model_pinn, 'pinn',
            xy_interior=xy_interior, u_interior=u_interior_norm,
            xy_boundary=xy_boundary, f_interior=f_interior_norm,
            u_mean=u_mean, u_std=u_std, f_mean=f_mean, f_std=f_std,
            d_flat=d_flat_p, n_flat=n_flat_p,
            alpha_range=(-1.0, 1.0), beta_range=(-1.0, 1.0),
            n_alpha=21, n_beta=21,
            out_prefix=f"landscape_K{K}_pinn",
            lambda_bd=1.0,
            device=device
        )

        # Data-driven landscape
        d_flat_d, n_flat_d = raw_random_directions(model_data, seed=seed+1, scale=0.5)
        loss_grid_data, contour_data, surface_data = compute_loss_landscape(
            model_data, 'data',
            xy_interior=xy_interior, u_interior=u_interior_norm,
            xy_boundary=xy_boundary, f_interior=f_interior_norm,
            u_mean=u_mean, u_std=u_std, f_mean=f_mean, f_std=f_std,
            d_flat=d_flat_d, n_flat=n_flat_d,
            alpha_range=(-1.0, 1.0), beta_range=(-1.0, 1.0),
            n_alpha=21, n_beta=21,
            out_prefix=f"landscape_K{K}_data",
            lambda_bd=0.0,
            device=device
        )

        contour_pairs.append((contour_pinn, contour_data))
        surface_pairs.append((surface_pinn, surface_data))

        print("Saved landscapes:", contour_pinn, surface_pinn, contour_data, surface_data)
    except Exception as e:
        print("Warning: landscape computation failed or was skipped:", e)

compose_pairs_grid(contour_pairs, out_file="all_contours.png", thumb_size=(600,450), padding=8)
compose_pairs_grid(surface_pairs, out_file="all_surfaces.png", thumb_size=(600,450), padding=8)

# Save result grids and loss curves
nK = len(K_list)
fig, axes = plt.subplots(nK, 4, figsize=(16, 4 * nK))

for i, K in enumerate(K_list):
    ax = axes[i, 0] if nK > 1 else axes[0]
    im = ax.imshow(results[K]['f'], cmap='viridis', origin='lower'); ax.set_title(f"f (K={K})"); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[i, 1] if nK > 1 else axes[1]
    im = ax.imshow(results[K]['u'], cmap='viridis', origin='lower'); ax.set_title("u exact"); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[i, 2] if nK > 1 else axes[2]
    im = ax.imshow(results[K]['u_pred_data'], cmap='viridis', origin='lower'); ax.set_title(f"Pred Data (L2={results[K]['l2_data']:.2e})"); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[i, 3] if nK > 1 else axes[3]
    im = ax.imshow(results[K]['u_pred_pinn'], cmap='viridis', origin='lower'); ax.set_title(f"Pred PINN (L2={results[K]['l2_pinn']:.2e})"); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("f / u_exact / pred_data / pred_pinn for each K", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig("results_grid.png", dpi=200); plt.close()

plt.figure(figsize=(8,6))
for K in K_list:
    hist_d = results[K]['hist_data']['loss']; hist_p = results[K]['hist_pinn']['loss']
    x_d = 50*np.arange(len(hist_d)); x_p = 50*np.arange(len(hist_p))
    plt.plot(x_d, hist_d, marker='o', linestyle='-', label=f"Data K={K}")
    plt.plot(x_p, hist_p, marker='x', linestyle='--', label=f"PINN K={K}")
plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)')
plt.title('Comparative loss curves for all K'); plt.legend(); plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout(); plt.savefig("loss_comparison.png", dpi=200); plt.close()
