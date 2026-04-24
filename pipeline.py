"""
Normalizing Kalman Filter — Full Pipeline
==========================================
Paper: de Bézenac et al., NeurIPS 2020
Data:  yfinance multi-level CSV  (Close / High / Low / Open / Volume  x 20 stocks)

Layout
------
  y  (observations)  : Close prices          shape (B, T, N=20)
  x  (covariates)    : High, Low, Open, Vol
                       + calendar features   shape (B, T, k)

When you add commodities / news, just concatenate them onto x.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH       = "stock.csv"   # change to your path
WINDOW         = 20            # shorter window → more training samples
HORIZON        = 5             # steps ahead to forecast
TRAIN_RATIO    = 0.7
VAL_RATIO      = 0.15          # remaining 0.15 → test
BATCH_SIZE     = 32
N_EPOCHS       = 50            # early stopping will kick in before this
LR             = 3e-4          # slower start reduces early overfitting
STATE_DIM      = 2             # level + trend per stock
FLOW_LAYERS    = 2             # fewer flow layers → less capacity
HIDDEN         = 32            # smaller coupling nets + LSTM
LSTM_LAYERS    = 1             # single LSTM layer
N_SAMPLES      = 200           # Monte Carlo forecast samples
PCA_COMPONENTS = 20            # reduce covariates via PCA
EARLY_STOP_PATIENCE = 7        # stop if val NLL doesn't improve
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Loading & Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str):
    """
    Parse yfinance multi-level CSV.
    Returns
    -------
    close  : pd.DataFrame  (T, 20)   — stock close prices
    covars : pd.DataFrame  (T, k)    — High/Low/Open/Vol per stock + calendar
    dates  : pd.DatetimeIndex
    """
    df = pd.read_csv(path, header=[0, 1], index_col=0, skiprows=[2])
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    tickers = df["Close"].columns.tolist()
    N = len(tickers)   # 20

    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    open_  = df["Open"].astype(float)
    volume = df["Volume"].astype(float)

    # ── Calendar features (same for all stocks, added once) ───────────────
    cal = pd.DataFrame(index=df.index)
    cal["day_of_week"]     = df.index.dayofweek / 4.0        # 0-1
    cal["month"]           = (df.index.month - 1) / 11.0     # 0-1
    cal["is_quarter_end"]  = df.index.is_quarter_end.astype(float)

    # ── Derived per-stock features ────────────────────────────────────────
    # Range = (High - Low) / Close  — normalised intra-day range
    intra_range = ((high - low) / close.replace(0, np.nan)).fillna(0)
    intra_range.columns = [f"range_{t}" for t in tickers]

    # Log volume (more Gaussian, avoids scale issues)
    log_vol = np.log1p(volume)
    log_vol.columns = [f"logvol_{t}" for t in tickers]

    # Stack all covariates  (T, k)
    covars = pd.concat([high, low, open_, intra_range, log_vol, cal], axis=1)
    covars.columns = (
        [f"high_{t}"  for t in tickers] +
        [f"low_{t}"   for t in tickers] +
        [f"open_{t}"  for t in tickers] +
        [f"range_{t}" for t in tickers] +
        [f"logvol_{t}" for t in tickers] +
        list(cal.columns)
    )

    # Drop any rows where close has NaN (missing trading days already dropped)
    mask = close.notna().all(axis=1) & covars.notna().all(axis=1)
    close  = close[mask]
    covars = covars[mask]

    print(f"Loaded  {len(close)} trading days  |  {N} stocks  |  {covars.shape[1]} covariates")
    print(f"Date range: {close.index[0].date()} → {close.index[-1].date()}")
    return close, covars, close.index


def split_and_scale(close, covars, train_r=0.7, val_r=0.15, pca_components=20):
    """
    Chronological train / val / test split.
    Steps:
      1. StandardScaler on covariates (fit on train only)
      2. PCA to reduce covariate dimensionality (fit on train only)
      3. StandardScaler on close prices (fit on train only)
    Returns numpy arrays and scalers needed to invert predictions.
    """
    T = len(close)
    t1 = int(T * train_r)
    t2 = int(T * (train_r + val_r))

    y_raw = close.values.astype(np.float32)   # (T, N)
    x_raw = covars.values.astype(np.float32)  # (T, k)

    # ── Scale covariates before PCA ───────────────────────────────
    x_scaler = StandardScaler().fit(x_raw[:t1])
    x_scaled = x_scaler.transform(x_raw)

    # ── PCA on covariates (fit on train only) ─────────────────────
    pca    = PCA(n_components=pca_components, random_state=42).fit(x_scaled[:t1])
    x      = pca.transform(x_scaled).astype(np.float32)
    var_ex = pca.explained_variance_ratio_.sum()
    print(f"PCA: {x_raw.shape[1]} → {pca_components} components  "
          f"({var_ex:.1%} variance explained)")

    # ── Scale close prices ────────────────────────────────────────
    y_scaler = StandardScaler().fit(y_raw[:t1])
    y        = y_scaler.transform(y_raw).astype(np.float32)

    splits = {
        "train": (y[:t1],   x[:t1]),
        "val":   (y[t1:t2], x[t1:t2]),
        "test":  (y[t2:],   x[t2:]),
    }
    print(f"Split sizes — train: {t1}  val: {t2-t1}  test: {T-t2}")
    return splits, y_scaler, x_scaler, pca


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class StockWindowDataset(Dataset):
    """
    Sliding window dataset.
    Each sample: (x_window, y_window, y_target)
      x_window : (WINDOW, k)        covariates for the input window
      y_window : (WINDOW, N)        close prices for the input window
      y_target : (HORIZON, N)       close prices to predict
    """
    def __init__(self, y: np.ndarray, x: np.ndarray,
                 window: int, horizon: int):
        self.y = torch.from_numpy(y)
        self.x = torch.from_numpy(x)
        self.window  = window
        self.horizon = horizon

    def __len__(self):
        return len(self.y) - self.window - self.horizon + 1

    def __getitem__(self, idx):
        s = idx
        e = idx + self.window
        return (
            self.x[s:e],                          # (W, k)
            self.y[s:e],                          # (W, N)
            self.y[e: e + self.horizon],          # (H, N)
        )


def make_loaders(splits, window, horizon, batch_size):
    loaders = {}
    for split, (y, x) in splits.items():
        ds = StockWindowDataset(y, x, window, horizon)
        shuffle = (split == "train")
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=shuffle, drop_last=False)
        print(f"{split:5s} loader: {len(ds)} windows, {len(loaders[split])} batches")
    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Components
# ─────────────────────────────────────────────────────────────────────────────

class CouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, (dim - half) * 2),
        )
        self.split = half

    def forward(self, z):
        z1, z2 = z[..., :self.split], z[..., self.split:]
        s, t   = self.net(z1).chunk(2, dim=-1)
        s      = torch.tanh(s)
        return torch.cat([z1, z2 * s.exp() + t], dim=-1), s.sum(-1)

    def inverse(self, y):
        y1, y2 = y[..., :self.split], y[..., self.split:]
        s, t   = self.net(y1).chunk(2, dim=-1)
        s      = torch.tanh(s)
        return torch.cat([y1, (y2 - t) * (-s).exp()], dim=-1), -s.sum(-1)


class RealNVP(nn.Module):
    def __init__(self, N: int, n_layers: int = 4, hidden: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([CouplingLayer(N, hidden) for _ in range(n_layers)])
        idx = torch.arange(N)
        self.register_buffer("perm",     torch.flip(idx, [0]))
        self.register_buffer("inv_perm", torch.argsort(torch.flip(idx, [0])))

    def forward(self, z):
        """z → y"""
        log_det = 0.
        for i, layer in enumerate(self.layers):
            z, ld = layer(z)
            log_det = log_det + ld
            if i < len(self.layers) - 1:
                z = z[..., self.perm]
        return z, log_det

    def inverse(self, y):
        """y → z"""
        log_det = 0.
        for i, layer in enumerate(reversed(self.layers)):
            if i > 0:
                y = y[..., self.inv_perm]
            y, ld = layer.inverse(y)
            log_det = log_det + ld
        return y, log_det


def kalman_step(mu, P, F, Q, A, obs_var, z_obs=None):
    """
    Batched Kalman predict + optional update.
    mu       (B, N, d)
    P        (B, N, d, d)
    F        (B, N, d, d)
    Q        (B, N, d, d)
    A        (B, N, d)       emission vector
    obs_var  (B, N)          observation noise variance
    z_obs    (B, N) or None
    """
    # Predict
    mu_p = (F @ mu.unsqueeze(-1)).squeeze(-1)
    P_p  = F @ P @ F.transpose(-1, -2) + Q

    if z_obs is None:
        return mu_p, P_p, torch.zeros(mu.shape[:2], device=mu.device)

    # Update
    A_   = A.unsqueeze(-1)                              # (B, N, d, 1)
    S    = (A_.transpose(-1,-2) @ P_p @ A_).squeeze(-1,-2) + obs_var  # (B, N)
    K    = (P_p @ A_) / S.unsqueeze(-1).unsqueeze(-1)  # (B, N, d, 1)
    z_hat = (A_.transpose(-1,-2) @ mu_p.unsqueeze(-1)).squeeze(-1,-2)
    innov = z_obs - z_hat
    mu_u  = mu_p + K.squeeze(-1) * innov.unsqueeze(-1)
    P_u   = (torch.eye(mu.shape[-1], device=mu.device) - K @ A_.transpose(-1,-2)) @ P_p
    S_    = S.squeeze(-1)
    log_lik = -0.5 * (innov**2 / S_ + S_.log() + np.log(2 * np.pi))
    return mu_u, P_u, log_lik


class ParameterNet(nn.Module):
    """
    Projects covariates down first, then runs LSTM → SSM parameters.
    covariate_dim (e.g. 20 after PCA) → proj_dim → LSTM → per-stock params.
    """
    def __init__(self, covariate_dim, N, state_dim=2, hidden=64, n_layers=1):
        super().__init__()
        self.N, self.d = N, state_dim
        proj_dim = min(hidden, max(covariate_dim // 2, 8))
        self.proj = nn.Sequential(
            nn.Linear(covariate_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(proj_dim, hidden, n_layers,
                            batch_first=True,
                            dropout=0.2 if n_layers > 1 else 0)
        self.head = nn.Linear(hidden, N * (1 + state_dim + 1))

    def forward(self, x, h=None):
        x      = self.proj(x)                  # (B, T, proj_dim)
        out, h = self.lstm(x, h)
        p      = self.head(out).view(*out.shape[:2], self.N, 2 + self.d)
        obs_var  = torch.exp(p[..., 0])        # Γ_t  (B, T, N)
        proc_var = torch.exp(p[..., 1])        # σ_t  (B, T, N)
        emit     = p[..., 2:]                  # A_t  (B, T, N, d)
        return obs_var, proc_var, emit, h


class NKF(nn.Module):
    """
    Normalizing Kalman Filter  (de Bézenac et al., NeurIPS 2020).

    y  (B, T, N)  — stock close prices   → observations for the KF
    x  (B, T, k)  — OHLV + calendar …   → covariates into LSTM
    """
    def __init__(self, N, covariate_dim, state_dim=2,
                 flow_layers=4, hidden=64, lstm_layers=2):
        super().__init__()
        self.N, self.d = N, state_dim
        self.flow      = RealNVP(N, flow_layers, hidden)
        self.param_net = ParameterNet(covariate_dim, N, state_dim, hidden, lstm_layers)

        F_base = torch.eye(state_dim)
        if state_dim >= 2:
            F_base[0, 1] = 1.0    # level += trend
        self.register_buffer("F_base", F_base)

    def _build_matrices(self, proc_var, emit):
        B, T, N = proc_var.shape
        d = self.d
        F = self.F_base.view(1,1,1,d,d).expand(B,T,N,d,d)
        Q = torch.diag_embed(proc_var.unsqueeze(-1).expand(B,T,N,d))
        return F, Q, emit   # A = emit  (B,T,N,d)

    def log_likelihood(self, y, x, mask=None):
        """Exact log-likelihood (Prop. 3). Used as training objective."""
        B, T, N = y.shape
        device  = y.device

        # f^{-1}(y) → pseudo-observations z
        z, log_det_inv = self.flow.inverse(y)           # z: (B,T,N), log_det: (B,T)

        obs_var, proc_var, emit, _ = self.param_net(x)
        F, Q, A = self._build_matrices(proc_var, emit)

        mu = torch.zeros(B, N, self.d, device=device)
        P  = torch.eye(self.d, device=device).view(1,1,self.d,self.d).expand(B,N,-1,-1).clone()

        total_kf_ll = torch.zeros(B, device=device)
        for t in range(T):
            obs_t = z[:,t,:] if mask is None else z[:,t,:] * mask[:,t,:]
            mu, P, step_ll = kalman_step(mu, P, F[:,t], Q[:,t], A[:,t], obs_var[:,t], obs_t)
            if mask is not None:
                step_ll = step_ll * mask[:,t,:]
            total_kf_ll += step_ll.sum(-1)

        total_ll = total_kf_ll + log_det_inv.sum(1)
        return total_ll.mean()

    @torch.no_grad()
    def forecast(self, y_hist, x_hist, x_future, n_samples=100):
        """
        Probabilistic forecast.
        y_hist   (B, T_hist, N)
        x_hist   (B, T_hist, k)
        x_future (B, T_fut,  k)
        Returns  (B, T_fut, N, n_samples)
        """
        B, T_hist, N = y_hist.shape
        T_fut  = x_future.shape[1]
        device = y_hist.device

        # Filter through history
        z_hist, _ = self.flow.inverse(y_hist)
        obs_var, proc_var, emit, h = self.param_net(x_hist)
        F, Q, A = self._build_matrices(proc_var, emit)

        mu = torch.zeros(B, N, self.d, device=device)
        P  = torch.eye(self.d, device=device).view(1,1,self.d,self.d).expand(B,N,-1,-1).clone()
        for t in range(T_hist):
            mu, P, _ = kalman_step(mu, P, F[:,t], Q[:,t], A[:,t], obs_var[:,t], z_hist[:,t])

        # Propagate latent state with samples
        obs_var_f, proc_var_f, emit_f, _ = self.param_net(x_future, h)
        F_f, Q_f, A_f = self._build_matrices(proc_var_f, emit_f)

        # Expand for samples: (B*S, N, d)
        S  = n_samples
        mu_s = mu.unsqueeze(1).expand(B, S, N, self.d).reshape(B*S, N, self.d)

        samples = []
        for t in range(T_fut):
            def expand(x_):
                return x_.unsqueeze(1).expand(B, S, *x_.shape[1:]).reshape(B*S, *x_.shape[1:])

            F_t     = expand(F_f[:,t])
            Q_t     = expand(Q_f[:,t])
            A_t     = expand(A_f[:,t])
            obs_v_t = expand(obs_var_f[:,t])

            # Sample process noise: l_t = F l_{t-1} + ε
            eps   = torch.randn_like(mu_s)
            L     = torch.linalg.cholesky(Q_t + 1e-6 * torch.eye(self.d, device=device))
            mu_s  = (F_t @ mu_s.unsqueeze(-1)).squeeze(-1) + (L @ eps.unsqueeze(-1)).squeeze(-1)

            # Emit z_t = A^T l_t + η
            z_t  = (A_t.unsqueeze(-2) @ mu_s.unsqueeze(-1)).squeeze(-1,-2)
            z_t  = z_t + torch.randn_like(z_t) * obs_v_t.sqrt()

            # Map z_t → y_t through flow
            y_t, _ = self.flow(z_t.view(B*S, N))
            samples.append(y_t.view(B, S, N))

        return torch.stack(samples, dim=2)  # (B, S, T_fut, N) → rearranged below


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────────────────

def train(model, loaders, n_epochs, lr, device, patience=7):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5, min_lr=1e-5
    )

    best_val_loss  = float("inf")
    best_state     = None
    epochs_no_impr = 0

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.
        for x_win, y_win, _ in loaders["train"]:
            x_win, y_win = x_win.to(device), y_win.to(device)
            opt.zero_grad()
            loss = -model.log_likelihood(y_win, x_win)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(loaders["train"])

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for x_win, y_win, _ in loaders["val"]:
                x_win, y_win = x_win.to(device), y_win.to(device)
                val_loss += (-model.log_likelihood(y_win, x_win)).item()
        val_loss /= len(loaders["val"])
        sched.step(val_loss)

        improved = val_loss < best_val_loss
        marker   = "  ✓" if improved else ""
        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"train NLL: {train_loss:.4f}  val NLL: {val_loss:.4f}  "
              f"lr: {opt.param_groups[0]['lr']:.2e}{marker}")

        if improved:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                print(f"\nEarly stopping — no improvement for {patience} epochs.")
                break

    print(f"\nBest val NLL: {best_val_loss:.4f} — restoring best weights")
    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation — CRPS-Sum (paper's metric)
# ─────────────────────────────────────────────────────────────────────────────

def crps_sum(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    CRPS-Sum (normalised).
    samples : (n_windows, n_samples, horizon, N)
    targets : (n_windows, horizon, N)
    """
    # Normalise each series by mean absolute value (paper's CRPS-Sum-N)
    scale = np.abs(targets).mean(axis=(0,1), keepdims=True) + 1e-8   # (1,1,N)
    targets_n = targets / scale
    samples_n = samples / scale

    # Sum over N  →  univariate CRPS on the sum
    tgt_sum = targets_n.sum(-1)         # (W, H)
    smp_sum = samples_n.sum(-1)         # (W, S, H)

    # Empirical CRPS: E|X-y| - 0.5 E|X-X'|
    crps_vals = []
    for w in range(tgt_sum.shape[0]):
        for h in range(tgt_sum.shape[1]):
            y   = tgt_sum[w, h]
            s   = smp_sum[w, :, h]
            e1  = np.abs(s - y).mean()
            e2  = 0.5 * np.abs(s[:, None] - s[None, :]).mean()
            crps_vals.append(e1 - e2)

    return float(np.mean(crps_vals))


@torch.no_grad()
def evaluate(model, loader, n_samples, horizon, device):
    model.eval()
    all_samples = []
    all_targets = []

    for x_win, y_win, y_tgt in loader:
        x_win = x_win.to(device)
        y_win = y_win.to(device)

        # We forecast HORIZON steps using the last WINDOW as history.
        # For simplicity we re-use x_win as x_future (real setup: pass future covariates).
        x_fut = x_win[:, -horizon:, :]   # (B, H, k)  — last H covariate steps

        samps = model.forecast(y_win, x_win, x_fut, n_samples=n_samples)
        # samps: (B, S, T_fut, N)
        all_samples.append(samps.cpu().numpy())
        all_targets.append(y_tgt.numpy())

    samples = np.concatenate(all_samples, axis=0)   # (W, S, H, N)
    targets = np.concatenate(all_targets, axis=0)   # (W, H, N)

    # Rearrange to (W, S, H, N) → already in that shape
    crps = crps_sum(samples.transpose(0,1,2,3), targets)
    mae  = np.abs(samples.mean(1) - targets).mean()
    print(f"  CRPS-Sum-N : {crps:.6f}")
    print(f"  MAE        : {mae:.6f}")
    return crps, mae


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt

@torch.no_grad()
def plot_predictions(model, loader, y_scaler, tickers, n_samples,
                     horizon, device, n_stocks=4, n_windows=3):
    """
    Plot predicted mean +/- std vs real close prices for a few stocks and windows.
    Prices are inverse-transformed back to original USD scale.
    """
    model.eval()

    x_win, y_win, y_tgt = next(iter(loader))
    x_win = x_win.to(device)
    y_win = y_win.to(device)
    x_fut = x_win[:, -horizon:, :]

    samps = model.forecast(y_win, x_win, x_fut, n_samples=n_samples)
    # samps: (B, S, T_fut, N)

    B, S, H, N = samps.shape
    samps_np = samps.cpu().numpy()
    y_tgt_np = y_tgt.numpy()
    y_win_np = y_win.cpu().numpy()

    # Inverse-transform back to original USD prices
    samps_orig = y_scaler.inverse_transform(
        samps_np.reshape(-1, N)).reshape(B, S, H, N)
    y_tgt_orig = y_scaler.inverse_transform(
        y_tgt_np.reshape(-1, N)).reshape(B, H, N)
    y_win_orig = y_scaler.inverse_transform(
        y_win_np.reshape(-1, N)).reshape(B, -1, N)

    W = y_win_orig.shape[1]
    x_hist = np.arange(W)
    x_fore = np.arange(W - 1, W + H)

    stock_idx = np.linspace(0, N - 1, n_stocks, dtype=int)
    n_windows  = min(n_windows, B)

    fig, axes = plt.subplots(n_windows, n_stocks,
                             figsize=(5 * n_stocks, 3 * n_windows),
                             squeeze=False)
    fig.suptitle("NKF: Predicted vs Real Close Prices (USD)", fontsize=14, y=1.01)

    for row, w in enumerate(range(n_windows)):
        for col, s in enumerate(stock_idx):
            ax = axes[row][col]

            hist   = y_win_orig[w, :, s]
            real   = y_tgt_orig[w, :, s]
            pred_m = samps_orig[w, :, :, s].mean(0)
            pred_s = samps_orig[w, :, :, s].std(0)

            ax.plot(x_hist, hist, color="steelblue", lw=1.2, label="History")

            # Bridge last history point to first forecast point
            ax.plot([W - 1, W], [hist[-1], real[0]],   color="green",  lw=1.2)
            ax.plot([W - 1, W], [hist[-1], pred_m[0]], color="tomato", lw=1.2)

            ax.plot(x_fore[1:], real,   color="green",  lw=1.5, label="Real")
            ax.plot(x_fore[1:], pred_m, color="tomato", lw=1.5,
                    linestyle="--", label="Pred mean")
            ax.fill_between(x_fore[1:],
                            pred_m - pred_s,
                            pred_m + pred_s,
                            color="tomato", alpha=0.2, label="+/-1 std")

            ax.axvline(W - 1, color="gray", linestyle=":", lw=1)
            ax.set_title(f"{tickers[s]}  (window {w})", fontsize=10)
            ax.set_xlabel("Trading days")
            if col == 0:
                ax.set_ylabel("Price (USD)")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("nkf_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved -> nkf_predictions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Normalizing Kalman Filter — Stock Forecasting Pipeline")
    print("=" * 60)

    # ── Load & process ──────────────────────────────────────────
    close, covars, dates = load_data(CSV_PATH)

    print(f"\nBefore drop_nulls: {close.shape}")
    mask = close.notna().all(axis=1) & covars.notna().all(axis=1)
    close  = close[mask]
    covars = covars[mask]
    print(f"After  drop_nulls: {close.shape}")

    splits, y_scaler, x_scaler, pca = split_and_scale(
        close, covars, TRAIN_RATIO, VAL_RATIO, PCA_COMPONENTS
    )

    loaders = make_loaders(splits, WINDOW, HORIZON, BATCH_SIZE)

    tickers = close.columns.tolist()
    N = close.shape[1]
    k = PCA_COMPONENTS

    # ── Build model ─────────────────────────────────────────────
    model = NKF(
        N             = N,
        covariate_dim = k,
        state_dim     = STATE_DIM,
        flow_layers   = FLOW_LAYERS,
        hidden        = HIDDEN,
        lstm_layers   = LSTM_LAYERS,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel on {DEVICE}  |  {n_params:,} trainable parameters")
    print(f"  N (stocks)   = {N}")
    print(f"  k (covars)   = {k}")
    print(f"  Window/Horiz = {WINDOW}/{HORIZON}")

    # ── Train ────────────────────────────────────────────────────
    print("\n── Training ──────────────────────────────────────────────")
    model = train(model, loaders, N_EPOCHS, LR, DEVICE, patience=EARLY_STOP_PATIENCE)

    # ── Evaluate ─────────────────────────────────────────────────
    print("\n── Validation set ────────────────────────────────────────")
    evaluate(model, loaders["val"], N_SAMPLES, HORIZON, DEVICE)

    print("\n── Test set ──────────────────────────────────────────────")
    evaluate(model, loaders["test"], N_SAMPLES, HORIZON, DEVICE)

    # ── Plot ─────────────────────────────────────────────────────
    print("\n── Forecast Plots ────────────────────────────────────────")
    plot_predictions(model, loaders["test"], y_scaler, tickers,
                     n_samples=N_SAMPLES, horizon=HORIZON, device=DEVICE,
                     n_stocks=4, n_windows=3)

    # ── Save ─────────────────────────────────────────────────────
    torch.save(model.state_dict(), "nkf_weights.pt")
    print("\nWeights saved to nkf_weights.pt")

    # ─────────────────────────────────────────────────────────────
    # NOTE: To add commodities / news, concatenate them to covars
    # before calling split_and_scale, e.g.:
    #
    #   comm_df  = pd.read_csv("commodities.csv", index_col=0, parse_dates=True)
    #   news_df  = pd.read_csv("news_sentiment.csv", index_col=0, parse_dates=True)
    #   covars   = pd.concat([covars, comm_df, news_df], axis=1).dropna()
    #   close    = close.loc[covars.index]
    # ─────────────────────────────────────────────────────────────