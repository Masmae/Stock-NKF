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
import ast

# ─────────────────────────────────────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────────────────────────────────────

CSV_path       = "stock.csv"   # stock data CSV_new_path
COMM_path      = "comm.csv"    # commodity data CSV_new_path
NEWS_path = "news.csv"         # news data CSV path
WINDOW         = 30            # longer window captures momentum/earnings cycles
HORIZON        = 5             # steps ahead to forecast
TRAIN_RATIO    = 0.7
VAL_RATIO      = 0.15          # remaining 0.15 → test
BATCH_SIZE     = 100
N_EPOCHS       = 100           # early stopping will kick in before this
LR             = 1e-3          # slower start reduces early overfitting
STATE_DIM      = 3             # level + trend + volatility state
FLOW_LAYERS    = 1             # fewer flow layers → less capacity
HIDDEN         = 8             # smaller coupling nets + LSTM
LSTM_LAYERS    = 2             # single LSTM layer
N_SAMPLES      = 1000          # Monte Carlo forecast samples
PCA_COMPONENTS = 50            # increased to capture ~90% variance
EARLY_STOP_PATIENCE = 7        # stop if val NLL doesn't improve
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Loading & Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ohlcv(df):
    """
    Parse a yfinance multi-level CSV (already loaded as DataFrame with MultiIndex columns).
    Returns dict of {price_type: DataFrame(T, n_tickers)}.
    """
    return {
        pt: df[pt].astype(float)
        for pt in ["Close", "High", "Low", "Open", "Volume"]
        if pt in df.columns.get_level_values(0)
    }


def _make_features(ohlcv: dict, label: str):
    """
    Build log-return features from an OHLCV dict.
    Returns a DataFrame of covariates aligned to the close_ret index.

    Lookahead note
    ──────────────
    x[t] contains same-day OHLCV (open/high/low of day t alongside close of day t).
    This is standard daily-bar practice — open/high/low are known before market close.
    Rolling features at t use days t-k..t, which is causal given we observe y[t].
    Prediction target is always y[e] where e > window end — strictly future.
    """
    close  = ohlcv["Close"]
    high   = ohlcv["High"]
    low    = ohlcv["Low"]
    open_  = ohlcv["Open"]
    volume = ohlcv["Volume"]
    tickers = close.columns.tolist()

    # ── Log returns ───────────────────────────────────────────────
    close_ret = np.log(close / close.shift(1)).iloc[1:]
    high_ret  = np.log(high  / high.shift(1) ).iloc[1:]
    low_ret   = np.log(low   / low.shift(1)  ).iloc[1:]
    open_ret  = np.log(open_ / open_.shift(1)).iloc[1:]

    high_ret.columns  = [f"{label}_high_{t}"  for t in tickers]
    low_ret.columns   = [f"{label}_low_{t}"   for t in tickers]
    open_ret.columns  = [f"{label}_open_{t}"  for t in tickers]

    # ── Intra-day range ───────────────────────────────────────────
    intra = ((high - low) / close.replace(0, np.nan)).iloc[1:].fillna(0)
    intra.columns = [f"{label}_range_{t}" for t in tickers]

    # ── Log volume ────────────────────────────────────────────────
    log_vol = np.log1p(volume.iloc[1:])
    log_vol.columns = [f"{label}_logvol_{t}" for t in tickers]

    # ── Volatility clustering (rolling std of returns) ────────────
    # Gives LSTM a direct signal to set Gamma_t on high-vol days
    vol10 = close_ret.rolling(10).std().fillna(0)
    vol20 = close_ret.rolling(20).std().fillna(0)
    vol10.columns = [f"{label}_vol10_{t}" for t in tickers]
    vol20.columns = [f"{label}_vol20_{t}" for t in tickers]

    # ── Momentum (cumulative returns) ─────────────────────────────
    mom5  = close_ret.rolling(5).sum().fillna(0)
    mom20 = close_ret.rolling(20).sum().fillna(0)
    mom5.columns  = [f"{label}_mom5_{t}"  for t in tickers]
    mom20.columns = [f"{label}_mom20_{t}" for t in tickers]

    return close_ret, pd.concat(
        [high_ret, low_ret, open_ret, intra, log_vol, vol10, vol20, mom5, mom20],
        axis=1
    )


def load_news(news_path: str) -> pd.DataFrame:
    """
    Parse a CSV with columns [Date, embedding(list)] where embedding is a serialized
    list like '([0.1, 0.2, ...])'.
    Returns a DataFrame indexed by Date with one column per embedding element.
    """
    df = pd.read_csv(news_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    tensor_col = df.iloc[:, 0]

    def parse_tensor(s):
        return np.array(ast.literal_eval(str(s).strip()))

    arrays = tensor_col.apply(parse_tensor)
    n_elements = max(len(a) for a in arrays)

    matrix = np.zeros((len(arrays), n_elements))
    for i, arr in enumerate(arrays):
        matrix[i, : len(arr)] = arr

    col_names = [f"emb_{i}" for i in range(n_elements)]
    return pd.DataFrame(matrix, index=df.index, columns=col_names)


def merge_tensor_covars(covars: pd.DataFrame, news_path: str) -> pd.DataFrame:
    """
    Merge news embeddings into the covariate DataFrame.

    Loads the news embedding CSV, left-joins it onto covars by date, and
    zero-pads any trading days with no corresponding news embedding.

    Parameters
    ----------
    covars : pd.DataFrame
        Existing covariate DataFrame indexed by Date, as returned by load_data
        before the news merge step.
    news_path : str
        Path to the news embeddings CSV (Date, list-serialized embedding vector).

    Returns
    -------
    pd.DataFrame
        covars extended with n_elements new columns named 'emb_0', 'emb_1', ...,
        'emb_{n-1}'. Rows with no matching news date are filled with zeros.
    """
    tensor_df = load_news(news_path)  # ← was load_tensor_csv
    n_elements = tensor_df.shape[1]

    merged = covars.join(tensor_df, how="left")
    emb_cols = [f"emb_{i}" for i in range(n_elements)]
    merged[emb_cols] = merged[emb_cols].fillna(0.0)

    print(f"Tensor features: {n_elements} elements | "
          f"{tensor_df.index.isin(covars.index).sum()} dates matched | "
          f"{(~tensor_df.index.isin(covars.index)).sum()} tensor dates outside covars range (dropped)")

    return merged


def load_data(stock_path: str, comm_path: str, news_path: str,
              use_comm: bool = True, use_news: bool = True):
    """
    Parse stock + commodity CSVs (yfinance multi-level format).

    Returns
    -------
    close_ret : pd.DataFrame  (T, 20)   log returns of stock closes  → y (observations)
    close_raw : pd.DataFrame  (T, 20)   raw stock close prices       → kept for reference
    covars    : pd.DataFrame  (T, k)    all features                 → x (covariates)
    dates     : pd.DatetimeIndex
    """
    def read_csv(CSV_new_path):
        df = pd.read_csv(CSV_new_path, header=[0, 1], index_col=0, skiprows=[2])
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df.sort_index()

    stock_df = read_csv(stock_path)
    comm_df  = read_csv(comm_path)

    stock_ohlcv = _parse_ohlcv(stock_df)

    # ── Observations: stock close log returns ─────────────────────
    close_raw                  = stock_ohlcv["Close"].copy()
    stock_close_ret, stock_cov = _make_features(stock_ohlcv, label="stk")

    # ── Calendar features ─────────────────────────────────────────
    cal = pd.DataFrame(index=stock_close_ret.index)
    cal["dow"]     = stock_close_ret.index.dayofweek / 4.0
    cal["month"]   = (stock_close_ret.index.month - 1) / 11.0
    cal["qtr_end"] = stock_close_ret.index.is_quarter_end.astype(float)

    parts = [stock_cov, cal]

    # ── Optional: commodities ─────────────────────────────────────
    if use_comm:
        comm_df   = read_csv(comm_path)
        comm_ohlcv = _parse_ohlcv(comm_df)
        _, comm_cov = _make_features(comm_ohlcv, label="com")
        parts.append(comm_cov)

    # ── Align all frames on common dates (inner join) ─────────────
    covars     = pd.concat(parts, axis=1)
    common_idx = stock_close_ret.index.intersection(covars.index)
    close_ret  = stock_close_ret.loc[common_idx]
    covars     = covars.loc[common_idx]
    close_raw  = close_raw.loc[close_raw.index.isin(common_idx)]

    # ── Drop any remaining NaNs ───────────────────────────────────
    mask      = close_ret.notna().all(axis=1) & covars.notna().all(axis=1)
    close_ret = close_ret[mask]
    covars    = covars[mask]
    close_raw = close_raw.loc[close_raw.index.isin(close_ret.index)]

    # ── Optional: news embeddings ─────────────────────────────────
    if use_news:
        covars = merge_tensor_covars(covars, news_path)

    N_stocks = len(stock_ohlcv["Close"].columns)
    n_comm   = len(_parse_ohlcv(read_csv(comm_path))["Close"].columns) if use_comm else 0
    print(f"Loaded  {len(close_ret)} days  |  "
          f"{N_stocks} stocks  +  {n_comm} comm  +  "
          f"{'news' if use_news else 'no news'}  |  "
          f"{covars.shape[1]} raw covariates")
    print(f"Date range: {close_ret.index[0].date()} → {close_ret.index[-1].date()}")
    return close_ret, close_raw, covars, close_ret.index


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
    # First fit with max components to find how many reach the threshold
    pca_full  = PCA(random_state=42).fit(x_scaled[:t1])
    cum_var   = np.cumsum(pca_full.explained_variance_ratio_)
    n_needed  = int(np.searchsorted(cum_var, 0.90)) + 1
    n_components = max(pca_components, n_needed)
    print(f"PCA: {x_raw.shape[1]} raw features  |  "
          f"{n_needed} components for 95% variance  |  "
          f"using {n_components}")

    pca    = PCA(n_components=n_components, random_state=42).fit(x_scaled[:t1])
    x      = pca.transform(x_scaled).astype(np.float32)
    var_ex = pca.explained_variance_ratio_.sum()
    print(f"     Variance explained: {var_ex:.1%}")

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

def train(model, loaders, n_epochs, lr, device, patience=7): # partition to window and horizon (p y_t+1 to y_t+T | p y_1 to y_t)
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
def evaluate(model, loader, n_samples, horizon, device, y_scaler, tickers):
    model.eval()
    all_samples = []
    all_targets = []

    for x_win, y_win, y_tgt in loader:
        x_win = x_win.to(device)
        y_win = y_win.to(device)
        x_fut = x_win[:, -1:, :].expand(-1, horizon, -1).clone()
        samps = model.forecast(y_win, x_win, x_fut, n_samples=n_samples)
        all_samples.append(samps.cpu().numpy())
        all_targets.append(y_tgt.numpy())

    samples = np.concatenate(all_samples, axis=0)   # (W, S, H, N)
    targets = np.concatenate(all_targets, axis=0)   # (W, H, N)

    # Inverse-transform to raw log returns for interpretable metrics
    W, S, H, N = samples.shape
    samp_ret = y_scaler.inverse_transform(
        samples.reshape(-1, N)).reshape(W, S, H, N)
    tgt_ret  = y_scaler.inverse_transform(
        targets.reshape(-1, N)).reshape(W, H, N)

    pred_mean = samp_ret.mean(axis=1)   # (W, H, N)
    pred_std  = samp_ret.std(axis=1)    # (W, H, N)

    # ── CRPS-Sum (paper metric) ───────────────────────────────────
    crps = crps_sum(samp_ret, tgt_ret)

    # ── MAE on log returns ────────────────────────────────────────
    mae = np.abs(pred_mean - tgt_ret).mean()

    # ── Directional accuracy (1-step ahead) ───────────────────────
    # % of days where predicted and real return have same sign
    dir_acc = (np.sign(pred_mean[:, 0, :]) == np.sign(tgt_ret[:, 0, :])).mean()

    # ── Calibration (1-step ahead) ────────────────────────────────
    # Fraction of real returns within predicted mean ± 1 std
    # Well-calibrated model → ~68%
    within_1std = (np.abs(tgt_ret[:, 0, :] - pred_mean[:, 0, :]) < pred_std[:, 0, :]).mean()

    print(f"  CRPS-Sum-N         : {crps:.6f}")
    print(f"  MAE (log return)   : {mae:.6f}")
    print(f"  Directional Acc    : {dir_acc:.2%}  (random baseline = 50%)")
    print(f"  Calibration ±1std  : {within_1std:.2%}  (ideal = 68%)")

    # ── Per-stock CRPS (1-step ahead) ─────────────────────────────
    print("\n  Per-stock CRPS (1-step):")
    per_stock_crps = []
    for s in range(N):
        s_samp = samp_ret[:, :, 0, s]   # (W, S)
        s_tgt  = tgt_ret[:,  0,    s]   # (W,)
        e1 = np.abs(s_samp - s_tgt[:, None]).mean()
        e2 = 0.5 * np.abs(s_samp[:, :, None] - s_samp[:, None, :]).mean()
        per_stock_crps.append(e1 - e2)
        print(f"    {tickers[s]:6s}: {per_stock_crps[-1]:.6f}")

    return crps, mae, dir_acc, within_1std, per_stock_crps


def evaluate_random_walk(loader, y_scaler, tickers, horizon):
    """
    Random Walk baseline: predict 0 return for all stocks at all horizons.
    This is the hardest baseline to beat in financial return forecasting.

    For CRPS we model the RW as a Gaussian with mean=0 and std estimated
    from the training window's empirical return std — this gives it a fair
    probabilistic score rather than penalising it for having no uncertainty.
    """
    all_targets  = []
    all_win_stds = []   # empirical std from each input window

    for _, y_win, y_tgt in loader:
        all_targets.append(y_tgt.numpy())               # (B, H, N)
        # std over the input window — used as the RW's predictive std
        all_win_stds.append(y_win.numpy().std(axis=1))  # (B, N)

    targets   = np.concatenate(all_targets,  axis=0)   # (W, H, N)
    win_stds  = np.concatenate(all_win_stds, axis=0)   # (W, N)

    # Inverse-transform to raw log returns
    W, H, N = targets.shape
    tgt_ret  = y_scaler.inverse_transform(
        targets.reshape(-1, N)).reshape(W, H, N)
    win_stds = win_stds * y_scaler.scale_               # scale std only

    # ── RW prediction: mean=0, std=empirical window std ──────────
    rw_mean = np.zeros_like(tgt_ret)                    # (W, H, N)
    # Broadcast win_stds across horizon steps
    rw_std  = win_stds[:, np.newaxis, :].repeat(H, axis=1)  # (W, H, N)

    # ── Build Gaussian samples for CRPS ──────────────────────────
    rng      = np.random.default_rng(42)
    S        = 500   # samples for CRPS estimation
    rw_samps = rw_mean[:, np.newaxis, :, :] + \
               rw_std[:, np.newaxis, :, :]  * \
               rng.standard_normal((W, S, H, N))       # (W, S, H, N)

    # ── Metrics ───────────────────────────────────────────────────
    crps     = crps_sum(rw_samps, tgt_ret)
    mae      = np.abs(rw_mean - tgt_ret).mean()
    dir_acc  = 0.50   # by definition — always predicts 0

    # Calibration: fraction of real returns within ±1 RW std
    within_1std = (np.abs(tgt_ret[:, 0, :]) < rw_std[:, 0, :]).mean()

    print(f"  CRPS-Sum-N         : {crps:.6f}")
    print(f"  MAE (log return)   : {mae:.6f}")
    print(f"  Directional Acc    : {dir_acc:.2%}  (random by definition)")
    print(f"  Calibration ±1std  : {within_1std:.2%}  (ideal = 68%)")

    # ── Per-stock CRPS ────────────────────────────────────────────
    print("\n  Per-stock CRPS (1-step):")
    per_stock_crps = []
    for s in range(N):
        s_samp = rw_samps[:, :, 0, s]
        s_tgt  = tgt_ret[:,  0,    s]
        e1 = np.abs(s_samp - s_tgt[:, None]).mean()
        e2 = 0.5 * np.abs(s_samp[:, :, None] - s_samp[:, None, :]).mean()
        per_stock_crps.append(e1 - e2)
        print(f"    {tickers[s]:6s}: {per_stock_crps[-1]:.6f}")

    return crps, mae, dir_acc, within_1std, per_stock_crps

import matplotlib.pyplot as plt

@torch.no_grad()
def plot_predictions(model, loader, y_scaler, tickers, n_samples,
                     horizon, device, n_stocks=4):
    """
    Plot predicted vs real log returns over the full test set.
    One subplot per stock. X-axis = test window index, Y-axis = log return.
    Takes only the 1-step-ahead forecast from each window.
    """
    model.eval()

    all_pred_mean = []
    all_pred_std  = []
    all_real      = []

    for x_win, y_win, y_tgt in loader:
        x_win = x_win.to(device)
        y_win = y_win.to(device)
        x_fut = x_win[:, -1:, :].expand(-1, horizon, -1).clone()

        samps = model.forecast(y_win, x_win, x_fut, n_samples=n_samples)
        B, S, H, N = samps.shape

        pred_t1 = samps[:, :, 0, :]                          # (B, S, N)
        all_pred_mean.append(pred_t1.mean(1).cpu().numpy())  # (B, N)
        all_pred_std.append( pred_t1.std(1).cpu().numpy())   # (B, N)
        all_real.append(y_tgt[:, 0, :].numpy())              # (B, N)

    pred_ret_mean = np.concatenate(all_pred_mean, axis=0)    # (W, N) — scaled returns
    pred_ret_std  = np.concatenate(all_pred_std,  axis=0)    # (W, N)
    real_ret      = np.concatenate(all_real,      axis=0)    # (W, N) — scaled returns

    # ── Inverse-transform: scaled returns → raw log returns ──────
    pred_ret_mean = y_scaler.inverse_transform(pred_ret_mean)
    real_ret      = y_scaler.inverse_transform(real_ret)
    pred_ret_std  = pred_ret_std * y_scaler.scale_

    W  = real_ret.shape[0]
    xs = np.arange(W)

    stock_idx = np.linspace(0, N - 1, n_stocks, dtype=int)

    fig, axes = plt.subplots(n_stocks, 1,
                             figsize=(14, 3.5 * n_stocks),
                             squeeze=False)
    fig.suptitle("NKF: Predicted vs Real — Full Test Set (1-step ahead, Log Return)",
                 fontsize=13, y=1.01)

    for row, s in enumerate(stock_idx):
        ax = axes[row][0]
        ax.plot(xs, real_ret[:, s],      color="steelblue", lw=1.0, label="Real return")
        ax.plot(xs, pred_ret_mean[:, s], color="tomato",    lw=1.0,
                linestyle="--", label="Pred mean")
        ax.fill_between(xs,
                        pred_ret_mean[:, s] - pred_ret_std[:, s],
                        pred_ret_mean[:, s] + pred_ret_std[:, s],
                        color="tomato", alpha=0.2, label="+/-1 std")
        ax.axhline(0, color="gray", lw=0.8, linestyle=":")
        ax.set_title(tickers[s], fontsize=11)
        ax.set_ylabel("Log return")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Test window index")
    import os, datetime
    os.makedirs("figs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"figs/nkf_test_predictions_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Handle registry
# ─────────────────────────────────────────────────────────────────────────────

import argparse, os, datetime

_HANDLES = {}

def handle(name: str):
    """
    Decorator that registers a function as a runnable handle.
    Usage:  python pipeline.py --run nkf
            python pipeline.py --run rw
            python pipeline.py --run all
    """
    def decorator(fn):
        _HANDLES[name] = fn
        return fn
    return decorator


def _load_shared(window=None, use_comm=True, use_news=True):
    """
    Load and preprocess data shared by all handles.
    window     : override WINDOW config if provided
    use_comm   : include commodity covariates
    use_news   : include news embedding covariates
    """
    print("=" * 60)
    print("Normalizing Kalman Filter — Stock Forecasting Pipeline")
    print("=" * 60)

    close, close_raw, covars, dates = load_data(
        CSV_path, COMM_path, NEWS_path,
        use_comm=use_comm, use_news=use_news
    )

    print(f"\nBefore drop_nulls: {close.shape}")
    mask      = close.notna().all(axis=1) & covars.notna().all(axis=1)
    close     = close[mask].sort_index()
    close_raw = close_raw.loc[close_raw.index.isin(close.index)].sort_index()
    covars    = covars[mask].sort_index()
    print(f"After  drop_nulls: {close.shape}")

    splits, y_scaler, x_scaler, pca = split_and_scale(
        close, covars, TRAIN_RATIO, VAL_RATIO, PCA_COMPONENTS
    )
    w = window if window is not None else WINDOW
    loaders = make_loaders(splits, w, HORIZON, BATCH_SIZE)
    tickers = close.columns.tolist()
    return loaders, tickers, y_scaler, pca


def _build_and_train(loaders, tickers, y_scaler, pca, label="NKF"):
    """
    Build, train, and evaluate an NKF model. Returns (model, results).
    """
    N = len(tickers)
    k = pca.n_components_

    model = NKF(
        N=N, covariate_dim=k, state_dim=STATE_DIM,
        flow_layers=FLOW_LAYERS, hidden=HIDDEN, lstm_layers=LSTM_LAYERS,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{label}] {n_params:,} params  |  N={N}  k={k}")

    model = train(model, loaders, N_EPOCHS, LR, DEVICE,
                  patience=EARLY_STOP_PATIENCE)

    print(f"\n── [{label}] Test set ────────────────────────────────────")
    results = evaluate(model, loaders["test"], N_SAMPLES, HORIZON,
                       DEVICE, y_scaler, tickers)
    return model, results


# ─────────────────────────────────────────────────────────────────────────────

@handle("rw")
def run_random_walk():
    """
    Evaluate Random Walk baseline on the test set.
    Run with:  python pipeline.py --run rw
    """
    loaders, tickers, y_scaler, _ = _load_shared()

    print("\n── Random Walk Baseline (test set) ───────────────────────")
    results = evaluate_random_walk(loaders["test"], y_scaler, tickers, HORIZON)
    return results


# ─────────────────────────────────────────────────────────────────────────────

@handle("nkf")
def run_nkf():
    """
    Train and evaluate the full NKF model.
    Run with:  python pipeline.py --run nkf
    """
    loaders, tickers, y_scaler, pca = _load_shared()

    N = len(tickers)
    k = pca.n_components_

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
    print(f"  N (stocks)   = {N}  |  k (covars) = {k}")
    print(f"  Window/Horiz = {WINDOW}/{HORIZON}")

    # ── Train ────────────────────────────────────────────────────
    print("\n── Training ──────────────────────────────────────────────")
    model = train(model, loaders, N_EPOCHS, LR, DEVICE, patience=EARLY_STOP_PATIENCE)

    # ── Evaluate ─────────────────────────────────────────────────
    print("\n── Validation set ────────────────────────────────────────")
    evaluate(model, loaders["val"], N_SAMPLES, HORIZON, DEVICE, y_scaler, tickers)

    print("\n── Test set ──────────────────────────────────────────────")
    results = evaluate(
        model, loaders["test"], N_SAMPLES, HORIZON, DEVICE, y_scaler, tickers
    )

    # ── Plot ─────────────────────────────────────────────────────
    print("\n── Forecast Plots ────────────────────────────────────────")
    plot_predictions(model, loaders["test"], y_scaler, tickers,
                     n_samples=N_SAMPLES, horizon=HORIZON, device=DEVICE,
                     n_stocks=4)

    # ── Save weights ─────────────────────────────────────────────
    torch.save(model.state_dict(), "nkf_weights.pt")
    print("\nWeights saved → nkf_weights.pt")

    return results, model


# ─────────────────────────────────────────────────────────────────────────────

@handle("all")
def run_all():
    """
    Run NKF + Random Walk and print a side-by-side comparison table.
    Run with:  python pipeline.py --run all
    """
    loaders, tickers, y_scaler, pca = _load_shared()

    N = len(tickers)
    k = pca.n_components_

    # ── Random Walk ──────────────────────────────────────────────
    print("\n── Random Walk Baseline (test set) ───────────────────────")
    rw_results = evaluate_random_walk(loaders["test"], y_scaler, tickers, HORIZON)

    # ── NKF ──────────────────────────────────────────────────────
    model = NKF(
        N=N, covariate_dim=k, state_dim=STATE_DIM,
        flow_layers=FLOW_LAYERS, hidden=HIDDEN, lstm_layers=LSTM_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel on {DEVICE}  |  {n_params:,} params  |  k={k}")

    print("\n── Training ──────────────────────────────────────────────")
    model = train(model, loaders, N_EPOCHS, LR, DEVICE, patience=EARLY_STOP_PATIENCE)

    print("\n── Validation set ────────────────────────────────────────")
    evaluate(model, loaders["val"], N_SAMPLES, HORIZON, DEVICE, y_scaler, tickers)

    print("\n── Test set (NKF) ────────────────────────────────────────")
    nkf_results = evaluate(
        model, loaders["test"], N_SAMPLES, HORIZON, DEVICE, y_scaler, tickers
    )

    # ── Comparison table ─────────────────────────────────────────
    print("\n── Summary: NKF vs Random Walk ───────────────────────────")
    print(f"  {'Metric':<22}  {'Random Walk':>12}  {'NKF':>12}  {'Better':>8}")
    print(f"  {'-'*60}")
    comparisons = [
        ("CRPS-Sum-N ↓",    rw_results[0], nkf_results[0], "lower"),
        ("MAE ↓",           rw_results[1], nkf_results[1], "lower"),
        ("Dir. Accuracy ↑", rw_results[2], nkf_results[2], "higher"),
        ("Calibration",     rw_results[3], nkf_results[3], "closer_68"),
    ]
    for name, rw, nkf, direction in comparisons:
        if direction == "lower":
            winner = "NKF ✓" if nkf < rw else "RW"
        elif direction == "higher":
            winner = "NKF ✓" if nkf > rw else "RW"
        else:
            winner = "NKF ✓" if abs(nkf - 0.68) < abs(rw - 0.68) else "RW"
        print(f"  {name:<22}  {rw:>12.4f}  {nkf:>12.4f}  {winner:>8}")

    # ── Plot ─────────────────────────────────────────────────────
    plot_predictions(model, loaders["test"], y_scaler, tickers,
                     n_samples=N_SAMPLES, horizon=HORIZON, device=DEVICE,
                     n_stocks=4)
    torch.save(model.state_dict(), "nkf_weights.pt")
    print("\nWeights saved → nkf_weights.pt")


# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────

@handle("window_search")
def run_window_search():
    """
    Compare NKF performance across different input window sizes.
    Data is loaded once; a fresh model is trained per window size.
    Run with:  python pipeline.py --run window_search

    Results table (test set):
        Window | CRPS-Sum | MAE | Dir.Acc | Calibration
    """
    WINDOWS = [10, 20, 30, 60, 120]

    # Load data once (full: stocks + comm + news)
    print("=" * 60)
    print("Window Search — Loading data once")
    print("=" * 60)

    close, close_raw, covars, dates = load_data(
        CSV_path, COMM_path, NEWS_path,
        use_comm=True, use_news=True
    )
    mask      = close.notna().all(axis=1) & covars.notna().all(axis=1)
    close     = close[mask].sort_index()
    covars    = covars[mask].sort_index()

    splits, y_scaler, x_scaler, pca = split_and_scale(
        close, covars, TRAIN_RATIO, VAL_RATIO, PCA_COMPONENTS
    )
    tickers = close.columns.tolist()

    results_table = []

    for w in WINDOWS:
        print(f"\n{'='*60}")
        print(f"Window = {w}")
        print(f"{'='*60}")

        loaders = make_loaders(splits, w, HORIZON, BATCH_SIZE)
        _, res  = _build_and_train(loaders, tickers, y_scaler, pca,
                                   label=f"W={w}")
        crps, mae, dir_acc, calib, _ = res
        results_table.append((w, crps, mae, dir_acc, calib))

    # ── Summary table ─────────────────────────────────────────────
    print("\n\n── Window Search Results (test set) ──────────────────────")
    print(f"  {'Window':>8}  {'CRPS-Sum':>10}  {'MAE':>10}  "
          f"{'Dir.Acc':>9}  {'Calib':>8}")
    print(f"  {'-'*52}")
    for w, crps, mae, dir_acc, calib in results_table:
        print(f"  {w:>8}  {crps:>10.6f}  {mae:>10.6f}  "
              f"{dir_acc:>9.2%}  {calib:>8.2%}")

    best_w = min(results_table, key=lambda x: x[1])[0]
    print(f"\n  Best window by CRPS-Sum: {best_w}")


# ─────────────────────────────────────────────────────────────────────────────

@handle("ablation")
def run_ablation():
    """
    Ablation: compare three data configurations on the test set.
        1. stocks only          (no commodities, no news)
        2. stocks + commodities (no news)
        3. stocks + comm + news (full model)

    All other hyperparameters are held fixed.
    Run with:  python pipeline.py --run ablation
    """
    configs = [
        ("Stocks only",        False, False),
        ("Stocks + Comm",      True,  False),
        ("Stocks + Comm + News", True, True),
    ]

    results_table = []

    for label, use_comm, use_news in configs:
        print(f"\n{'='*60}")
        print(f"Ablation: {label}")
        print(f"{'='*60}")

        loaders, tickers, y_scaler, pca = _load_shared(
            use_comm=use_comm, use_news=use_news
        )
        _, res = _build_and_train(loaders, tickers, y_scaler, pca,
                                  label=label)
        crps, mae, dir_acc, calib, _ = res
        results_table.append((label, crps, mae, dir_acc, calib))

    # ── Summary table ─────────────────────────────────────────────
    print("\n\n── Ablation Results (test set) ───────────────────────────")
    print(f"  {'Config':<26}  {'CRPS-Sum':>10}  {'MAE':>10}  "
          f"{'Dir.Acc':>9}  {'Calib':>8}")
    print(f"  {'-'*68}")
    for label, crps, mae, dir_acc, calib in results_table:
        print(f"  {label:<26}  {crps:>10.6f}  {mae:>10.6f}  "
              f"{dir_acc:>9.2%}  {calib:>8.2%}")

    # Mark best per metric
    print("\n  Best per metric:")
    for metric_idx, metric_name in [(1, "CRPS-Sum"), (2, "MAE"),
                                    (3, "Dir.Acc"), (4, "Calib")]:
        if metric_name == "Calib":
            best = min(results_table, key=lambda x: abs(x[metric_idx] - 0.68))
        elif metric_name == "Dir.Acc":
            best = max(results_table, key=lambda x: x[metric_idx])
        else:
            best = min(results_table, key=lambda x: x[metric_idx])
        print(f"    {metric_name}: {best[0]}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NKF Pipeline")
    parser.add_argument(
        "--run",
        choices=list(_HANDLES.keys()),
        default="all",
        help=f"Which handle to run. Choices: {list(_HANDLES.keys())}",
    )
    args = parser.parse_args()
    print(f"\nRunning handle: '{args.run}'\n")
    _HANDLES[args.run]()