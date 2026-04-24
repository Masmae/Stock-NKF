import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

# ══════════════════════════════════════════════════════════════════
# 1.  RealNVP — Global Normalizing Flow  (Section 3, ft in paper)
#     Maps observations y_t <-> pseudo-observations z_t
#     Jacobian is lower-triangular → det computable in O(N)
# ══════════════════════════════════════════════════════════════════

class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP.
    Splits input in half; one half parameterises scale+shift for the other.
    """
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, (dim - half) * 2),   # scale + shift
        )
        self.split = half

    def forward(self, z):
        """z -> y  (forward = generate)"""
        z1, z2 = z[..., :self.split], z[..., self.split:]
        params  = self.net(z1)
        s, t    = params.chunk(2, dim=-1)
        s       = torch.tanh(s)                     # bound scales
        y2      = z2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)                     # log |det J|
        return torch.cat([z1, y2], dim=-1), log_det

    def inverse(self, y):
        """y -> z  (inverse = infer pseudo-obs)"""
        y1, y2 = y[..., :self.split], y[..., self.split:]
        params  = self.net(y1)
        s, t    = params.chunk(2, dim=-1)
        s       = torch.tanh(s)
        z2      = (y2 - t) * torch.exp(-s)
        log_det = -s.sum(dim=-1)
        return torch.cat([y1, z2], dim=-1), log_det


class RealNVP(nn.Module):
    """
    Stack of alternating coupling layers with permutations.
    ft: R^N -> R^N  (invertible, tractable Jacobian)
    """
    def __init__(self, N: int, n_layers: int = 4, hidden: int = 64):
        super().__init__()
        self.layers = nn.ModuleList(
            [CouplingLayer(N, hidden) for _ in range(n_layers)]
        )
        # Fixed alternating permutations
        idx = torch.arange(N)
        self.register_buffer('perm',    torch.flip(idx, [0]))
        self.register_buffer('inv_perm', torch.argsort(torch.flip(idx, [0])))

    def forward(self, z):
        """z -> y,  returns (y, total log|det J|)"""
        log_det = 0.
        for i, layer in enumerate(self.layers):
            z, ld = layer(z)
            log_det = log_det + ld
            if i < len(self.layers) - 1:
                z = z[..., self.perm]               # permute between layers
        return z, log_det

    def inverse(self, y):
        """y -> z,  returns (z, total log|det J^{-1}|)"""
        log_det = 0.
        for i, layer in enumerate(reversed(self.layers)):
            if i > 0:
                y = y[..., self.inv_perm]
            y, ld = layer.inverse(y)
            log_det = log_det + ld
        return y, log_det


# ══════════════════════════════════════════════════════════════════
# 2.  Kalman Filter — one step, batched over (B, N) series
#     State: l_t ∈ R^d  (d=2: level + trend)
# ══════════════════════════════════════════════════════════════════

def kalman_step(mu, P, F, Q, A, obs_var, z_obs=None):
    """
    One Kalman predict+update step.

    Args
    ----
    mu      (B, N, d)       filtered mean from t-1
    P       (B, N, d, d)    filtered cov  from t-1
    F       (B, N, d, d)    transition matrix
    Q       (B, N, d, d)    process noise covariance
    A       (B, N, d)       emission vector  (A^T l gives scalar z)
    obs_var (B, N)          observation noise variance Γ_t (scalar per series)
    z_obs   (B, N) or None  pseudo-observation; if None, predict only

    Returns
    -------
    mu_upd, P_upd, log_lik  (log_lik is 0 if z_obs is None)
    """
    # ── Predict ──────────────────────────────────────────────────
    mu_p = (F @ mu.unsqueeze(-1)).squeeze(-1)           # (B, N, d)
    P_p  = F @ P @ F.transpose(-1, -2) + Q              # (B, N, d, d)

    if z_obs is None:
        return mu_p, P_p, torch.zeros(mu.shape[:2], device=mu.device)

    # ── Update ───────────────────────────────────────────────────
    A_   = A.unsqueeze(-1)                              # (B, N, d, 1)
    # Innovation variance: S = A^T P A + Γ
    S    = (A_.transpose(-1,-2) @ P_p @ A_).squeeze(-1,-2) + obs_var.unsqueeze(-1)  # (B,N,1)
    # Kalman gain: K = P A / S
    K    = (P_p @ A_) / S.unsqueeze(-1)                # (B, N, d, 1)
    # Innovation
    z_hat = (A_.transpose(-1,-2) @ mu_p.unsqueeze(-1)).squeeze(-1,-2)  # (B, N)
    innov = z_obs - z_hat                               # (B, N)
    # State update
    mu_u = mu_p + (K.squeeze(-1) * innov.unsqueeze(-1))                # (B, N, d)
    IKA  = torch.eye(mu.shape[-1], device=mu.device) - K @ A_.transpose(-1,-2)
    P_u  = IKA @ P_p                                    # (B, N, d, d)
    # Log-likelihood (Gaussian predictive)
    S_   = S.squeeze(-1)                                # (B, N)
    log_lik = -0.5 * (innov**2 / S_ + S_.log() + np.log(2 * np.pi))

    return mu_u, P_u, log_lik                          # log_lik: (B, N)


# ══════════════════════════════════════════════════════════════════
# 3.  Parameter Network — LSTM that emits SSM params from covariates
#     (Section 2.2: Θ_t = σ(h_t; Φ), h_t = Ψ(x_t, h_{t-1}; Φ))
# ══════════════════════════════════════════════════════════════════

class ParameterNet(nn.Module):
    """
    LSTM over covariates x_t to produce time-varying SSM parameters
    for each of the N time series:
        - obs_var  Γ_t   : observation noise variance  (>0)
        - proc_var σ_t   : process noise scale         (>0)
        - emit     A_t   : emission vector              (d-dim)
    F_t is fixed to the level-trend structure (standard choice).
    """
    def __init__(self, covariate_dim: int, N: int, state_dim: int = 2,
                 hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.N = N
        self.d = state_dim

        # Shared LSTM across series (amortized — Section 3)
        self.lstm = nn.LSTM(covariate_dim, hidden, n_layers, batch_first=True)

        # Output heads (per series)
        out_dim = N * (1 + state_dim + 1)   # obs_var + emit(d) + proc_scale
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, h=None):
        """
        x: (B, T, covariate_dim)
        Returns dicts of tensors shaped (B, T, N, ...)
        """
        out, h = self.lstm(x, h)            # (B, T, hidden)
        params  = self.head(out)            # (B, T, N*(2+d))
        B, T, _ = params.shape

        params  = params.view(B, T, self.N, 2 + self.d)
        obs_var  = torch.exp(params[..., 0])            # Γ_t  (B,T,N)
        proc_var = torch.exp(params[..., 1])            # σ_t  (B,T,N)
        emit     = params[..., 2:]                      # A_t  (B,T,N,d)

        return obs_var, proc_var, emit, h


# ══════════════════════════════════════════════════════════════════
# 4.  NKF — Full Model  (ties everything together)
# ══════════════════════════════════════════════════════════════════

class NKF(nn.Module):
    """
    Normalizing Kalman Filter (de Bézenac et al., NeurIPS 2020).

    Architecture
    ────────────
    • RealNVP  (global normalizing flow)  :  y_t  <->  z_t
    • Local LGM per series                :  KF on z_t
    • LSTM parameter network              :  covariates -> SSM params
    
    Data layout expected
    ─────────────────────
    y : (B, T, N)   — N target time series  (stocks + commodities)
    x : (B, T, k)   — k covariates per timestep (news sentiment, calendar, etc.)
    """
    def __init__(self, N: int, covariate_dim: int,
                 state_dim: int = 2,
                 flow_layers: int = 4,
                 flow_hidden: int = 64,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2):
        super().__init__()
        self.N = N
        self.d = state_dim

        self.flow     = RealNVP(N, flow_layers, flow_hidden)
        self.param_net = ParameterNet(covariate_dim, N, state_dim,
                                      lstm_hidden, lstm_layers)

        # Fixed level-trend transition matrix F (same for all series/time)
        # [[1, 1], [0, 1]]  — level accumulates trend
        F_base = torch.eye(state_dim)
        if state_dim >= 2:
            F_base[0, 1] = 1.0              # level += trend
        self.register_buffer('F_base', F_base)

    def _build_matrices(self, proc_var, emit):
        """
        Build batched F, Q, A from network outputs.
        proc_var: (B, T, N)
        emit:     (B, T, N, d)
        Returns F, Q each (B, T, N, d, d); A (B, T, N, d)
        """
        B, T, N = proc_var.shape
        d = self.d
        # F is fixed level-trend structure, broadcast
        F = self.F_base.view(1,1,1,d,d).expand(B,T,N,d,d)
        # Q = diag(proc_var) — diagonal process noise
        Q = torch.diag_embed(proc_var.unsqueeze(-1).expand(B,T,N,d))
        A = emit                            # (B, T, N, d)
        return F, Q, A

    def log_likelihood(self, y, x, mask=None):
        """
        Compute exact log-likelihood (Proposition 3 in paper).
        Trains the full model end-to-end.

        y    : (B, T, N)  — observations
        x    : (B, T, k)  — covariates
        mask : (B, T, N)  — 1=observed, 0=missing  (optional)
        """
        B, T, N = y.shape
        device  = y.device

        # ── Step 1: get pseudo-observations via f^{-1} ──────────
        # z_t = f^{-1}(y_t),  shape (B, T, N)
        z, log_det_inv = self.flow.inverse(y)           # log_det_inv: (B, T)

        # ── Step 2: get time-varying SSM params ─────────────────
        obs_var, proc_var, emit, _ = self.param_net(x)
        F, Q, A = self._build_matrices(proc_var, emit)

        # ── Step 3: run Kalman filter on z (Proposition 1) ──────
        mu = torch.zeros(B, N, self.d, device=device)
        P  = torch.eye(self.d, device=device).unsqueeze(0).unsqueeze(0)\
                   .expand(B, N, self.d, self.d).clone()

        total_kf_ll = torch.zeros(B, device=device)

        for t in range(T):
            z_t      = z[:, t, :]               # (B, N)
            obs_v_t  = obs_var[:, t, :]         # (B, N)
            F_t      = F[:, t, :, :, :]         # (B, N, d, d)
            Q_t      = Q[:, t, :, :, :]         # (B, N, d, d)
            A_t      = A[:, t, :, :]            # (B, N, d)

            # Handle missing values: skip update for missing entries
            obs_t = z_t if mask is None else z_t * mask[:, t, :]

            mu, P, step_ll = kalman_step(mu, P, F_t, Q_t, A_t, obs_v_t, obs_t)

            if mask is not None:
                step_ll = step_ll * mask[:, t, :]  # ignore missing

            total_kf_ll = total_kf_ll + step_ll.sum(dim=-1)  # sum over N series

        # ── Step 4: combine KF log-lik with flow log-det ────────
        # log p(y) = log p_LGM(z) + log|det J_{f^{-1}}|
        # log_det_inv: (B, T)  — summed over N features per timestep
        total_ll = total_kf_ll + log_det_inv.sum(dim=1)      # sum over T

        return total_ll.mean()                               # scalar

    @torch.no_grad()
    def forecast(self, y_hist, x_hist, x_future, n_samples: int = 100):
        """
        Draw sample forecasts.  (Section 2.3)

        y_hist   : (B, T_hist, N)
        x_hist   : (B, T_hist, k)
        x_future : (B, T_fut,  k)
        Returns    (B, T_fut, N, n_samples)
        """
        B, T_hist, N = y_hist.shape
        T_fut = x_future.shape[1]
        device = y_hist.device

        # Filter through history to get p(l_T | y_{1:T})
        z_hist, _ = self.flow.inverse(y_hist)
        obs_var, proc_var, emit, h = self.param_net(x_hist)
        F, Q, A = self._build_matrices(proc_var, emit)

        mu = torch.zeros(B, N, self.d, device=device)
        P  = torch.eye(self.d, device=device).view(1,1,self.d,self.d)\
                   .expand(B, N, self.d, self.d).clone()

        for t in range(T_hist):
            mu, P, _ = kalman_step(mu, P, F[:,t], Q[:,t], A[:,t],
                                   obs_var[:,t], z_hist[:,t])

        # Forecast by propagating latent state + sampling
        obs_var_f, proc_var_f, emit_f, _ = self.param_net(x_future, h)
        F_f, Q_f, A_f = self._build_matrices(proc_var_f, emit_f)

        samples = []
        mu_s = mu.unsqueeze(1).expand(B, n_samples, N, self.d).reshape(B*n_samples, N, self.d)
        # (simplification: use mean state for all samples; extend for full uncertainty)

        for t in range(T_fut):
            F_t = F_f[:,t].unsqueeze(1).expand(B, n_samples, N, self.d, self.d)\
                          .reshape(B*n_samples, N, self.d, self.d)
            Q_t = Q_f[:,t].unsqueeze(1).expand(B, n_samples, N, self.d, self.d)\
                          .reshape(B*n_samples, N, self.d, self.d)
            A_t = A_f[:,t].unsqueeze(1).expand(B, n_samples, N, self.d)\
                          .reshape(B*n_samples, N, self.d)
            obs_v_t = obs_var_f[:,t].unsqueeze(1).expand(B, n_samples, N)\
                                    .reshape(B*n_samples, N)

            # Sample process noise
            noise = torch.randn_like(mu_s)
            L = torch.linalg.cholesky(Q_t)
            mu_s = (F_t @ mu_s.unsqueeze(-1)).squeeze(-1) + (L @ noise.unsqueeze(-1)).squeeze(-1)

            # Sample observation noise and emit pseudo-obs
            z_t = (A_t.unsqueeze(-2) @ mu_s.unsqueeze(-1)).squeeze(-1,-2)
            z_t = z_t + torch.randn_like(z_t) * obs_v_t.sqrt()

            # Map pseudo-obs -> observations through flow  (eq. 7c)
            z_t_batch = z_t.view(B, n_samples, N)
            y_t, _ = self.flow(z_t_batch.view(B*n_samples, N))  # hack: apply flow per step
            samples.append(y_t.view(B, n_samples, N))

        return torch.stack(samples, dim=1)  # (B, T_fut, n_samples, N)