from model import NKF
import torch


def train_nkf(model, train_loader, n_epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.
        for y, x, mask in train_loader:
            # y: (B, T, N), x: (B, T, k), mask: (B, T, N)
            optimizer.zero_grad()
            loss = -model.log_likelihood(y, x, mask)   # maximize LL
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{n_epochs}  NLL: {total_loss/len(train_loader):.4f}")


N = 30          # 20 stocks + 10 commodities
k = 10          # covariates: news sentiment scores, calendar features, etc.

model = NKF(
    N              = N,
    covariate_dim  = k,
    state_dim      = 2,     # level + trend per series
    flow_layers    = 4,     # RealNVP depth
    flow_hidden    = 64,
    lstm_hidden    = 64,
    lstm_layers    = 2,
)