"""Tabular PyTorch MLP, used to compare against the gradient boosters.

A feed-forward net with BatchNorm, dropout and early stopping. It's here to check
whether deep learning beats boosting on this data (it doesn't). The
``KMP_DUPLICATE_LIB_OK`` line works around the Windows/Anaconda OpenMP clash
before torch is imported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E402  (must precede torch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from bmw_sales.config import get_settings  # noqa: E402
from bmw_sales.models.evaluate import (  # noqa: E402
    classification_metrics,
    regression_metrics,
)
from bmw_sales.models.preprocessing import Dataset, build_preprocessor  # noqa: E402


class TabularMLP(nn.Module):
    """A compact feed-forward net for tabular data (BatchNorm + dropout)."""

    def __init__(self, in_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden:
            layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class DLResult:
    """Held-out metrics for the DL model plus training metadata."""

    task: str
    metrics: dict[str, float]
    epochs_run: int
    n_params: int


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(arr, dtype=np.float32))


def train_tabular_nn(
    dataset: Dataset,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 6,
) -> DLResult:
    """Train the MLP on ``dataset`` with early stopping; return held-out metrics."""
    seed = get_settings().random_seed
    _set_seed(seed)

    pre = build_preprocessor(dataset.numeric, dataset.categorical)
    x_train = pre.fit_transform(dataset.X_train)
    x_val = pre.transform(dataset.X_val)
    x_test = pre.transform(dataset.X_test)

    is_reg = dataset.task == "regression"
    y_train = dataset.y_train.to_numpy(dtype=np.float32)
    y_val = dataset.y_val.to_numpy(dtype=np.float32)
    y_test = dataset.y_test.to_numpy(dtype=np.float32)

    # Standardise the regression target for stable optimisation; invert later.
    y_mean, y_std = (float(y_train.mean()), float(y_train.std() or 1.0)) if is_reg else (0.0, 1.0)
    y_train_t = (y_train - y_mean) / y_std

    train_loader = DataLoader(
        TensorDataset(_to_tensor(x_train), _to_tensor(y_train_t)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = TabularMLP(in_features=x_train.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn: nn.Module = nn.MSELoss() if is_reg else nn.BCEWithLogitsLoss()

    x_val_t, y_val_t = _to_tensor(x_val), _to_tensor((y_val - y_mean) / y_std)
    best_val = float("inf")
    best_state = model.state_dict()
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t))
        if val_loss < best_val - 1e-4:
            best_val, best_state, epochs_no_improve = val_loss, model.state_dict(), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(_to_tensor(x_test))
        if is_reg:
            pred = raw.numpy() * y_std + y_mean
            metrics = regression_metrics(y_test, pred).as_dict()
        else:
            proba = torch.sigmoid(raw).numpy()
            pred = (proba >= 0.5).astype(int)
            metrics = classification_metrics(y_test.astype(int), pred, proba).as_dict()

    n_params = sum(p.numel() for p in model.parameters())
    return DLResult(dataset.task, metrics, epochs_run, n_params)
