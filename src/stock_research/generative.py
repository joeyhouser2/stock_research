"""Autoregressive TCN generative model for synthetic daily-return paths.

A dilated **causal** temporal-convolution network learns the one-step-ahead
conditional distribution of daily returns, ``p(r_t | r_{t-1}, ..., r_{t-k})``,
with a **Student-t head** (conditional location, scale, degrees-of-freedom) so it
captures fat tails and volatility clustering the way GARCH-t does — but with a
richer, learned dependence on the recent path rather than a fixed recursion.

It's trained by maximum likelihood (stable, deterministically testable), and
samples paths autoregressively on the GPU. Crucially, the returns are **de-meaned**
before training, so the network learns the *shock* process and drift stays an
external, pluggable input (same convention as the GBM/GARCH engines) — you feed
the valuation drift in at sample time.

Sampled paths are wrapped in a :class:`simulate.Simulation`, so the learned model
plugs straight into the existing touch / terminal / VaR / drawdown readouts and is
scored by the same ``backtest`` harness against the GARCH baseline. PyTorch is
required (``pip install -e ".[gpu]"``); it runs on CUDA when available, else CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import simulate

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Network.
# --------------------------------------------------------------------------- #
class _CausalConv1d(nn.Module):
    """Left-padded 1-D conv so output at t depends only on inputs <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCN(nn.Module):
    """Stack of dilated causal conv blocks -> Student-t parameters per step."""

    def __init__(self, channels: int = 32, kernel: int = 2,
                 dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.res = nn.ModuleList()
        in_ch = 1
        for d in dilations:
            self.blocks.append(nn.Sequential(
                _CausalConv1d(in_ch, channels, kernel, d), nn.GELU(),
                _CausalConv1d(channels, channels, kernel, d), nn.GELU(),
            ))
            self.res.append(nn.Conv1d(in_ch, channels, 1) if in_ch != channels
                            else nn.Identity())
            in_ch = channels
        self.head = nn.Conv1d(channels, 3, 1)     # loc, log_scale, log_df
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)

    def forward(self, x):                          # x: (B, 1, L)
        h = x
        for block, res in zip(self.blocks, self.res):
            h = block(h) + res(h)
        out = self.head(h)                         # (B, 3, L)
        loc = out[:, 0, :]
        scale = F.softplus(out[:, 1, :]) + 1e-4
        df = 2.0 + F.softplus(out[:, 2, :])        # df > 2 -> finite variance
        return loc, scale, df


@dataclass
class TCNModel:
    """A trained network plus the standardization needed to use it.

    For a single-name fit, ``mean``/``std`` are that name's. For a pooled fit the
    net is name-agnostic (mean=0, std=1) and per-name standardization comes from
    ``scalers`` (ticker -> (mean, std)), applied at sample time.
    """

    net: TCN
    mean: float                 # per-step return mean removed before training
    std: float                  # per-step return std used to standardize
    device: str
    receptive_field: int
    final_loss: float
    scalers: dict | None = None

    @property
    def annual_vol(self) -> float:
        return self.std * math.sqrt(TRADING_DAYS)


def _select_device(force_cpu: bool) -> str:
    return "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"


# --------------------------------------------------------------------------- #
# Training.
# --------------------------------------------------------------------------- #
def _train_net(
    z_series: list[np.ndarray],
    *,
    channels: int,
    kernel: int,
    dilations,
    window: int,
    batch: int,
    steps: int,
    lr: float,
    seed: int,
    device: str,
    verbose: bool,
) -> tuple[TCN, float]:
    """Train a TCN by MLE on a list of (already standardized) return series.

    Each minibatch sample draws a window from a randomly chosen series (weighted
    by length), so pooled training never bridges two names within one window.
    """
    tensors = [torch.from_numpy(z.astype(np.float32)).to(device)
               for z in z_series if len(z) >= window + 2]
    if not tensors:
        raise ValueError(f"no series with >= {window + 2} returns to train on")
    lengths = np.array([t.shape[0] for t in tensors], dtype=float)
    weights = lengths / lengths.sum()

    torch.manual_seed(seed)
    net = TCN(channels, kernel, dilations).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    last = float("nan")
    net.train()
    for step in range(steps):
        sidx = rng.choice(len(tensors), size=batch, p=weights)
        xs, ys = [], []
        for si in sidx:
            t = tensors[si]
            s = int(rng.integers(0, t.shape[0] - window - 1))
            xs.append(t[s:s + window])
            ys.append(t[s + 1:s + window + 1])
        x = torch.stack(xs).unsqueeze(1)            # (B, 1, W)
        y = torch.stack(ys)                         # (B, W)
        loc, scale, df = net(x)
        nll = -torch.distributions.StudentT(df, loc, scale).log_prob(y).mean()
        opt.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        last = float(nll.item())
        if verbose and (step % 100 == 0 or step == steps - 1):
            print(f"  step {step:4d}  nll {last:.4f}")

    net.eval()
    return net, last


def _standardize(returns: np.ndarray) -> tuple[np.ndarray, float, float]:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    mean, std = float(r.mean()), float(r.std())
    std = std if std > 0 else 1.0
    return (r - mean) / std, mean, std


def fit_tcn(
    returns: np.ndarray,
    *,
    channels: int = 32,
    kernel: int = 2,
    dilations=(1, 2, 4, 8, 16),
    window: int = 64,
    batch: int = 128,
    steps: int = 800,
    lr: float = 1e-3,
    seed: int = 0,
    force_cpu: bool = False,
    verbose: bool = False,
) -> TCNModel:
    """Fit the TCN to a single 1-D daily-return series by maximum likelihood."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < window + 2:
        raise ValueError(f"need >= {window + 2} returns to train, got {r.size}")
    z, mean, std = _standardize(r)
    device = _select_device(force_cpu)
    net, last = _train_net([z], channels=channels, kernel=kernel, dilations=dilations,
                           window=window, batch=batch, steps=steps, lr=lr, seed=seed,
                           device=device, verbose=verbose)
    return TCNModel(net=net, mean=mean, std=std, device=device,
                    receptive_field=net.receptive_field, final_loss=last)


def fit_tcn_pooled(
    returns_by_ticker: dict,
    *,
    channels: int = 32,
    kernel: int = 2,
    dilations=(1, 2, 4, 8, 16),
    window: int = 64,
    batch: int = 128,
    steps: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
    force_cpu: bool = False,
    verbose: bool = False,
) -> TCNModel:
    """Train one TCN on returns pooled across many names.

    Each name is standardized by its OWN (mean, std) before pooling, so the net
    learns universal standardized dynamics. Those per-name scalers are stored on
    the model and applied at sample time. Names with too little data are dropped.
    """
    z_series, scalers = [], {}
    for ticker, returns in returns_by_ticker.items():
        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        if r.size < window + 2:
            continue
        z, mean, std = _standardize(r)
        z_series.append(z)
        scalers[ticker] = (mean, std)
    if not z_series:
        raise ValueError("no ticker had enough returns to train on")

    device = _select_device(force_cpu)
    net, last = _train_net(z_series, channels=channels, kernel=kernel, dilations=dilations,
                           window=window, batch=batch, steps=steps, lr=lr, seed=seed,
                           device=device, verbose=verbose)
    if verbose:
        print(f"  pooled {len(z_series)} names, {sum(len(z) for z in z_series):,} returns")
    return TCNModel(net=net, mean=0.0, std=1.0, device=device,
                    receptive_field=net.receptive_field, final_loss=last, scalers=scalers)


# --------------------------------------------------------------------------- #
# Sampling.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_paths(
    model: TCNModel,
    context_returns: np.ndarray,
    horizon: int,
    n_paths: int,
    *,
    mu: float = 0.0,
    seed: int = 0,
    scaler: tuple[float, float] | None = None,
) -> np.ndarray:
    """Autoregressively sample ``n_paths`` real return paths of length ``horizon``.

    The standardized context (last ``receptive_field`` returns) seeds every path;
    each step samples from the learned Student-t and slides the window. Drift is
    added back as ``mu/252`` per step (the network models zero-mean shocks).
    ``scaler`` = (mean, std) overrides the model's own standardization — required
    for a pooled model, where it's the target name's train-set scaler.
    """
    mean, std = scaler if scaler is not None else (model.mean, model.std)
    rf = model.receptive_field
    ctx = np.asarray(context_returns, dtype=np.float64)
    ctx = ctx[np.isfinite(ctx)]
    z_ctx = (ctx[-rf:] - mean) / std
    if z_ctx.size < rf:                                  # left-pad short context
        z_ctx = np.concatenate([np.zeros(rf - z_ctx.size), z_ctx])

    torch.manual_seed(seed)
    dev = model.device
    buf = torch.tensor(z_ctx, dtype=torch.float32, device=dev).repeat(n_paths, 1).unsqueeze(1)
    out = torch.empty((n_paths, horizon), device=dev)
    for t in range(horizon):
        loc, scale, df = model.net(buf)
        dist = torch.distributions.StudentT(df[:, -1], loc[:, -1], scale[:, -1])
        z = dist.sample()
        out[:, t] = z
        buf = torch.cat([buf[:, :, 1:], z.view(n_paths, 1, 1)], dim=2)

    z_paths = out.cpu().numpy().astype(np.float64)
    return z_paths * std + mu / TRADING_DAYS


def sample_simulation(
    model: TCNModel,
    *,
    spot: float,
    context_returns: np.ndarray,
    horizon_days: int,
    mu: float = 0.04,
    n_paths: int = 20_000,
    sigma: float | None = None,
    seed: int = 0,
    scaler: tuple[float, float] | None = None,
) -> simulate.Simulation:
    """Sample paths and wrap them in a :class:`simulate.Simulation`."""
    n_steps = max(1, round(horizon_days * TRADING_DAYS / 365.0))
    rets = sample_paths(model, context_returns, n_steps, n_paths, mu=mu, seed=seed,
                        scaler=scaler)
    if sigma is None:
        std = scaler[1] if scaler is not None else model.std
        sigma = std * math.sqrt(TRADING_DAYS)

    # Price paths including the spot at t0, then the running readouts.
    steps = spot * np.exp(np.cumsum(rets, axis=1))
    prices = np.concatenate([np.full((n_paths, 1), spot), steps], axis=1)
    cummax = np.maximum.accumulate(prices, axis=1)
    return simulate.Simulation(
        model="tcn", spot=spot, mu=mu, sigma=sigma,
        horizon_days=horizon_days, n_steps=n_steps, t_years=n_steps / TRADING_DAYS,
        backend=f"{model.device} (tcn)",
        terminal=prices[:, -1], path_max=prices.max(axis=1), path_min=prices.min(axis=1),
        max_dd=(1.0 - prices / cummax).max(axis=1),
    )


# --------------------------------------------------------------------------- #
# Validation: frozen TCN vs refit-GARCH on a held-out split.
# --------------------------------------------------------------------------- #
def compare_on_split(
    closes,
    *,
    train_frac: float = 0.7,
    horizon_days: int,
    target_pct: float = 0.05,
    lookback: int = 504,
    n_paths: int = 20_000,
    mu: float = 0.085,
    var_level: float = 0.95,
    seed: int = 0,
    force_cpu: bool = False,
    tcn_hp: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Train the TCN on the first ``train_frac`` of history, then backtest the
    frozen model vs a refit GARCH on the held-out tail — identical windows.

    The TCN never sees the test period and each forecast conditions only on real
    trailing returns, so this is a clean out-of-sample comparison.
    """
    from . import backtest  # local: backtest stays importable without torch

    prices = np.asarray(closes, dtype=float)
    split = int(len(prices) * train_frac)
    train_rets = np.diff(np.log(prices[:split]))
    model = fit_tcn(train_rets, seed=seed, force_cpu=force_cpu, verbose=verbose,
                    **(tcn_hp or {}))

    # Evaluate on the tail; include `lookback` pre-split context for the first window.
    test_closes = closes.iloc[max(split - lookback, 0):]

    def tcn_fn(spot, ctx_rets, horizon, sd):
        return sample_simulation(model, spot=spot, context_returns=ctx_rets,
                                 horizon_days=horizon, mu=mu, n_paths=n_paths, seed=sd)

    common = dict(horizon_days=horizon_days, target_pct=target_pct, lookback=lookback,
                  n_paths=n_paths, mu=mu, var_level=var_level, seed=seed,
                  force_cpu=force_cpu)
    garch_recs = backtest.backtest_series(test_closes, model="garch", **common)
    tcn_recs = backtest.backtest_series(test_closes, simulate_fn=tcn_fn, **common)
    return {
        "model": model,
        "windows": len(tcn_recs),
        "garch": backtest.score(garch_recs, var_level=var_level),
        "tcn": backtest.score(tcn_recs, var_level=var_level),
    }


def compare_universe(
    closes_by_ticker: dict,
    *,
    train_frac: float = 0.7,
    horizon_days: int,
    target_pct: float = 0.05,
    lookback: int = 504,
    n_paths: int = 20_000,
    mu: float = 0.085,
    var_level: float = 0.95,
    seed: int = 0,
    force_cpu: bool = False,
    tcn_hp: dict | None = None,
    verbose: bool = False,
) -> dict:
    """The decisive test: train ONE TCN pooled across names, then compare the
    frozen pooled model vs a per-name refit GARCH on held-out windows.

    A single global split DATE separates train from test for every name, so the
    pooled model never sees any name's test period (no cross-sectional leakage),
    and per-name scalers come from train data only.
    """
    from . import backtest

    h = max(1, round(horizon_days * TRADING_DAYS / 365.0))
    starts = [c.index[0] for c in closes_by_ticker.values()]
    ends = [c.index[-1] for c in closes_by_ticker.values()]
    split_date = min(starts) + (max(ends) - min(starts)) * train_frac

    # Pre-split returns per name -> pooled training set (scalers stored on model).
    train_returns = {}
    for ticker, closes in closes_by_ticker.items():
        pre = closes[closes.index < split_date]
        if len(pre) > lookback // 2:
            train_returns[ticker] = np.diff(np.log(pre.values))
    if len(train_returns) < 2:
        raise ValueError("need >= 2 names with pre-split history")

    model = fit_tcn_pooled(train_returns, seed=seed, force_cpu=force_cpu,
                           verbose=verbose, **(tcn_hp or {}))

    garch_recs, tcn_recs, n_used = [], [], 0
    for ticker, closes in closes_by_ticker.items():
        if ticker not in model.scalers:
            continue
        split_pos = int((closes.index < split_date).sum())
        test_closes = closes.iloc[max(split_pos - lookback, 0):]
        if len(test_closes) < lookback + h + 1:
            continue
        sc = model.scalers[ticker]

        def tcn_fn(spot, ctx, horizon, sd, sc=sc):
            return sample_simulation(model, spot=spot, context_returns=ctx,
                                     horizon_days=horizon, mu=mu, n_paths=n_paths,
                                     seed=sd, scaler=sc)

        common = dict(horizon_days=horizon_days, target_pct=target_pct, lookback=lookback,
                      n_paths=n_paths, mu=mu, var_level=var_level, seed=seed,
                      force_cpu=force_cpu, ticker=ticker)
        garch_recs += backtest.backtest_series(test_closes, model="garch", **common)
        tcn_recs += backtest.backtest_series(test_closes, simulate_fn=tcn_fn, **common)
        n_used += 1
        if verbose:
            print(f"  tested {ticker}: {len(tcn_recs)} pooled windows so far")

    return {
        "model": model,
        "n_tickers": n_used,
        "windows": len(tcn_recs),
        "garch": backtest.score(garch_recs, var_level=var_level),
        "tcn": backtest.score(tcn_recs, var_level=var_level),
    }
