"""Monte-Carlo risk and touch-probability simulation for a single underlying.

This answers three distinct questions that are easy to conflate:

* **Touch / first-passage** — P(the stock reaches a price *at any point* before T).
  This is what "will it hit $X" usually means and is the relevant number for
  American-style assignment. It's materially larger than the terminal probability.
* **Terminal** — P(S_T is above/below a level *at* expiry T). This is the
  assignment probability of a European write and matches ``blackscholes.prob_otm``.
* **Risk** — the shape of the loss distribution: VaR, CVaR / expected shortfall,
  and the max-drawdown distribution over the path.

For GBM both probabilities have closed forms (the touch probability via the
reflection principle), which we keep as a cross-check. The Monte-Carlo engine
goes beyond GBM with fat-tailed (Student-t), volatility-clustering (GARCH-t) and
non-parametric (block-bootstrap) path models, and is the part that scales on a
GPU. It runs on PyTorch+CUDA when available and falls back to NumPy on CPU; the
math is identical, only the backend differs.

Conventions: we simulate in **trading days** (dt = 1/252) with an annualized
drift ``mu`` and vol ``sigma``. A calendar ``horizon_days`` (to match option DTE)
is converted to trading steps, and the reported ``t_years`` is trading-time
(steps / 252) so the closed-form and the simulation use an identical horizon.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t_dist

from . import expected_return

TRADING_DAYS = 252.0
DT = 1.0 / TRADING_DAYS


# --------------------------------------------------------------------------- #
# Backends: NumPy (CPU) and PyTorch (GPU). Both expose the same tiny op set so
# the simulation loop is written once.
# --------------------------------------------------------------------------- #
class _NumpyBackend:
    name = "cpu (numpy)"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def normal(self, n):
        return self.rng.standard_normal(n)

    def student_t(self, n, nu):
        # Scale to unit variance: a raw t(nu) has variance nu/(nu-2).
        return self.rng.standard_t(nu, size=n) * math.sqrt((nu - 2.0) / nu)

    def randint(self, high, n):
        return self.rng.integers(0, max(high, 1), size=n)

    def gather(self, arr, idx):
        return arr[idx]

    def zeros(self, n):
        return np.zeros(n)

    def full(self, n, v):
        return np.full(n, float(v))

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def exp(self, a):
        return np.exp(a)

    def sqrt(self, a):
        return np.sqrt(a)

    def asarray(self, a):
        return np.asarray(a, dtype=float)

    def to_numpy(self, a):
        return np.asarray(a, dtype=float)


class _TorchBackend:
    def __init__(self, seed: int, device: str):
        import torch  # noqa: PLC0415 - optional dependency

        self.torch = torch
        self.device = device
        self.dtype = torch.float32
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        self.name = f"gpu ({torch.cuda.get_device_name(device)})" if device.startswith("cuda") \
            else "cpu (torch)"

    def normal(self, n):
        return self.torch.randn(n, device=self.device, dtype=self.dtype)

    def student_t(self, n, nu):
        t = self.torch
        z = t.randn(n, device=self.device, dtype=self.dtype)
        # chi2(nu) = Gamma(shape=nu/2, rate=1/2); raw t = z / sqrt(chi2/nu).
        g = t.distributions.Gamma(
            t.tensor(nu / 2.0, device=self.device),
            t.tensor(0.5, device=self.device),
        ).sample((n,)).to(self.dtype)
        raw = z / t.sqrt(g / nu)
        return raw * math.sqrt((nu - 2.0) / nu)

    def randint(self, high, n):
        return self.torch.randint(0, max(high, 1), (n,), device=self.device)

    def gather(self, arr, idx):
        return arr[idx]

    def zeros(self, n):
        return self.torch.zeros(n, device=self.device, dtype=self.dtype)

    def full(self, n, v):
        return self.torch.full((n,), float(v), device=self.device, dtype=self.dtype)

    def maximum(self, a, b):
        return self.torch.maximum(a, b)

    def minimum(self, a, b):
        return self.torch.minimum(a, b)

    def exp(self, a):
        return self.torch.exp(a)

    def sqrt(self, a):
        return self.torch.sqrt(a)

    def asarray(self, a):
        return self.torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def to_numpy(self, a):
        return a.detach().to("cpu").numpy().astype(float)


def select_backend(seed: int = 12345, *, force_cpu: bool = False):
    """Pick a GPU backend when one is available, else CPU NumPy."""
    if not force_cpu:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return _TorchBackend(seed, "cuda")
        except Exception:
            pass
    return _NumpyBackend(seed)


# --------------------------------------------------------------------------- #
# Closed-form GBM references (real-world drift mu, not risk-neutral).
# --------------------------------------------------------------------------- #
def terminal_prob_above_gbm(spot: float, level: float, t_years: float,
                            mu: float, sigma: float) -> float:
    """P(S_T >= level) under GBM with drift ``mu``."""
    if t_years <= 0 or sigma <= 0:
        return 1.0 if spot >= level else 0.0
    nu = mu - 0.5 * sigma * sigma
    d = (math.log(spot / level) + nu * t_years) / (sigma * math.sqrt(t_years))
    return float(norm.cdf(d))


def first_passage_prob_gbm(spot: float, barrier: float, t_years: float,
                           mu: float, sigma: float) -> float:
    """P(the path touches ``barrier`` at any time in [0, T]) under GBM.

    Reflection-principle result for the running max/min of a drifted Brownian
    motion. Auto-detects an up-barrier (barrier > spot) vs a down-barrier.
    """
    if t_years <= 0 or sigma <= 0:
        return 1.0 if barrier == spot else 0.0
    nu = mu - 0.5 * sigma * sigma          # log-price drift
    s = sigma * math.sqrt(t_years)
    b = math.log(barrier / spot)           # >0 up, <0 down, =0 already there
    if b == 0:
        return 1.0
    if b > 0:                              # P(running max >= b)
        return float(norm.cdf((nu * t_years - b) / s)
                     + math.exp(2 * nu * b / (sigma * sigma)) * norm.cdf((-nu * t_years - b) / s))
    # down-barrier: P(running min <= b), b < 0
    return float(norm.cdf((b - nu * t_years) / s)
                 + math.exp(2 * nu * b / (sigma * sigma)) * norm.cdf((b + nu * t_years) / s))


# --------------------------------------------------------------------------- #
# GARCH(1,1) maximum-likelihood fit.
# --------------------------------------------------------------------------- #
@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    nu: float | None          # Student-t dof; None for a normal fit
    last_var: float           # filtered conditional variance at the last obs
    last_eps2: float          # last squared residual (demeaned)
    mean: float               # sample mean daily return
    loglik: float
    dist: str

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def uncond_var(self) -> float:
        p = self.persistence
        return self.omega / (1.0 - p) if p < 1.0 else self.last_var


def _garch_filter(omega, alpha, beta, resid):
    """Run the variance recursion; return the filtered variance series."""
    n = resid.shape[0]
    var = np.empty(n)
    var[0] = resid.var() if n > 1 else (resid[0] ** 2 + 1e-12)
    for i in range(1, n):
        var[i] = omega + alpha * resid[i - 1] ** 2 + beta * var[i - 1]
    return var


def fit_garch(returns: np.ndarray, *, dist: str = "t") -> GarchFit | None:
    """MLE fit of GARCH(1,1) with normal or standardized-Student-t innovations.

    Returns ``None`` if there isn't enough data or the optimiser fails, so callers
    can fall back to a constant-volatility model.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 60:
        return None
    mean = float(r.mean())
    resid = r - mean
    var0 = max(resid.var(), 1e-10)

    def nll(params):
        omega, alpha, beta = params[0], params[1], params[2]
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e12
        var = _garch_filter(omega, alpha, beta, resid)
        if not np.all(np.isfinite(var)) or np.any(var <= 0):
            return 1e12
        if dist == "normal":
            ll = norm.logpdf(resid / np.sqrt(var)) - 0.5 * np.log(var)
        else:
            nu = params[3]
            if nu <= 2.01:
                return 1e12
            lam = math.sqrt((nu - 2.0) / nu)              # std-t scale
            z = resid / (np.sqrt(var) * lam)
            ll = student_t_dist.logpdf(z, nu) - np.log(np.sqrt(var) * lam)
        s = ll.sum()
        return -s if np.isfinite(s) else 1e12

    if dist == "normal":
        x0 = [0.05 * var0, 0.05, 0.90]
        bounds = [(1e-12, None), (0.0, 0.999), (0.0, 0.999)]
    else:
        x0 = [0.05 * var0, 0.05, 0.90, 6.0]
        bounds = [(1e-12, None), (0.0, 0.999), (0.0, 0.999), (2.05, 60.0)]

    try:
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    except Exception:
        return None
    if not res.success and res.fun >= 1e11:
        return None

    omega, alpha, beta = float(res.x[0]), float(res.x[1]), float(res.x[2])
    nu = float(res.x[3]) if dist != "normal" else None
    var = _garch_filter(omega, alpha, beta, resid)
    return GarchFit(
        omega=omega, alpha=alpha, beta=beta, nu=nu,
        last_var=float(var[-1]), last_eps2=float(resid[-1] ** 2),
        mean=mean, loglik=float(-res.fun), dist=dist,
    )


# --------------------------------------------------------------------------- #
# The Monte-Carlo engine (single streaming loop, no full path stored).
# --------------------------------------------------------------------------- #
def _simulate_core(b, *, model, spot, n_paths, n_steps, mu, sigma,
                   nu_t, garch, emp, block, recenter):
    """Step the ensemble forward, tracking terminal price, running max/min and
    max drawdown per path. Returns those four arrays (length ``n_paths``)."""
    price = b.full(n_paths, spot)
    peak = b.full(n_paths, spot)
    trough = b.full(n_paths, spot)
    max_dd = b.zeros(n_paths)

    mu_daily = mu * DT

    if model in ("gbm", "t"):
        drift = (mu - 0.5 * sigma * sigma) * DT
        vol = sigma * math.sqrt(DT)
    elif model == "garch":
        var = b.full(n_paths, garch.last_var)
        eps_prev2 = b.full(n_paths, garch.last_eps2)
    elif model == "bootstrap":
        adj = emp - emp.mean() + mu_daily if recenter else emp
        emp_arr = b.asarray(adj)
        n_emp = len(adj)
        src = None
    else:
        raise ValueError(f"unknown model {model!r}")

    for step in range(n_steps):
        if model == "gbm":
            incr = drift + vol * b.normal(n_paths)
        elif model == "t":
            incr = drift + vol * b.student_t(n_paths, nu_t)
        elif model == "garch":
            var = garch.omega + garch.alpha * eps_prev2 + garch.beta * var
            z = b.student_t(n_paths, garch.nu) if garch.dist != "normal" else b.normal(n_paths)
            eps = b.sqrt(var) * z
            eps_prev2 = eps * eps
            incr = mu_daily + eps
        else:  # bootstrap
            offset = step % block
            if offset == 0:
                src = b.randint(n_emp - block + 1, n_paths)
            incr = b.gather(emp_arr, src + offset)

        price = price * b.exp(incr)
        peak = b.maximum(peak, price)
        trough = b.minimum(trough, price)
        max_dd = b.maximum(max_dd, 1.0 - price / peak)

    return (b.to_numpy(price), b.to_numpy(peak),
            b.to_numpy(trough), b.to_numpy(max_dd))


@dataclass
class Simulation:
    """Outcome of a path ensemble. Query touch/terminal probabilities for any
    barrier and risk stats off the stored per-path arrays."""

    model: str
    spot: float
    mu: float
    sigma: float
    horizon_days: int
    n_steps: int
    t_years: float
    backend: str
    terminal: np.ndarray
    path_max: np.ndarray
    path_min: np.ndarray
    max_dd: np.ndarray
    garch: GarchFit | None = field(default=None)

    # -- probabilities ------------------------------------------------------ #
    def prob_touch(self, barrier: float, *, continuity_correct: bool = True) -> float:
        """P(path reaches ``barrier`` at any point) — running max above spot, min below.

        Daily steps only see closes, so they undercount intraday barrier crossings.
        The Broadie-Glasserman-Kou correction shifts the barrier inward by
        ``exp(+/- 0.5826 * sigma * sqrt(dt))`` so the discrete estimate approximates
        the continuous-monitoring probability (what "does it ever hit X" means).
        """
        up = barrier >= self.spot
        if continuity_correct and self.sigma > 0 and self.n_steps > 0:
            beta = 0.5825971579  # -zeta(1/2)/sqrt(2*pi)
            shift = beta * self.sigma * math.sqrt(self.t_years / self.n_steps)
            barrier = barrier * math.exp(-shift) if up else barrier * math.exp(shift)
        if up:
            return float(np.mean(self.path_max >= barrier))
        return float(np.mean(self.path_min <= barrier))

    def prob_terminal_above(self, level: float) -> float:
        return float(np.mean(self.terminal >= level))

    def prob_terminal_below(self, level: float) -> float:
        return float(np.mean(self.terminal <= level))

    # -- distribution & risk ------------------------------------------------ #
    @property
    def returns(self) -> np.ndarray:
        return self.terminal / self.spot - 1.0

    def expected_terminal(self) -> float:
        return float(self.terminal.mean())

    def terminal_quantiles(self, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[float, float]:
        return {q: float(np.quantile(self.terminal, q)) for q in qs}

    def var(self, level: float = 0.95) -> float:
        """Value-at-Risk on the holding-period return, as a positive loss fraction."""
        return float(-np.quantile(self.returns, 1.0 - level))

    def cvar(self, level: float = 0.95) -> float:
        """Expected shortfall: mean loss in the worst ``1-level`` tail."""
        cutoff = np.quantile(self.returns, 1.0 - level)
        tail = self.returns[self.returns <= cutoff]
        return float(-tail.mean()) if tail.size else float(-cutoff)

    def drawdown_quantiles(self, qs=(0.5, 0.95, 0.99)) -> dict[float, float]:
        return {q: float(np.quantile(self.max_dd, q)) for q in qs}

    def mean_max_drawdown(self) -> float:
        return float(self.max_dd.mean())


def simulate(
    *,
    spot: float,
    returns: np.ndarray | None,
    horizon_days: int,
    model: str = "garch",
    mu: float = 0.04,
    sigma: float | None = None,
    nu_t: float | None = None,
    garch: GarchFit | None = None,
    block: int = 5,
    recenter: bool = True,
    n_paths: int = 100_000,
    seed: int = 12345,
    force_cpu: bool = False,
) -> Simulation:
    """Run a Monte-Carlo ensemble and return a :class:`Simulation`.

    ``returns`` is the trailing daily log-return series (needed for ``sigma``
    estimation and the ``garch``/``bootstrap`` models). ``mu`` and ``sigma`` are
    annualized; ``sigma`` is estimated from ``returns`` when not given.
    """
    n_steps = max(1, round(horizon_days * TRADING_DAYS / 365.0))
    t_years = n_steps / TRADING_DAYS

    if sigma is None:
        sigma = _annualized_vol(returns)
    if nu_t is None:
        nu_t = _estimate_nu(returns)

    if model == "garch" and garch is None:
        garch = fit_garch(returns) if returns is not None else None
        if garch is None:                      # not enough data / fit failed
            model = "t"
    if model == "bootstrap" and (returns is None or len(returns) < block + 1):
        model = "gbm"

    b = select_backend(seed, force_cpu=force_cpu)
    terminal, pmax, pmin, mdd = _simulate_core(
        b, model=model, spot=spot, n_paths=n_paths, n_steps=n_steps,
        mu=mu, sigma=sigma, nu_t=nu_t, garch=garch,
        emp=np.asarray(returns, dtype=float) if returns is not None else None,
        block=block, recenter=recenter,
    )
    return Simulation(
        model=model, spot=spot, mu=mu, sigma=sigma, horizon_days=horizon_days,
        n_steps=n_steps, t_years=t_years, backend=b.name,
        terminal=terminal, path_max=pmax, path_min=pmin, max_dd=mdd, garch=garch,
    )


def _annualized_vol(returns: np.ndarray | None) -> float:
    if returns is None or len(returns) < 2:
        return 0.25                            # neutral fallback
    s = float(np.std(np.asarray(returns, dtype=float), ddof=1))
    return s * math.sqrt(TRADING_DAYS) if math.isfinite(s) and s > 0 else 0.25


def _estimate_nu(returns: np.ndarray | None) -> float:
    """Crude dof from excess kurtosis: for a t, excess kurtosis = 6/(nu-4)."""
    if returns is None or len(returns) < 30:
        return 6.0
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std()
    if sd <= 0:
        return 30.0
    g = float(np.mean(((r - r.mean()) / sd) ** 4) - 3.0)   # excess kurtosis
    if g <= 0.1:
        return 30.0
    return float(min(max(6.0 / g + 4.0, 3.0), 50.0))


# --------------------------------------------------------------------------- #
# Shared per-name setup + summary (used by deepdive, screener and riskscan so a
# name's drift and GARCH fit are computed once, then reused across every expiry).
# --------------------------------------------------------------------------- #
def prepare_drift_and_garch(*, info, price, dividend_yield, returns, settings, sim_model):
    """Estimate the valuation drift and (for the garch model) fit GARCH once.

    The returned ``garch`` fit is horizon-independent, so a single fit feeds every
    expiry's simulation for that name.
    """
    drift = expected_return.estimate(
        info, price, dividend_yield,
        model=settings.drift_model, rf=settings.risk_free_rate,
        erp=settings.equity_risk_premium, pe_anchor=settings.pe_anchor,
        reversion_years=settings.pe_reversion_years, shrink=settings.pe_reversion_shrink,
    )
    garch = None
    if sim_model == "garch" and returns is not None and len(returns):
        garch = fit_garch(returns)
    return drift, garch


def summarize_name(sim: "Simulation", drift, *, target_pct: float = 0.05,
                   rf: float = 0.04, var_level: float = 0.95) -> dict:
    """A one-row risk summary for the universe scan: drift, vol, touch odds, tail."""
    up = sim.spot * (1.0 + target_pct)
    dn = sim.spot * (1.0 - target_pct)
    sharpe = (drift.annual_drift - rf) / sim.sigma if sim.sigma > 0 else None
    return {
        "spot": round(sim.spot, 2),
        "drift": round(drift.annual_drift, 4),
        "sigma": round(sim.sigma, 4),
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "exp_return": round(sim.expected_terminal() / sim.spot - 1.0, 4),
        "prob_up": round(sim.prob_touch(up), 4),
        "prob_down": round(sim.prob_touch(dn), 4),
        "prob_term_up": round(sim.prob_terminal_above(up), 4),
        "var": round(sim.var(var_level), 4),
        "cvar": round(sim.cvar(var_level), 4),
        "mdd": round(sim.mean_max_drawdown(), 4),
        "horizon_days": sim.horizon_days,
        "model": sim.model,
    }


# --------------------------------------------------------------------------- #
# Orchestration + reporting (used by the CLI).
# --------------------------------------------------------------------------- #
def run(
    ticker: str,
    settings,
    *,
    targets: list[float] | None = None,
    target_pcts: list[float] | None = None,
    horizon_days: int | None = None,
    model: str = "garch",
    mu: float | None = None,
    drift_model: str | None = None,
    sigma: float | None = None,
    lookback: int = 504,
    block: int = 5,
    n_paths: int = 100_000,
    var_level: float = 0.95,
    seed: int = 12345,
    force_cpu: bool = False,
) -> tuple[Simulation, dict]:
    """Fetch data for ``ticker``, run the simulation, and assemble a report dict."""
    from . import data  # local import keeps the math network-free

    snap = data.get_snapshot(ticker)
    if snap is None:
        raise ValueError(f"No price/size data for {ticker!r}.")
    rets = data.daily_log_returns(ticker, lookback)

    horizon_days = horizon_days or settings.max_dte

    # Drift: an explicit --drift number wins; otherwise estimate it from the
    # underlying's fundamentals via the chosen expected-return model.
    if mu is not None:
        drift = expected_return.DriftEstimate(mu, "fixed (--drift)", {"baseline": mu}, [])
    else:
        drift = expected_return.estimate(
            snap.info, snap.price, snap.dividend_yield,
            model=drift_model or settings.drift_model,
            rf=settings.risk_free_rate, erp=settings.equity_risk_premium,
            pe_anchor=settings.pe_anchor, reversion_years=settings.pe_reversion_years,
            shrink=settings.pe_reversion_shrink,
        )
    mu = drift.annual_drift

    sim = simulate(
        spot=snap.price, returns=rets, horizon_days=horizon_days, model=model,
        mu=mu, sigma=sigma, block=block, n_paths=n_paths, seed=seed,
        force_cpu=force_cpu,
    )

    levels = list(targets or [])
    levels += [round(snap.price * (1.0 + p), 4) for p in (target_pcts or [])]
    if not levels:                             # sensible defaults: +-10%
        levels = [round(snap.price * 1.10, 2), round(snap.price * 0.90, 2)]

    report = {
        "ticker": ticker,
        "quote_type": snap.quote_type,
        "spot": snap.price,
        "drift": drift,
        "targets": _target_rows(sim, levels, var_level),
        "var_level": var_level,
    }
    return sim, report


def _target_rows(sim: Simulation, levels, var_level) -> list[dict]:
    rows = []
    for k in sorted(levels, reverse=True):
        up = k >= sim.spot
        rows.append({
            "level": k,
            "pct": k / sim.spot - 1.0,
            "direction": "up" if up else "down",
            "prob_touch_mc": sim.prob_touch(k),
            "prob_touch_gbm": first_passage_prob_gbm(sim.spot, k, sim.t_years, sim.mu, sim.sigma),
            "prob_terminal_mc": sim.prob_terminal_above(k) if up else sim.prob_terminal_below(k),
            "prob_terminal_gbm": (terminal_prob_above_gbm(sim.spot, k, sim.t_years, sim.mu, sim.sigma)
                                  if up else
                                  1.0 - terminal_prob_above_gbm(sim.spot, k, sim.t_years, sim.mu, sim.sigma)),
        })
    return rows


def render_text(sim: Simulation, report: dict) -> str:
    """Format the simulation report for the console."""
    lines = []
    g = sim.garch
    model_note = sim.model
    if sim.model == "garch" and g is not None:
        model_note += (f" (omega={g.omega:.2e}, alpha={g.alpha:.3f}, beta={g.beta:.3f}, "
                       f"persistence={g.persistence:.3f}"
                       + (f", nu={g.nu:.1f})" if g.nu else ")"))
    lines.append(
        f"\n{report['ticker']}  ({report['quote_type']})   spot ${sim.spot:.2f}   "
        f"horizon {sim.horizon_days}d ({sim.n_steps} trading steps)\n"
        f"  path model: {model_note}\n"
        f"  vol sigma={sim.sigma:.1%}/yr   paths={len(sim.terminal):,}   backend={sim.backend}"
    )
    drift = report.get("drift")
    if drift is not None:
        lines.append(f"  drift: {drift.summary()}")
        for note in drift.notes:
            lines.append(f"         note: {note}")

    lines.append("\n  Price targets (MC = simulated, GBM = closed-form cross-check):")
    lines.append(f"    {'target':>10} {'move':>8} {'dir':>5} "
                 f"{'P(touch)':>10} {'[GBM]':>8} {'P(term)':>9} {'[GBM]':>8}")
    for r in report["targets"]:
        lines.append(
            f"    {r['level']:>10.2f} {r['pct']:>+7.1%} {r['direction']:>5} "
            f"{r['prob_touch_mc']:>10.1%} {r['prob_touch_gbm']:>8.1%} "
            f"{r['prob_terminal_mc']:>9.1%} {r['prob_terminal_gbm']:>8.1%}"
        )

    q = sim.terminal_quantiles()
    lines.append("\n  Terminal price distribution:")
    lines.append(f"    expected ${sim.expected_terminal():.2f}   "
                 f"p05 ${q[0.05]:.2f}   p25 ${q[0.25]:.2f}   median ${q[0.5]:.2f}   "
                 f"p75 ${q[0.75]:.2f}   p95 ${q[0.95]:.2f}")

    lvl = report["var_level"]
    dd = sim.drawdown_quantiles()
    lines.append(f"\n  Risk over the horizon:")
    lines.append(f"    VaR({lvl:.0%}) {sim.var(lvl):.1%}   CVaR({lvl:.0%}) {sim.cvar(lvl):.1%}   "
                 f"(holding-period loss)")
    lines.append(f"    max drawdown: mean {sim.mean_max_drawdown():.1%}   "
                 f"median {dd[0.5]:.1%}   p95 {dd[0.95]:.1%}   p99 {dd[0.99]:.1%}")
    return "\n".join(lines)
