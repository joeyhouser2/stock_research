"""Streamlit dashboard for stock_research.

Run with:  streamlit run app.py

Every tab is a thin wrapper over the same functions the CLI uses, so the UI and
CLI stay in lockstep. Network/GPU work runs only on an explicit "Run" button and
is stashed in session_state so results persist as you tweak other widgets.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

# Make `stock_research` importable when running from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from stock_research import (config, deepdive, longcall, riskscan, screener, simulate,
                            valuescan)
from stock_research.config import load_universe

st.set_page_config(page_title="stock_research", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in re.split(r"[,\s]+", raw or "") if t.strip()]


def universe_input(key: str, default: str = "AAPL MSFT NVDA AMZN GOOGL"):
    """Ticker box + 'use universe.yaml' toggle. Returns (tickers, use_universe).

    ``key`` namespaces the widgets so the same helper can appear on several tabs.
    """
    use_uni = st.checkbox("Use config/universe.yaml", value=False, key=f"{key}_useuni")
    raw = st.text_input("Tickers", value=default, disabled=use_uni, key=f"{key}_tickers")
    if use_uni:
        try:
            return load_universe(), True
        except Exception:
            st.warning("Could not load universe.yaml; using the text box.")
    return parse_tickers(raw), False


def download_df(df: pd.DataFrame, name: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", buf.getvalue(), file_name=name, mime="text/csv",
                       key=f"dl_{name}")


def sidebar_settings() -> config.Settings:
    """Global model settings shared by every tab."""
    st.sidebar.header("Model settings")
    base = config.load_settings()
    with st.sidebar.expander("Risk / DTE / OTM", expanded=False):
        rf = st.number_input("Risk-free rate", 0.0, 0.20, float(base.risk_free_rate), 0.005, format="%.3f")
        min_dte = st.number_input("Min DTE", 1, 365, int(base.min_dte))
        max_dte = st.number_input("Max DTE", 1, 365, int(base.max_dte))
        min_otm = st.number_input("Min OTM", 0.0, 0.5, float(base.min_otm), 0.01, format="%.2f")
        max_otm = st.number_input("Max OTM", 0.0, 1.0, float(base.max_otm), 0.01, format="%.2f")
    with st.sidebar.expander("Valuation drift", expanded=False):
        drift_model = st.selectbox("Drift model", ["fundamental", "fixed", "analyst", "blend"],
                                   index=["fundamental", "fixed", "analyst", "blend"].index(base.drift_model))
        erp = st.number_input("Equity risk premium", 0.0, 0.20, float(base.equity_risk_premium), 0.005, format="%.3f")
        rev_years = st.number_input("P/E reversion years", 0.5, 20.0, float(base.pe_reversion_years), 0.5)
        rev_shrink = st.number_input("Reversion shrink", 0.0, 1.0, float(base.pe_reversion_shrink), 0.1)
    return config.override(
        base, risk_free_rate=rf, min_dte=int(min_dte), max_dte=int(max_dte),
        min_otm=min_otm, max_otm=max_otm, drift_model=drift_model,
        equity_risk_premium=erp, pe_reversion_years=rev_years, pe_reversion_shrink=rev_shrink,
    )


def _gpu_status():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return "GPU: none (CPU)"
    except Exception:
        return "torch not installed (CPU only)"


def reliability_chart(rel, title: str):
    st.markdown(f"**{title}** — Brier `{rel.brier:.4f}`, ECE `{rel.ece:.4f}`, n={rel.n}")
    if not rel.table:
        return
    tdf = pd.DataFrame(rel.table)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.plot(tdf["mean_pred"], tdf["obs_freq"], "o-")
    ax.set_xlabel("predicted"); ax.set_ylabel("observed"); ax.set_title("calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    st.pyplot(fig)


# --------------------------------------------------------------------------- #
# Tabs.
# --------------------------------------------------------------------------- #
def tab_screen(settings):
    st.subheader("OTM call writing — rank tradable out-of-the-money call sales")
    tickers, use_uni = universe_input("screen")
    c1, c2, c3 = st.columns(3)
    sort_by = c1.selectbox("Sort by", screener.SORT_KEYS, key="screen_sort",
                           help="score = yield × P(OTM); score_adj also weights by IV/HV (rich vol).")
    expiry = c2.selectbox("Expiries", ["any", "weekly", "monthly"], key="screen_exp")
    min_iv_hv = c3.number_input("Min IV/HV (rich-vol gate, 0 = off)", 0.0, 3.0, 0.0, 0.1,
                                key="screen_ivhv",
                                help="Keep only contracts where IV ≥ this × realized vol. "
                                     "Pair with sort=score_adj for best bang-for-buck.")
    c4, c5 = st.columns(2)
    with_value = c4.checkbox("Value metrics", value=False, key="screen_value")
    sim_risk = c5.checkbox("Add sim columns (fat-tailed)", value=False, key="screen_simcol")
    s = config.override(settings, expiry_type=expiry,
                        min_iv_hv=(min_iv_hv if min_iv_hv > 0 else None))

    if st.button("Run screen", type="primary"):
        with st.spinner(f"Screening {len(tickers)} tickers..."):
            st.session_state["screen"] = screener.run(
                tickers, s, sort_by=sort_by, with_value=with_value,
                simulate_risk=sim_risk, verbose=False)
    df = st.session_state.get("screen")
    if df is not None:
        st.caption(f"{len(df)} ranked contracts")
        st.dataframe(df, use_container_width=True, height=520)
        download_df(df, "screen.csv")


def tab_riskscan(settings):
    st.subheader("Risk scan — rank underlyings by simulated risk/return")
    tickers, _ = universe_input("riskscan")
    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.number_input("Horizon (days)", 5, 365, int(settings.max_dte), key="rs_h")
    target_pct = c2.number_input("Touch band ±", 0.01, 0.5, 0.05, 0.01, format="%.2f", key="rs_band")
    model = c3.selectbox("Path model", ["garch", "t", "bootstrap", "gbm"], key="rs_model")
    sort_by = c4.selectbox("Sort by", riskscan.SORT_KEYS, key="rs_sort")
    paths = st.slider("Monte-Carlo paths", 5_000, 100_000, 30_000, 5_000, key="rs_paths")

    if st.button("Run risk scan", type="primary"):
        with st.spinner(f"Simulating {len(tickers)} names..."):
            st.session_state["riskscan"] = riskscan.run(
                tickers, settings, horizon_days=int(horizon), target_pct=target_pct,
                model=model, n_paths=paths, sort_by=sort_by, verbose=False)
    df = st.session_state.get("riskscan")
    if df is not None:
        st.dataframe(df, use_container_width=True, height=520)
        download_df(df, "riskscan.csv")


def tab_deepdive(settings):
    st.subheader("Deep dive — full OTM-call grid for one ticker")
    c1, c2, c3, c4 = st.columns(4)
    ticker = c1.text_input("Ticker", value="MSFT", key="dd_tk")
    expiry = c2.selectbox("Expiries", ["any", "weekly", "monthly"], key="dd_exp")
    sim_risk = c3.checkbox("Sim columns", value=True, key="dd_sim")
    with_value = c4.checkbox("Value metrics", value=True, key="dd_value")
    s = config.override(settings, expiry_type=expiry)

    if st.button("Run deep dive", type="primary"):
        with st.spinner(f"Building grid for {ticker}..."):
            try:
                grid, header = deepdive.build(ticker.upper(), s, with_value=with_value,
                                              simulate_risk=sim_risk)
                st.session_state["deepdive"] = (grid, header)
            except ValueError as exc:
                st.session_state["deepdive"] = None
                st.error(str(exc))
    out = st.session_state.get("deepdive")
    if out:
        grid, header = out
        h = header
        st.markdown(f"**{h['ticker']}** ({h['quote_type']}) — spot ${h['price']:.2f}, "
                    f"size ${h['size_b']:.1f}B, HV {h['hv']*100:.1f}%"
                    if h.get("hv") else f"**{h['ticker']}** spot ${h['price']:.2f}")
        if h.get("drift") is not None:
            st.caption(h["drift"].summary())
        if grid.empty:
            st.info("No OTM strikes in the configured DTE / OTM window.")
            return
        st.dataframe(grid, use_container_width=True, height=420)
        # Annualized yield vs %OTM by expiry.
        fig, ax = plt.subplots(figsize=(7, 4))
        for expiry_val, grp in grid.groupby("expiry"):
            ax.plot(grp["pct_otm"] * 100, grp["annual_yield"] * 100, "o-", label=str(expiry_val))
        ax.set_xlabel("% out of the money"); ax.set_ylabel("Annualized yield (%)")
        ax.legend(fontsize=7, title="expiry"); ax.grid(alpha=0.3)
        st.pyplot(fig)


def tab_value(settings):
    st.subheader("Value buys — rules-based value/quality rank + likely prices")
    st.caption("Uses current fundamentals: a live ranking lens, not a backtested model. "
               "Long-equity value/quality screen.")
    tickers, _ = universe_input(
        "value", default="AAPL MSFT NVDA AMZN GOOGL META JPM XOM KO JNJ PG WMT")
    c1, c2, c3 = st.columns(3)
    sort_by = c1.selectbox("Sort by", valuescan.SORT_KEYS, key="val_sort")
    min_cap = c2.number_input("Min cap ($B)", 0.0, 5000.0, 1.0, 0.5, key="val_mincap")
    max_cap = c3.number_input("Max cap ($B, 0 = none)", 0.0, 5000.0, 0.0, 1.0, key="val_maxcap",
                              help="Value-picker ceiling — find smaller names below this size.")
    f1, f2, f3, f4 = st.columns(4)
    max_pe = f1.number_input("Max P/E (0=off)", 0.0, 200.0, 0.0, 1.0, key="val_maxpe")
    max_peg = f2.number_input("Max PEG (0=off)", 0.0, 20.0, 0.0, 0.1, key="val_maxpeg")
    min_roe = f3.number_input("Min ROE (0=off)", 0.0, 2.0, 0.0, 0.05, key="val_minroe")
    min_upside = f4.number_input("Min upside (0=off)", 0.0, 2.0, 0.0, 0.05, key="val_minups")

    if st.button("Run value scan", type="primary"):
        s = config.override(settings, min_market_cap=min_cap * 1e9,
                            max_market_cap=(max_cap * 1e9 if max_cap > 0 else None))
        filters = {k: v for k, v in {"max_pe": max_pe or None, "max_peg": max_peg or None,
                                     "min_roe": min_roe or None, "min_upside": min_upside or None}.items()
                   if v is not None}
        with st.spinner(f"Ranking {len(tickers)} names..."):
            st.session_state["value"] = valuescan.run(tickers, s, sort_by=sort_by,
                                                      filters=filters, verbose=False)
    df = st.session_state.get("value")
    if df is None or df.empty:
        return
    st.caption(f"{len(df)} ranked names — value_score blends cheapness (50%), quality (30%), "
               "analyst upside (20%)")
    st.dataframe(df, use_container_width=True, height=420)
    download_df(df, "value.csv")

    st.markdown("### Likely price distribution")
    c1, c2, c3 = st.columns(3)
    pick = c1.selectbox("Simulate a name", df["ticker"].tolist(), key="val_pick")
    horizon = c2.number_input("Horizon (days)", 5, 365, 90, key="val_h")
    paths = c3.slider("Paths", 10_000, 200_000, 50_000, 10_000, key="val_paths")
    if st.button("Simulate price", key="val_sim_btn"):
        with st.spinner(f"Simulating {pick}..."):
            try:
                st.session_state["value_sim"] = simulate.run(
                    pick, settings, horizon_days=int(horizon), n_paths=paths)
            except ValueError as exc:
                st.session_state["value_sim"] = None
                st.error(str(exc))
    out = st.session_state.get("value_sim")
    if out:
        sim, report = out
        d = report.get("drift")
        st.markdown(f"**{report['ticker']}** spot ${sim.spot:.2f} — {sim.model}, "
                    f"vol {sim.sigma:.1%}/yr, {sim.horizon_days}d")
        if d is not None:
            st.caption(f"drift: {d.summary()}")
        q = sim.terminal_quantiles()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected", f"${sim.expected_terminal():.2f}")
        m2.metric("Median", f"${q[0.5]:.2f}")
        m3.metric("p05 – p95", f"${q[0.05]:.0f}–${q[0.95]:.0f}")
        m4.metric("VaR 95%", f"{sim.var(0.95):.1%}")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(sim.terminal, bins=80, color="#8172b3", alpha=0.85)
        ax.axvline(sim.spot, color="black", linestyle="--", label="spot")
        ax.set_xlabel("Terminal price"); ax.set_ylabel("paths"); ax.legend()
        st.pyplot(fig)


def tab_longcall(settings):
    st.subheader("Long calls — P&L of BUYING a call (vega-aware)")
    st.caption("Buying pays the volatility premium, so this gives the honest odds. "
               "Find the expiry/strike on the Deep dive or OTM tab first.")
    c1, c2, c3, c4 = st.columns(4)
    ticker = c1.text_input("Ticker", "AAPL", key="lc_tk")
    expiry = c2.text_input("Expiry (YYYY-MM-DD)", "", key="lc_exp")
    strike = c3.number_input("Strike", 0.0, 100_000.0, 0.0, key="lc_strike")
    hold = c4.number_input("Hold days (0 = to expiry)", 0, 365, 0, key="lc_hold")
    paths = st.slider("Paths", 10_000, 200_000, 50_000, 10_000, key="lc_paths")

    if st.button("Simulate long call", type="primary", key="lc_run"):
        if not expiry or strike <= 0:
            st.error("Enter an expiry date and a strike.")
        else:
            with st.spinner(f"Simulating buying {ticker} {strike:g} {expiry}..."):
                try:
                    st.session_state["longcall"] = longcall.run(
                        ticker.upper(), settings, expiry=expiry, strike=strike,
                        hold_days=(hold or None), n_paths=paths)
                except ValueError as exc:
                    st.session_state["longcall"] = None
                    st.error(str(exc))
    out = st.session_state.get("longcall")
    if not out:
        return
    res, report = out
    flag = "CHEAP" if res.iv_rv < 0.95 else "RICH" if res.iv_rv > 1.05 else "fair"
    st.markdown(f"**{report['ticker']} {res.strike:g} call exp {report['expiry']}** — "
                f"spot ${res.spot:.2f}, premium ${res.premium:.2f}, "
                f"breakeven ${res.breakeven_spot:.2f}")
    st.caption(f"IV {res.iv0:.1%} vs realized {res.rv0:.1%} → IV/RV {res.iv_rv:.2f} "
               f"[{flag}] · hold {res.hold_days}d of {res.dte}d · {res.backend}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(profit)", f"{res.prob_profit():.1%}")
    m2.metric("P(2×)", f"{res.prob_multiple(2):.1%}")
    m3.metric("P(lose ≥90%)", f"{res.prob_total_loss():.1%}")
    m4.metric("Expected return", f"{res.expected_return():+.1%}")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(res.call_value, bins=80, color="#c44e52", alpha=0.85)
    ax.axvline(res.premium, color="black", linestyle="--", label=f"premium ${res.premium:.2f}")
    ax.set_xlabel("Exit value of the call"); ax.set_ylabel("paths"); ax.legend()
    st.pyplot(fig)


def tab_simulate(settings):
    st.subheader("Simulate — touch / terminal probabilities, VaR, drawdown")
    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", value="NVDA", key="sim_tk")
    horizon = c2.number_input("Horizon (days)", 5, 365, 45, key="sim_h")
    model = c3.selectbox("Path model", ["garch", "t", "bootstrap", "gbm"], key="sim_model")
    band = st.text_input("Target moves (±, comma-separated)", value="0.10, -0.10", key="sim_band")
    paths = st.slider("Paths", 10_000, 500_000, 100_000, 10_000, key="sim_paths")
    pcts = [float(x) for x in re.split(r"[,\s]+", band) if x.strip()]

    if st.button("Run simulation", type="primary"):
        with st.spinner("Simulating..."):
            try:
                sim, report = simulate.run(
                    ticker.upper(), settings, target_pcts=pcts, horizon_days=int(horizon),
                    model=model, n_paths=paths)
                st.session_state["simulate"] = (sim, report)
            except ValueError as exc:
                st.session_state["simulate"] = None
                st.error(str(exc))
    out = st.session_state.get("simulate")
    if out:
        sim, report = out
        d = report.get("drift")
        st.markdown(f"**{report['ticker']}** spot ${sim.spot:.2f} — {sim.model}, "
                    f"vol {sim.sigma:.1%}/yr, {len(sim.terminal):,} paths, {sim.backend}")
        if d is not None:
            st.caption(f"drift: {d.summary()}")
        st.dataframe(pd.DataFrame(report["targets"]), use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected", f"${sim.expected_terminal():.2f}")
        m2.metric(f"VaR {report['var_level']:.0%}", f"{sim.var(report['var_level']):.1%}")
        m3.metric(f"CVaR {report['var_level']:.0%}", f"{sim.cvar(report['var_level']):.1%}")
        m4.metric("Mean max DD", f"{sim.mean_max_drawdown():.1%}")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(sim.terminal, bins=80, color="#4c72b0", alpha=0.85)
        ax.axvline(sim.spot, color="black", linestyle="--", label="spot")
        ax.set_xlabel("Terminal price"); ax.set_ylabel("paths"); ax.legend()
        st.pyplot(fig)


def tab_backtest(settings):
    st.subheader("Backtest — walk-forward calibration of the forecasts")
    st.caption("No look-ahead. Validates the price model (drift held flat).")
    c1, c2, c3 = st.columns(3)
    tickers = parse_tickers(c1.text_input("Tickers (pooled)", value="SPY AAPL MSFT XOM", key="bt_tks"))
    horizon = c2.number_input("Horizon (days)", 5, 120, 30, key="bt_h")
    model = c3.selectbox("Model", ["garch", "t", "bootstrap", "gbm"], key="bt_model")
    c4, c5 = st.columns(2)
    history = c4.number_input("History (days)", 600, 6000, 1825, 100, key="bt_hist")
    paths = c5.slider("Paths/window", 5_000, 50_000, 20_000, 5_000, key="bt_paths")

    if st.button("Run backtest", type="primary"):
        from stock_research import backtest
        with st.spinner("Rolling walk-forward backtest (can take a minute)..."):
            _recs, scores = backtest.run(
                tickers, settings, horizon_days=int(horizon), model=model,
                history_days=int(history), n_paths=paths, verbose=False)
            st.session_state["backtest"] = scores
    scores = st.session_state.get("backtest")
    if scores:
        if not scores["n"]:
            st.warning("No windows — not enough history.")
            return
        st.markdown(f"**{scores['n']} windows**")
        v = scores["var"]
        ok = v["kupiec_p"] == v["kupiec_p"] and v["kupiec_p"] > 0.05
        c1, c2, c3 = st.columns(3)
        c1.metric("VaR observed", f"{v['observed_rate']:.1%}", f"exp {v['expected_rate']:.1%}")
        c2.metric("Kupiec p", f"{v['kupiec_p']:.3f}", "OK" if ok else "reject")
        c3.metric("Pinball", f"{scores['pinball']['overall']:.4f}")
        a, b, c = st.columns(3)
        with a:
            reliability_chart(scores["touch_up"], "P(touch +band)")
        with b:
            reliability_chart(scores["touch_down"], "P(touch -band)")
        with c:
            reliability_chart(scores["term_up"], "P(finish above +band)")


def _scorecard_df(pairs: dict) -> pd.DataFrame:
    return pd.DataFrame(pairs)


def tab_generate(settings):
    st.subheader("Generative model — TCN, pooled, and cross-asset joint")
    st.caption(_gpu_status())
    mode = st.radio("Mode", ["Single-ticker simulate", "Single vs GARCH (compare)",
                             "Pooled universe vs GARCH", "Cross-asset joint (portfolio)"],
                    horizontal=False, key="gen_mode")
    steps = st.slider("Training steps", 300, 5000, 1500, 100, key="gen_steps")
    paths = st.slider("Paths", 5_000, 50_000, 20_000, 5_000, key="gen_paths")

    if mode == "Single-ticker simulate":
        ticker = st.text_input("Ticker", value="NVDA", key="gen_tk")
        horizon = st.number_input("Horizon (days)", 5, 365, 45, key="gen_h")
        if st.button("Train + simulate", type="primary"):
            from stock_research import data, expected_return, generative
            with st.spinner("Training TCN + sampling..."):
                snap = data.get_snapshot(ticker.upper())
                if snap is None:
                    st.error(f"No data for {ticker}."); return
                rets = data.daily_log_returns(ticker.upper(), 2520)
                drift = expected_return.estimate(
                    snap.info, snap.price, snap.dividend_yield, model=settings.drift_model,
                    rf=settings.risk_free_rate, erp=settings.equity_risk_premium,
                    pe_anchor=settings.pe_anchor, reversion_years=settings.pe_reversion_years,
                    shrink=settings.pe_reversion_shrink)
                model = generative.fit_tcn(rets, steps=steps)
                sim = generative.sample_simulation(model, spot=snap.price, context_returns=rets,
                                                   horizon_days=int(horizon), mu=drift.annual_drift,
                                                   n_paths=paths)
                st.session_state["gen_single"] = (sim, drift, model)
        out = st.session_state.get("gen_single")
        if out:
            sim, drift, model = out
            st.caption(f"TCN NLL {model.final_loss:.4f}, RF {model.receptive_field}d, "
                       f"{model.device}; drift {drift.summary()}")
            cc = st.columns(3)
            cc[0].metric("Expected", f"${sim.expected_terminal():.2f}")
            cc[1].metric("VaR 95%", f"{sim.var(0.95):.1%}")
            cc[2].metric("Mean max DD", f"{sim.mean_max_drawdown():.1%}")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.hist(sim.terminal, bins=80, color="#55a868", alpha=0.85)
            ax.axvline(sim.spot, color="black", linestyle="--")
            ax.set_xlabel("Terminal price"); st.pyplot(fig)
        return

    # Comparison modes share a ticker set + horizon.
    tickers = parse_tickers(st.text_input(
        "Tickers", value="SPY QQQ AAPL MSFT NVDA AMZN GOOGL META JPM XOM KO JNJ", key="gen_tks"))
    horizon = st.number_input("Horizon (days)", 5, 120, 30, key="gen_ch")

    if st.button("Run comparison", type="primary"):
        from stock_research import data, generative
        mu = settings.risk_free_rate + settings.equity_risk_premium
        with st.spinner("Fetching history, training, backtesting (a few minutes)..."):
            closes = {}
            for tk in tickers:
                c = data.close_history(tk.upper(), 2520)
                if len(c) > 600:
                    closes[tk.upper()] = c
            if len(closes) < 2:
                st.error("Need >= 2 tickers with history."); return
            if mode == "Single vs GARCH (compare)":
                res = generative.compare_on_split(
                    closes[list(closes)[0]], horizon_days=int(horizon), n_paths=paths,
                    mu=mu, tcn_hp={"steps": steps})
                st.session_state["gen_cmp"] = ("single", res)
            elif mode == "Pooled universe vs GARCH":
                res = generative.compare_universe(
                    closes, horizon_days=int(horizon), n_paths=paths, mu=mu,
                    tcn_hp={"steps": steps})
                st.session_state["gen_cmp"] = ("pooled", res)
            else:
                from stock_research import joint
                res = joint.compare_joint(
                    closes, horizon_days=int(horizon), n_paths=paths, mu=mu,
                    tcn_hp={"steps": steps})
                st.session_state["gen_cmp"] = ("joint", res)

    cmp = st.session_state.get("gen_cmp")
    if cmp:
        kind, res = cmp
        if kind in ("single", "pooled"):
            g, t = res["garch"], res["tcn"]
            st.markdown(f"**{res['windows']} held-out windows** — frozen TCN vs refit GARCH")
            df = pd.DataFrame({
                "metric": ["touch+ ECE", "touch- ECE", "term+ ECE", "VaR obs", "VaR Kupiec p", "pinball"],
                "GARCH": [g["touch_up"].ece, g["touch_down"].ece, g["term_up"].ece,
                          g["var"]["observed_rate"], g["var"]["kupiec_p"], g["pinball"]["overall"]],
                "TCN": [t["touch_up"].ece, t["touch_down"].ece, t["term_up"].ece,
                        t["var"]["observed_rate"], t["var"]["kupiec_p"], t["pinball"]["overall"]],
            })
            st.dataframe(df, use_container_width=True)
        else:
            p, s = res["portfolio"], res["single_name"]
            st.markdown(f"**Portfolio tail risk** — joint factor model vs independent GARCH "
                        f"({res['windows']} windows)")
            st.dataframe(pd.DataFrame({
                "metric": ["VaR observed", "VaR Kupiec p", "pinball"],
                "independent GARCH": [p["independent"]["var"]["observed_rate"],
                                      p["independent"]["var"]["kupiec_p"],
                                      p["independent"]["pinball"]["overall"]],
                "joint model": [p["joint"]["var"]["observed_rate"],
                                p["joint"]["var"]["kupiec_p"],
                                p["joint"]["pinball"]["overall"]],
            }), use_container_width=True)
            st.markdown("**Single-name marginals** — joint model vs GARCH")
            st.dataframe(pd.DataFrame({
                "metric": ["VaR observed", "term+ ECE", "pinball"],
                "GARCH": [s["garch"]["var"]["observed_rate"], s["garch"]["term_up"].ece,
                          s["garch"]["pinball"]["overall"]],
                "joint model": [s["joint"]["var"]["observed_rate"], s["joint"]["term_up"].ece,
                                s["joint"]["pinball"]["overall"]],
            }), use_container_width=True)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    st.title("📈 stock_research")
    st.caption("OTM call-writing screener · Monte-Carlo risk · generative & cross-asset models")
    settings = sidebar_settings()
    st.sidebar.caption(_gpu_status())

    tabs = st.tabs(["OTM calls", "Long calls", "Value buys", "Risk scan", "Deep dive",
                    "Simulate", "Backtest", "Generative"])
    with tabs[0]:
        tab_screen(settings)
    with tabs[1]:
        tab_longcall(settings)
    with tabs[2]:
        tab_value(settings)
    with tabs[3]:
        tab_riskscan(settings)
    with tabs[4]:
        tab_deepdive(settings)
    with tabs[5]:
        tab_simulate(settings)
    with tabs[6]:
        tab_backtest(settings)
    with tabs[7]:
        tab_generate(settings)


if __name__ == "__main__":
    main()
