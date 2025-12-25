import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH, TICKERS, SCALES, PCT_WINDOWS, RZ_WINDOWS, HURST_ROC_HORIZONS, MIN_ROLLING_WINDOW

import os
import urllib.request

DB_URL = "https://github.com/SistemaLinkResearch/dashboard-prototype/releases/download/DB/research_programme.db"

def ensure_db(local_path: Path, url: str):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 10_000_000:
        return
    urllib.request.urlretrieve(url, local_path)

ensure_db(DB_PATH, DB_URL)

# Custom diverging colorscale (SistemaLink style)
SISTEMALINK_COLORSCALE = [
    [0.00, "#7F2020"],
    [0.10, "#9C2C2C"],
    [0.25, "#C65A5A"],
    [0.40, "#E6B3B3"],
    [0.45, "#BFBFBF"],
    [0.55, "#BFBFBF"],
    [0.60, "#D9E8DC"],
    [0.75, "#8FBC9A"],
    [0.90, "#3C8C55"],
    [1.00, "#2F6F46"],
]

st.set_page_config(page_title="SistemaLink Prototype", layout="wide")

# ---------------------------
# DB helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_prices(db_path: Path, ticker: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT date, close FROM prices WHERE ticker=? ORDER BY date ASC;",
        con,
        params=(ticker,),
    )
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


@st.cache_data(show_spinner=False)
def load_hurst_pct(db_path: Path, ticker: str, pct_window: int) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT date, scale_label, pct
        FROM hurst_pct
        WHERE ticker=? AND pct_window=?
        ORDER BY date ASC;
        """,
        con,
        params=(ticker, int(pct_window)),
    )
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_hurst_rz(db_path: Path, ticker: str, rz_window: int) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT date, scale_label, rz
        FROM hurst_rz
        WHERE ticker=? AND rz_window=?
        ORDER BY date ASC;
        """,
        con,
        params=(ticker, int(rz_window)),
    )
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_hurst_roc(db_path: Path, ticker: str, horizon: int) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT date, scale_label, roc
        FROM hurst_roc
        WHERE ticker=? AND horizon=?
        ORDER BY date ASC;
        """,
        con,
        params=(ticker, int(horizon)),
    )
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df



@st.cache_data(show_spinner=False)
def load_iqr_scale_raw(db_path: Path, ticker: str) -> pd.DataFrame:
    """Load scale-wise IQR (agent disagreement) from iqr_metrics.

    Convention: scale_raw rows are stored with scale_label='' and duration_window=-1.
    """
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT date, value
        FROM iqr_metrics
        WHERE ticker=?
          AND metric='scale_raw'
          AND scale_label=''
          AND duration_window=-1
        ORDER BY date ASC;
        """,
        con,
        params=(ticker,),
    )
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


# Entropy strip loader
@st.cache_data(show_spinner=False)
def load_entropy_scale_raw(db_path: Path, ticker: str) -> pd.DataFrame:
    """Load scale-wise entropy (agent disorder) from entropy_metrics.

    Convention: scale_raw rows are stored with scale_label='' and duration_window=-1.
    Values are normalized entropy in [0,1].
    """
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT date, value
        FROM entropy_metrics
        WHERE ticker=?
          AND metric='scale_raw'
          AND scale_label=''
          AND duration_window=-1
        ORDER BY date ASC;
        """,
        con,
        params=(ticker,),
    )
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def to_heatmap(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Returns a wide matrix: index=date, columns=scale_label, values=value_col."""
    wide = df.pivot(index="date", columns="scale_label", values=value_col).sort_index()
    cols = [c for c in SCALES if c in wide.columns]
    return wide[cols]


def heatmap_figure(mat: pd.DataFrame, title: str, zmin=None, zmax=None):
    # Use graph_objects Heatmap to ensure custom colorscale is applied reliably.
    # x-axis: dates, y-axis: scales
    x = mat.index
    y = list(mat.columns)
    z = mat.values.T  # shape: (scales, dates)

    hm = go.Heatmap(
        x=x,
        y=y,
        z=z,
        colorscale=SISTEMALINK_COLORSCALE,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=title),
    )

    fig = go.Figure(data=[hm])
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Scale",
    )
    return fig


def downsample_matrix(mat: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    """Downsample columns (dates) to reduce heatmap size."""
    if mat.empty:
        return mat
    if freq_label == "Daily":
        return mat
    rule = "W-FRI" if freq_label == "Weekly" else "M"
    return mat.resample(rule).last()


def robust_limits(values: np.ndarray, clip_percent: float, symmetric: bool = False):
    """Compute robust (clipped) color limits."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None, None

    p = float(clip_percent)
    lo = np.percentile(v, p)
    hi = np.percentile(v, 100.0 - p)

    if symmetric:
        m = float(max(abs(lo), abs(hi)))
        return -m, m

    return float(lo), float(hi)


def rolling_percentile_series(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile (0..100) of s[t] vs s[t-window:t]."""
    vals = s.values.astype(float)
    out = np.full(vals.shape, np.nan, dtype=float)
    w = int(window)
    if w < int(MIN_ROLLING_WINDOW):
        w = int(MIN_ROLLING_WINDOW)

    for i in range(len(vals)):
        if i < w:
            continue
        hist = vals[i - w : i]
        hist = hist[~np.isnan(hist)]
        if hist.size == 0 or np.isnan(vals[i]):
            continue
        out[i] = 100.0 * float(np.sum(hist <= vals[i])) / float(hist.size)

    return pd.Series(out, index=s.index)


def regime_strip_figure(y: pd.Series, title: str):
    """Line plot with background regime bands (0–100)."""
    fig = go.Figure()

    # Background bands (paper-wide)
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=0,
        y1=50,
        fillcolor="#7F2020",
        opacity=0.12,
        line_width=0,
    )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=50,
        y1=100,
        fillcolor="#2F6F46",
        opacity=0.10,
        line_width=0,
    )
    # Neutral mid-band for readability
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=45,
        y1=55,
        fillcolor="#E5E5E5",
        opacity=0.35,
        line_width=0,
    )

    fig.add_trace(
        go.Scatter(
            x=y.index,
            y=y.values,
            mode="lines",
            line=dict(width=2),
            name=title,
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Percentile (0–100)",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
    )
    return fig


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("Controls")

ticker = st.sidebar.selectbox("Ticker", TICKERS, index=0)

# date range based on prices
prices = load_prices(DB_PATH, ticker)
if prices.empty:
    st.error("No prices found in DB for this ticker.")
    st.stop()

min_d, max_d = prices.index.min().date(), prices.index.max().date()

# Default view: last 5 years (avoids rendering gigantic heatmaps by default)
default_start = (prices.index.max() - pd.DateOffset(years=5)).date()
if default_start < min_d:
    default_start = min_d

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_d),
    min_value=min_d,
    max_value=max_d,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    d0, d1 = prices.index.min(), prices.index.max()

pct_window = st.sidebar.selectbox("Percentile window (days)", PCT_WINDOWS, index=len(PCT_WINDOWS) - 1)
rz_window = st.sidebar.selectbox("Robust Z window (days)", RZ_WINDOWS, index=len(RZ_WINDOWS) - 1)
roc_h = st.sidebar.selectbox("RoC horizon (days)", HURST_ROC_HORIZONS, index=1)

iqr_pct_window = st.sidebar.selectbox(
    "IQR strip window (days)",
    [63, 126, 189, 252],
    index=3,
    help="Rolling lookback used to convert scale_raw IQR into a 0–100 percentile strip.",
)


smooth_iqr = st.sidebar.checkbox(
    "Smooth IQR strip",
    value=True,
    help="Apply a moving-average smoother to the IQR percentile strip for readability.",
)

smooth_iqr_days = st.sidebar.slider(
    "IQR smoothing (trading days)",
    min_value=1,
    max_value=30,
    value=7,
    step=1,
    help="Moving-average window applied to the IQR percentile strip (1 = no smoothing).",
)

# Entropy strip sidebar controls
entropy_pct_window = st.sidebar.selectbox(
    "Entropy strip window (days)",
    [63, 126, 189, 252],
    index=3,
    help="Rolling lookback used to convert scale_raw entropy into a 0–100 percentile strip.",
)

smooth_entropy = st.sidebar.checkbox(
    "Smooth entropy strip",
    value=True,
    help="Apply a moving-average smoother to the entropy percentile strip for readability.",
)

smooth_entropy_days = st.sidebar.slider(
    "Entropy smoothing (trading days)",
    min_value=1,
    max_value=60,
    value=14,
    step=1,
    help="Moving-average window applied to the entropy percentile strip (1 = no smoothing).",
)

heatmap_freq = st.sidebar.selectbox(
    "Heatmap frequency",
    ["Daily", "Weekly", "Monthly"],
    index=1,
    help="Downsample heatmaps to improve performance on long date ranges.",
)

clip_pct = st.sidebar.slider(
    "Color clip (percent)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="Clip the color scale to ignore extreme outliers.",
)

use_symmetric = st.sidebar.checkbox(
    "Symmetric around 0 (z / RoC)",
    value=True,
    help="For z-scores and RoC, use symmetric color limits around 0.",
)


# ---------------------------
# Layout
# ---------------------------
st.title("SLC Research Programme")

# ---------------------------
# Page sections (no tabs)
# ---------------------------

# 1) Price
st.subheader("1) Price (Close)")
p = prices.loc[d0:d1].copy()
fig = px.line(p, y="close")
fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# 2) Hurst Percentiles
st.subheader(f"2) Hurst Percentiles (window={pct_window})")
df = load_hurst_pct(DB_PATH, ticker, pct_window)
df = df[(df["date"] >= d0) & (df["date"] <= d1)]
if df.empty:
    st.warning("No percentile rows for this selection.")
else:
    mat = to_heatmap(df, "pct")
    mat = downsample_matrix(mat, heatmap_freq)
    if mat.shape[0] > 2500:
        st.info("Large date range selected; heatmap may be slow. Consider Weekly/Monthly frequency or shorter date range.")

    # Percentiles naturally live in [0,100]. Optional clipping improves contrast.
    zlo, zhi = 0.0, 100.0
    if clip_pct > 0:
        lo, hi = robust_limits(mat.values, clip_pct, symmetric=False)
        if lo is not None and hi is not None:
            zlo = max(0.0, lo)
            zhi = min(100.0, hi)

    st.plotly_chart(heatmap_figure(mat, "Percentile", zmin=zlo, zmax=zhi), use_container_width=True)
    st.caption("Interpretation: higher = more persistent/trending relative to the last N days (duration tier).")

st.divider()

# 3) Robust Z-score
st.subheader(f"3) Robust Z-score (window={rz_window})")
df = load_hurst_rz(DB_PATH, ticker, rz_window)
df = df[(df["date"] >= d0) & (df["date"] <= d1)]
if df.empty:
    st.warning("No robust z rows for this selection.")
else:
    mat = to_heatmap(df, "rz")
    mat = downsample_matrix(mat, heatmap_freq)
    if mat.shape[0] > 2500:
        st.info("Large date range selected; heatmap may be slow. Consider Weekly/Monthly frequency or shorter date range.")

    zmin, zmax = robust_limits(mat.values, clip_pct, symmetric=use_symmetric)
    if zmin is None or zmax is None:
        zmin, zmax = -3.0, 3.0

    st.plotly_chart(heatmap_figure(mat, "Robust Z", zmin=zmin, zmax=zmax), use_container_width=True)
    st.caption("Interpretation: distance from recent regime median (scaled by MAD). Big |z| suggests structural deviation.")

st.divider()

# 4) Hurst RoC
st.subheader(f"4) Hurst RoC (ΔH in % points over {roc_h} trading days)")
df = load_hurst_roc(DB_PATH, ticker, roc_h)
df = df[(df["date"] >= d0) & (df["date"] <= d1)]
if df.empty:
    st.warning("No RoC rows for this selection.")
else:
    mat = to_heatmap(df, "roc") * 100.0  # convert to percent points
    mat = downsample_matrix(mat, heatmap_freq)
    if mat.shape[0] > 2500:
        st.info("Large date range selected; heatmap may be slow. Consider Weekly/Monthly frequency or shorter date range.")

    zmin, zmax = robust_limits(mat.values, clip_pct, symmetric=use_symmetric)
    st.plotly_chart(heatmap_figure(mat, f"RoC (% points over {roc_h}d)", zmin=zmin, zmax=zmax), use_container_width=True)

    latest = mat.dropna(how="all").iloc[-1].rename("latest").to_frame()
    st.write("Latest RoC by scale (% points)")
    st.dataframe(latest.T, use_container_width=True)

st.divider()

# 5) IQR Strip
st.subheader("5) Agent Disagreement Strip (scale-wise IQR)")

df_full = load_iqr_scale_raw(DB_PATH, ticker)

if df_full.empty:
    st.warning("No iqr_metrics rows found for metric='scale_raw'. Run Step 7 to populate iqr_metrics.")
else:
    # Compute percentile on the FULL history so the chosen date range only affects what is shown,
    # while the percentile window defines the local memory used for ranking.
    pct_full = rolling_percentile_series(df_full["value"], window=int(iqr_pct_window))

    # Now apply the UI date filter for display
    df = df_full.loc[d0:d1]
    pct = pct_full.loc[d0:d1].dropna()
    pct_plot = pct
    if smooth_iqr and not pct.empty and int(smooth_iqr_days) > 1:
        pct_plot = pct.rolling(int(smooth_iqr_days), min_periods=1).mean()

    if pct.empty:
        st.info(
            "Not enough pre-history inside the selected date range to compute rolling percentiles. "
            "Try an earlier start date or a shorter IQR strip window."
        )
    else:
        st.plotly_chart(
            regime_strip_figure(pct_plot, f"{ticker} — scale_raw IQR percentile ({iqr_pct_window}d)"),
            use_container_width=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    if not pct.empty:
        c1.metric("Latest percentile", f"{pct.iloc[-1]:.1f}")
    else:
        c1.metric("Latest percentile", "—")

    if not df.empty and df["value"].dropna().size > 0:
        c2.metric("Latest raw IQR", f"{df['value'].dropna().iloc[-1]:.4f}")
        c3.metric("Max raw IQR (range)", f"{df['value'].max():.4f}")
    else:
        c2.metric("Latest raw IQR", "—")
        c3.metric("Max raw IQR (range)", "—")

    if smooth_iqr and not pct_plot.empty:
        c4.metric("Latest percentile (smoothed)", f"{pct_plot.iloc[-1]:.1f}")
    else:
        c4.metric("Latest percentile (smoothed)", "—")


    st.caption(
        "Interpretation: this is the percentile rank of today’s cross-scale Hurst dispersion (agent disagreement) versus the last N days. "
        "High = unusually fractured cross-horizon structure; low = unusually coherent."
    )


st.divider()

# 6) Entropy Strip
st.subheader("6) Agent Disorder Strip (scale-wise Entropy)")

df_full = load_entropy_scale_raw(DB_PATH, ticker)

if df_full.empty:
    st.warning("No entropy_metrics rows found for metric='scale_raw'. Run Step 8 to populate entropy_metrics.")
else:
    # Convert normalized entropy (0..1) into a rolling percentile strip (0..100)
    pct_full = rolling_percentile_series(df_full["value"], window=int(entropy_pct_window))

    # Apply UI date filter for display
    df = df_full.loc[d0:d1]
    pct = pct_full.loc[d0:d1].dropna()

    pct_plot = pct
    if smooth_entropy and not pct.empty and int(smooth_entropy_days) > 1:
        pct_plot = pct.rolling(int(smooth_entropy_days), min_periods=1).mean()

    if pct.empty:
        st.info(
            "Not enough pre-history inside the selected date range to compute rolling percentiles. "
            "Try an earlier start date or a shorter entropy strip window."
        )
    else:
        st.plotly_chart(
            regime_strip_figure(pct_plot, f"{ticker} — scale_raw entropy percentile ({entropy_pct_window}d)"),
            use_container_width=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    if not pct.empty:
        c1.metric("Latest percentile", f"{pct.iloc[-1]:.1f}")
    else:
        c1.metric("Latest percentile", "—")

    if not df.empty and df["value"].dropna().size > 0:
        c2.metric("Latest raw entropy", f"{df['value'].dropna().iloc[-1]:.4f}")
        c3.metric("Max raw entropy (range)", f"{df['value'].max():.4f}")
    else:
        c2.metric("Latest raw entropy", "—")
        c3.metric("Max raw entropy (range)", "—")

    if smooth_entropy and not pct_plot.empty:
        c4.metric("Latest percentile (smoothed)", f"{pct_plot.iloc[-1]:.1f}")
    else:
        c4.metric("Latest percentile (smoothed)", "—")

    st.caption(
        "Interpretation: this is the percentile rank of today’s scale-wise entropy (how disordered the cross-agent Hurst configuration is) versus the last N days. "
        "High = unusually scattered agent structure; low = unusually ordered/coherent structure."
    )
