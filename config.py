"""Shared configuration for the SistemaLink Hurst Research Programme.

Keep *constants* here (paths, tickers, windows). Avoid business logic.
All pipeline steps should import from this module so changes are made once.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------
# Project paths (robust to VS Code Run button)
# ----------------------------
# Base directory = folder where this config.py lives
BASE_DIR: Path = Path(__file__).resolve().parent

# Project root = parent of research_programme/
PROJECT_ROOT: Path = BASE_DIR.parent

# SQLite database file (single source of truth for derived data)
DB_PATH: Path = BASE_DIR / "research_programme.db"

# Primary Excel input (keep Excel as upstream only)
DATA_DIR: Path = PROJECT_ROOT / "data"
EXCEL_PATH: Path = DATA_DIR / "Data_Hurst.xlsx"

# SQLite export path (for convenience)
EXPORT_DIR = PROJECT_ROOT / "exports"
EXPORT_XLSX_PATH = EXPORT_DIR / "research_programme_dump.xlsx"

# ----------------------------
# Universe
# ----------------------------
TICKERS: list[str] = [
    "S&P 500",
]

# ----------------------------
# Hurst scales
# ----------------------------
WINDOW_SIZES: list[int] = [32, 64, 128, 256, 512, 1024]

SCALE_LABELS: dict[int, str] = {
    32: "1.5M",
    64: "3M",
    128: "6M",
    256: "1Y",
    512: "2Y",
    1024: "4Y",
}

# Convenience: labels only, in the order of WINDOW_SIZES
SCALES: list[str] = [SCALE_LABELS[w] for w in WINDOW_SIZES]

# Default subwindows used in your R/S implementation
SUBWINDOWS: list[int] = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# ----------------------------
# Duration tiers (rolling context)
# ----------------------------
# Trading-day windows: 1M, 3M, 6M, 9M, 12M
DURATION_WINDOWS: dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "9M": 189,
    "12M": 252,
}

PCT_WINDOWS: list[int] = list(DURATION_WINDOWS.values())
RZ_WINDOWS: list[int] = list(DURATION_WINDOWS.values())

# Extra subwindows to stabilize short horizons (used only when window_size <= 64)
SHORT_SUBWINDOW_EXTRAS: list[int] = [6, 12, 24]
SHORT_SUBWINDOW_MAX_WINDOW: int = 64

# ----------------------------
# Numeric guards / pipeline knobs
# ----------------------------
MIN_ROLLING_WINDOW: int = 21
BUFFER_ROWS: int = 50

# For Step 1 ingestion (if you keep it Excel-based)
PRICES_SHEET_NAME: str = "Prices HC"

# Optional: sanity check flag
STRICT_MODE: bool = False

# ----------------------------
# Hurst Rate-of-Change horizons (trading days)
# ----------------------------
HURST_ROC_HORIZONS: list[int] = [1, 5, 21, 63]  # daily, weekly, monthly, quarterly

# IQR windows for time-series (agent vs own history)
IQR_WINDOWS: list[int] = [21, 63, 126, 189, 252]