"""Calendar and weather feature engineering.

The model sees two feature sets:
- PAST features: everything, including the (normalized) load itself.
- FUTURE features: only covariates known in advance — calendar and weather
  (weather enters as a forecast at inference time; the backtest uses actual
  weather, i.e. a perfect-weather-forecast assumption, stated in the README).
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

# Covariates known for the future. Order matters: kept stable for checkpoints.
FUTURE_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_holiday",
    "is_holiday_adjacent",
    "temp_norm",
    "hdd",
    "cdd",
]
# The past additionally sees the target itself.
PAST_FEATURES = ["load_norm", *FUTURE_FEATURES]

HDD_BASE_C = 16.0  # heating starts below this temperature
CDD_BASE_C = 21.0  # cooling starts above this temperature


def _cyc(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def italian_holiday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Holiday flag evaluated on local (Europe/Rome) calendar dates."""
    local_dates = index.tz_convert("Europe/Rome").date
    years = range(index[0].year, index[-1].year + 1)
    it_holidays = holidays.country_holidays("IT", years=years)
    return np.array([d in it_holidays for d in local_dates], dtype=bool)


def build_features(
    df: pd.DataFrame, load_scaler, temp_mean: float, temp_std: float
) -> pd.DataFrame:
    """Return a copy of df with all model features appended."""
    out = df.copy()
    idx = out.index
    local = idx.tz_convert("Europe/Rome")

    out["load_norm"] = load_scaler.transform(out["load_mw"].to_numpy())

    hour_sin, hour_cos = _cyc(local.hour.to_numpy().astype(float), 24)
    dow_sin, dow_cos = _cyc(local.dayofweek.to_numpy().astype(float), 7)
    doy_sin, doy_cos = _cyc(local.dayofyear.to_numpy().astype(float), 365.25)
    out["hour_sin"], out["hour_cos"] = hour_sin, hour_cos
    out["dow_sin"], out["dow_cos"] = dow_sin, dow_cos
    out["doy_sin"], out["doy_cos"] = doy_sin, doy_cos
    out["is_weekend"] = (local.dayofweek >= 5).astype(float)

    hol = italian_holiday_mask(idx)
    # A day is "holiday adjacent" when the previous or next local day is a
    # holiday (ponte behaviour: demand dips around holidays too).
    hol_series = pd.Series(hol, index=idx)
    daily = hol_series.groupby(local.date).max()
    prev_or_next = daily.shift(1, fill_value=False) | daily.shift(-1, fill_value=False)
    adjacent = np.array(
        [prev_or_next.get(d, False) for d in local.date], dtype=bool
    ) & ~hol
    out["is_holiday"] = hol.astype(float)
    out["is_holiday_adjacent"] = adjacent.astype(float)

    temp = out["temp_c"].to_numpy()
    out["temp_norm"] = (temp - temp_mean) / temp_std
    out["hdd"] = np.maximum(0.0, HDD_BASE_C - temp) / 10.0
    out["cdd"] = np.maximum(0.0, temp - CDD_BASE_C) / 10.0
    return out
