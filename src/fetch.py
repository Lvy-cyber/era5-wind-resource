"""ERA5 数据获取。

两种方式，任选其一：

1) 本脚本（默认）：Open-Meteo Archive API，底层就是 ERA5，无需注册。
   https://open-meteo.com/en/docs/historical-weather-api
2) Copernicus CDS（原始网格）：需要账号 + cdsapi key，见 README 附录。

注意：Open-Meteo 会把 ERA5 0.25° 网格插值到请求点，返回的是"点序列"，
不是原始网格。做风资源评估时这是已知局限，报告里会写明。
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

API = "https://archive-api.open-meteo.com/v1/archive"

HOURLY = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
]


def fetch_era5(
    lat: float,
    lon: float,
    start: str,
    end: str,
    out_csv: str,
    meta_json: str | None = None,
    models: str = "era5",
    retries: int = 3,
) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY),
        "models": models,
        "wind_speed_unit": "ms",  # 关键：默认是 km/h
        "timezone": "UTC",
    }

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(API, params=params, timeout=90)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    else:  # pragma: no cover
        raise RuntimeError("ERA5 获取失败") from last_err

    if "hourly" not in payload:
        raise RuntimeError(f"接口返回异常：{str(payload)[:300]}")

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(
        columns={
            "wind_speed_10m": "ws10",
            "wind_direction_10m": "wd10",
            "wind_speed_100m": "ws100",
            "wind_direction_100m": "wd100",
            "temperature_2m": "temp_c",
            "surface_pressure": "pressure_hpa",
        }
    )
    df = df[["time", "ws10", "wd10", "ws100", "wd100", "temp_c", "pressure_hpa"]]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta = {
        "source": "Open-Meteo Archive API (ERA5)",
        "model": models,
        "requested_lat": lat,
        "requested_lon": lon,
        "returned_lat": payload.get("latitude"),
        "returned_lon": payload.get("longitude"),
        "grid_elevation_m": payload.get("elevation"),
        "start": start,
        "end": end,
        "rows": int(len(df)),
        "units": payload.get("hourly_units"),
        "license": "ERA5 © Copernicus Climate Change Service; API by Open-Meteo (CC BY 4.0)",
        "fetched_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    if meta_json:
        os.makedirs(os.path.dirname(meta_json), exist_ok=True)
        with open(meta_json, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    return df


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df
