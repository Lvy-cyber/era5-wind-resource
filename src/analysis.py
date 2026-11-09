"""核心分析：质控、Weibull、风切变、风玫瑰、发电量估算。

设计原则：每一步都能单独调用、单独解释。报告里的每个数字都从这里出来。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn

from . import config as cfg


# ---------------------------------------------------------------- 数据质控

@dataclass
class QCReport:
    total: int
    missing: int
    out_of_range: int
    stuck: int
    spike: int
    sigma_outliers: int
    filled: int

    @property
    def missing_rate(self) -> float:
        return self.missing / self.total if self.total else 0.0

    @property
    def valid_rate(self) -> float:
        return 1.0 - self.missing_rate

    def as_dict(self) -> dict[str, float | int]:
        return {
            "总时次": self.total,
            "缺测": self.missing,
            "缺测率": round(self.missing_rate * 100, 2),
            "超物理限值": self.out_of_range,
            "僵值": self.stuck,
            "变化率异常": self.spike,
            "3σ 离群(仅标记)": self.sigma_outliers,
            "线性插补": self.filled,
        }


def qc_series(
    s: pd.Series,
    lo: float = 0.0,
    hi: float = 50.0,
    max_delta: float = 15.0,
    stuck_n: int = 6,
    max_gap: int = 3,
) -> tuple[pd.Series, QCReport]:
    """质控 -> 返回清洗后的序列（含插补）与报告。

    规则（可按项目实际调整）：
    - 物理限值：风速 < 0 或 > 50 m/s 判为无效
    - 僵值：连续 >= 6 个时次数值完全相同（传感器卡死）
    - 变化率：相邻小时变化 > 15 m/s（远超真实气象过程）
    - 3σ：只统计、不剔除。风速分布右偏，3σ 会误杀真实大风。
    - 插补：仅对 <= 3 h 的连续缺测做线性插值，并保留标志列
    """
    s = s.astype(float).copy()
    total = len(s)

    missing = int(s.isna().sum())

    invalid = (s < lo) | (s > hi)
    out_of_range = int(invalid.sum())
    s[invalid] = np.nan

    # 僵值：滚动窗口内极差为 0
    stuck_mask = pd.Series(False, index=s.index)
    if stuck_n > 1:
        roll_max = s.rolling(stuck_n, min_periods=stuck_n).max()
        roll_min = s.rolling(stuck_n, min_periods=stuck_n).min()
        stuck_mask = (roll_max - roll_min) < 1e-9
    stuck = int(stuck_mask.fillna(False).sum())

    prev = s.shift(1)
    spike_mask = (s - prev).abs() > max_delta
    spike = int(spike_mask.fillna(False).sum())

    valid = s.dropna()
    sigma_outliers = 0
    if len(valid) > 10:
        mu, sd = valid.mean(), valid.std()
        sigma_outliers = int(((valid - mu).abs() > 3 * sd).sum())

    filled = 0
    if 0 < max_gap:
        s = s.interpolate(method="linear", limit=max_gap, limit_area="inside")
        filled = int(s.notna().sum() - (total - out_of_range - missing))

    report = QCReport(
        total=total,
        missing=missing,
        out_of_range=out_of_range,
        stuck=stuck,
        spike=spike,
        sigma_outliers=sigma_outliers,
        filled=max(filled, 0),
    )
    return s, report


# ---------------------------------------------------------------- Weibull

@dataclass
class WeibullFit:
    k: float
    c: float
    mean_ms: float
    mean_cube_ms3: float
    power_density_wm2: float
    rho: float
    ks_pvalue: float
    n: int

    def pdf(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        out = np.zeros_like(v)
        m = v > 0
        out[m] = (self.k / self.c) * (v[m] / self.c) ** (self.k - 1) * np.exp(
            -(v[m] / self.c) ** self.k
        )
        return out

    def cdf(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return 1.0 - np.exp(-np.maximum(v, 0) ** self.k / self.c**self.k)


def weibull_fit(v: pd.Series, rho: float = cfg.RHO_REF) -> WeibullFit:
    """两参数 Weibull，极大似然估计（MLE）。

    MLE 方程：1/k + mean(ln x) - sum(x^k ln x)/sum(x^k) = 0
    对 k 用 brentq 求根；c = (mean(x^k))^(1/k)。
    """
    x = np.asarray(pd.Series(v).dropna().values, dtype=float)
    x = x[x > 1e-6]
    if x.size < 24:
        raise ValueError("有效样本太少，无法拟合 Weibull")

    def score(k: float) -> float:
        xk = x**k
        return 1.0 / k + np.mean(np.log(x)) - np.sum(xk * np.log(x)) / np.sum(xk)

    try:
        lo, hi = 0.3, 15.0
        flo, fhi = score(lo), score(hi)
        if flo * fhi > 0:
            raise ValueError("区间内无符号变化")
        k = brentq(score, lo, hi, xtol=1e-6)
    except Exception:  # 退回矩估计
        mean, std = x.mean(), x.std(ddof=1)
        k = max((std / mean) ** -1.086, 0.1)

    c = float(np.mean(x**k) ** (1.0 / k))
    mean_ms = float(c * gamma_fn(1.0 + 1.0 / k))
    mean_cube = float(c**3 * gamma_fn(1.0 + 3.0 / k))
    ks = stats.kstest(x, lambda z: 1.0 - np.exp(-(np.maximum(z, 0) ** k) / c**k))

    return WeibullFit(
        k=float(k),
        c=c,
        mean_ms=mean_ms,
        mean_cube_ms3=mean_cube,
        power_density_wm2=float(0.5 * rho * mean_cube),
        rho=rho,
        ks_pvalue=float(ks.pvalue),
        n=int(x.size),
    )


# ---------------------------------------------------------------- 风切变

@dataclass
class ShearResult:
    alpha_median: float
    alpha_p25: float
    alpha_p75: float
    alpha_ols: float
    n: int


def shear_exponent(
    ws_low: pd.Series, ws_high: pd.Series, z_low: float = 10.0, z_high: float = 100.0
) -> ShearResult:
    """逐时风切变指数 α = ln(v_high/v_low) / ln(z_high/z_low)。

    只保留两层都 >= 2 m/s 的时次——低风速下比值不稳定。
    """
    df = pd.DataFrame({"lo": ws_low, "hi": ws_high}).dropna()
    df = df[(df.lo >= 2.0) & (df.hi >= 2.0)]
    if len(df) < 24:
        raise ValueError("有效样本太少，无法估计风切变")

    alpha = np.log(df.hi / df.lo) / np.log(z_high / z_low)
    x = np.log(df.lo.values)
    y = np.log(df.hi.values)
    alpha_ols = float(np.polyfit(x, y, 1)[0] / np.log(z_high / z_low))

    return ShearResult(
        alpha_median=float(alpha.median()),
        alpha_p25=float(alpha.quantile(0.25)),
        alpha_p75=float(alpha.quantile(0.75)),
        alpha_ols=alpha_ols,
        n=int(len(df)),
    )


def extrapolate(v_ref: pd.Series, alpha: float, z_ref: float, z_target: float) -> pd.Series:
    """幂律外推：v2 = v1 * (z2/z1)^α"""
    return v_ref * (z_target / z_ref) ** alpha


# ---------------------------------------------------------------- 风玫瑰

SECTORS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
SPEED_BINS = [0, 2, 4, 6, 8, 10, 12, 15, 20, np.inf]
SPEED_LABELS = ["<2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-15", "15-20", ">20"]


def wind_rose(ws: pd.Series, wd: pd.Series) -> pd.DataFrame:
    """按 16 方位 × 风速区间统计频次（%）。风向为风的来向。"""
    df = pd.DataFrame({"ws": ws, "wd": wd}).dropna()
    idx = (((df.wd % 360) + 11.25) % 360 / 22.5).astype(int)
    df["sector"] = idx.map(lambda i: SECTORS[i % 16])
    df["bin"] = pd.cut(df.ws, bins=SPEED_BINS, labels=SPEED_LABELS, right=False)

    table = (
        df.groupby(["sector", "bin"], observed=False).size().unstack(fill_value=0)
        / len(df) * 100.0
    )
    table = table.reindex(SECTORS).fillna(0.0)
    table["合计"] = table.sum(axis=1)
    return table


def sector_mean_speed(ws: pd.Series, wd: pd.Series) -> pd.Series:
    df = pd.DataFrame({"ws": ws, "wd": wd}).dropna()
    idx = (((df.wd % 360) + 11.25) % 360 / 22.5).astype(int)
    df["sector"] = idx.map(lambda i: SECTORS[i % 16])
    return df.groupby("sector")["ws"].mean().reindex(SECTORS)


# ---------------------------------------------------------------- 空气密度 & 功率

def air_density(pressure_hpa: pd.Series, temp_c: pd.Series) -> pd.Series:
    """理想气体：ρ = p / (R·T)"""
    t_k = temp_c + 273.15
    return pressure_hpa * 100.0 / (cfg.R_D * t_k)


def power_curve(v: np.ndarray, turbine=cfg.TURBINE) -> np.ndarray:
    xs = np.array([p[0] for p in turbine.curve], dtype=float)
    ys = np.array([p[1] for p in turbine.curve], dtype=float)
    v = np.asarray(v, dtype=float)
    p = np.interp(v, xs, ys, left=0.0, right=0.0)
    p[(v < turbine.cut_in_ms) | (v > turbine.cut_out_ms)] = 0.0
    return p


def density_corrected_speed(v: pd.Series, rho: pd.Series) -> pd.Series:
    """IEC 61400-12 等功率法：v_eq = v * (ρ/ρ_ref)^(1/3)"""
    return v * (rho / cfg.RHO_REF) ** (1.0 / 3.0)


# ---------------------------------------------------------------- 发电量

@dataclass
class AEPResult:
    gross_gwh: float
    net_gwh: float
    capacity_factor: float
    full_load_hours: float
    loss_total: float
    hours: float
    per_turbine_kw: float


def estimate_aep(
    ws_hub: pd.Series,
    rho: pd.Series | None = None,
    turbine=cfg.TURBINE,
    losses: dict[str, float] | None = None,
    hours: float = 8760.0,
    bin_width: float = 0.25,
) -> AEPResult:
    """基于观测风速分布 + 机型功率曲线的年发电量估算。

    做法：把逐时风速按 rho 修正到等效风速 -> 分箱统计频率
          -> AEP_gross = Σ P(v_i)·f_i·hours
          -> 逐项折减得到 net
    """
    losses = cfg.LOSSES if losses is None else losses
    v = pd.Series(ws_hub).dropna().astype(float)
    if rho is not None:
        v = density_corrected_speed(v, pd.Series(rho).reindex(v.index))

    edges = np.arange(0.0, 30.0 + bin_width, bin_width)
    counts, edges = np.histogram(v.values, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    freq = counts / counts.sum()

    p_kw = power_curve(centres, turbine)
    gross_kwh = float(np.sum(p_kw * freq) * hours)
    loss_total = 1.0 - np.prod([1.0 - x for x in losses.values()])
    net_kwh = gross_kwh * (1.0 - loss_total)

    return AEPResult(
        gross_gwh=gross_kwh / 1e6,
        net_gwh=net_kwh / 1e6,
        capacity_factor=net_kwh / (turbine.rated_kw * hours),
        full_load_hours=net_kwh / turbine.rated_kw,
        loss_total=float(loss_total),
        hours=hours,
        per_turbine_kw=turbine.rated_kw,
    )


def monthly_aep(
    ws_hub: pd.Series, rho: pd.Series, turbine=cfg.TURBINE, losses: dict[str, float] | None = None
) -> pd.DataFrame:
    """逐月 AEP 与容量系数（每月按当月实际小时数外推）。"""
    df = pd.DataFrame({"ws": ws_hub, "rho": rho}).dropna()
    df["month"] = df.index.month
    rows = []
    for m, g in df.groupby("month"):
        hours = float(len(g))
        r = estimate_aep(g.ws, g.rho, turbine=turbine, losses=losses, hours=hours)
        rows.append(
            {
                "月份": int(m),
                "平均风速": round(float(g.ws.mean()), 2),
                "样本小时": int(hours),
                "毛发电量_GWh": round(r.gross_gwh, 3),
                "净发电量_GWh": round(r.net_gwh, 3),
                "容量系数": round(r.capacity_factor, 4),
            }
        )
    return pd.DataFrame(rows)


def sensitivity(
    ws_hub: pd.Series,
    rho: pd.Series,
    turbine=cfg.TURBINE,
    losses: dict[str, float] | None = None,
) -> pd.DataFrame:
    """风速 ±10%、折减 ±3pp 对净发电量的影响。"""
    base = estimate_aep(ws_hub, rho, turbine, losses).net_gwh
    rows = []

    for label, factor in [("-10%", 0.9), ("基准", 1.0), ("+10%", 1.1)]:
        r = estimate_aep(ws_hub * factor, rho, turbine, losses)
        rows.append(
            {
                "情景": f"风速 {label}",
                "净发电量_GWh": round(r.net_gwh, 3),
                "相对基准": f"{(r.net_gwh / base - 1) * 100:+.1f}%",
            }
        )

    los = dict(cfg.LOSSES if losses is None else losses)
    for label, delta in [("折减 +3pp", 0.03), ("折减 -3pp", -0.03)]:
        adj = {k: max(v + delta, 0.0) for k, v in los.items()}
        r = estimate_aep(ws_hub, rho, turbine, adj)
        rows.append(
            {
                "情景": label,
                "净发电量_GWh": round(r.net_gwh, 3),
                "相对基准": f"{(r.net_gwh / base - 1) * 100:+.1f}%",
            }
        )

    return pd.DataFrame(rows)
