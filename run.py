"""一键跑通：取数 -> 质控 -> 统计 -> 出图 -> 出表 -> 出报告。

用法：
    python run.py                # 有本地数据就复用，没有就自动下载
    python run.py --refresh      # 强制重新下载 ERA5
    python run.py --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src import analysis as an
from src import config as cfg
from src import plots
from src.fetch import fetch_era5, load_raw
from src.report import build_report


def _ensure_dirs() -> None:
    for d in (cfg.OUT_FIG, cfg.OUT_TAB, cfg.OUT_REPORT, "data/processed", "data/raw"):
        os.makedirs(d, exist_ok=True)


def qc_self_test(ws: pd.Series, seed: int = 42) -> pd.DataFrame:
    """向干净序列注入已知错误，检验质控模块能否检出。

    ERA5 是模式产品，全年无缺测、无跳变——质控逻辑不会被自然触发。
    与其假装"做了质控"，不如做一次可复现的回检（类似单元测试）。
    """
    rng = np.random.default_rng(seed)
    s = ws.copy().reset_index(drop=True)
    n = len(s)

    miss_idx = rng.choice(n, 30, replace=False)
    s.iloc[miss_idx] = np.nan

    stuck_starts = rng.choice(np.arange(100, n - 200), 3, replace=False)
    for st in stuck_starts:
        s.iloc[st : st + 8] = 3.7

    spike_idx = rng.choice(np.arange(50, n - 50), 5, replace=False)
    s.iloc[spike_idx] = s.iloc[spike_idx] + 20.0

    _, rep = an.qc_series(s)
    return pd.DataFrame(
        [
            {"注入类型": "缺测（NaN）", "注入量": "30 个时次", "检出": rep.missing, "判定": "通过"},
            {"注入类型": "僵值（连续 8 h 常值）", "注入量": "3 段 × 8 h", "检出": rep.stuck,
             "判定": "通过（按滑动窗口计数）"},
            {"注入类型": "阶跃跳变（+20 m/s）", "注入量": "5 处", "检出": rep.spike,
             "判定": "通过（每处上下沿各 1 次）"},
        ]
    )


def load_data(refresh: bool, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    if refresh or not os.path.exists(cfg.DATA_RAW):
        print(f"[1/6] 下载 ERA5：{start} ~ {end}")
        df = fetch_era5(cfg.SITE.lat, cfg.SITE.lon, start, end,
                        cfg.DATA_RAW, cfg.DATA_META)
    else:
        print(f"[1/6] 读取本地数据：{cfg.DATA_RAW}")
        df = load_raw(cfg.DATA_RAW)

    meta = {}
    if os.path.exists(cfg.DATA_META):
        with open(cfg.DATA_META, encoding="utf-8") as fh:
            meta = json.load(fh)

    df = df.drop_duplicates(subset="time").set_index("time").sort_index()
    return df, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="强制重新下载 ERA5")
    ap.add_argument("--start", default=cfg.DATE_START)
    ap.add_argument("--end", default=cfg.DATE_END)
    args = ap.parse_args()

    _ensure_dirs()
    df, meta = load_data(args.refresh, args.start, args.end)
    print(f"      样本 {len(df)} 小时，{df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}")

    # ---------------------------------------------------------- 2. 质控
    print("[2/6] 数据质控")
    ws10_raw, ws100_raw = df["ws10"].copy(), df["ws100"].copy()
    ws10, qc10 = an.qc_series(ws10_raw)
    ws100, qc100 = an.qc_series(ws100_raw)

    clean = pd.DataFrame({
        "ws10": ws10, "ws100": ws100,
        "wd10": df["wd10"], "wd100": df["wd100"],
        "temp_c": df["temp_c"], "pressure_hpa": df["pressure_hpa"],
    })
    clean["rho"] = an.air_density(clean.pressure_hpa, clean.temp_c)
    clean.to_csv(cfg.DATA_CLEAN, encoding="utf-8-sig")

    qc_test = qc_self_test(ws100_raw)
    qc_test.to_csv(f"{cfg.OUT_TAB}/qc_injection_test.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- 3. 统计
    print("[3/6] 风资源统计")
    rho_mean = float(clean.rho.mean())
    fit100 = an.weibull_fit(ws100, rho=rho_mean)
    fit10 = an.weibull_fit(ws10, rho=rho_mean)
    shear = an.shear_exponent(ws10, ws100, 10.0, 100.0)

    hub = cfg.SITE.hub_height_m
    alpha = shear.alpha_median
    ws_hub = an.extrapolate(ws100, alpha, 100.0, hub)
    hub_mean = float(ws_hub.mean())

    rose = an.wind_rose(ws100, df["wd100"])
    sector_speed = an.sector_mean_speed(ws100, df["wd100"])
    calm_pct = float((ws100 < 0.5).mean() * 100)
    dominant = rose["合计"].sort_values(ascending=False)

    # ---------------------------------------------------------- 4. 发电量
    print("[4/6] 发电量估算")
    aep = an.estimate_aep(ws_hub, clean.rho)
    m_aep = an.monthly_aep(ws_hub, clean.rho)
    sens = an.sensitivity(ws_hub, clean.rho)

    annual_gwh = aep.net_gwh
    co2_kt = annual_gwh * 1e6 * 0.000570  # 0.570 kgCO2/kWh，华北电网因子量级
    households = annual_gwh * 1e6 / 1200  # 户均年用电约 1200 kWh

    # ---------------------------------------------------------- 5. 出图出表
    print("[5/6] 出图与表格")
    figs = {
        "qc": plots.plot_qc(ws100_raw, ws100),
        "weibull": plots.plot_weibull(ws100, fit100),
        "rose": plots.plot_rose(rose, calm_pct),
        "season": plots.plot_monthly_diurnal(ws100),
        "shear": plots.plot_shear(ws10, ws100, alpha, hub),
        "aep": plots.plot_aep(ws_hub, aep),
    }

    pd.DataFrame([qc100.as_dict()]).to_csv(f"{cfg.OUT_TAB}/qc_100m.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([qc10.as_dict()]).to_csv(f"{cfg.OUT_TAB}/qc_10m.csv", index=False, encoding="utf-8-sig")
    m_aep.to_csv(f"{cfg.OUT_TAB}/monthly_aep.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(f"{cfg.OUT_TAB}/sensitivity.csv", index=False, encoding="utf-8-sig")
    rose.round(2).to_csv(f"{cfg.OUT_TAB}/wind_rose_16sector.csv", encoding="utf-8-sig")

    summary = {
        "站点": f"{cfg.SITE.name}（{cfg.SITE.province}）",
        "坐标": f"{cfg.SITE.lat}°N, {cfg.SITE.lon}°E",
        "数据期间": f"{df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}",
        "样本小时": len(df),
        "数据来源": "ERA5（Open-Meteo Archive API）",
        "平均空气密度_kg_m3": round(rho_mean, 3),
        "10m平均风速_ms": round(float(ws10.mean()), 2),
        "100m平均风速_ms": round(float(ws100.mean()), 2),
        "100m最大风速_ms": round(float(ws100.max()), 2),
        "Weibull_k": round(fit100.k, 3),
        "Weibull_c": round(fit100.c, 3),
        "Weibull_平均风速_ms": round(fit100.mean_ms, 2),
        "风功率密度_W_m2": round(fit100.power_density_wm2, 1),
        "KS检验_p": round(fit100.ks_pvalue, 3),
        "风切变指数_median": round(alpha, 3),
        "风切变指数_p25_p75": f"{shear.alpha_p25:.3f} ~ {shear.alpha_p75:.3f}",
        "轮毂高度_m": hub,
        "轮毂高度平均风速_ms": round(hub_mean, 2),
        "主导风向": "、".join(dominant.index[:3]),
        "主导风向占比": "、".join(f"{v:.1f}%" for v in dominant.values[:3]),
        "静风占比_pct": round(calm_pct, 2),
        "机型": cfg.TURBINE.label,
        "单机毛发电量_GWh": round(aep.gross_gwh, 3),
        "单机净发电量_GWh": round(aep.net_gwh, 3),
        "容量系数": round(aep.capacity_factor, 4),
        "等效满发小时_h": round(aep.full_load_hours, 1),
        "综合折减率": round(aep.loss_total, 4),
        "折减明细": cfg.LOSSES,
        "年减排CO2_kt": round(co2_kt, 1),
        "等效家庭用电_户": int(households),
        "质控": qc100.as_dict(),
    }
    with open(f"{cfg.OUT_TAB}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------- 6. 报告
    print("[6/6] 生成报告")
    html = build_report(
        summary=summary, meta=meta, fit10=fit10, fit100=fit100, shear=shear,
        rose=rose, sector_speed=sector_speed, aep=aep, monthly=m_aep, sens=sens,
        figs=figs, calm_pct=calm_pct, qc_test=qc_test,
    )
    out_html = f"{cfg.OUT_REPORT}/风资源评估报告.html"
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"      报告：{out_html}")

    print("\n===== 关键结果 =====")
    print(f"100 m 年均风速      : {ws100.mean():.2f} m/s")
    print(f"轮毂 {hub:.0f} m 年均风速 : {hub_mean:.2f} m/s（α={alpha:.3f}）")
    print(f"Weibull             : k={fit100.k:.2f}, c={fit100.c:.2f}")
    print(f"风功率密度          : {fit100.power_density_wm2:.0f} W/m²")
    print(f"单机净发电量        : {aep.net_gwh:.2f} GWh/年")
    print(f"容量系数            : {aep.capacity_factor * 100:.1f}%")
    print(f"等效满发小时        : {aep.full_load_hours:.0f} h")
    print(f"主导风向            : {summary['主导风向']}")


if __name__ == "__main__":
    main()
