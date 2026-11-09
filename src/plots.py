"""出图。所有图统一风格，尺寸适配报告与 GitHub README。"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from . import config as cfg
from . import analysis as an

# ---- 中文字体（Windows 自带雅黑；Linux 上会回退）
for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.edgecolor": "#9aa9b4",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#24333f",
        "text.color": "#1a1a1a",
        "xtick.color": "#4a5b6a",
        "ytick.color": "#4a5b6a",
        "grid.color": "#dfe6ea",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    }
)

NAVY = "#0F3D5C"
GOLD = "#C8A45C"
RED = "#B0392E"
GREY = "#8AA2B2"


def _save(fig, name: str, out_dir: str | None = None, tight: bool = True) -> str:
    out_dir = out_dir or cfg.OUT_FIG
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if tight:
        fig.savefig(path, facecolor="white", bbox_inches="tight")
    else:
        fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_qc(raw: pd.Series, clean: pd.Series, out_dir: str | None = None) -> str:
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.3), height_ratios=[2, 1])

    ax[0].plot(raw.index, raw.values, lw=0.4, color=GREY, label="原始值")
    ax[0].plot(clean.index, clean.values, lw=0.5, color=NAVY, label="质控/插补后")
    bad = raw.index[raw.isna()]
    if len(bad):
        ax[0].scatter(bad, np.full(len(bad), ax[0].get_ylim()[0]), s=8, color=RED,
                      label="缺测时次", zorder=3)
    ax[0].set_ylabel("风速 (m/s, 100 m)")
    ax[0].set_title("图1　100 m 风速序列与数据质控", loc="left", fontsize=11, fontweight="bold")
    ax[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    ax[0].grid(True, axis="y")

    monthly = clean.groupby(clean.index.month).mean()
    ax[1].bar(monthly.index, monthly.values, color=NAVY, alpha=0.85, width=0.65)
    ax[1].set_ylabel("月均风速 (m/s)")
    ax[1].set_xlabel("月份")
    ax[1].set_xticks(range(1, 13))
    ax[1].grid(True, axis="y")

    return _save(fig, "fig1_qc.png", out_dir)


def plot_weibull(v: pd.Series, fit, out_dir: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 3.9))

    v = v.dropna()
    ax.hist(v, bins=np.arange(0, 30, 0.5), density=True, color=GREY, alpha=0.55,
            edgecolor="white", linewidth=0.5, label="观测频率")

    x = np.linspace(0.01, 30, 400)
    ax.plot(x, fit.pdf(x), color=NAVY, lw=2.2, label=f"Weibull 拟合 (k={fit.k:.2f}, c={fit.c:.2f})")
    ax.axvline(fit.mean_ms, color=GOLD, ls="--", lw=1.4, label=f"平均风速 {fit.mean_ms:.2f} m/s")

    ax.set_xlabel("风速 (m/s, 100 m)")
    ax.set_ylabel("概率密度")
    ax.set_xlim(0, 30)
    ax.set_title("图2　风速分布与 Weibull 拟合", loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8.5)

    txt = (
        f"风功率密度 {fit.power_density_wm2:.0f} W/m²\n"
        f"KS 检验 p = {fit.ks_pvalue:.2f}\n"
        f"样本 n = {fit.n}"
    )
    ax.text(0.98, 0.72, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F2F6F9", ec="#D8E0E6"))

    return _save(fig, "fig2_weibull.png", out_dir)


def plot_rose(rose_table: pd.DataFrame, calm_pct: float, out_dir: str | None = None) -> str:
    labels = list(rose_table.index)
    n = len(labels)
    theta = np.deg2rad(np.arange(n) * 360.0 / n)
    width = np.deg2rad(360.0 / n) * 0.92

    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    colors = plt.cm.Blues(np.linspace(0.28, 0.95, len(an.SPEED_LABELS)))
    bottom = np.zeros(n)
    for label, color in zip(an.SPEED_LABELS, colors):
        vals = rose_table[label].values if label in rose_table.columns else np.zeros(n)
        ax.bar(theta, vals, width=width, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.4, label=f"{label} m/s")
        bottom += vals

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title("图3　100 m 风向玫瑰（风的来向，单位 %）", loc="left",
                 fontsize=11, fontweight="bold", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False,
              fontsize=7.4, title="风速区间", title_fontsize=7.4, handlelength=1.4,
              columnspacing=1.1, handletextpad=0.5)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.26)
    return _save(fig, "fig3_windrose.png", out_dir, tight=False)


def plot_monthly_diurnal(ws: pd.Series, out_dir: str | None = None) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))

    m = ws.groupby(ws.index.month).mean()
    s = ws.groupby(ws.index.month).std()
    ax[0].bar(m.index, m.values, yerr=s.values, capsize=3, color=NAVY, alpha=0.85,
              error_kw=dict(ecolor=GREY, lw=0.8), width=0.65)
    ax[0].set_xticks(range(1, 13))
    ax[0].set_xlabel("月份")
    ax[0].set_ylabel("风速 (m/s)")
    ax[0].set_title("月平均风速（误差棒为 1σ）", fontsize=10)
    ax[0].grid(True, axis="y")

    h = ws.groupby(ws.index.hour).mean()
    ax[1].plot(h.index, h.values, color=GOLD, lw=2.2, marker="o", ms=3.5,
               markerfacecolor="white", markeredgewidth=1.2)
    ax[1].set_xticks(range(0, 24, 3))
    ax[1].set_xlabel("小时 (UTC)")
    ax[1].set_ylabel("风速 (m/s)")
    ax[1].set_title("日变化", fontsize=10)
    ax[1].grid(True, axis="y")

    fig.suptitle("图4　风速的时间变化特征", x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fig4_monthly_diurnal.png", out_dir)


def plot_shear(ws10: pd.Series, ws100: pd.Series, alpha: float, hub: float,
               out_dir: str | None = None) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1, 1.25]})

    z = np.array([10.0, 100.0, hub])
    v = np.array([ws10.mean(), ws100.mean(), ws100.mean() * (hub / 100.0) ** alpha])
    ax[0].plot(v[:2], z[:2], "o-", color=NAVY, lw=1.6, ms=7, label="ERA5 层结")
    ax[0].plot([v[1], v[2]], [z[1], z[2]], "--", color=GOLD, lw=1.6, label="幂律外推")
    ax[0].scatter([v[2]], [z[2]], marker="D", s=55, color=GOLD, zorder=4)
    ax[0].annotate(f"{v[2]:.2f} m/s\n@ {hub:.0f} m", (v[2], z[2]), textcoords="offset points",
                   xytext=(10, -2), fontsize=8.5, color="#4a5b6a")
    ax[0].set_xlabel("平均风速 (m/s)")
    ax[0].set_ylabel("高度 (m)")
    ax[0].set_title(f"垂直廓线（α = {alpha:.3f}）", fontsize=10)
    ax[0].grid(True)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    df = pd.DataFrame({"lo": ws10, "hi": ws100}).dropna()
    df = df[(df.lo >= 2) & (df.hi >= 2)]
    a_i = np.log(df.hi / df.lo) / np.log(10.0)
    ax[1].hist(a_i, bins=np.arange(-0.2, 0.9, 0.03), color=GREY, alpha=0.65,
               edgecolor="white", linewidth=0.4)
    ax[1].axvline(alpha, color=NAVY, lw=2, label=f"中位数 {alpha:.3f}")
    ax[1].axvline(np.log(df.hi.mean() / df.lo.mean()) / np.log(10.0), color=GOLD,
                  ls="--", lw=1.6, label="均值比")
    ax[1].set_xlabel("逐时风切变指数 α")
    ax[1].set_ylabel("时次数")
    ax[1].set_title("α 的分布（仅 v≥2 m/s）", fontsize=10)
    ax[1].grid(True, axis="y")
    ax[1].legend(frameon=False, fontsize=8)

    fig.suptitle("图5　风切变与轮毂高度外推", x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig5_shear.png", out_dir)


def plot_aep(ws_hub: pd.Series, aep, turbine=cfg.TURBINE, out_dir: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8.2, 3.7))

    v = ws_hub.dropna()
    counts, edges = np.histogram(v, bins=np.arange(0, 30.5, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])
    freq = counts / counts.sum() * 100

    ax.bar(centres, freq, width=0.85, color=GREY, alpha=0.55, edgecolor="white",
           linewidth=0.4, label="风速频率")
    ax.set_xlabel("风速 (m/s)")
    ax.set_ylabel("频率 (%)")
    ax.set_xlim(0, 30)
    ax.grid(True, axis="y")

    ax2 = ax.twinx()
    xs = np.linspace(0, 30, 600)
    ax2.plot(xs, np.interp(xs, [p[0] for p in turbine.curve], [p[1] for p in turbine.curve],
                           left=0, right=0) / 1000.0, color=NAVY, lw=2.2, label="功率曲线")
    ax2.set_ylabel("单机功率 (MW)")
    ax2.set_ylim(0, turbine.rated_kw / 1000 * 1.25)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper right")

    txt = (
        f"单机净发电量 {aep.net_gwh:.2f} GWh/年\n"
        f"容量系数 {aep.capacity_factor * 100:.1f}%\n"
        f"等效满发 {aep.full_load_hours:.0f} h"
    )
    ax.text(0.02, 0.72, txt, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F2F6F9", ec="#D8E0E6"))
    ax.set_title("图6　风速分布与机型功率曲线匹配", loc="left", fontsize=11, fontweight="bold")

    return _save(fig, "fig6_aep.png", out_dir)
