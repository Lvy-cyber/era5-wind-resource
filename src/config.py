"""项目配置：站点、机型、损失假设。

所有"假设"都集中在这里，报告里会原样引用——这是可复现的前提，
也是面试时能被追问的部分（改一个参数，AEP 会变多少）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Site:
    name: str
    province: str
    lat: float
    lon: float
    elevation_m: int
    hub_height_m: float
    # ERA5 在 Open-Meteo 上可取的层结高度（m）
    levels_m: tuple[int, ...] = (10, 100)


@dataclass(frozen=True)
class Turbine:
    """代表性机型曲线，非特定厂商型号。"""

    label: str
    rated_kw: float
    rotor_d_m: float
    cut_in_ms: float
    rated_ms: float
    cut_out_ms: float
    # (风速 m/s, 功率 kW) 分段线性
    curve: tuple[tuple[float, float], ...]


SITE = Site(
    name="辉腾锡勒",
    province="内蒙古自治区乌兰察布市察哈尔右翼中旗",
    lat=41.25,
    lon=112.75,
    elevation_m=2002,
    hub_height_m=110.0,
)

# 5 MW 级、156 m 叶轮：陆上大基地主流机型量级
TURBINE = Turbine(
    label="5.0 MW / Φ156 m（代表性曲线）",
    rated_kw=5000.0,
    rotor_d_m=156.0,
    cut_in_ms=3.0,
    rated_ms=10.5,
    cut_out_ms=25.0,
    curve=(
        (0.0, 0.0),
        (3.0, 0.0),
        (4.0, 250.0),
        (5.0, 550.0),
        (6.0, 1000.0),
        (7.0, 1600.0),
        (8.0, 2400.0),
        (9.0, 3300.0),
        (10.0, 4300.0),
        (10.5, 5000.0),
        (25.0, 5000.0),
        (25.1, 0.0),
    ),
)

# 逐项折减（乘性），合计约 12%
LOSSES: dict[str, float] = {
    "尾流损失": 0.06,
    "机组可利用率": 0.03,
    "电气与线损": 0.02,
    "限电": 0.02,
}

DATE_START = "2024-01-01"
DATE_END = "2024-12-31"

RHO_REF = 1.225  # kg/m3，功率曲线标定工况
R_D = 287.05  # 干空气气体常数 J/(kg·K)

DATA_RAW = "data/raw/era5_huitengxile_2024.csv"
DATA_META = "data/raw/era5_huitengxile_2024.meta.json"
DATA_CLEAN = "data/processed/clean_hourly.csv"
OUT_FIG = "output/figures"
OUT_TAB = "output/tables"
OUT_REPORT = "report"
