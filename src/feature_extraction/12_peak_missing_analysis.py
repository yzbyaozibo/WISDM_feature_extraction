from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path

import pandas as pd


# ============================================================
# STEP 1
# 读取43特征数据
# ============================================================

INPUT_FILE = project_path('data/processed/04_features/10_final_43_features.csv')

OUTPUT_DIR = project_path('data/processed/04_features')

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# STEP 2
# 标记PEAK是否缺失
# ============================================================

df["XPEAK_missing"] = (
    df["XPEAK"].isna()
)

df["YPEAK_missing"] = (
    df["YPEAK"].isna()
)

df["ZPEAK_missing"] = (
    df["ZPEAK"].isna()
)


# 任意一个PEAK缺失
df["ANY_PEAK_missing"] = (
    df[
        [
            "XPEAK",
            "YPEAK",
            "ZPEAK"
        ]
    ]
    .isna()
    .any(axis=1)
)


# ============================================================
# STEP 3
# 总体缺失窗口数量
# ============================================================

print()
print(
    "========== PEAK总体缺失 =========="
)

print(
    "XPEAK缺失：",
    df["XPEAK_missing"].sum()
)

print(
    "YPEAK缺失：",
    df["YPEAK_missing"].sum()
)

print(
    "ZPEAK缺失：",
    df["ZPEAK_missing"].sum()
)

print(
    "至少一个PEAK缺失的窗口数：",
    df["ANY_PEAK_missing"].sum()
)


missing_rate = (
    df["ANY_PEAK_missing"].mean()
    *
    100
)

print(
    "至少一个PEAK缺失的窗口比例：",
    f"{missing_rate:.4f}%"
)


# ============================================================
# STEP 4
# 按活动类别统计
# ============================================================

class_stats = (
    df.groupby("class")
    .agg(

        total_windows=(
            "window_id",
            "count"
        ),

        XPEAK_missing=(
            "XPEAK_missing",
            "sum"
        ),

        YPEAK_missing=(
            "YPEAK_missing",
            "sum"
        ),

        ZPEAK_missing=(
            "ZPEAK_missing",
            "sum"
        ),

        ANY_PEAK_missing=(
            "ANY_PEAK_missing",
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# STEP 5
# 计算缺失率
# ============================================================

class_stats[
    "ANY_PEAK_missing_percent"
] = (

    class_stats[
        "ANY_PEAK_missing"
    ]

    /

    class_stats[
        "total_windows"
    ]

    *

    100
)


print()
print(
    "========== 各活动PEAK缺失情况 =========="
)

print(
    class_stats.to_string(
        index=False
    )
)


# ============================================================
# STEP 6
# 保存
# ============================================================

class_stats.to_csv(
    output_path('results/metrics/04_features', "12_peak_missing_by_class.csv"),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "结果已保存到："
    "data/processed/04_features"
    "12_peak_missing_by_class.csv"
)