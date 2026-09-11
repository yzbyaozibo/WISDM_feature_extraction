from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path

import pandas as pd
import numpy as np


# ============================================================
# STEP 1
# 路径
# ============================================================

INPUT_FILE = project_path('data/processed/01_cleaning/01_clean_data.csv')

OUTPUT_DIR = project_path('results/metrics/02_statistics')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 2
# 读取清洗后的数据
# ============================================================

print("开始读取清洗后的数据...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "数据记录数：",
    len(df)
)


# ============================================================
# STEP 3
# 计算相邻记录是否属于同一用户和同一活动
# ============================================================

same_id = (
    df["id"]
    ==
    df["id"].shift(1)
)

same_class = (
    df["class"]
    ==
    df["class"].shift(1)
)

same_segment = (
    same_id
    &
    same_class
)


# ============================================================
# STEP 4
# 计算相邻时间戳差
# ============================================================

df["timestamp_diff"] = (
    df["timestamp"]
    -
    df["timestamp"].shift(1)
)


# 只保留：
# 1. 同一用户
# 2. 同一活动
# 3. 前后时间戳均大于0
# 4. 时间差为正
valid_mask = (

    same_segment

    &

    (df["timestamp"] > 0)

    &

    (df["timestamp"].shift(1) > 0)

    &

    (df["timestamp_diff"] > 0)
)


valid_diffs = (
    df.loc[
        valid_mask,
        "timestamp_diff"
    ]
    .copy()
)


# ============================================================
# STEP 5
# 纳秒转换为秒
# ============================================================

valid_diff_seconds = (
    valid_diffs
    /
    1_000_000_000
)


# ============================================================
# STEP 6
# 统计采样间隔
# ============================================================

median_interval = (
    valid_diff_seconds.median()
)

mean_interval = (
    valid_diff_seconds.mean()
)

std_interval = (
    valid_diff_seconds.std()
)

min_interval = (
    valid_diff_seconds.min()
)

max_interval = (
    valid_diff_seconds.max()
)


# 根据中位时间间隔估算采样率
estimated_sampling_rate = (
    1 / median_interval
)


print()
print(
    "========== 时间戳统计 =========="
)

print(
    "有效相邻时间间隔数量：",
    len(valid_diff_seconds)
)

print(
    "平均采样间隔：",
    f"{mean_interval:.6f} 秒"
)

print(
    "中位采样间隔：",
    f"{median_interval:.6f} 秒"
)

print(
    "采样间隔标准差：",
    f"{std_interval:.6f} 秒"
)

print(
    "最小采样间隔：",
    f"{min_interval:.6f} 秒"
)

print(
    "最大采样间隔：",
    f"{max_interval:.6f} 秒"
)

print(
    "根据中位数估算采样率：",
    f"{estimated_sampling_rate:.4f} Hz"
)


# ============================================================
# STEP 7
# 计算关键分位数
# ============================================================

quantiles = (
    valid_diff_seconds
    .quantile(
        [
            0.01,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.99
        ]
    )
)

print()
print(
    "========== 采样间隔分位数 =========="
)

print(
    quantiles
)


# ============================================================
# STEP 8
# 保存统计结果
# ============================================================

summary = pd.DataFrame(
    [
        {
            "valid_interval_count":
                len(valid_diff_seconds),

            "mean_interval_seconds":
                mean_interval,

            "median_interval_seconds":
                median_interval,

            "std_interval_seconds":
                std_interval,

            "min_interval_seconds":
                min_interval,

            "max_interval_seconds":
                max_interval,

            "estimated_sampling_rate_hz":
                estimated_sampling_rate
        }
    ]
)


summary.to_csv(
    output_path('results/metrics/02_statistics', "03_timestamp_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)


# 分位数单独保存
quantile_df = (
    quantiles
    .reset_index()
)

quantile_df.columns = [
    "quantile",
    "interval_seconds"
]


quantile_df.to_csv(
    output_path('results/metrics/02_statistics', "03_timestamp_quantiles.csv"),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "时间戳统计已保存到：",
    OUTPUT_DIR
)