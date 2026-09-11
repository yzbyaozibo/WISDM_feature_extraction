from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path

import pandas as pd


# ============================================================
# STEP 1
# 定义输入和输出路径
# ============================================================

INPUT_FILE = project_path('data/processed/01_cleaning/01_clean_data.csv')

OUTPUT_DIR = project_path('results/metrics/02_statistics')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 2
# 读取清洗后的完整数据
# ============================================================

print("开始读取清洗后的数据...")

df = pd.read_csv(
    INPUT_FILE
)

print("数据读取完成")

print(
    "完整数据记录数：",
    len(df)
)


# ============================================================
# STEP 3
# 统计用户数量
# ============================================================

user_count = df["id"].nunique()

print()
print(
    "========== 用户统计 =========="
)

print(
    "用户数量：",
    user_count
)


# ============================================================
# STEP 4
# 统计活动类别数量
# ============================================================

class_count = df["class"].nunique()

class_names = sorted(
    df["class"].unique()
)

print()
print(
    "========== 活动类别统计 =========="
)

print(
    "活动类别数量：",
    class_count
)

print(
    "活动类别：",
    class_names
)


# ============================================================
# STEP 5
# 统计每个活动类别的原始采样点数量
# ============================================================

class_statistics = (
    df
    .groupby("class")
    .size()
    .reset_index(
        name="raw_sample_count"
    )
)


# 计算每个类别占全部数据的百分比
class_statistics[
    "percentage"
] = (
    class_statistics[
        "raw_sample_count"
    ]
    /
    len(df)
    *
    100
)


# 按样本数量从大到小排列
class_statistics = (
    class_statistics
    .sort_values(
        by="raw_sample_count",
        ascending=False
    )
    .reset_index(drop=True)
)


print()
print(
    "========== 各活动类别原始数据量 =========="
)

print(
    class_statistics.to_string(
        index=False
    )
)


# 保存类别统计
class_statistics.to_csv(
    output_path('results/metrics/02_statistics', "02_raw_class_statistics.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STEP 6
# 统计每个用户的原始采样点数量
# ============================================================

user_statistics = (
    df
    .groupby("id")
    .size()
    .reset_index(
        name="raw_sample_count"
    )
)


# 按用户ID排序
user_statistics = (
    user_statistics
    .sort_values(
        by="id"
    )
    .reset_index(drop=True)
)


print()
print(
    "========== 前10个用户的数据量 =========="
)

print(
    user_statistics
    .head(10)
    .to_string(
        index=False
    )
)


# 保存用户统计
user_statistics.to_csv(
    output_path('results/metrics/02_statistics', "02_raw_user_statistics.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STEP 7
# 统计每个用户 × 每种活动的数据量
# ============================================================

user_class_statistics = (
    df
    .groupby(
        ["id", "class"]
    )
    .size()
    .reset_index(
        name="raw_sample_count"
    )
)


user_class_statistics.to_csv(
    output_path('results/metrics/02_statistics', "02_user_class_statistics.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STEP 8
# 保存本阶段总体摘要
# ============================================================

summary = pd.DataFrame(
    [
        {
            "total_records":
                len(df),

            "user_count":
                user_count,

            "class_count":
                class_count
        }
    ]
)


summary.to_csv(
    output_path('results/metrics/02_statistics', "02_statistics_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "========== 统计完成 =========="
)

print(
    "统计结果已保存到：",
    OUTPUT_DIR
)