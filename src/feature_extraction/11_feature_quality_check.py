from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path

import pandas as pd
import numpy as np


# ============================================================
# STEP 1
# 读取最终43特征文件
# ============================================================

INPUT_FILE = project_path('data/processed/04_features/10_final_43_features.csv')

OUTPUT_DIR = project_path('data/processed/04_features')


print("开始读取43特征数据...")

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# STEP 2
# 基础结构检查
# ============================================================

print()
print("========== 数据结构 ==========")

print(
    "样本数：",
    len(df)
)

print(
    "总列数：",
    len(df.columns)
)


metadata_columns = [
    "window_id",
    "id",
    "class"
]

feature_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]


print(
    "特征数量：",
    len(feature_columns)
)


# ============================================================
# STEP 3
# 缺失值检查
# ============================================================

print()
print("========== 缺失值 ==========")

missing = (
    df[feature_columns]
    .isna()
    .sum()
)

missing = missing[
    missing > 0
]

if len(missing) == 0:

    print(
        "没有发现缺失特征值"
    )

else:

    print(
        missing.to_string()
    )


# ============================================================
# STEP 4
# 无穷值检查
# ============================================================

numeric_features = df[
    feature_columns
].select_dtypes(
    include=[np.number]
)


inf_count = np.isinf(
    numeric_features
).sum().sum()


print()
print("========== 无穷值 ==========")

print(
    "Inf数量：",
    inf_count
)


# ============================================================
# STEP 5
# 重复窗口检查
# ============================================================

duplicate_count = df.duplicated(
    subset=feature_columns + ["class"]
).sum()


print()
print("========== 重复特征记录 ==========")

print(
    "完全重复的特征记录数：",
    duplicate_count
)


# ============================================================
# STEP 6
# 类别分布检查
# ============================================================

class_count = (
    df["class"]
    .value_counts()
)


print()
print("========== 类别分布 ==========")

print(
    class_count.to_string()
)


# ============================================================
# STEP 7
# 检查PEAK缺失
# ============================================================

print()
print("========== PEAK检查 ==========")

for col in [
    "XPEAK",
    "YPEAK",
    "ZPEAK"
]:

    print(
        col,
        "缺失数量：",
        df[col].isna().sum()
    )


# ============================================================
# STEP 8
# 保存质量统计
# ============================================================

quality_summary = pd.DataFrame(
    {
        "feature":
            feature_columns,

        "missing_count":
            [
                df[col].isna().sum()
                for col in feature_columns
            ],

        "min":
            [
                df[col].min()
                for col in feature_columns
            ],

        "max":
            [
                df[col].max()
                for col in feature_columns
            ],

        "mean":
            [
                df[col].mean()
                for col in feature_columns
            ],

        "std":
            [
                df[col].std()
                for col in feature_columns
            ]
    }
)


quality_summary.to_csv(
    output_path('results/metrics/04_features', "11_feature_quality_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "质量检查结果已保存到："
    "data/processed/04_features"
    "11_feature_quality_summary.csv"
)