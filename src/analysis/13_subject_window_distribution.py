from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path

import pandas as pd


# ============================================================
# STEP 1
# 读取最终43特征数据
# ============================================================

INPUT_FILE = project_path('data/processed/04_features/10_final_43_features.csv')

OUTPUT_DIR = project_path('data/processed/05_split')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("开始读取43特征数据...")

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# STEP 2
# 统计每个受试者、每个活动的窗口数量
# ============================================================

subject_class_table = pd.crosstab(
    df["id"],
    df["class"]
)


# ============================================================
# STEP 3
# 保证六类活动列顺序固定
# ============================================================

activity_order = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing"
]


for activity in activity_order:

    if activity not in subject_class_table.columns:

        subject_class_table[
            activity
        ] = 0


subject_class_table = subject_class_table[
    activity_order
]


# ============================================================
# STEP 4
# 增加每个受试者的总窗口数
# ============================================================

subject_class_table[
    "Total"
] = subject_class_table.sum(
    axis=1
)


# ============================================================
# STEP 5
# 统计每个受试者覆盖了多少类活动
# ============================================================

subject_class_table[
    "Activity_count"
] = (
    subject_class_table[
        activity_order
    ]
    > 0
).sum(
    axis=1
)


# ============================================================
# STEP 6
# 输出总体信息
# ============================================================

print()
print(
    "========== 受试者窗口分布 =========="
)

print(
    subject_class_table
    .to_string()
)


print()
print(
    "受试者总数：",
    len(subject_class_table)
)


print()
print(
    "========== 每位受试者活动覆盖数 =========="
)

print(
    subject_class_table[
        "Activity_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# STEP 7
# 检查哪些受试者六类活动都有
# ============================================================

all_six = subject_class_table[
    subject_class_table[
        "Activity_count"
    ] == 6
]


print()
print(
    "六类活动都有的受试者数量：",
    len(all_six)
)


print(
    "这些受试者ID："
)

print(
    list(
        all_six.index
    )
)


# ============================================================
# STEP 8
# 保存结果
# ============================================================

subject_class_table.to_csv(
    output_path('results/metrics/05_split', "13_subject_window_distribution.csv"),
    encoding="utf-8-sig"
)


print()
print(
    "结果已保存到："
    "data/processed/05_split"
    "13_subject_window_distribution.csv"
)