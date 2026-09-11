from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path

import pandas as pd


# ============================================================
# STEP 1
# 文件路径
# ============================================================

INPUT_FILE = project_path('data/processed/04_features/10_final_43_features.csv')

OUTPUT_DIR = project_path('data/processed/05_split')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 2
# 读取43特征数据
# ============================================================

print("开始读取43特征数据...")

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# STEP 3
# 定义测试受试者
#
# 这些受试者完整留作测试集
# 训练阶段完全不可见
# ============================================================

TEST_SUBJECTS = [
    5,
    6,
    7,
    13,
    24,
    33,
    34
]


# ============================================================
# STEP 4
# 划分训练集和测试集
# ============================================================

test_df = df[
    df["id"].isin(
        TEST_SUBJECTS
    )
].copy()


train_df = df[
    ~df["id"].isin(
        TEST_SUBJECTS
    )
].copy()


# ============================================================
# STEP 5
# 检查受试者是否重叠
# ============================================================

train_subjects = set(
    train_df["id"].unique()
)

test_subjects = set(
    test_df["id"].unique()
)

overlap = (
    train_subjects
    &
    test_subjects
)


print()
print(
    "========== 受试者独立性检查 =========="
)

print(
    "训练受试者数量：",
    len(train_subjects)
)

print(
    "测试受试者数量：",
    len(test_subjects)
)

print(
    "训练/测试重复受试者：",
    overlap
)


# ============================================================
# STEP 6
# 样本数量检查
# ============================================================

print()
print(
    "========== 样本数量 =========="
)

print(
    "全部窗口：",
    len(df)
)

print(
    "训练窗口：",
    len(train_df)
)

print(
    "测试窗口：",
    len(test_df)
)


train_percent = (
    len(train_df)
    /
    len(df)
    *
    100
)

test_percent = (
    len(test_df)
    /
    len(df)
    *
    100
)


print(
    "训练集比例：",
    f"{train_percent:.2f}%"
)

print(
    "测试集比例：",
    f"{test_percent:.2f}%"
)


# ============================================================
# STEP 7
# 类别分布
# ============================================================

print()
print(
    "========== 训练集类别分布 =========="
)

print(
    train_df[
        "class"
    ]
    .value_counts()
    .to_string()
)


print()
print(
    "========== 测试集类别分布 =========="
)

print(
    test_df[
        "class"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# STEP 8
# 保存划分结果
# ============================================================

train_df.to_csv(
    output_path('data/processed/05_split', "14_train_subject_independent.csv"),
    index=False,
    encoding="utf-8-sig"
)


test_df.to_csv(
    output_path('data/processed/05_split', "14_test_subject_independent.csv"),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "训练集已保存："
    "data/processed/05_split"
    "14_train_subject_independent.csv"
)

print(
    "测试集已保存："
    "data/processed/05_split"
    "14_test_subject_independent.csv"
)