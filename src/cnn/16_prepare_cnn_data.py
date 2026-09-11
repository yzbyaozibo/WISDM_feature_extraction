from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path


import numpy as np
import pandas as pd


# ============================================================
# STEP 1
# 文件路径
# ============================================================

RAW_FILE = project_path('data/processed/01_cleaning/01_clean_data.csv')

FEATURE_FILE = project_path('data/processed/04_features/10_final_43_features.csv')

OUTPUT_DIR = project_path('results/metrics/07_cnn')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 2
# 参数
# ============================================================

WINDOW_SIZE = 200

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
# STEP 3
# 读取数据
# ============================================================

print("读取原始清洗数据...")

raw_df = pd.read_csv(
    RAW_FILE
)


print("读取43特征数据...")

feature_df = pd.read_csv(
    FEATURE_FILE
)


print()
print(
    "原始记录数：",
    len(raw_df)
)

print(
    "特征窗口数：",
    len(feature_df)
)


# ============================================================
# STEP 4
# 定义43个统计特征列
# ============================================================

NON_FEATURE_COLUMNS = [
    "window_id",
    "id",
    "class"
]


FEATURE_COLUMNS = [
    col
    for col in feature_df.columns
    if col not in NON_FEATURE_COLUMNS
]


print(
    "统计特征数量：",
    len(FEATURE_COLUMNS)
)


# ============================================================
# STEP 5
# 按之前完全相同的方法重新构建200点窗口
# ============================================================

raw_windows = []

window_ids = []

subject_ids = []

labels = []


grouped = raw_df.groupby(
    [
        "id",
        "class"
    ],
    sort=False
)


window_id = 1


for (user_id, activity), group in grouped:

    group = group.reset_index(
        drop=True
    )

    complete_windows = (
        len(group)
        //
        WINDOW_SIZE
    )


    for i in range(
        complete_windows
    ):

        start = (
            i
            *
            WINDOW_SIZE
        )

        end = (
            start
            +
            WINDOW_SIZE
        )


        window = group.iloc[
            start:end
        ]


        # ----------------------------------------------------
        # 当前窗口：
        # shape = (200, 3)
        # ----------------------------------------------------

        xyz = window[
            [
                "x",
                "y",
                "z"
            ]
        ].to_numpy(
            dtype=np.float32
        )


        raw_windows.append(
            xyz
        )

        window_ids.append(
            window_id
        )

        subject_ids.append(
            user_id
        )

        labels.append(
            activity
        )


        window_id += 1


# ============================================================
# STEP 6
# 转为numpy数组
# ============================================================

X_raw = np.stack(
    raw_windows
)

window_ids = np.array(
    window_ids
)

subject_ids = np.array(
    subject_ids
)

labels = np.array(
    labels
)


print()
print(
    "========== 原始CNN窗口 =========="
)

print(
    "X_raw shape：",
    X_raw.shape
)


# ============================================================
# STEP 7
# 读取43维统计特征
# ============================================================

X_stats = feature_df[
    FEATURE_COLUMNS
].to_numpy(
    dtype=np.float32
)


print(
    "X_stats shape：",
    X_stats.shape
)


# ============================================================
# STEP 8
# 检查窗口数量一致
# ============================================================

if len(X_raw) != len(feature_df):

    raise ValueError(
        "原始窗口数量和43特征窗口数量不一致！"
    )


# ============================================================
# STEP 9
# 检查 id / class 是否逐窗口一致
# ============================================================

feature_subject_ids = feature_df[
    "id"
].to_numpy()


feature_labels = feature_df[
    "class"
].to_numpy()


id_match = np.array_equal(
    subject_ids,
    feature_subject_ids
)

label_match = np.array_equal(
    labels,
    feature_labels
)


print()
print(
    "========== 对齐检查 =========="
)

print(
    "ID逐行一致：",
    id_match
)

print(
    "class逐行一致：",
    label_match
)


if not id_match:

    raise ValueError(
        "原始窗口和特征数据的ID顺序不一致！"
    )


if not label_match:

    raise ValueError(
        "原始窗口和特征数据的class顺序不一致！"
    )


# ============================================================
# STEP 10
# 按受试者划分训练 / 测试
# ============================================================

test_mask = np.isin(
    subject_ids,
    TEST_SUBJECTS
)

train_mask = ~test_mask


X_raw_train = X_raw[
    train_mask
]

X_raw_test = X_raw[
    test_mask
]


X_stats_train = X_stats[
    train_mask
]

X_stats_test = X_stats[
    test_mask
]


y_train = labels[
    train_mask
]

y_test = labels[
    test_mask
]


train_subject_ids = subject_ids[
    train_mask
]

test_subject_ids = subject_ids[
    test_mask
]


train_window_ids = window_ids[
    train_mask
]

test_window_ids = window_ids[
    test_mask
]


# ============================================================
# STEP 11
# 输出形状检查
# ============================================================

print()
print(
    "========== CNN数据划分 =========="
)

print(
    "X_raw_train：",
    X_raw_train.shape
)

print(
    "X_stats_train：",
    X_stats_train.shape
)

print(
    "y_train：",
    y_train.shape
)


print()

print(
    "X_raw_test：",
    X_raw_test.shape
)

print(
    "X_stats_test：",
    X_stats_test.shape
)

print(
    "y_test：",
    y_test.shape
)


# ============================================================
# STEP 12
# 检查43维统计特征中的NaN
# ============================================================

print()
print(
    "========== 统计特征NaN =========="
)

print(
    "训练集NaN数量：",
    np.isnan(
        X_stats_train
    ).sum()
)

print(
    "测试集NaN数量：",
    np.isnan(
        X_stats_test
    ).sum()
)


# ============================================================
# STEP 13
# 保存CNN训练集
# ============================================================

np.savez_compressed(

    output_path('data/processed/07_cnn', "16_cnn_train_data.npz"),

    X_raw=X_raw_train,

    X_stats=X_stats_train,

    y=y_train,

    subject_id=train_subject_ids,

    window_id=train_window_ids
)


# ============================================================
# STEP 14
# 保存CNN测试集
# ============================================================

np.savez_compressed(

    output_path('data/processed/07_cnn', "16_cnn_test_data.npz"),

    X_raw=X_raw_test,

    X_stats=X_stats_test,

    y=y_test,

    subject_id=test_subject_ids,

    window_id=test_window_ids
)


print()
print(
    "CNN训练/测试数据准备完成。"
)

print(
    "保存目录：",
    OUTPUT_DIR
)