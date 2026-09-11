from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.paths import project_path, output_path


import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dropout,
    Flatten,
    Concatenate,
    Dense
)


# ============================================================
# STEP 1
# 固定随机种子
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# STEP 2
# 路径
# ============================================================

TRAIN_FILE = project_path('data/processed/07_cnn/16_cnn_train_data.npz')

VALID_FILE = project_path('data/processed/07_cnn/16_cnn_test_data.npz')

OUTPUT_DIR = project_path('results/metrics/07_cnn')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 3
# 读取数据
# ============================================================

print("读取CNN训练数据...")

train_data = np.load(
    TRAIN_FILE,
    allow_pickle=True
)

valid_data = np.load(
    VALID_FILE,
    allow_pickle=True
)


X_raw_train = train_data["X_raw"]
X_stats_train = train_data["X_stats"]
y_train_text = train_data["y"]

X_raw_valid = valid_data["X_raw"]
X_stats_valid = valid_data["X_stats"]
y_valid_text = valid_data["y"]


print(
    "X_raw_train:",
    X_raw_train.shape
)

print(
    "X_stats_train:",
    X_stats_train.shape
)

print(
    "X_raw_valid:",
    X_raw_valid.shape
)

print(
    "X_stats_valid:",
    X_stats_valid.shape
)


# ============================================================
# STEP 4
# PEAK NaN 中位数填补
#
# 只 fit 训练集
# validation 只能 transform
# ============================================================

print()
print(
    "========== PEAK缺失值处理 =========="
)

imputer = SimpleImputer(
    strategy="median"
)

X_stats_train = imputer.fit_transform(
    X_stats_train
)

X_stats_valid = imputer.transform(
    X_stats_valid
)


print(
    "训练集剩余NaN：",
    np.isnan(
        X_stats_train
    ).sum()
)

print(
    "验证集剩余NaN：",
    np.isnan(
        X_stats_valid
    ).sum()
)


# ============================================================
# STEP 5
# 标签编码
# ============================================================

CLASS_NAMES = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing"
]


class_to_index = {
    name: index
    for index, name
    in enumerate(CLASS_NAMES)
}


y_train = np.array(
    [
        class_to_index[label]
        for label in y_train_text
    ]
)

y_valid = np.array(
    [
        class_to_index[label]
        for label in y_valid_text
    ]
)


# ============================================================
# STEP 6
# 将三轴拆开
#
# 论文描述为三轴并行处理
# 每个分支输入 shape:
# (200, 1)
# ============================================================

X_train_x = X_raw_train[:, :, 0:1]
X_train_y = X_raw_train[:, :, 1:2]
X_train_z = X_raw_train[:, :, 2:3]

X_valid_x = X_raw_valid[:, :, 0:1]
X_valid_y = X_raw_valid[:, :, 1:2]
X_valid_z = X_raw_valid[:, :, 2:3]


print()
print(
    "单轴训练输入：",
    X_train_x.shape
)


# ============================================================
# STEP 7
# 定义单轴CNN分支
#
# 对照论文：
#
# filters = 196
# kernel = 12
# stride = 1
# activation = sigmoid
# dropout = 0.15
# pool_size = 4
# ============================================================

def build_axis_branch(
    input_name
):

    input_layer = Input(
        shape=(200, 1),
        name=input_name
    )

    x = Conv1D(
        filters=196,
        kernel_size=12,
        strides=1,
        activation="sigmoid",
        padding="same"
    )(
        input_layer
    )

    x = Dropout(
        0.15
    )(
        x
    )

    x = MaxPooling1D(
        pool_size=4
    )(
        x
    )

    x = Flatten()(
        x
    )

    return (
        input_layer,
        x
    )


# ============================================================
# STEP 8
# 三轴并行分支
# ============================================================

x_input, x_branch = build_axis_branch(
    "x_axis"
)

y_input, y_branch = build_axis_branch(
    "y_axis"
)

z_input, z_branch = build_axis_branch(
    "z_axis"
)


# ============================================================
# STEP 9
# 43维统计特征输入
# ============================================================

stats_input = Input(
    shape=(43,),
    name="statistical_features"
)


# ============================================================
# STEP 10
# 融合
#
# 论文：
# 卷积结果展平
# +
# 统计特征
# ============================================================

merged = Concatenate(
    name="feature_fusion"
)(
    [
        x_branch,
        y_branch,
        z_branch,
        stats_input
    ]
)


# ============================================================
# STEP 11
# 全连接输出
#
# 六种活动
# ============================================================

output = Dense(
    6,
    activation="softmax",
    name="activity_output"
)(
    merged
)


model = Model(
    inputs=[
        x_input,
        y_input,
        z_input,
        stats_input
    ],
    outputs=output
)


# ============================================================
# STEP 12
# 查看网络结构
# ============================================================

print()
print(
    "========== CNN模型结构 =========="
)

model.summary()


# ============================================================
# STEP 13
# 编译模型
#
# 注意：
# Adam / learning_rate
# 不是论文明确给出的参数
# 属于我们的实现选择
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# STEP 14
# 训练参数
#
# batch_size / epochs
# 原论文未明确完整给出
#
# 这里先采用：
# batch_size = 32
# epochs = 50
# ============================================================

BATCH_SIZE = 32

EPOCHS = 50


print()
print(
    "开始训练CNN..."
)


history = model.fit(

    [
        X_train_x,
        X_train_y,
        X_train_z,
        X_stats_train
    ],

    y_train,

    validation_data=(

        [
            X_valid_x,
            X_valid_y,
            X_valid_z,
            X_stats_valid
        ],

        y_valid
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    verbose=1
)


# ============================================================
# STEP 15
# 验证集预测
# ============================================================

print()
print(
    "========== CNN预测 =========="
)


probabilities = model.predict(

    [
        X_valid_x,
        X_valid_y,
        X_valid_z,
        X_stats_valid
    ],

    verbose=0
)


y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# STEP 16
# 总体评价指标
# ============================================================

accuracy = accuracy_score(
    y_valid,
    y_pred
)

macro_recall = recall_score(
    y_valid,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_valid,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_valid,
    y_pred,
    average="weighted",
    zero_division=0
)


print()

print(
    "Overall Accuracy:",
    f"{accuracy:.4f}"
)

print(
    "Macro Recall:",
    f"{macro_recall:.4f}"
)

print(
    "Macro F1:",
    f"{macro_f1:.4f}"
)

print(
    "Weighted F1:",
    f"{weighted_f1:.4f}"
)


# ============================================================
# STEP 17
# 每类指标
# ============================================================

print()
print(
    "========== Classification Report =========="
)

report = classification_report(

    y_valid,
    y_pred,

    labels=range(6),

    target_names=CLASS_NAMES,

    zero_division=0
)

print(
    report
)


# ============================================================
# STEP 18
# 混淆矩阵
# ============================================================

cm = confusion_matrix(
    y_valid,
    y_pred,
    labels=range(6)
)


cm_df = pd.DataFrame(
    cm,
    index=[
        f"true_{x}"
        for x in CLASS_NAMES
    ],
    columns=[
        f"pred_{x}"
        for x in CLASS_NAMES
    ]
)


cm_df.to_csv(
    output_path('results/confusion_matrix/07_cnn', "17_cnn_confusion_matrix.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# STEP 19
# 保存训练历史
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    output_path('results/metrics/07_cnn', "17_cnn_training_history.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STEP 20
# 保存总体指标
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "model":
                "CNN",

            "Accuracy":
                accuracy,

            "Macro_Recall":
                macro_recall,

            "Macro_F1":
                macro_f1,

            "Weighted_F1":
                weighted_f1
        }
    ]
)


metrics_df.to_csv(
    output_path('results/metrics/07_cnn', "17_cnn_metrics.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STEP 21
# 保存模型
# ============================================================

model.save(
    output_path('models', "17_cnn_model.keras")
)


print()
print(
    "CNN训练完成。"
)

print(
    "结果已保存到：",
    OUTPUT_DIR
)