from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path


import numpy as np
import pandas as pd


# ==============================
# Keras 3 写法
# ==============================

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

from tensorflow.keras.optimizers import Adam


from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==============================
# 随机种子
# ==============================

np.random.seed(42)


# ==============================
# 路径
# ==============================

TRAIN_FILE = project_path('data/processed/07_cnn/16_cnn_train_data.npz')

TEST_FILE = project_path('data/processed/07_cnn/16_cnn_test_data.npz')

OUTPUT_DIR = project_path('results/metrics/07_cnn')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================
# 读取数据
# ==============================

print("读取CNN数据...")


train = np.load(
    TRAIN_FILE,
    allow_pickle=True
)

test = np.load(
    TEST_FILE,
    allow_pickle=True
)


X_raw_train = train["X_raw"]
X_stats_train = train["X_stats"]
y_train_text = train["y"]


X_raw_test = test["X_raw"]
X_stats_test = test["X_stats"]
y_test_text = test["y"]


print(
    "X_raw_train:",
    X_raw_train.shape
)

print(
    "X_stats_train:",
    X_stats_train.shape
)

print(
    "X_raw_test:",
    X_raw_test.shape
)

print(
    "X_stats_test:",
    X_stats_test.shape
)



# ==============================
# PEAK NaN处理
# 只使用训练集计算median
# ==============================


print()
print("处理PEAK缺失...")


imputer = SimpleImputer(
    strategy="median"
)


X_stats_train = imputer.fit_transform(
    X_stats_train
)


X_stats_test = imputer.transform(
    X_stats_test
)


print(
    "训练NaN:",
    np.isnan(X_stats_train).sum()
)


print(
    "测试NaN:",
    np.isnan(X_stats_test).sum()
)



# ==============================
# 标签编码
# ==============================


classes = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing"
]


label_map = {
    c:i
    for i,c in enumerate(classes)
}


y_train = np.array(
    [
        label_map[x]
        for x in y_train_text
    ]
)


y_test = np.array(
    [
        label_map[x]
        for x in y_test_text
    ]
)



# ==============================
# 三轴拆分
# ==============================


X_train_x = X_raw_train[:,:,0:1]
X_train_y = X_raw_train[:,:,1:2]
X_train_z = X_raw_train[:,:,2:3]


X_test_x = X_raw_test[:,:,0:1]
X_test_y = X_raw_test[:,:,1:2]
X_test_z = X_raw_test[:,:,2:3]



# ==============================
# 单轴CNN模块
# ==============================


def create_branch(name):

    inp = Input(
        shape=(200,1),
        name=name
    )


    x = Conv1D(
        filters=196,
        kernel_size=12,
        strides=1,
        padding="same",
        activation="sigmoid"
    )(inp)


    x = Dropout(
        0.15
    )(x)


    x = MaxPooling1D(
        pool_size=4
    )(x)


    x = Flatten()(x)


    return inp,x



# ==============================
# 建立三个轴分支
# ==============================


input_x, feature_x = create_branch(
    "axis_x"
)


input_y, feature_y = create_branch(
    "axis_y"
)


input_z, feature_z = create_branch(
    "axis_z"
)



# ==============================
# 统计特征输入
# ==============================


input_stats = Input(
    shape=(43,),
    name="statistics"
)



# ==============================
# 特征融合
# ==============================


fusion = Concatenate()(
    [
        feature_x,
        feature_y,
        feature_z,
        input_stats
    ]
)



# ==============================
# 分类层
# ==============================


output = Dense(
    6,
    activation="softmax"
)(fusion)



model = Model(
    inputs=[
        input_x,
        input_y,
        input_z,
        input_stats
    ],
    outputs=output
)



print()
print(
    "==========模型结构=========="
)

model.summary()



# ==============================
# 编译
# ==============================


model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)



# ==============================
# 训练
# ==============================


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
            X_test_x,
            X_test_y,
            X_test_z,
            X_stats_test
        ],

        y_test

    ),


    epochs=50,

    batch_size=32,

    verbose=1

)



# ==============================
# 预测
# ==============================


print()
print(
    "开始预测..."
)


prob = model.predict(

    [
        X_test_x,
        X_test_y,
        X_test_z,
        X_stats_test
    ]

)


y_pred = np.argmax(
    prob,
    axis=1
)



# ==============================
# 指标
# ==============================


acc = accuracy_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)


f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print()
print(
    "========== CNN结果 =========="
)

print(
    "Accuracy:",
    acc
)

print(
    "Macro Recall:",
    recall
)

print(
    "Macro F1:",
    f1
)



# ==============================
# 分类报告
# ==============================


print()

print(
    classification_report(
        y_test,
        y_pred,
        target_names=classes
    )
)



# ==============================
# 混淆矩阵
# ==============================


cm = confusion_matrix(
    y_test,
    y_pred
)


cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)


cm_df.to_csv(
    output_path('results/confusion_matrix/07_cnn', "17_cnn_confusion_matrix.csv"),
    encoding="utf-8-sig"
)



# ==============================
# 保存训练历史
# ==============================


pd.DataFrame(
    history.history
).to_csv(
    output_path('results/metrics/07_cnn', "17_cnn_history.csv"),
    index=False
)



# ==============================
# 保存模型
# ==============================


model.save(
    output_path('models', "17_cnn_model.keras")
)



print()

print(
    "CNN训练完成"
)