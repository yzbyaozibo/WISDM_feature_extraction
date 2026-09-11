from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path


import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# STEP 1
# 文件路径
# ============================================================

INPUT_FILE = project_path('data/processed/04_features/10_final_43_features.csv')

OUTPUT_DIR = project_path('results/metrics/06_model_comparison')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 2
# 固定测试受试者
#
# 这些受试者训练阶段完全不可见
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
# STEP 3
# 活动顺序
# ============================================================

ACTIVITY_ORDER = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing"
]


# ============================================================
# STEP 4
# 读取43特征
# ============================================================

print("开始读取最终43特征数据...")

df = pd.read_csv(
    INPUT_FILE
)


print(
    "总窗口数：",
    len(df)
)


# ============================================================
# STEP 5
# 按受试者划分训练集 / 测试集
# ============================================================

train_df = df[
    ~df["id"].isin(
        TEST_SUBJECTS
    )
].copy()


test_df = df[
    df["id"].isin(
        TEST_SUBJECTS
    )
].copy()


print()
print(
    "========== Subject-independent split =========="
)

print(
    "训练窗口：",
    len(train_df)
)

print(
    "测试窗口：",
    len(test_df)
)

print(
    "训练受试者：",
    sorted(
        train_df["id"].unique()
    )
)

print(
    "测试受试者：",
    sorted(
        test_df["id"].unique()
    )
)


# ============================================================
# STEP 6
# 定义43个机器学习特征
#
# window_id：追踪用
# id：受试者编号
# class：标签
#
# 三者都不能作为输入特征
# ============================================================

NON_FEATURE_COLUMNS = [
    "window_id",
    "id",
    "class"
]


FEATURE_COLUMNS = [
    col
    for col in df.columns
    if col not in NON_FEATURE_COLUMNS
]


print()
print(
    "原始特征数量：",
    len(FEATURE_COLUMNS)
)


X_train = train_df[
    FEATURE_COLUMNS
].copy()

X_test = test_df[
    FEATURE_COLUMNS
].copy()


y_train = train_df[
    "class"
].copy()

y_test = test_df[
    "class"
].copy()


# ============================================================
# STEP 7
# 检查NaN
# ============================================================

print()
print(
    "========== 划分后的NaN =========="
)

print(
    "训练集NaN："
)

print(
    X_train.isna()
    .sum()
    .loc[
        lambda x: x > 0
    ]
    .to_string()
)


print()
print(
    "测试集NaN："
)

print(
    X_test.isna()
    .sum()
    .loc[
        lambda x: x > 0
    ]
    .to_string()
)


# ============================================================
# STEP 8
# 定义三种NaN处理策略
#
# 重要：
# Imputer只能fit训练集
# 测试集只能transform
# ============================================================

STRATEGIES = {

    # ----------------------------------------
    # 实验A
    # 训练集均值填补
    # ----------------------------------------

    "A_mean": SimpleImputer(
        strategy="mean",
        add_indicator=False
    ),


    # ----------------------------------------
    # 实验B
    # 训练集中位数填补
    # ----------------------------------------

    "B_median": SimpleImputer(
        strategy="median",
        add_indicator=False
    ),


    # ----------------------------------------
    # 实验C
    # 中位数填补 + Missing Indicator
    #
    # 会增加：
    # XPEAK_missing
    # YPEAK_missing
    # ZPEAK_missing
    #
    # 实际增加多少由训练集缺失情况决定
    # ----------------------------------------

    "C_median_indicator": SimpleImputer(
        strategy="median",
        add_indicator=True
    )
}


# ============================================================
# STEP 9
# 保存最终比较结果
# ============================================================

summary_rows = []


# ============================================================
# STEP 10
# 对三种NaN方案分别实验
# ============================================================

for strategy_name, imputer in STRATEGIES.items():

    print()
    print(
        "=" * 70
    )

    print(
        "当前NaN方案：",
        strategy_name
    )

    print(
        "=" * 70
    )


    # ========================================================
    # 10.1
    # 只在训练集学习填补参数
    # ========================================================

    X_train_processed = (
        imputer.fit_transform(
            X_train
        )
    )


    # ========================================================
    # 10.2
    # 测试集只使用训练阶段学习到的参数
    # ========================================================

    X_test_processed = (
        imputer.transform(
            X_test
        )
    )


    print(
        "处理后训练特征数：",
        X_train_processed.shape[1]
    )

    print(
        "处理后测试特征数：",
        X_test_processed.shape[1]
    )


    print(
        "处理后训练NaN数：",
        np.isnan(
            X_train_processed
        ).sum()
    )

    print(
        "处理后测试NaN数：",
        np.isnan(
            X_test_processed
        ).sum()
    )


    # ========================================================
    # STEP 11
    # 定义论文中的两个传统模型
    #
    # KNN:
    # K = 5
    # Euclidean distance
    #
    # RF:
    # n_estimators = 100
    # ========================================================

    models = {

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean"
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }


    # ========================================================
    # STEP 12
    # 分别训练KNN和RF
    # ========================================================

    for model_name, model in models.items():

        print()
        print(
            "----------",
            strategy_name,
            "+",
            model_name,
            "----------"
        )


        # ----------------------------------------------------
        # 训练
        # ----------------------------------------------------

        model.fit(
            X_train_processed,
            y_train
        )


        # ----------------------------------------------------
        # 测试集预测
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test_processed
        )


        # ====================================================
        # STEP 13
        # 总体指标
        # ====================================================

        overall_accuracy = accuracy_score(
            y_test,
            y_pred
        )


        macro_recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )


        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )


        weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )


        print(
            "Overall Accuracy：",
            f"{overall_accuracy:.4f}"
        )

        print(
            "Macro Recall：",
            f"{macro_recall:.4f}"
        )

        print(
            "Macro F1：",
            f"{macro_f1:.4f}"
        )

        print(
            "Weighted F1：",
            f"{weighted_f1:.4f}"
        )


        # ====================================================
        # STEP 14
        # 混淆矩阵
        # ====================================================

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=ACTIVITY_ORDER
        )


        cm_df = pd.DataFrame(
            cm,
            index=[
                f"true_{x}"
                for x in ACTIVITY_ORDER
            ],
            columns=[
                f"pred_{x}"
                for x in ACTIVITY_ORDER
            ]
        )


        cm_df.to_csv(
            output_path('results/confusion_matrix/06_model_comparison', f"{strategy_name}_"
                f"{model_name}_"
                f"confusion_matrix.csv"),
            encoding="utf-8-sig"
        )


        # ====================================================
        # STEP 15
        # 每个类别：
        #
        # Accuracy
        # Recall
        # F1
        #
        # 与论文表3形式尽量对应
        # ====================================================

        class_metric_rows = []


        for class_index, activity in enumerate(
            ACTIVITY_ORDER
        ):

            TP = cm[
                class_index,
                class_index
            ]

            FN = (
                cm[
                    class_index,
                    :
                ].sum()
                -
                TP
            )

            FP = (
                cm[
                    :,
                    class_index
                ].sum()
                -
                TP
            )

            TN = (
                cm.sum()
                -
                TP
                -
                FN
                -
                FP
            )


            # ----------------------------------------
            # 类别Accuracy：
            # (TP + TN) / 全部测试样本
            # ----------------------------------------

            class_accuracy = (
                (TP + TN)
                /
                cm.sum()
            )


            # ----------------------------------------
            # Recall
            # ----------------------------------------

            if TP + FN == 0:

                class_recall = 0.0

            else:

                class_recall = (
                    TP
                    /
                    (TP + FN)
                )


            # ----------------------------------------
            # Precision
            # ----------------------------------------

            if TP + FP == 0:

                class_precision = 0.0

            else:

                class_precision = (
                    TP
                    /
                    (TP + FP)
                )


            # ----------------------------------------
            # F1
            # ----------------------------------------

            if (
                class_precision
                +
                class_recall
                ==
                0
            ):

                class_f1 = 0.0

            else:

                class_f1 = (

                    2

                    *

                    class_precision

                    *

                    class_recall

                    /

                    (
                        class_precision
                        +
                        class_recall
                    )
                )


            class_metric_rows.append({

                "strategy":
                    strategy_name,

                "model":
                    model_name,

                "class":
                    activity,

                "Accuracy":
                    class_accuracy,

                "Precision":
                    class_precision,

                "Recall":
                    class_recall,

                "F1":
                    class_f1,

                "support":
                    TP + FN
            })


        class_metrics_df = pd.DataFrame(
            class_metric_rows
        )


        class_metrics_df.to_csv(
            output_path('results/metrics/06_model_comparison', f"{strategy_name}_"
                f"{model_name}_"
                f"class_metrics.csv"),
            index=False,
            encoding="utf-8-sig"
        )


        # ====================================================
        # STEP 16
        # 保存每个窗口预测结果
        # ====================================================

        prediction_df = pd.DataFrame({

            "window_id":
                test_df[
                    "window_id"
                ].values,

            "id":
                test_df[
                    "id"
                ].values,

            "true_class":
                y_test.values,

            "predicted_class":
                y_pred,

            "correct":
                (
                    y_test.values
                    ==
                    y_pred
                )
        })


        prediction_df.to_csv(
            output_path('results/metrics/06_model_comparison', f"{strategy_name}_"
                f"{model_name}_"
                f"predictions.csv"),
            index=False,
            encoding="utf-8-sig"
        )


        # ====================================================
        # STEP 17
        # 保存总体结果
        # ====================================================

        summary_rows.append({

            "NaN_strategy":
                strategy_name,

            "model":
                model_name,

            "input_feature_count":
                X_train_processed.shape[1],

            "train_samples":
                len(y_train),

            "test_samples":
                len(y_test),

            "Overall_Accuracy":
                overall_accuracy,

            "Macro_Recall":
                macro_recall,

            "Macro_F1":
                macro_f1,

            "Weighted_F1":
                weighted_f1
        })


# ============================================================
# STEP 18
# 汇总六组实验
#
# 三种NaN方案 × 两种模型
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


# 按Macro F1从高到低排列
summary_df = summary_df.sort_values(
    by="Macro_F1",
    ascending=False
)


print()
print(
    "=" * 80
)

print(
    "========== 最终实验对比 =========="
)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 19
# 保存最终对比表
# ============================================================

summary_df.to_csv(
    output_path('results/metrics/06_model_comparison', "15_nan_strategy_model_comparison.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STEP 20
# 自动显示每个模型最优NaN策略
# ============================================================

print()
print(
    "========== 每种模型最佳方案 =========="
)


for model_name in [
    "KNN",
    "RandomForest"
]:

    model_results = summary_df[
        summary_df[
            "model"
        ]
        ==
        model_name
    ]


    best_row = (
        model_results
        .sort_values(
            "Macro_F1",
            ascending=False
        )
        .iloc[0]
    )


    print()

    print(
        model_name
    )

    print(
        "最佳NaN方案：",
        best_row[
            "NaN_strategy"
        ]
    )

    print(
        "Accuracy：",
        f"{best_row['Overall_Accuracy']:.4f}"
    )

    print(
        "Macro Recall：",
        f"{best_row['Macro_Recall']:.4f}"
    )

    print(
        "Macro F1：",
        f"{best_row['Macro_F1']:.4f}"
    )


print()

print(
    "所有结果已保存到：",
    OUTPUT_DIR
)