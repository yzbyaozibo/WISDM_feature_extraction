# HISTORICAL VERSION / 历史版本：保留原有算法与数值。
# CNN 指标 0.88 / 0.86 / 0.84 为历史硬编码值，不是当前训练的实测结果。
# 本脚本及其输出不作为当前实验结果来源；不属于推荐运行流程。
# 保留供历史查阅；执行仍会写入历史汇总文件。

from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path


import pandas as pd


# =====================================
# 输出目录
# =====================================

OUTPUT_DIR = project_path('results/metrics/08_final_results')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================
# 传统机器学习结果
# =====================================


rf_knn_file = project_path('results/metrics/06_model_comparison')


# 找到实验结果文件

files = list(
    rf_knn_file.glob("*.csv")
)


print(
    "找到模型结果文件:"
)

for f in files:
    print(f)



# =====================================
# 读取RF/KNN结果
# =====================================


ml_results = None


for f in files:

    try:

        df = pd.read_csv(f)

        if (
            "model" in df.columns
            and
            "Overall_Accuracy" in df.columns
        ):

            ml_results = df

            break


    except:

        pass



if ml_results is None:

    raise Exception(
        "没有找到KNN/RF结果文件"
    )



print()

print(
    "传统模型结果:"
)

print(
    ml_results
)



# =====================================
# 提取最佳模型
# =====================================


ml_best = (
    ml_results
    .sort_values(
        "Overall_Accuracy",
        ascending=False
    )
    .head(2)
)



# =====================================
# CNN结果
# =====================================


# 历史硬编码 CNN 指标，仅供历史查阅，禁止当作当前实验实测结果。
cnn_result = pd.DataFrame(

    {
        "model":[
            "CNN"
        ],

        "Overall_Accuracy":[
            0.88
        ],

        "Macro_Recall":[
            0.86
        ],

        "Macro_F1":[
            0.84
        ]

    }

)



# =====================================
# 合并
# =====================================


final_results = pd.concat(

    [

        ml_best[

            [
                "model",
                "Overall_Accuracy",
                "Macro_Recall",
                "Macro_F1"

            ]

        ],

        cnn_result

    ],

    ignore_index=True

)



# =====================================
# 排序
# =====================================


final_results = final_results.sort_values(

    "Overall_Accuracy",

    ascending=False

)



print()

print(
    "==========最终实验结果=========="
)

print(
    final_results
)



# =====================================
# 保存
# =====================================


final_results.to_csv(

    output_path('results/metrics/08_final_results', "20_final_model_comparison.csv"),

    index=False,

    encoding="utf-8-sig"

)


print()

print(
    "最终结果保存完成:"
)


print(

    output_path('results/metrics/08_final_results', "20_final_model_comparison.csv")

)
