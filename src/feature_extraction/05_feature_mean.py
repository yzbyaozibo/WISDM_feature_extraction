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

INPUT_FILE = project_path('data/processed/01_cleaning/01_clean_data.csv')

OUTPUT_DIR = project_path('data/processed/04_features')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 2
# 窗口大小
# ============================================================

WINDOW_SIZE = 200


# ============================================================
# STEP 3
# 读取清洗后的数据
# ============================================================

print("开始读取清洗后的数据...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "完整数据记录数：",
    len(df)
)


# ============================================================
# STEP 4
# 按 id + class 分组
# ============================================================

grouped = df.groupby(
    ["id", "class"],
    sort=False
)


# ============================================================
# STEP 5
# 对每个200点窗口计算均值
# ============================================================

feature_rows = []

window_id = 1


for (user_id, activity), group in grouped:

    group = group.reset_index(drop=True)

    complete_windows = (
        len(group) // WINDOW_SIZE
    )

    for i in range(complete_windows):

        start = (
            i * WINDOW_SIZE
        )

        end = (
            start + WINDOW_SIZE
        )

        window = group.iloc[
            start:end
        ]


        # -------------------------
        # 计算三个轴的平均值
        # -------------------------

        x_avg = window["x"].mean()

        y_avg = window["y"].mean()

        z_avg = window["z"].mean()


        # -------------------------
        # 保存结果
        # -------------------------

        feature_rows.append({

            "window_id":
                window_id,

            "id":
                user_id,

            "class":
                activity,

            "XAVG":
                x_avg,

            "YAVG":
                y_avg,

            "ZAVG":
                z_avg
        })


        window_id += 1


# ============================================================
# STEP 6
# 转换结果
# ============================================================

feature_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# STEP 7
# 输出检查
# ============================================================

print()
print(
    "========== 均值特征提取结果 =========="
)

print(
    "窗口数量：",
    len(feature_df)
)

print()
print(
    "前5个窗口："
)

print(
    feature_df
    .head()
    .to_string(index=False)
)


# ============================================================
# STEP 8
# 保存
# ============================================================

feature_df.to_csv(
    output_path('data/processed/04_features', "05_mean_features.csv"),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "均值特征已保存到："
    "data/processed/04_features/05_mean_features.csv"
)