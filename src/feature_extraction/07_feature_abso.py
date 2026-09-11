from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path

import pandas as pd
import numpy as np


# ============================================================
# STEP 1
# 路径
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
# 读取数据
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
# 特征提取
# ============================================================

feature_rows = []

window_id = 1


for (user_id, activity), group in grouped:

    group = group.reset_index(drop=True)

    complete_windows = (
        len(group) // WINDOW_SIZE
    )

    for i in range(complete_windows):

        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window = group.iloc[
            start:end
        ]


        # ----------------------------------------------------
        # 取出三个轴
        # ----------------------------------------------------

        x = window["x"].to_numpy()
        y = window["y"].to_numpy()
        z = window["z"].to_numpy()


        # ----------------------------------------------------
        # AVG
        # ----------------------------------------------------

        x_avg = np.mean(x)
        y_avg = np.mean(y)
        z_avg = np.mean(z)


        # ----------------------------------------------------
        # STAND
        # 目前使用总体标准差 ddof=0
        # ----------------------------------------------------

        x_std = np.std(
            x,
            ddof=0
        )

        y_std = np.std(
            y,
            ddof=0
        )

        z_std = np.std(
            z,
            ddof=0
        )


        # ----------------------------------------------------
        # ABSO
        #
        # mean(|xi - x平均值|)
        # ----------------------------------------------------

        x_abso = np.mean(
            np.abs(
                x - x_avg
            )
        )

        y_abso = np.mean(
            np.abs(
                y - y_avg
            )
        )

        z_abso = np.mean(
            np.abs(
                z - z_avg
            )
        )


        # ----------------------------------------------------
        # 保存结果
        # ----------------------------------------------------

        feature_rows.append({

            "window_id":
                window_id,

            "id":
                user_id,

            "class":
                activity,

            "XAVG":
                x_avg,

            "XABSO":
                x_abso,

            "XSTAND":
                x_std,

            "YAVG":
                y_avg,

            "YABSO":
                y_abso,

            "YSTAND":
                y_std,

            "ZAVG":
                z_avg,

            "ZABSO":
                z_abso,

            "ZSTAND":
                z_std
        })


        window_id += 1


# ============================================================
# STEP 6
# 转DataFrame
# ============================================================

feature_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# STEP 7
# 检查
# ============================================================

print()
print(
    "========== AVG + ABSO + STAND =========="
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
    output_path('data/processed/04_features', "07_avg_abso_std_features.csv"),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "结果已保存到："
    "data/processed/04_features/07_avg_abso_std_features.csv"
)