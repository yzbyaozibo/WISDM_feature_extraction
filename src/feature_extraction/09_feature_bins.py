from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path


import numpy as np
import pandas as pd


# ============================================================
# STEP 1
# 路径和窗口参数
# ============================================================

INPUT_FILE = project_path('data/processed/01_cleaning/01_clean_data.csv')

OUTPUT_DIR = project_path('data/processed/04_features')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WINDOW_SIZE = 200


# ============================================================
# STEP 2
# 定义10-bin特征计算函数
# ============================================================

def calculate_bins(values):

    values = np.asarray(
        values,
        dtype=float
    )

    min_value = np.min(values)
    max_value = np.max(values)

    # --------------------------------------------------------
    # 极端情况：
    # 如果200个值完全相同，则无法正常划分范围
    # 这里将全部数据放入第0个bin
    # --------------------------------------------------------

    if min_value == max_value:

        proportions = np.zeros(10)

        proportions[0] = 1.0

        return proportions


    # --------------------------------------------------------
    # 将 min ~ max 均分成10个区间
    # --------------------------------------------------------

    counts, bin_edges = np.histogram(
        values,
        bins=10,
        range=(
            min_value,
            max_value
        )
    )


    # --------------------------------------------------------
    # 转换为比例
    #
    # 一个窗口200点，因此10个比例之和应当为1
    # --------------------------------------------------------

    proportions = (
        counts
        /
        len(values)
    )

    return proportions


# ============================================================
# STEP 3
# 读取清洗数据
# ============================================================

print(
    "开始读取清洗后的数据..."
)

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
    [
        "id",
        "class"
    ],
    sort=False
)


# ============================================================
# STEP 5
# 提取特征
# ============================================================

feature_rows = []

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


        # ====================================================
        # 获取三轴数据
        # ====================================================

        x = window[
            "x"
        ].to_numpy()

        y = window[
            "y"
        ].to_numpy()

        z = window[
            "z"
        ].to_numpy()


        # ====================================================
        # 1. 10-bin特征
        # ====================================================

        x_bins = calculate_bins(x)

        y_bins = calculate_bins(y)

        z_bins = calculate_bins(z)


        # ====================================================
        # 2. AVG
        # ====================================================

        x_avg = np.mean(x)

        y_avg = np.mean(y)

        z_avg = np.mean(z)


        # ====================================================
        # 3. ABSO
        # ====================================================

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


        # ====================================================
        # 4. STAND
        # ====================================================

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


        # ====================================================
        # 5. RESULTANT
        # ====================================================

        resultant = np.mean(

            np.sqrt(

                x ** 2
                +
                y ** 2
                +
                z ** 2

            )

        )


        # ====================================================
        # 6. 建立当前窗口的特征记录
        # ====================================================

        row = {

            "window_id":
                window_id,

            "id":
                user_id,

            "class":
                activity
        }


        # ----------------------------------------------------
        # X0 ~ X9
        # ----------------------------------------------------

        for j in range(10):

            row[
                f"X{j}"
            ] = x_bins[j]


        row[
            "XAVG"
        ] = x_avg

        row[
            "XABSO"
        ] = x_abso

        row[
            "XSTAND"
        ] = x_std


        # ----------------------------------------------------
        # Y0 ~ Y9
        # ----------------------------------------------------

        for j in range(10):

            row[
                f"Y{j}"
            ] = y_bins[j]


        row[
            "YAVG"
        ] = y_avg

        row[
            "YABSO"
        ] = y_abso

        row[
            "YSTAND"
        ] = y_std


        # ----------------------------------------------------
        # Z0 ~ Z9
        # ----------------------------------------------------

        for j in range(10):

            row[
                f"Z{j}"
            ] = z_bins[j]


        row[
            "ZAVG"
        ] = z_avg

        row[
            "ZABSO"
        ] = z_abso

        row[
            "ZSTAND"
        ] = z_std


        row[
            "RESULTANT"
        ] = resultant


        feature_rows.append(
            row
        )

        window_id += 1


# ============================================================
# STEP 6
# 转成DataFrame
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
    "========== 40个特征提取结果 =========="
)

print(
    "窗口数量：",
    len(feature_df)
)

print(
    "当前特征数量：",
    len(feature_df.columns) - 3
)

# -3：
# window_id
# id
# class


print()

print(
    "第1个窗口："
)

print(
    feature_df
    .head(1)
    .to_string(index=False)
)


# ============================================================
# STEP 8
# 验证bin比例
# ============================================================

first_row = feature_df.iloc[0]

x_bin_sum = sum(
    first_row[
        f"X{i}"
    ]
    for i in range(10)
)

y_bin_sum = sum(
    first_row[
        f"Y{i}"
    ]
    for i in range(10)
)

z_bin_sum = sum(
    first_row[
        f"Z{i}"
    ]
    for i in range(10)
)


print()

print(
    "========== 第1窗口bin检查 =========="
)

print(
    "X0~X9之和：",
    x_bin_sum
)

print(
    "Y0~Y9之和：",
    y_bin_sum
)

print(
    "Z0~Z9之和：",
    z_bin_sum
)


# ============================================================
# STEP 9
# 保存
# ============================================================

feature_df.to_csv(

    output_path('data/processed/04_features', "09_features_40_without_peak.csv"),

    index=False,

    encoding="utf-8-sig"

)


print()

print(
    "结果已保存到："
    "data/processed/04_features"
    "09_features_40_without_peak.csv"
)