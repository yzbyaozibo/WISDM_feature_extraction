from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path


import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ============================================================
# STEP 1
# 路径与参数
# ============================================================

RAW_FILE = project_path('data/processed/01_cleaning/01_clean_data.csv')

FEATURE_FILE = project_path('data/processed/04_features/09_features_40_without_peak.csv')

OUTPUT_DIR = project_path('data/processed/04_features')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


WINDOW_SIZE = 200

SAMPLING_RATE = 20.0

SAMPLING_INTERVAL = (
    1.0 / SAMPLING_RATE
)


# ============================================================
# STEP 2
# PEAK计算函数
# ============================================================

def calculate_peak_time(values):

    values = np.asarray(
        values,
        dtype=float
    )

    # --------------------------------------------------------
    # 先找所有局部峰值
    # --------------------------------------------------------

    peak_indices, _ = find_peaks(
        values
    )


    # 连3个局部峰都没有
    if len(peak_indices) < 3:

        return np.nan


    peak_values = values[
        peak_indices
    ]


    # --------------------------------------------------------
    # 找最高峰
    # --------------------------------------------------------

    highest_peak = np.max(
        peak_values
    )


    # --------------------------------------------------------
    # 阈值比例：
    # 90%、85%、80% ... 10%
    #
    # 这是我们的可重复实现规则，
    # 不是论文明确给出的具体百分比
    # --------------------------------------------------------

    threshold_ratios = np.arange(
        0.90,
        0.05,
        -0.05
    )


    selected_peaks = None


    for ratio in threshold_ratios:

        threshold = (
            highest_peak
            *
            ratio
        )


        current_peaks = peak_indices[
            peak_values >= threshold
        ]


        if len(current_peaks) >= 3:

            selected_peaks = (
                current_peaks
            )

            break


    # --------------------------------------------------------
    # 最低阈值下仍不足3个峰
    # --------------------------------------------------------

    if selected_peaks is None:

        return np.nan


    # --------------------------------------------------------
    # 相邻峰之间相差多少个采样点
    # --------------------------------------------------------

    sample_differences = np.diff(
        selected_peaks
    )


    # --------------------------------------------------------
    # 20 Hz:
    #
    # 每个采样点间隔 = 0.05 秒
    #
    # 转成真实时间
    # --------------------------------------------------------

    time_differences = (
        sample_differences
        *
        SAMPLING_INTERVAL
    )


    # --------------------------------------------------------
    # 平均峰间时间
    # --------------------------------------------------------

    average_peak_time = np.mean(
        time_differences
    )


    return average_peak_time


# ============================================================
# STEP 3
# 读取数据
# ============================================================

print(
    "开始读取数据..."
)

df = pd.read_csv(
    RAW_FILE
)

feature_df = pd.read_csv(
    FEATURE_FILE
)


print(
    "原始记录数：",
    len(df)
)

print(
    "已有特征窗口数：",
    len(feature_df)
)


# ============================================================
# STEP 4
# 按与前面完全相同的方法划分窗口
# ============================================================

grouped = df.groupby(
    [
        "id",
        "class"
    ],
    sort=False
)


peak_rows = []


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


        x = window[
            "x"
        ].to_numpy()

        y = window[
            "y"
        ].to_numpy()

        z = window[
            "z"
        ].to_numpy()


        # ----------------------------------------------------
        # 三轴平均峰间时间
        # ----------------------------------------------------

        x_peak = calculate_peak_time(
            x
        )

        y_peak = calculate_peak_time(
            y
        )

        z_peak = calculate_peak_time(
            z
        )


        peak_rows.append({

            "XPEAK":
                x_peak,

            "YPEAK":
                y_peak,

            "ZPEAK":
                z_peak
        })


# ============================================================
# STEP 5
# 检查窗口数是否一致
# ============================================================

peak_df = pd.DataFrame(
    peak_rows
)


print()

print(
    "PEAK窗口数量：",
    len(peak_df)
)


if len(peak_df) != len(feature_df):

    raise ValueError(
        "PEAK窗口数与前面40特征窗口数不一致！"
    )


# ============================================================
# STEP 6
# 合并PEAK特征
# ============================================================

feature_df[
    "XPEAK"
] = peak_df[
    "XPEAK"
]

feature_df[
    "YPEAK"
] = peak_df[
    "YPEAK"
]

feature_df[
    "ZPEAK"
] = peak_df[
    "ZPEAK"
]


# ============================================================
# STEP 7
# 按论文字段顺序重新排列
# ============================================================

ordered_columns = [

    "window_id",
    "id",

    "X0", "X1", "X2", "X3", "X4",
    "X5", "X6", "X7", "X8", "X9",
    "XAVG",
    "XPEAK",
    "XABSO",
    "XSTAND",

    "Y0", "Y1", "Y2", "Y3", "Y4",
    "Y5", "Y6", "Y7", "Y8", "Y9",
    "YAVG",
    "YPEAK",
    "YABSO",
    "YSTAND",

    "Z0", "Z1", "Z2", "Z3", "Z4",
    "Z5", "Z6", "Z7", "Z8", "Z9",
    "ZAVG",
    "ZPEAK",
    "ZABSO",
    "ZSTAND",

    "RESULTANT",

    "class"
]


feature_df = feature_df[
    ordered_columns
]


# ============================================================
# STEP 8
# 检查43特征
# ============================================================

metadata_columns = [
    "window_id",
    "id",
    "class"
]

feature_count = (
    len(feature_df.columns)
    -
    len(metadata_columns)
)


print()

print(
    "========== 43特征结果 =========="
)

print(
    "窗口数量：",
    len(feature_df)
)

print(
    "特征数量：",
    feature_count
)


# ============================================================
# STEP 9
# 检查无法计算PEAK的窗口
# ============================================================

print()

print(
    "========== PEAK缺失统计 =========="
)

print(
    "XPEAK无法计算：",
    feature_df[
        "XPEAK"
    ].isna().sum()
)

print(
    "YPEAK无法计算：",
    feature_df[
        "YPEAK"
    ].isna().sum()
)

print(
    "ZPEAK无法计算：",
    feature_df[
        "ZPEAK"
    ].isna().sum()
)


# ============================================================
# STEP 10
# 查看前5个窗口
# ============================================================

print()

print(
    "前5个窗口的PEAK："
)

print(
    feature_df[
        [
            "window_id",
            "id",
            "class",
            "XPEAK",
            "YPEAK",
            "ZPEAK"
        ]
    ]
    .head()
    .to_string(
        index=False
    )
)


# ============================================================
# STEP 11
# 保存最终43特征
# ============================================================

feature_df.to_csv(

    output_path('data/processed/04_features', "10_final_43_features.csv"),

    index=False,

    encoding="utf-8-sig"
)


print()

print(
    "最终43特征已保存到："
    "data/processed/04_features"
    "10_final_43_features.csv"
)