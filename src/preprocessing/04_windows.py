from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path

import pandas as pd


# ============================================================
# STEP 1
# 输入与输出路径
# ============================================================

INPUT_FILE = project_path('data/processed/01_cleaning/01_clean_data.csv')

OUTPUT_DIR = project_path('results/metrics/03_windows')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 2
# 窗口参数
# ============================================================

WINDOW_SIZE = 200


# ============================================================
# STEP 3
# 读取清洗后的完整数据
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
#
# 作用：
# 保证一个窗口中不会混入不同用户或不同活动
# ============================================================

grouped = df.groupby(
    ["id", "class"],
    sort=False
)


# ============================================================
# STEP 5
# 创建统计容器
# ============================================================

group_statistics = []

total_windows = 0
total_used_samples = 0
total_discarded_samples = 0


# ============================================================
# STEP 6
# 在每个 id + class 组内划分200点窗口
# ============================================================

for (user_id, activity), group in grouped:

    # 保留当前组原有顺序
    group = group.reset_index(drop=True)

    sample_count = len(group)

    # 可以组成多少个完整200点窗口
    complete_windows = (
        sample_count // WINDOW_SIZE
    )

    # 剩余不足200点的尾部
    discarded_tail_samples = (
        sample_count % WINDOW_SIZE
    )

    # 实际用于窗口的数据量
    used_samples = (
        complete_windows * WINDOW_SIZE
    )

    # 累计总体统计
    total_windows += (
        complete_windows
    )

    total_used_samples += (
        used_samples
    )

    total_discarded_samples += (
        discarded_tail_samples
    )

    # 保存当前用户+活动统计
    group_statistics.append({

        "id":
            user_id,

        "class":
            activity,

        "raw_sample_count":
            sample_count,

        "complete_windows":
            complete_windows,

        "used_samples":
            used_samples,

        "discarded_tail_samples":
            discarded_tail_samples
    })


# ============================================================
# STEP 7
# 转换统计结果
# ============================================================

group_stats_df = pd.DataFrame(
    group_statistics
)


# ============================================================
# STEP 8
# 输出总体窗口统计
# ============================================================

print()
print(
    "========== 窗口划分总体结果 =========="
)

print(
    "窗口大小：",
    WINDOW_SIZE
)

print(
    "总完整窗口数：",
    total_windows
)

print(
    "用于窗口的数据点数：",
    total_used_samples
)

print(
    "尾部舍弃数据点数：",
    total_discarded_samples
)

print(
    "原始完整记录数：",
    len(df)
)


# 检查总量是否一致
print(
    "used + discarded：",
    total_used_samples
    +
    total_discarded_samples
)


# ============================================================
# STEP 9
# 数据利用率
# ============================================================

utilization_rate = (
    total_used_samples
    /
    len(df)
    *
    100
)

discard_rate = (
    total_discarded_samples
    /
    len(df)
    *
    100
)

print(
    "数据利用率：",
    f"{utilization_rate:.4f}%"
)

print(
    "尾部舍弃率：",
    f"{discard_rate:.4f}%"
)


# ============================================================
# STEP 10
# 保存每个用户×活动的窗口统计
# ============================================================

group_stats_df.to_csv(
    output_path('results/metrics/03_windows', "04_window_statistics_by_user_class.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STEP 11
# 按活动类别汇总
# ============================================================

class_stats = (
    group_stats_df
    .groupby("class")
    .agg(

        raw_sample_count=(
            "raw_sample_count",
            "sum"
        ),

        complete_windows=(
            "complete_windows",
            "sum"
        ),

        used_samples=(
            "used_samples",
            "sum"
        ),

        discarded_tail_samples=(
            "discarded_tail_samples",
            "sum"
        )
    )
    .reset_index()
)


class_stats[
    "window_percentage"
] = (
    class_stats[
        "complete_windows"
    ]
    /
    total_windows
    *
    100
)


print()
print(
    "========== 各活动窗口统计 =========="
)

print(
    class_stats.to_string(
        index=False
    )
)


# 保存类别窗口统计
class_stats.to_csv(
    output_path('results/metrics/03_windows', "04_window_statistics_by_class.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# STEP 12
# 保存总体摘要
# ============================================================

summary = pd.DataFrame(
    [
        {
            "window_size":
                WINDOW_SIZE,

            "total_windows":
                total_windows,

            "total_used_samples":
                total_used_samples,

            "total_discarded_samples":
                total_discarded_samples,

            "utilization_rate_percent":
                utilization_rate,

            "discard_rate_percent":
                discard_rate
        }
    ]
)


summary.to_csv(
    output_path('results/metrics/03_windows', "04_window_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "窗口统计已保存到：",
    OUTPUT_DIR
)