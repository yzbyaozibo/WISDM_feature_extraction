from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import project_path, output_path

import shutil


def organize_output_files():

    print("开始整理 output 文件夹")

    output_dir = project_path('data/processed/legacy_output')

    cleaning_dir = output_dir / "01_cleaning"
    statistics_dir = output_dir / "02_statistics"
    windows_dir = output_dir / "03_windows"
    features_dir = output_dir / "04_features"
    final_dir = output_dir / "final"
    archive_dir = output_dir / "archive"

    # 创建文件夹
    folders = [
        cleaning_dir,
        statistics_dir,
        windows_dir,
        features_dir,
        final_dir,
        archive_dir
    ]

    for folder in folders:
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        print("已确认文件夹：", folder)


    # 需要移动的文件
    files_to_move = {

        "01_clean_data.csv":
            cleaning_dir,

        "01_parsing_statistics.csv":
            cleaning_dir,

        "01_unparsed_examples.csv":
            cleaning_dir,

        "invalid_examples.csv":
            archive_dir
    }


    for file_name, destination_folder in files_to_move.items():

        source = output_dir / file_name

        destination = (
            destination_folder
            / file_name
        )

        print()
        print("检查文件：", source)

        if source.exists():

            print("找到文件：", source)

            if not destination.exists():

                shutil.move(
                    str(source),
                    str(destination)
                )

                print(
                    "移动完成：",
                    source,
                    "→",
                    destination
                )

            else:

                print(
                    "目标文件已经存在，不覆盖：",
                    destination
                )

        else:

            print(
                "当前没有找到：",
                source
            )

    print()
    print("output 文件整理完成")


# ============================================================
# 真正执行函数
# ============================================================

if __name__ == "__main__":

    print("程序已经开始运行")

    organize_output_files()

    print("程序执行结束")