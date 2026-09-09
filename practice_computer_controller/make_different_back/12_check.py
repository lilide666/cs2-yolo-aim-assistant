import shutil
from pathlib import Path


# ============================================================
# 配置
# ============================================================

DATASET_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\10_different_back_dataset"
)

IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

# 检查结果输出目录
EMPTY_CHECK_DIR = DATASET_DIR / "empty_check"

# 报告
REPORT_PATH = EMPTY_CHECK_DIR / "empty_check_report.txt"


# ============================================================
# 图片扩展名
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 判断 YOLO 标签是否有效
# ============================================================

def check_label(label_path):
    """
    返回：

    {
        "empty": True/False,
        "reason": "...",
        "content": "原始标签内容"
    }
    """

    # --------------------------------------------------------
    # 标签不存在
    # --------------------------------------------------------

    if not label_path.exists():

        return {
            "empty": True,
            "reason": "标签文件不存在",
            "content": "",
        }


    # --------------------------------------------------------
    # 读取标签
    # --------------------------------------------------------

    try:

        content = label_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        try:

            content = label_path.read_text(
                encoding="utf-8-sig"
            )

        except Exception as e:

            return {
                "empty": True,
                "reason": f"标签读取失败：{e}",
                "content": "",
            }

    except Exception as e:

        return {
            "empty": True,
            "reason": f"标签读取失败：{e}",
            "content": "",
        }


    # --------------------------------------------------------
    # 完全为空
    # --------------------------------------------------------

    if not content.strip():

        return {
            "empty": True,
            "reason": "标签文件为空",
            "content": content,
        }


    # --------------------------------------------------------
    # 检查每一行
    # --------------------------------------------------------

    lines = content.splitlines()

    valid_lines = 0
    invalid_lines = []

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        # 空行直接跳过
        if not line:
            continue

        parts = line.split()

        # YOLO detect 格式：
        #
        # class x_center y_center width height
        #
        # 共 5 个数字

        if len(parts) != 5:

            invalid_lines.append(
                (
                    line_number,
                    line,
                    "字段数量不是5"
                )
            )

            continue


        try:

            class_id = int(
                float(parts[0])
            )

            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])


        except ValueError:

            invalid_lines.append(
                (
                    line_number,
                    line,
                    "包含无法解析的数字"
                )
            )

            continue


        # ----------------------------------------------------
        # 基础 YOLO 检查
        # ----------------------------------------------------

        if class_id < 0:

            invalid_lines.append(
                (
                    line_number,
                    line,
                    "class_id < 0"
                )
            )

            continue


        if not (
            0 <= x <= 1
            and
            0 <= y <= 1
            and
            0 < w <= 1
            and
            0 < h <= 1
        ):

            invalid_lines.append(
                (
                    line_number,
                    line,
                    "bbox 坐标范围异常"
                )
            )

            continue


        valid_lines += 1


    # --------------------------------------------------------
    # 完全没有有效框
    # --------------------------------------------------------

    if valid_lines == 0:

        return {
            "empty": True,
            "reason": "没有有效 YOLO 标注框",
            "content": content,
        }


    # --------------------------------------------------------
    # 有无效行
    #
    # 注意：
    # 这种情况不算“空标签”，
    # 但我们可以报告出来。
    # --------------------------------------------------------

    if invalid_lines:

        return {
            "empty": False,
            "reason": (
                f"存在 {len(invalid_lines)} 个无效标签行，"
                f"但仍有 {valid_lines} 个有效框"
            ),
            "content": content,
        }


    # --------------------------------------------------------
    # 正常标签
    # --------------------------------------------------------

    return {
        "empty": False,
        "reason": f"正常，共 {valid_lines} 个框",
        "content": content,
    }


# ============================================================
# 复制文件
# ============================================================

def copy_empty_image(
    image_path,
    split,
    index
):

    # --------------------------------------------------------
    # 为了避免 train / val 同名文件冲突
    #
    # 使用：
    #
    # train_0001_xxx.jpg
    # val_0002_xxx.jpg
    # --------------------------------------------------------

    new_name = (
        f"{split}_{index:04d}_"
        f"{image_path.name}"
    )

    destination = (
        EMPTY_CHECK_DIR /
        new_name
    )

    shutil.copy2(
        image_path,
        destination
    )

    return destination


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 80)
    print("YOLO 空标注图片检查工具")
    print("=" * 80)

    print()

    print(
        f"数据集：\n{DATASET_DIR}"
    )

    print()

    print(
        f"图片目录：\n{IMAGE_DIR}"
    )

    print()

    print(
        f"标签目录：\n{LABEL_DIR}"
    )

    print()

    print(
        f"检查输出：\n{EMPTY_CHECK_DIR}"
    )

    print()


    # ========================================================
    # 检查目录
    # ========================================================

    if not DATASET_DIR.exists():

        raise RuntimeError(
            f"数据集不存在：\n{DATASET_DIR}"
        )


    if not IMAGE_DIR.exists():

        raise RuntimeError(
            f"图片目录不存在：\n{IMAGE_DIR}"
        )


    if not LABEL_DIR.exists():

        raise RuntimeError(
            f"标签目录不存在：\n{LABEL_DIR}"
        )


    # ========================================================
    # 创建输出目录
    # ========================================================

    EMPTY_CHECK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 清理旧报告
    #
    # 不删除旧图片。
    # ========================================================

    if REPORT_PATH.exists():

        REPORT_PATH.unlink()


    # ========================================================
    # 找 train / val
    # ========================================================

    splits = [
        "train",
        "val",
    ]


    total_images = 0
    total_empty = 0

    missing_label_count = 0
    empty_file_count = 0
    invalid_only_count = 0


    report_lines = []


    report_lines.append(
        "=" * 80
    )

    report_lines.append(
        "YOLO 空标注图片检查报告"
    )

    report_lines.append(
        "=" * 80
    )

    report_lines.append("")

    report_lines.append(
        f"数据集：{DATASET_DIR}"
    )

    report_lines.append("")


    # ========================================================
    # 开始扫描
    # ========================================================

    for split in splits:

        image_split_dir = (
            IMAGE_DIR /
            split
        )

        label_split_dir = (
            LABEL_DIR /
            split
        )


        print("=" * 80)

        print(
            f"检查 {split.upper()}"
        )

        print("=" * 80)


        if not image_split_dir.exists():

            print(
                f"[跳过] 图片目录不存在："
                f"{image_split_dir}"
            )

            print()

            continue


        # ----------------------------------------------------
        # 获取图片
        # ----------------------------------------------------

        image_files = sorted(
            [
                p
                for p in image_split_dir.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            ]
        )


        print(
            f"图片数量："
            f"{len(image_files)}"
        )

        print()


        split_empty_count = 0


        for image_index, image_path in enumerate(
            image_files,
            start=1
        ):

            total_images += 1


            # ------------------------------------------------
            # 对应标签
            # ------------------------------------------------

            label_path = (
                label_split_dir /
                f"{image_path.stem}.txt"
            )


            # ------------------------------------------------
            # 检查
            # ------------------------------------------------

            result = check_label(
                label_path
            )


            # ------------------------------------------------
            # 正常图片
            # ------------------------------------------------

            if not result["empty"]:

                continue


            # ------------------------------------------------
            # 发现空标签
            # ------------------------------------------------

            total_empty += 1
            split_empty_count += 1


            reason = result["reason"]

            content = result["content"]


            if reason == "标签文件不存在":

                missing_label_count += 1

            elif reason == "标签文件为空":

                empty_file_count += 1

            elif reason == "没有有效 YOLO 标注框":

                invalid_only_count += 1


            # ------------------------------------------------
            # 复制图片
            # ------------------------------------------------

            destination = copy_empty_image(
                image_path,
                split,
                split_empty_count
            )


            # ------------------------------------------------
            # 终端输出
            # ------------------------------------------------

            print(
                f"[发现] {split}/{image_path.name}"
            )

            print(
                f"       原图：{image_path}"
            )

            print(
                f"       标签：{label_path}"
            )

            print(
                f"       原因：{reason}"
            )

            print(
                f"       复制：{destination}"
            )

            print(
                "       标签内容："
            )


            if content:

                for line in content.splitlines():

                    print(
                        f"           {line}"
                    )

            else:

                print(
                    "           <空>"
                )


            print()


            # ------------------------------------------------
            # 写报告
            # ------------------------------------------------

            report_lines.append(
                "=" * 80
            )

            report_lines.append(
                f"编号：{total_empty}"
            )

            report_lines.append(
                f"Split：{split}"
            )

            report_lines.append(
                f"图片：{image_path.name}"
            )

            report_lines.append(
                f"原图片路径：{image_path}"
            )

            report_lines.append(
                f"标签路径：{label_path}"
            )

            report_lines.append(
                f"原因：{reason}"
            )

            report_lines.append(
                f"复制到：{destination}"
            )

            report_lines.append(
                "原始标签内容："
            )


            if content:

                report_lines.extend(
                    [
                        f"    {line}"
                        for line
                        in content.splitlines()
                    ]
                )

            else:

                report_lines.append(
                    "    <空>"
                )


            report_lines.append("")


        print(
            f"{split.upper()} 空标注："
            f"{split_empty_count}"
        )

        print()


    # ========================================================
    # 写报告
    # ========================================================

    report_lines.append(
        "=" * 80
    )

    report_lines.append(
        "统计"
    )

    report_lines.append(
        "=" * 80
    )

    report_lines.append("")

    report_lines.append(
        f"总图片：{total_images}"
    )

    report_lines.append(
        f"空标注图片：{total_empty}"
    )

    report_lines.append(
        f"标签不存在：{missing_label_count}"
    )

    report_lines.append(
        f"标签文件为空：{empty_file_count}"
    )

    report_lines.append(
        f"没有有效 YOLO 框：{invalid_only_count}"
    )

    report_lines.append("")


    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )


    # ========================================================
    # 最终输出
    # ========================================================

    print("=" * 80)
    print("检查完成")
    print("=" * 80)

    print()

    print(
        f"总图片："
        f"{total_images}"
    )

    print(
        f"空标注图片："
        f"{total_empty}"
    )

    print()

    print(
        f"标签不存在："
        f"{missing_label_count}"
    )

    print(
        f"标签文件为空："
        f"{empty_file_count}"
    )

    print(
        f"没有有效 YOLO 框："
        f"{invalid_only_count}"
    )

    print()

    print(
        f"图片已复制到："
        f"\n{EMPTY_CHECK_DIR}"
    )

    print()

    print(
        f"详细报告："
        f"\n{REPORT_PATH}"
    )

    print()

    print("=" * 80)


# ============================================================
# Windows
# ============================================================

if __name__ == "__main__":

    main()