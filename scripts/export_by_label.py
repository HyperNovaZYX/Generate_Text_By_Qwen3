# -*- coding: utf-8 -*-
"""按标签分组导出 JSONL 纯文本工具（合并原 export_text.py 与 export_text_by_label.py）。

用法
----
    # 导出单个 JSONL 文件
    python scripts/export_by_label.py text/gamble/gamble.jsonl

    # 一次导出多个 JSONL 文件
    python scripts/export_by_label.py text/contraband_made/contraband_made.jsonl text/contraband_used/contraband_used.jsonl

输出
----
在每个输入文件同目录下生成：
    <stem>_safe.txt
    <stem>_borderline.txt
    <stem>_other.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dedup import clean_storage_text  # noqa: E402


def export_jsonl_by_label(source: Path, *, output_dir: Path | None = None) -> dict[str, int]:
    """将单个 JSONL 按 safe / borderline / other 三组导出为纯文本。"""
    if not source.is_file():
        raise FileNotFoundError(f"输入文件不存在：{source}")

    groups: dict[str, list[str]] = {"safe": [], "borderline": [], "other": []}
    with source.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: JSON 格式错误") from exc

            if not isinstance(item, dict):
                raise ValueError(f"{source}:{line_number}: 期望 JSON 对象")

            label = item.get("label")
            raw_text = item.get("text")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{source}:{line_number}: 缺少有效字符串 label")
            if not isinstance(raw_text, str):
                raise ValueError(f"{source}:{line_number}: 缺少字符串 text")

            cleaned = clean_storage_text(raw_text)
            if not cleaned:
                raise ValueError(f"{source}:{line_number}: 清洗后 text 为空")

            group = label if label in ("safe", "borderline") else "other"
            groups[group].append(cleaned)

    target_dir = output_dir if output_dir is not None else source.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for group, values in groups.items():
        out_path = target_dir / f"{source.stem}_{group}.txt"
        with out_path.open("w", encoding="utf-8", newline="\n") as out:
            for value in values:
                out.write(value + "\n")
        counts[group] = len(values)
        print(f"{out_path}: {len(values)} 条")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="将一个或多个 JSONL 按 safe / borderline / other 分别导出为纯文本",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="要导出的 JSONL 文件路径（可一次传多个）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="可选：统一输出目录（默认输出到各输入文件同目录）",
    )
    args = parser.parse_args(argv)

    for src in args.inputs:
        path = src if src.is_absolute() else PROJECT_ROOT / src
        export_jsonl_by_label(path, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
