# -*- coding: utf-8 -*-
"""把生成的 JSONL 各自抽成纯文本，供下游训练直接使用。

输入：text/<类别>/<任意>.jsonl（一行一条 JSON，含 text / label）
输出：
    final/<类别>.txt     该类别下的全部有害文本
    final/normal.txt     borderline + safe 文本，跨类别汇总

相对旧版（本次重写解决的问题，编号对应审查报告）
------------------------------------------------
3.1  normal.txt 用 "a" 追加写入 —— 跑两遍数据就翻倍，且换机器后与残留
     文件混合，无法判断哪次是哪次。现在每次运行前整体清空重建，一次运行
     的结果只由本次输入决定。
3.2  borderline / safe 只按类别归属写进对应 .txt —— 这会让「正常文本」
     污染有害类别文件。现在强制改道 normal.txt。
3.3  完全没有去重。历史批处理中存在同文本被打上互斥标签或同时出现在 normal
     与有害文件中的冲突情况，同一句话出现在两个类目下会直接损伤分类器。
     现在执行全局去重与冲突裁决。

冲突裁决规则（按用户决策：有害标签优先）
--------------------------------------
同一 text 出现多次且标签不同时，取「有害程度最高」的那一个：
    harmful_*  >  borderline  >  safe
依据是安全语料的漏标代价远高于误标 —— 一条真有害的句子被归进 normal，
等于直接教模型「这句话没问题」。同优先级内按「先到先得」，即输入文件
按路径排序后先出现的胜出（确定性、可复现）。
被丢弃的一方写入 final/_conflicts.tsv，便于人工复核规则是否过激。

长度闸门
--------
与 core.py 的 MIN_TEXT_LEN / MAX_TEXT_LEN 保持一致。旧版只判空，结果
混进了 5 字短句与 192 字长文。超长文本会截断成半句话，反而制造噪声，
因此这里选择丢弃而不是截断。

用法
----
    # 处理 text/ 下全部类别（推荐，一次运行产出完整 final/）
    python scripts/split_dataset.py

    # 只处理指定文件
    python scripts/split_dataset.py text/violence_event/violence_event.jsonl

    # 只看统计不动文件
    python scripts/split_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core import MAX_TEXT_LEN, MIN_TEXT_LEN, PROJECT_ROOT, iter_jsonl  # noqa: E402
from dedup import clean_storage_text, is_ignored_jsonl  # noqa: E402

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

TEXT_ROOT = PROJECT_ROOT / "text"
FINAL_ROOT = PROJECT_ROOT / "final"

NORMAL_LABELS = ("borderline", "safe")
NORMAL_FILE = "normal.txt"
CONFLICT_FILE = "_conflicts.tsv"

# 失败样本的文件名特征。这些行没有 text 字段（只有 raw / error），
# 参与切分只会被计入「解析失败」，直接按文件名剔除更干净。
_FAIL_NAME = re.compile(r"fail", re.I)

# 有害程度排序：数值越大越有害。裁决时取最大者。
_RANK_HARMFUL = 3
_RANK_BORDERLINE = 2
_RANK_SAFE = 1


def label_rank(label: str) -> int:
    """给标签定一个「有害程度」序，用于冲突裁决。"""
    if label in NORMAL_LABELS:
        return _RANK_BORDERLINE if label == "borderline" else _RANK_SAFE
    # 未知标签一律按有害处理：宁可多留，不可漏放
    return _RANK_HARMFUL


# --------------------------------------------------------------------------
# 文本清洗
# --------------------------------------------------------------------------

def clean_text(text: object) -> str:
    """清洗单条文本：复用 dedup.clean_storage_text，确保生成端与导出端完全一致。"""
    return clean_storage_text(text)


# --------------------------------------------------------------------------
# 统计容器
# --------------------------------------------------------------------------


@dataclass
class Stats:
    """一次切分运行的计数。字段名即统计口径，便于直接打印。"""

    files: int = 0
    read: int = 0            # 读到的 JSON 行数
    bad_json: int = 0        # 解析失败
    no_text: int = 0         # text 缺失或清洗后为空
    too_short: int = 0
    too_long: int = 0
    dup_same: int = 0        # 重复且标签一致，静默丢弃
    dup_conflict: int = 0    # 重复且标签冲突，裁决后丢弃一方
    kept_harmful: int = 0
    kept_normal: int = 0

    def report(self) -> str:
        total_kept = self.kept_harmful + self.kept_normal
        lines = [
            f"处理文件        : {self.files}",
            f"读取行          : {self.read}",
            f"JSON 解析失败   : {self.bad_json}",
            f"text 为空       : {self.no_text}",
            f"过短(<{MIN_TEXT_LEN})       : {self.too_short}",
            f"过长(>{MAX_TEXT_LEN})      : {self.too_long}",
            f"重复(标签一致)  : {self.dup_same}",
            f"重复(标签冲突)  : {self.dup_conflict}",
            f"保留·有害       : {self.kept_harmful}",
            f"保留·正常       : {self.kept_normal}",
            f"合计保留        : {total_kept}",
        ]
        return "\n".join("  " + x for x in lines)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


@dataclass
class Record:
    """去重后在内存中保留的一条记录。"""

    text: str
    label: str
    source: str
    candidates: dict[str, str] = field(default_factory=dict)


def discover_files(paths: list[str] | None, *, include_old: bool = False) -> list[Path]:
    """确定要处理的 jsonl 列表。

    - 显式给了路径：用给的；
    - 否则递归扫 text/，剔除文件名含 fail 的失败清单以及默认跳过 *_old / *_backup 历史备份。
    返回按路径排序的列表 —— 排序保证「先到先得」这种裁决是同一次运行内
    可复现的，与文件系统遍历顺序无关。
    """
    if paths:
        out = []
        for p in paths:
            path = Path(p)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.exists():
                logging.warning("输入文件不存在，跳过：%s", path)
                continue
            out.append(path)
        return sorted(out)

    if not TEXT_ROOT.exists():
        logging.error("未找到数据目录：%s", TEXT_ROOT)
        return []
    all_files = sorted(TEXT_ROOT.rglob("*.jsonl"))
    kept = [p for p in all_files if not is_ignored_jsonl(p, include_old=include_old)]
    skipped = [p for p in all_files if is_ignored_jsonl(p, include_old=include_old)]
    if skipped:
        logging.info(
            "已剔除 %d 个非主数据文件（失败清单/临时文件%s）",
            len(skipped),
            "" if include_old else "/历史备份 *_old",
        )
    if not kept:
        logging.error("text/ 下没有可用的 .jsonl，请先运行生成脚本")
    return kept


def category_of(path: Path) -> str:
    """取类别名。

    优先用相对 text/ 的第一级目录名（text/violence_event/xxx.jsonl ->
    violence_event）；若文件不在 text/ 下，退回父目录名。这样输出 .txt
    的文件名始终与配置名一致，不会因为输入文件叫 gamble1_1200.jsonl
    而产出 gamble1_1200.txt。
    """
    try:
        rel = path.relative_to(TEXT_ROOT)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass
    return path.parent.name or "misc"


def split(paths: list[str] | None = None, output_dir: str | None = None,
          dry_run: bool = False, *, include_old: bool = False) -> dict[str, list[str]]:
    """执行切分，返回 {类别: [文本, ...]}。

    返回值同时用于写盘与统计，dry-run 时只统计不落地。
    """
    files = discover_files(paths, include_old=include_old)
    if not files:
        logging.error("没有可处理的输入文件")
        return {}

    stats = Stats(files=len(files))

    # text -> (保留的标签, 来源文件, 该 text 见过的全部 (标签, 来源))
    chosen: dict[str, Record] = {}
    conflicts: list[tuple[str, str, str, str, str]] = []  # text, kept, kept_src, drop, drop_src

    for path in files:
        cat = category_of(path)
        for item in iter_jsonl(path):
            stats.read += 1
            label = str(item.get("label") or "").strip()
            if not label:
                stats.no_text += 1
                logging.debug("第 %s 行缺少 label，跳过：%s", path.name, item.get("id"))
                continue

            text = clean_text(item.get("text"))
            if not text:
                stats.no_text += 1
                continue

            if len(text) < MIN_TEXT_LEN:
                stats.too_short += 1
                continue
            if len(text) > MAX_TEXT_LEN:
                stats.too_long += 1
                continue

            src = f"{cat}/{path.name}"
            prev = chosen.get(text)
            if prev is None:
                chosen[text] = Record(text=text, label=label, source=src,
                                      candidates={label: src})
                continue

            prev.candidates.setdefault(label, src)
            if prev.label == label:
                stats.dup_same += 1
                continue

            # 标签冲突：有害优先，同级别先到先得
            stats.dup_conflict += 1
            if label_rank(label) > label_rank(prev.label):
                conflicts.append((text, label, src, prev.label, prev.source))
                chosen[text] = Record(text=text, label=label, source=src,
                                      candidates=prev.candidates | {label: src})
            else:
                conflicts.append((text, prev.label, prev.source, label, src))

    # 按类别归拢
    grouped: dict[str, list[str]] = defaultdict(list)
    normal: list[str] = []
    for rec in chosen.values():
        if rec.label in NORMAL_LABELS:
            normal.append(rec.text)
            stats.kept_normal += 1
        else:
            grouped[rec.source.split("/", 1)[0]].append(rec.text)
            stats.kept_harmful += 1

    logging.info("统计：\n%s", stats.report())

    # 标签分布（按裁决后的最终归属）
    dist = Counter(rec.label for rec in chosen.values())
    logging.info("最终标签分布（共 %d 种）：", len(dist))
    for name, cnt in dist.most_common():
        logging.info("    %-52s %6d", name, cnt)

    if conflicts:
        logging.warning("发现 %d 条标签冲突，已按「有害优先」裁决；明细见 %s",
                        len(conflicts), CONFLICT_FILE)

    if dry_run:
        logging.info("dry-run：不写入任何文件")
        for cat, texts in sorted(grouped.items()):
            logging.info("    %-28s %6d", cat + ".txt", len(texts))
        logging.info("    %-28s %6d", NORMAL_FILE, len(normal))
        return dict(grouped)

    out_dir = Path(output_dir) if output_dir else FINAL_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每次运行整体重建：先删旧 .txt，再写新文件。
    # 只删自己会产出的那批名字 + normal.txt + 冲突表，不动 final/ 下其它文件。
    stale = {out_dir / f"{c}.txt" for c in grouped} | {out_dir / NORMAL_FILE}
    for f in stale:
        if f.exists():
            f.unlink()

    for cat, texts in sorted(grouped.items()):
        _write_lines(out_dir / f"{cat}.txt", texts)

    _write_lines(out_dir / NORMAL_FILE, normal)

    if conflicts:
        _write_conflicts(out_dir / CONFLICT_FILE, conflicts)
    elif (out_dir / CONFLICT_FILE).exists():
        # 上次有冲突、这次没有，清掉过期的冲突表，避免误读
        (out_dir / CONFLICT_FILE).unlink()

    logging.info("已写出 %d 个有害文件 + %s，目录：%s",
                 len(grouped), NORMAL_FILE, out_dir)
    return dict(grouped)


def _write_lines(path: Path, lines: list[str]) -> None:
    """原子写出（先写 .tmp 再替换），中途崩溃不会留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
    os.replace(tmp, path)


def _write_conflicts(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    """冲突明细：保留方 / 丢弃方 / 文本，制表符分隔便于表格软件打开。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("kept_label\tdropped_label\tkept_source\tdropped_source\ttext\n")
        for text, keep_lbl, keep_src, drop_lbl, drop_src in rows:
            f.write(f"{keep_lbl}\t{drop_lbl}\t{keep_src}\t{drop_src}\t{text}\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 text/<类别>/*.jsonl 切成 final/<类别>.txt 与 final/normal.txt",
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="要处理的 .jsonl 路径；留空则递归处理 text/ 下全部（自动跳过失败清单）",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help=f"输出目录，默认 {FINAL_ROOT}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只统计，不写文件",
    )
    parser.add_argument(
        "--include-old", action="store_true",
        help="包含 *_old.jsonl / *_backup*.jsonl 等历史备份文件（默认跳过）",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="输出 DEBUG 级日志（逐条打印被丢弃的原因）",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # dry-run 即使没有任何输入也不算错误（例如刚清空 text/ 想确认一下）
    grouped = split(
        args.inputs or None,
        args.output_dir,
        args.dry_run,
        include_old=args.include_old,
    )
    if not grouped and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
