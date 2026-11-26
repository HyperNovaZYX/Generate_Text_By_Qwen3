# -*- coding: utf-8 -*-
"""样本生成统一入口。

用法
----
    python run.py --list                                 列出全部类别与当前进度
    python run.py violence_event                         扩容/续跑该类别至默认目标（12000 条）
    python run.py violence_event --total 12000           显式指定目标总条数
    python run.py violence_event --add 4000              基于已有最大 id 显式新增 4000 条
    python run.py violence_event --limit 20              只生成 20 条，用于试跑
    python run.py violence_event --dry-run               只打印提示词与分布，不加载模型/不写库
    python run.py violence_event --retry-failed          重试失败清单中的 id
    python run.py --all --total 12000 --max-rounds 5     依次运行全部 18 类（共享单 vLLM 实例）
    python run.py cat1 cat2 cat3 --total 12000           多卡分组时在单卡内串行运行指定类别

环境变量
--------
    GEN_MODEL_PATH    覆盖模型路径（默认 Qwen/Qwen3-4B-AWQ，建议指向本地模型目录）
    GEN_MAX_MODEL_LEN 覆盖上下文长度（默认 3072）
    GEN_GPU_UTIL      覆盖显存占比（默认 0.55）
    GEN_DEDUP_DB      覆盖 SQLite 去重库路径（默认 data/dedup_index.sqlite）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEDUP_SCOPE,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MODEL_PATH,
    DEFAULT_NEAR_DUP_THRESHOLD,
    _init_llm,
    list_configs,
    load_config,
    load_existing_ids,
    load_failed_ids,
    run_generation,
)


def _load(name: str):
    return load_config(name)


def _needs_llm(cfg, args) -> bool:
    """判断该类别是否确实有待生成样本，避免无任务时白白加载大模型。"""
    existing = load_existing_ids(cfg.default_output())
    if args.retry_failed:
        return bool(load_failed_ids(cfg.default_failed()) - existing)
    if args.add is not None:
        return args.add > 0
    target = args.total if args.total is not None else cfg.total
    if args.ensure_unique_total:
        return True
    return any(i not in existing for i in range(1, target + 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="有害文本分类样本生成（支持 12,000 扩容、持久化去重与自动重试）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "categories",
        nargs="*",
        help="一个或多个类别名，如 violence_event gamble",
    )
    parser.add_argument("--all", action="store_true", help="依次运行全部类别")
    parser.add_argument("--list", action="store_true", help="列出全部类别与现有条数")
    parser.add_argument("--total", type=int, default=None, help="目标总条数（默认取配置值 12000）")
    parser.add_argument("--add", type=int, default=None, help="增量模式：在现有最大 id 之后新增 N 条")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（小样本试跑）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help=f"含主轮次在内的最大自动重试轮数（默认 {DEFAULT_MAX_ROUNDS}）",
    )
    parser.add_argument(
        "--dedup-scope",
        choices=("global", "category"),
        default=DEFAULT_DEDUP_SCOPE,
        help=f"去重作用域：global 跨全类精确去重 / category 仅类内（默认 {DEFAULT_DEDUP_SCOPE}）",
    )
    parser.add_argument(
        "--dedup-db",
        type=str,
        default=None,
        help="SQLite 去重索引路径（默认 data/dedup_index.sqlite 或 GEN_DEDUP_DB）",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=DEFAULT_NEAR_DUP_THRESHOLD,
        help=f"类内 3-gram Jaccard 近似去重阈值，设为 0 关闭（默认 {DEFAULT_NEAR_DUP_THRESHOLD}）",
    )
    parser.add_argument(
        "--extra-existing",
        action="append",
        default=None,
        help="额外纳入去重基准的已有 JSONL 文件或目录（可多次指定）",
    )
    parser.add_argument(
        "--ensure-unique-total",
        action="store_true",
        help="若存量存在历史重复，额外追加新 id 使唯一文本总数 >= 目标值",
    )
    parser.add_argument("--retry-failed", action="store_true", help="仅重试失败清单")
    parser.add_argument("--dry-run", action="store_true", help="不加载模型、不写文件，仅自检")
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args(argv)

    if args.total is not None and args.add is not None:
        parser.error("--total 与 --add 不能同时指定")

    if args.list:
        for name in list_configs():
            cfg = _load(name)
            existing = len(load_existing_ids(cfg.default_output()))
            failed = len(load_failed_ids(cfg.default_failed()) - load_existing_ids(cfg.default_output()))
            print(
                f"{name:26} 标签 {len(cfg.labels):2} 个  "
                f"已有 {existing:6d} 条  失败 {failed:4d} 条  目标 {cfg.total:6d} 条"
            )
        return 0

    if args.all:
        names = list_configs()
    elif args.categories:
        names = args.categories
    else:
        parser.print_help()
        return 1

    model_path = os.environ.get("GEN_MODEL_PATH") or DEFAULT_MODEL_PATH
    max_model_len = int(os.environ.get("GEN_MAX_MODEL_LEN", "3072"))
    gpu_util = float(os.environ.get("GEN_GPU_UTIL", "0.55"))

    failures: list[str] = []
    loaded_cfgs = []
    for name in names:
        try:
            loaded_cfgs.append(_load(name))
        except FileNotFoundError:
            print(f"[跳过] 未知类别：{name}", file=sys.stderr)
            failures.append(name)

    # 多类别运行时共享单个 vLLM 实例，避免重复加载导致耗时与显存碎片
    shared_llm = None
    reuse_llm = len(loaded_cfgs) > 1 and not args.dry_run

    for cfg in loaded_cfgs:
        if reuse_llm and shared_llm is None and _needs_llm(cfg, args):
            shared_llm = _init_llm(model_path, max_model_len, gpu_util)

        stats = run_generation(
            cfg,
            retry_failed=args.retry_failed,
            limit=args.limit,
            batch_size=args.batch_size,
            model_path=model_path,
            max_model_len=max_model_len,
            gpu_util=gpu_util,
            seed=args.seed,
            dry_run=args.dry_run,
            target_total=args.total,
            add_count=args.add,
            max_rounds=args.max_rounds,
            dedup_scope=args.dedup_scope,
            dedup_db=args.dedup_db,
            near_dup_threshold=args.near_dup_threshold,
            extra_existing=args.extra_existing,
            ensure_unique_total=args.ensure_unique_total,
            llm=shared_llm,
        )
        print(f"[{cfg.name}] {stats}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
