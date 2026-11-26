# -*- coding: utf-8 -*-
"""数据集扩容全量只读验收校验脚本。

安全声明
--------
本脚本仅输出聚合统计数字、类别名、标签计数与通过/失败状态，
绝不向标准输出或 JSON 报告中打印任何具体文本正文内容。

校验维度
--------
1. 类别完整性：核对全部（或指定）类别数据文件存在。
2. 数量与 ID 完整性：总行数、唯一 ID 数、坏 JSON 行、重复 ID、缺失 ID、新增段条数、失败清单残留。
3. 标签与配比一致性：label 合法性、label == cfg.label_at(id)、与 20 槽位理论配比偏差。
4. 字段与生成规则一致性：7 字段完整、severity 合法、safe/borderline 非 high、text/reason 长度、主分类合法。
5. 去重零违约校验：
   - 新增段内部（清洗精确 / 强规范化 / 近似 Jaccard）；
   - 新增段 vs 存量段（强规范化交集）；
   - 跨 18 类别全局（新增段 vs 其他类别全量）。
6. SQLite 去重库只读一致性校验（若去重库存在）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (  # noqa: E402
    DEFAULT_DEDUP_DB,
    DEFAULT_NEAR_DUP_THRESHOLD,
    DEFAULT_TOTAL,
    MAX_REASON_LEN,
    MAX_TEXT_LEN,
    MIN_TEXT_LEN,
    list_configs,
    load_config,
    load_failed_ids,
)
from dedup import (  # noqa: E402
    NearDupIndex,
    canonical_text,
    clean_storage_text,
    text_sha256,
)


def _check_sqlite_coverage(
    db_path: Path,
    category_hashes: dict[str, set[str]],
) -> dict[str, Any]:
    """只读检查 SQLite 去重库是否覆盖全部 JSONL 规范化指纹。"""
    if not db_path.exists():
        return {"exists": False, "db_rows": 0, "missing_by_category": {}}

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM seen_texts;")
        total_rows = int(cur.fetchone()[0])

        db_by_cat: dict[str, set[str]] = defaultdict(set)
        cur.execute("SELECT category, canon_hash FROM seen_texts;")
        for cat, ch in cur:
            db_by_cat[cat].add(ch)

        missing: dict[str, int] = {}
        for cat, hashes in category_hashes.items():
            diff = len(hashes - db_by_cat.get(cat, set()))
            if diff:
                missing[cat] = diff
        return {
            "exists": True,
            "db_rows": total_rows,
            "missing_by_category": missing,
        }
    finally:
        conn.close()


def verify_dataset(
    *,
    categories: Sequence[str] | None = None,
    expected_total: int = DEFAULT_TOTAL,
    baseline_max_id: int = 8000,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    dedup_db: Path | None = None,
    allow_existing_dups: bool = True,
    require_db_sync: bool = False,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """执行全量校验并返回结构化报告（不含任何具体文本）。"""
    all_names = list(categories) if categories else list_configs()
    category_reports: dict[str, dict[str, Any]] = {}

    # 用于跨类别去重统计：canon_hash -> [(category, rec_id, is_new), ...]
    global_canon_owners: dict[str, list[tuple[str, int, bool]]] = defaultdict(list)
    category_canon_sets: dict[str, set[str]] = {}

    for name in all_names:
        cfg = load_config(name)
        if data_root is not None:
            output_file = data_root / name / f"{name}.jsonl"
            failed_file = data_root / name / f"{name}_failed.jsonl"
        else:
            output_file = cfg.default_output()
            failed_file = cfg.default_failed()

        required_fields = {
            "id",
            "text",
            "label",
            cfg.category_field,
            "risk_type",
            "severity",
            "reason",
        }
        valid_categories = set(cfg.dimensions[cfg.category_field])
        valid_labels = set(cfg.labels)
        valid_severities = set(cfg.severities)

        rep: dict[str, Any] = {
            "category": name,
            "file_exists": output_file.exists(),
            "total_rows": 0,
            "bad_json_lines": 0,
            "unique_ids": 0,
            "duplicate_ids": 0,
            "missing_ids_count": 0,
            "baseline_rows": 0,
            "added_rows": 0,
            "expected_added": max(0, expected_total - baseline_max_id) if baseline_max_id > 0 else expected_total,
            "failed_pending": 0,
            "label_illegal": 0,
            "label_slot_mismatch": 0,
            "label_ratio_mismatch": 0,
            "schema_errors_new": 0,
            "schema_errors_existing": 0,
            "existing_internal_canon_dups": 0,
            "new_internal_exact_dups": 0,
            "new_internal_canon_dups": 0,
            "new_vs_existing_canon_dups": 0,
            "new_near_dups": 0,
            "cross_category_new_dups": 0,
            "label_distribution": {},
            "passed": False,
            "reasons": [],
        }

        if not output_file.exists():
            rep["missing_ids_count"] = expected_total
            rep["reasons"].append("数据文件不存在")
            category_reports[name] = rep
            category_canon_sets[name] = set()
            continue

        seen_ids: set[int] = set()
        label_counter: Counter[str] = Counter()

        # 先读取全部有效行，分离存量段（id <= baseline_max_id）与新增段（id > baseline_max_id）
        parsed_rows: list[tuple[int, dict[str, Any]]] = []
        with open(output_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    rep["bad_json_lines"] += 1
                    continue
                if not isinstance(obj, dict):
                    rep["bad_json_lines"] += 1
                    continue
                rep["total_rows"] += 1
                try:
                    rec_id = int(obj["id"])
                except (KeyError, TypeError, ValueError):
                    rep["schema_errors_new"] += 1
                    continue
                if rec_id in seen_ids:
                    rep["duplicate_ids"] += 1
                else:
                    seen_ids.add(rec_id)
                parsed_rows.append((rec_id, obj))

        rep["unique_ids"] = len(seen_ids)
        missing_ids = [i for i in range(1, expected_total + 1) if i not in seen_ids]
        rep["missing_ids_count"] = len(missing_ids)

        failed_ids = load_failed_ids(failed_file) - seen_ids
        rep["failed_pending"] = len(failed_ids)

        # 按 id 排序，保证先将存量段载入去重索引，再逐条检验新增段
        parsed_rows.sort(key=lambda x: x[0])

        existing_canon_hashes: set[str] = set()
        new_clean_hashes: set[str] = set()
        new_canon_hashes: set[str] = set()
        cat_all_canon_hashes: set[str] = set()
        near_index = NearDupIndex(threshold=near_dup_threshold)

        for rec_id, obj in parsed_rows:
            is_new = rec_id > baseline_max_id if baseline_max_id > 0 else True
            if is_new:
                rep["added_rows"] += 1
            else:
                rep["baseline_rows"] += 1

            # 1. 字段完整性与生成规则检查
            row_schema_bad = False
            if not required_fields.issubset(obj.keys()):
                row_schema_bad = True

            label = str(obj.get("label") or "").strip()
            severity = str(obj.get("severity") or "").strip()
            cat_val = str(obj.get(cfg.category_field) or "").strip()
            risk_type = str(obj.get("risk_type") or "").strip()
            clean_txt = clean_storage_text(obj.get("text"))
            clean_reason = clean_storage_text(obj.get("reason"))
            canon = canonical_text(clean_txt)

            if label not in valid_labels:
                rep["label_illegal"] += 1
            else:
                label_counter[label] += 1
                if label != cfg.label_at(rec_id):
                    rep["label_slot_mismatch"] += 1

            if severity not in valid_severities:
                row_schema_bad = True
            if label in ("safe", "borderline") and severity == "high":
                row_schema_bad = True
            if not (MIN_TEXT_LEN <= len(clean_txt) <= MAX_TEXT_LEN) or not canon:
                row_schema_bad = True
            if not clean_reason or len(clean_reason) > MAX_REASON_LEN:
                row_schema_bad = True
            if cat_val not in valid_categories or not risk_type:
                row_schema_bad = True

            if row_schema_bad:
                if is_new:
                    rep["schema_errors_new"] += 1
                else:
                    rep["schema_errors_existing"] += 1

            if not canon:
                continue

            clean_h = text_sha256(clean_txt)
            canon_h = text_sha256(canon)
            cat_all_canon_hashes.add(canon_h)
            global_canon_owners[canon_h].append((name, rec_id, is_new))

            if not is_new:
                if canon_h in existing_canon_hashes:
                    rep["existing_internal_canon_dups"] += 1
                else:
                    existing_canon_hashes.add(canon_h)
                    if near_index.enabled:
                        near_index.add(canon, rec_id)
            else:
                if canon_h in existing_canon_hashes:
                    rep["new_vs_existing_canon_dups"] += 1
                elif canon_h in new_canon_hashes:
                    rep["new_internal_canon_dups"] += 1
                    if clean_h in new_clean_hashes:
                        rep["new_internal_exact_dups"] += 1
                else:
                    if near_index.enabled and near_index.find_match(canon) is not None:
                        rep["new_near_dups"] += 1
                    new_canon_hashes.add(canon_h)
                    new_clean_hashes.add(clean_h)
                    if near_index.enabled:
                        near_index.add(canon, rec_id)

        category_canon_sets[name] = cat_all_canon_hashes
        rep["label_distribution"] = dict(sorted(label_counter.items()))

        # 校验理论 20 槽位标签数量（当 expected_total 为 20 的整数倍时做精确配比校验）
        if expected_total % 20 == 0:
            cycles = expected_total // 20
            for lbl, slots in cfg.label_slots.items():
                expected_cnt = len(slots) * cycles
                actual_cnt = label_counter.get(lbl, 0)
                if actual_cnt != expected_cnt:
                    rep["label_ratio_mismatch"] += abs(actual_cnt - expected_cnt)

        category_reports[name] = rep

    # 跨类别去重统计：凡新增记录与其他类别存在相同 canon_hash，计入该新增记录所属类别
    for canon_h, owners in global_canon_owners.items():
        distinct_cats = {cat for cat, _, _ in owners}
        if len(distinct_cats) <= 1:
            continue
        for cat, _, is_new in owners:
            if is_new:
                category_reports[cat]["cross_category_new_dups"] += 1

    # SQLite 只读一致性检查
    db_info = _check_sqlite_coverage(
        dedup_db if dedup_db is not None else DEFAULT_DEDUP_DB,
        category_canon_sets,
    )

    # 综合判定每类是否通过
    all_passed = True
    for name, rep in category_reports.items():
        reasons = rep["reasons"]
        if rep["file_exists"]:
            if rep["bad_json_lines"] > 0:
                reasons.append(f"坏JSON行={rep['bad_json_lines']}")
            if rep["total_rows"] < expected_total or rep["unique_ids"] < expected_total:
                reasons.append(f"数量不足({rep['unique_ids']}/{expected_total})")
            if rep["duplicate_ids"] > 0:
                reasons.append(f"重复ID={rep['duplicate_ids']}")
            if rep["missing_ids_count"] > 0:
                reasons.append(f"缺失ID={rep['missing_ids_count']}")
            if baseline_max_id > 0 and rep["added_rows"] < rep["expected_added"]:
                reasons.append(f"新增不足({rep['added_rows']}/{rep['expected_added']})")
            if rep["failed_pending"] > 0:
                reasons.append(f"失败清单未清空={rep['failed_pending']}")
            if rep["label_illegal"] > 0:
                reasons.append(f"非法标签={rep['label_illegal']}")
            if rep["label_slot_mismatch"] > 0:
                reasons.append(f"槽位标签错配={rep['label_slot_mismatch']}")
            if rep["label_ratio_mismatch"] > 0:
                reasons.append(f"标签配比偏差={rep['label_ratio_mismatch']}")
            if rep["schema_errors_new"] > 0:
                reasons.append(f"新增字段/规则违规={rep['schema_errors_new']}")
            if not allow_existing_dups and rep["existing_internal_canon_dups"] > 0:
                reasons.append(f"存量内部重复={rep['existing_internal_canon_dups']}")
            if rep["new_vs_existing_canon_dups"] > 0:
                reasons.append(f"新增与存量重复={rep['new_vs_existing_canon_dups']}")
            if rep["new_internal_canon_dups"] > 0:
                reasons.append(f"新增内部重复={rep['new_internal_canon_dups']}")
            if near_dup_threshold > 0 and rep["new_near_dups"] > 0:
                reasons.append(f"新增近似重复={rep['new_near_dups']}")
            if rep["cross_category_new_dups"] > 0:
                reasons.append(f"新增跨类重复={rep['cross_category_new_dups']}")
            if require_db_sync:
                db_miss = db_info["missing_by_category"].get(name, 0)
                if not db_info["exists"] or db_miss > 0:
                    reasons.append(f"SQLite未同步={db_miss}")

        rep["passed"] = len(reasons) == 0
        if not rep["passed"]:
            all_passed = False

    summary = {
        "categories_checked": len(all_names),
        "categories_passed": sum(1 for r in category_reports.values() if r["passed"]),
        "expected_total_per_category": expected_total,
        "baseline_max_id": baseline_max_id,
        "near_dup_threshold": near_dup_threshold,
        "total_rows_all": sum(r["total_rows"] for r in category_reports.values()),
        "total_added_all": sum(r["added_rows"] for r in category_reports.values()),
        "total_new_dups": sum(
            r["new_vs_existing_canon_dups"]
            + r["new_internal_canon_dups"]
            + r["new_near_dups"]
            + r["cross_category_new_dups"]
            for r in category_reports.values()
        ),
        "dedup_db": db_info,
        "all_passed": all_passed,
    }

    return {
        "summary": summary,
        "categories": category_reports,
    }


def _print_table(report: dict[str, Any]) -> None:
    s = report["summary"]
    header = (
        f"{'类别':25} {'总条数':>6} {'新增':>6} {'缺ID':>5} {'失败':>5} "
        f"{'标错':>5} {'格错':>5} {'新内重':>6} {'撞旧重':>6} {'近似重':>6} {'跨类重':>6} {'状态'}"
    )
    print(header)
    print("-" * len(header) + "--------")
    for name, r in report["categories"].items():
        status = "PASS" if r["passed"] else "FAIL: " + "; ".join(r["reasons"])
        print(
            f"{name:25} {r['total_rows']:6d} {r['added_rows']:6d} "
            f"{r['missing_ids_count']:5d} {r['failed_pending']:5d} "
            f"{r['label_illegal'] + r['label_slot_mismatch']:5d} "
            f"{r['schema_errors_new']:5d} "
            f"{r['new_internal_canon_dups']:6d} {r['new_vs_existing_canon_dups']:6d} "
            f"{r['new_near_dups']:6d} {r['cross_category_new_dups']:6d} {status}"
        )
    print("-" * len(header) + "--------")
    db_info = s["dedup_db"]
    print(
        f"汇总：通过 {s['categories_passed']}/{s['categories_checked']} 类 | "
        f"全量行数 {s['total_rows_all']} | 新增段行数 {s['total_added_all']} | "
        f"新增重复总计 {s['total_new_dups']} | "
        f"SQLite存在={db_info['exists']}（记录数={db_info['db_rows']}） | "
        f"验收结论={'PASS' if s['all_passed'] else 'FAIL'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="有害文本数据集扩容只读验收工具（仅输出统计结果，不输出具体文本）",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="只校验指定类别，留空则校验全部 18 类",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=DEFAULT_TOTAL,
        help=f"每类期望总条数（默认 {DEFAULT_TOTAL}）",
    )
    parser.add_argument(
        "--baseline-max-id",
        type=int,
        default=8000,
        help="扩容前存量最大 ID（用于区分存量段与新增段，默认 8000；设为 0 表示全量统一校验）",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=DEFAULT_NEAR_DUP_THRESHOLD,
        help=f"新增段近似重复校验阈值，设为 0 关闭（默认 {DEFAULT_NEAR_DUP_THRESHOLD}）",
    )
    parser.add_argument(
        "--dedup-db",
        type=Path,
        default=None,
        help="SQLite 去重库路径（默认 data/dedup_index.sqlite）",
    )
    parser.add_argument(
        "--strict-all",
        action="store_true",
        help="严格模式：存量段内部历史重复也判为失败（默认仅作为信息汇报）",
    )
    parser.add_argument(
        "--require-db-sync",
        action="store_true",
        help="要求 SQLite 去重库存在且与全部 JSONL 完全同步",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="可选：将结构化统计报告写入 JSON 文件（不含任何具体文本）",
    )
    args = parser.parse_args(argv)

    report = verify_dataset(
        categories=args.categories,
        expected_total=args.expected_total,
        baseline_max_id=args.baseline_max_id,
        near_dup_threshold=args.near_dup_threshold,
        dedup_db=args.dedup_db,
        allow_existing_dups=not args.strict_all,
        require_db_sync=args.require_db_sync,
    )
    _print_table(report)

    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"结构化校验报告已写入：{args.json_report}")

    return 0 if report["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
