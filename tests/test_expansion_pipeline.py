# -*- coding: utf-8 -*-
"""数据集 8,000 → 12,000 扩容方案离线单元测试。

全部测试使用无害占位文本与 Mock LLM，无需 GPU 或 vLLM，可在任意开发机/服务器上运行。
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core  # noqa: E402
from dedup import (  # noqa: E402
    DuplicateTextError,
    NearDupIndex,
    PersistentDeduplicator,
    canonical_text,
    clean_storage_text,
    is_ignored_jsonl,
    text_sha256,
)
from scripts.export_by_label import export_jsonl_by_label  # noqa: E402
from scripts.verify_dataset import verify_dataset  # noqa: E402


def _make_dummy_config(
    name: str,
    root: Path,
    *,
    category_field: str = "scene_category",
) -> core.GeneratorConfig:
    return core.GeneratorConfig(
        name=name,
        theme="占位测试主题",
        labels=["harmful_alpha", "harmful_beta", "borderline", "safe"],
        label_slots={
            "harmful_alpha": tuple(range(1, 11)),       # 10/20
            "harmful_beta": tuple(range(11, 17)),       # 6/20
            "borderline": (17, 18, 19),                 # 3/20
            "safe": (0,),                               # 1/20
        },
        severities=["low", "medium", "high"],
        dimensions={
            category_field: [f"场景{i}" for i in range(10)],
            "text_style": [f"形态{i}" for i in range(15)],
            "user_intent": [f"意图{i}" for i in range(12)],
            "language_style": [f"风格{i}" for i in range(10)],
            "resource_hint": [f"线索{i}" for i in range(12)],
        },
        risk_type_map={
            "harmful_alpha": "类型A",
            "harmful_beta": "类型B",
            "borderline": "边界类型",
            "safe": "正常类型",
        },
        prompt_body=(
            "IDX={idx} LABEL={target_label} CAT={"
            + category_field
            + "} STYLE={text_style} INTENT={user_intent} "
            "LANG={language_style} HINT={resource_hint}"
        ),
        category_field=category_field,
        total=120,
        output_file=str(root / name / f"{name}.jsonl"),
        failed_file=str(root / name / f"{name}_failed.jsonl"),
    )


def _unique_harmless_text(tag: str, num: int) -> str:
    """生成彼此高度不同（Jaccard 极低）且完全无害的占位文本。"""
    # 使用不同汉字/字母组合打散 3-gram，避免触发近似去重
    alphabet_a = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
    alphabet_b = "春夏秋冬东南西北金木水火土风花雪月山川湖海星云"
    c1 = alphabet_a[num % len(alphabet_a)]
    c2 = alphabet_b[(num // len(alphabet_a)) % len(alphabet_b)]
    c3 = alphabet_a[(num * 7 + 3) % len(alphabet_a)]
    c4 = alphabet_b[(num * 11 + 5) % len(alphabet_b)]
    token = text_sha256(f"{tag}:{num}")[:14]
    return f"测试占位{tag}{c1}{c2}{c3}{c4}编号{num:05d}标识{token}"


def _mock_sampling_params(
    *,
    attempt: int = 0,
    truncated_retry: bool = False,
    max_model_len: int = 3072,
    seed: int | None = None,
):
    max_tokens = core.MAX_NEW_TOKENS
    if truncated_retry:
        upper = max(1024, max_model_len - 512)
        max_tokens = min(max(core.MAX_NEW_TOKENS + 512, 2560), upper)
    return SimpleNamespace(
        attempt=attempt,
        truncated_retry=truncated_retry,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        seed=seed,
    )


class TestDedupUnit(unittest.TestCase):
    def test_clean_and_canonical_normalization(self):
        raw_a = "\ufeff  示例测试文本\r\n\tＡＢＣ，１２３！  "
        raw_b = "示例测试文本 abc 123?"
        self.assertEqual(clean_storage_text(raw_a), "示例测试文本 ＡＢＣ，１２３！")
        self.assertEqual(canonical_text(raw_a), canonical_text(raw_b))
        self.assertEqual(canonical_text("  ...！？\u200b  "), "")

    def test_ignored_jsonl_detection(self):
        self.assertTrue(is_ignored_jsonl(Path("gamble_failed.jsonl")))
        self.assertTrue(is_ignored_jsonl(Path("gamble.jsonl.tmp")))
        self.assertTrue(is_ignored_jsonl(Path("politics_cult_old.jsonl")))
        self.assertFalse(is_ignored_jsonl(Path("politics_cult_old.jsonl"), include_old=True))
        self.assertFalse(is_ignored_jsonl(Path("gamble.jsonl")))

    def test_near_dup_index(self):
        idx = NearDupIndex(threshold=0.85)
        base = canonical_text(
            "这是一条用于单元测试的占位文本样例用来验证近似重复索引在长句末尾改动时的召回能力一"
        )
        near = canonical_text(
            "这是一条用于单元测试的占位文本样例用来验证近似重复索引在长句末尾改动时的召回能力二"
        )
        diff = canonical_text("完全不同的另外一段测试记录用于检验误报情况九八七六五")
        idx.add(base, record_id=1)

        hit = idx.find_match(near)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], 1)
        self.assertGreaterEqual(hit[1], 0.85)
        self.assertIsNone(idx.find_match(diff))

        disabled = NearDupIndex(threshold=0.0)
        disabled.add(base, record_id=1)
        self.assertIsNone(disabled.find_match(near))

    def test_persistent_dedup_rollback_rebuild_and_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "dedup.sqlite"
            out_a = tmp_path / "cat_a.jsonl"
            out_b = tmp_path / "cat_b.jsonl"

            # 1. 写盘失败时必须回滚 SQLite，不残留脏指纹
            with PersistentDeduplicator(db_path, "cat_a", scope="global") as d:
                rec = {"id": 1, "text": _unique_harmless_text("rb", 1), "label": "safe"}

                def _broken_append(p, item):
                    raise OSError("模拟磁盘写失败")

                with self.assertRaises(OSError):
                    d.check_and_persist(rec, out_a, _broken_append)
                self.assertEqual(d.count_in_db("cat_a"), 0)

                # 正常写盘应成功
                d.check_and_persist(rec, out_a, core.append_jsonl)
                self.assertEqual(d.count_in_db("cat_a"), 1)

                # 标点/全角变体命中 canonical 重复
                rec_var = {"id": 2, "text": rec["text"] + "！！！", "label": "safe"}
                with self.assertRaises(DuplicateTextError) as cm:
                    d.check_and_persist(rec_var, out_a, core.append_jsonl)
                self.assertEqual(cm.exception.dup_type, "canonical")

            # 2. 跨类别全局拦截 vs 仅类内放行
            with PersistentDeduplicator(db_path, "cat_b", scope="global") as d_global:
                d_global.load_memory_cache()
                rec_b = {"id": 1, "text": _unique_harmless_text("rb", 1), "label": "safe"}
                with self.assertRaises(DuplicateTextError) as cm:
                    d_global.check_and_persist(rec_b, out_b, core.append_jsonl)
                self.assertEqual(cm.exception.matched_category, "cat_a")

            with PersistentDeduplicator(db_path, "cat_b", scope="category") as d_cat:
                d_cat.load_memory_cache()
                rec_b = {"id": 1, "text": _unique_harmless_text("rb", 1), "label": "safe"}
                d_cat.check_and_persist(rec_b, out_b, core.append_jsonl)

            # 3. 删除 SQLite 后从 JSONL 无损冷启动重建
            db_path.unlink()
            with PersistentDeduplicator(db_path, "cat_a", scope="global") as d_rebuild:
                st = d_rebuild.sync_from_jsonl(out_a, "cat_a", load_near_index=True)
                self.assertEqual(st["inserted_db"], 1)
                rec_dup = {"id": 9, "text": _unique_harmless_text("rb", 1), "label": "safe"}
                with self.assertRaises(DuplicateTextError):
                    d_rebuild.check_and_persist(rec_dup, out_a, core.append_jsonl)

            # 4. 多线程并发写入同一 SQLite（模拟多卡并发）
            shared_text = _unique_harmless_text("race", 999)
            results = {"ok": 0, "dup": 0, "err": 0}
            lock = threading.Lock()

            def worker(tid: int):
                out_w = tmp_path / f"worker_{tid}.jsonl"
                try:
                    with PersistentDeduplicator(db_path, f"cat_w{tid}", scope="global") as dw:
                        # 故意不预热内存缓存，直接考验 SQLite BEGIN IMMEDIATE 原子互斥
                        try:
                            dw.check_and_persist(
                                {"id": tid, "text": shared_text, "label": "safe"},
                                out_w,
                                core.append_jsonl,
                            )
                            with lock:
                                results["ok"] += 1
                        except DuplicateTextError:
                            with lock:
                                results["dup"] += 1
                except Exception:
                    with lock:
                        results["err"] += 1

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(results["err"], 0)
            self.assertEqual(results["ok"], 1)
            self.assertEqual(results["dup"], 5)


class TestDimensionDiversity(unittest.TestCase):
    def test_dimension_coverage_and_12k_diversity(self):
        # 1. 使用内置占位配置验证跨轮次组合多样性（在不含私有 configs/*.py 的干净仓库也可通过）
        with tempfile.TemporaryDirectory() as tmp:
            dummy_cfg = _make_dummy_config("dummy_div", Path(tmp))
            picker = core.DimensionPicker(dummy_cfg.dimensions, seed=20261102)
            for key, seen in picker.coverage().items():
                self.assertEqual(seen, picker.length(key))

            combos = {
                tuple(picker.pick(k, i) for k in picker.names())
                for i in range(1, 12001)
            }
            self.assertGreater(len(combos), 1000)

            p0 = core.build_prompt(dummy_cfg, picker, idx=8001, attempt=0)
            p1 = core.build_prompt(dummy_cfg, picker, idx=8001, attempt=1)
            self.assertNotEqual(p0, p1)

        # 2. 若本地存在业务配置，逐一执行加载、12,000 默认目标与维度全覆盖校验
        names = core.list_configs()
        self.assertNotIn("_example", names)
        for name in names:
            cfg = core.load_config(name)
            self.assertEqual(cfg.total, 12000)
            picker = core.DimensionPicker(cfg.dimensions, seed=20261102)
            for key, seen in picker.coverage().items():
                self.assertEqual(seen, picker.length(key), f"{name}.{key} 覆盖不足")


class _FaultInjectingMockLLM:
    """可编程故障注入 Mock LLM，验证全部重试、去重与截断分支。"""

    def __init__(
        self,
        cfg: core.GeneratorConfig,
        picker: core.DimensionPicker,
        *,
        existing_sample: str,
        cross_cat_sample: str,
    ) -> None:
        self.cfg = cfg
        self.picker = picker
        self.existing_sample = existing_sample
        self.cross_cat_sample = cross_cat_sample
        self.calls_by_id: dict[int, int] = {}
        self.max_tokens_seen: dict[int, list[int]] = {}
        self.batch_crash_done = False

    def _extract_idx(self, prompt: str) -> int:
        for tok in prompt.split():
            if tok.startswith("IDX="):
                return int(tok.split("=", 1)[1])
        raise ValueError("missing IDX")

    def _ok_json(self, idx: int, text: str) -> str:
        lbl = self.cfg.label_at(idx)
        sev = "low" if lbl in ("safe", "borderline") else "high"
        cat_val = self.picker.pick(self.cfg.category_field, idx, attempt=0)
        payload = {
            "id": idx,
            "text": text,
            "label": lbl,
            self.cfg.category_field: cat_val,
            "risk_type": self.cfg.risk_type_map[lbl],
            "severity": sev,
            "reason": f"占位判定理由编号{idx}",
        }
        return json.dumps(payload, ensure_ascii=False)

    def generate(self, prompts, sampling_params):
        # 故意对包含 id=89 的首个多条批次抛出异常，测试自动单条降级重试
        ids = [self._extract_idx(p) for p in prompts]
        if len(ids) > 1 and 89 in ids and not self.batch_crash_done:
            self.batch_crash_done = True
            raise RuntimeError("模拟批次推理瞬时异常")

        outs = []
        for idx in ids:
            attempt = self.calls_by_id.get(idx, 0)
            self.calls_by_id[idx] = attempt + 1
            self.max_tokens_seen.setdefault(idx, []).append(sampling_params.max_tokens)

            if attempt == 0:
                if idx == 81:
                    # 1. 撞已有文本（exact）
                    raw = self._ok_json(idx, self.existing_sample)
                elif idx == 82:
                    # 2. 撞已有文本标点变体（canonical）
                    raw = self._ok_json(idx, self.existing_sample + "！！")
                elif idx == 83:
                    # 3. 撞已有文本近似改写（near）
                    near_txt = self.existing_sample[:-1] + ("乙" if not self.existing_sample.endswith("乙") else "甲")
                    raw = self._ok_json(idx, near_txt)
                elif idx == 84:
                    # 4. 撞其他类别已有文本（global cross-category）
                    raw = self._ok_json(idx, self.cross_cat_sample)
                elif idx == 85:
                    # 5. <think> 未闭合截断
                    raw = "<think>正在推理但是预算耗尽未闭合"
                elif idx == 86:
                    # 6. 非法标签拼写
                    bad = json.loads(self._ok_json(idx, _unique_harmless_text("new", idx)))
                    bad["label"] = "harmful_typo_xxx"
                    raw = json.dumps(bad, ensure_ascii=False)
                elif idx == 88:
                    # 7. 新增内部互相撞车（与 id=87 产出同文）
                    raw = self._ok_json(idx, _unique_harmless_text("new", 87))
                else:
                    raw = self._ok_json(idx, _unique_harmless_text("new", idx))
            else:
                # 重试轮产出全新不重复文本（带 <think> 闭合测试清洗）
                raw = "<think>推理完成</think>\n" + self._ok_json(
                    idx, _unique_harmless_text(f"retry{attempt}", idx)
                )

            outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=raw)]))
        return outs


class TestExpansionPipelineEndToEnd(unittest.TestCase):
    def test_expand_80_to_120_with_retries_dedup_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "text"
            db_path = root / "data" / "dedup_index.sqlite"

            cfg_a = _make_dummy_config("demo_cat_a", data_root)
            cfg_b = _make_dummy_config("demo_cat_b", data_root)
            picker_a = core.DimensionPicker(cfg_a.dimensions, seed=20261102)
            picker_b = core.DimensionPicker(cfg_b.dimensions, seed=20261102)

            # 构造存量 80 条（模拟 8,000 条基线），其中 id=10 与 id=2 故意重复以验证存量保护口径
            out_a = cfg_a.default_output()
            for idx in range(1, 81):
                lbl = cfg_a.label_at(idx)
                sev = "low" if lbl in ("safe", "borderline") else "high"
                txt = _unique_harmless_text("oldA", 2 if idx == 10 else idx)
                core.append_jsonl(
                    out_a,
                    {
                        "id": idx,
                        "text": txt,
                        "label": lbl,
                        cfg_a.category_field: picker_a.pick(cfg_a.category_field, idx, attempt=0),
                        "risk_type": cfg_a.risk_type_map[lbl],
                        "severity": sev,
                        "reason": f"存量理由{idx}",
                    },
                )

            # 构造另一个类别的已有文件，验证全局跨类去重
            out_b = cfg_b.default_output()
            cross_sample = _unique_harmless_text("oldB", 1)
            for idx in range(1, 121):
                lbl = cfg_b.label_at(idx)
                sev = "low" if lbl in ("safe", "borderline") else "high"
                core.append_jsonl(
                    out_b,
                    {
                        "id": idx,
                        "text": cross_sample if idx == 1 else _unique_harmless_text("oldB", idx),
                        "label": lbl,
                        cfg_b.category_field: picker_b.pick(cfg_b.category_field, idx, attempt=0),
                        "risk_type": cfg_b.risk_type_map[lbl],
                        "severity": sev,
                        "reason": f"B类理由{idx}",
                    },
                )

            existing_sample = _unique_harmless_text("oldA", 1)
            mock_llm = _FaultInjectingMockLLM(
                cfg_a,
                picker_a,
                existing_sample=existing_sample,
                cross_cat_sample=cross_sample,
            )

            with (
                mock.patch.object(core, "PROJECT_ROOT", root),
                mock.patch.object(core, "_build_sampling_params", _mock_sampling_params),
            ):
                stats = core.run_generation(
                    cfg_a,
                    batch_size=4,
                    target_total=120,
                    max_rounds=5,
                    dedup_scope="global",
                    dedup_db=db_path,
                    near_dup_threshold=0.85,
                    extra_existing=[out_b],
                    llm=mock_llm,
                )

            self.assertEqual(stats["skipped"], 80)
            self.assertEqual(stats["success"], 40)
            self.assertEqual(stats["failed"], 0)
            self.assertEqual(stats["total_after"], 120)
            self.assertGreaterEqual(stats["dup_blocked"], 5)
            # 确认截断样本 id=85 在重试时获得了更高的 max_tokens 预算
            self.assertEqual(len(mock_llm.max_tokens_seen[85]), 2)
            self.assertGreater(mock_llm.max_tokens_seen[85][1], mock_llm.max_tokens_seen[85][0])
            # 失败清单应已完全收敛清空
            self.assertEqual(len(list(core.iter_jsonl(cfg_a.default_failed()))), 0)

            # 再次运行应直接跳过，0 新增
            with (
                mock.patch.object(core, "PROJECT_ROOT", root),
                mock.patch.object(core, "_build_sampling_params", _mock_sampling_params),
            ):
                stats_again = core.run_generation(
                    cfg_a,
                    target_total=120,
                    dedup_db=db_path,
                    llm=mock_llm,
                )
            self.assertEqual(stats_again["success"], 0)
            self.assertEqual(stats_again["skipped"], 120)

            # 使用 verify_dataset 验收两类数据（打桩配置加载器）
            cfg_map = {"demo_cat_a": cfg_a, "demo_cat_b": cfg_b}
            with mock.patch("scripts.verify_dataset.load_config", side_effect=lambda n: cfg_map[n]):
                rep = verify_dataset(
                    categories=["demo_cat_a", "demo_cat_b"],
                    expected_total=120,
                    baseline_max_id=80,
                    near_dup_threshold=0.85,
                    dedup_db=db_path,
                    allow_existing_dups=True,
                    require_db_sync=True,
                    data_root=data_root,
                )
            self.assertTrue(rep["summary"]["all_passed"], rep["categories"]["demo_cat_a"]["reasons"])
            self.assertEqual(rep["categories"]["demo_cat_a"]["added_rows"], 40)
            self.assertEqual(rep["categories"]["demo_cat_a"]["existing_internal_canon_dups"], 1)
            self.assertEqual(rep["categories"]["demo_cat_a"]["new_internal_canon_dups"], 0)
            self.assertEqual(rep["categories"]["demo_cat_a"]["new_vs_existing_canon_dups"], 0)
            self.assertEqual(rep["categories"]["demo_cat_a"]["new_near_dups"], 0)
            self.assertEqual(rep["categories"]["demo_cat_a"]["cross_category_new_dups"], 0)

            # 验证 --add 20 增量模式（并且先删 SQLite 验证冷启动自愈重建）
            db_path.unlink()
            with (
                mock.patch.object(core, "PROJECT_ROOT", root),
                mock.patch.object(core, "_build_sampling_params", _mock_sampling_params),
            ):
                stats_add = core.run_generation(
                    cfg_a,
                    add_count=20,
                    max_rounds=3,
                    dedup_scope="global",
                    dedup_db=db_path,
                    extra_existing=[out_b],
                    llm=mock_llm,
                )
            self.assertEqual(stats_add["success"], 20)
            self.assertEqual(stats_add["total_after"], 140)

            # 验证 verify_dataset 能准确捕获注入的新增重复与缺量故障
            core.append_jsonl(
                out_a,
                {
                    "id": 141,
                    "text": existing_sample,  # 故意注入撞存量重复
                    "label": cfg_a.label_at(141),
                    cfg_a.category_field: picker_a.pick(cfg_a.category_field, 141, attempt=0),
                    "risk_type": cfg_a.risk_type_map[cfg_a.label_at(141)],
                    "severity": "high",
                    "reason": "故意故障测试",
                },
            )
            with mock.patch("scripts.verify_dataset.load_config", side_effect=lambda n: cfg_map[n]):
                rep_bad = verify_dataset(
                    categories=["demo_cat_a"],
                    expected_total=140,
                    baseline_max_id=80,
                    dedup_db=db_path,
                    data_root=data_root,
                )
            self.assertFalse(rep_bad["summary"]["all_passed"])
            self.assertEqual(rep_bad["categories"]["demo_cat_a"]["new_vs_existing_canon_dups"], 1)

            # 验证合并后的按标签导出工具
            export_counts = export_jsonl_by_label(out_b, output_dir=root / "exported")
            self.assertEqual(sum(export_counts.values()), 120)
            self.assertTrue((root / "exported" / "demo_cat_b_safe.txt").exists())
            self.assertTrue((root / "exported" / "demo_cat_b_borderline.txt").exists())
            self.assertTrue((root / "exported" / "demo_cat_b_other.txt").exists())

            # 释放 Windows 临时目录下的日志 FileHandler，确保 TemporaryDirectory 正常清理
            logger = logging.getLogger("demo_cat_a")
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)

    def test_ensure_unique_total_compensates_existing_dups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "text"
            db_path = root / "data" / "dedup_index.sqlite"
            cfg = _make_dummy_config("demo_unique", data_root)
            picker = core.DimensionPicker(cfg.dimensions, seed=20261102)

            # 构造 20 条存量，其中 id=2,3 与 id=1 重复（存在 2 条历史重复 -> 唯一数 18）
            out = cfg.default_output()
            for idx in range(1, 21):
                lbl = cfg.label_at(idx)
                sev = "low" if lbl in ("safe", "borderline") else "high"
                txt = _unique_harmless_text("uniq", 1 if idx in (1, 2, 3) else idx)
                core.append_jsonl(
                    out,
                    {
                        "id": idx,
                        "text": txt,
                        "label": lbl,
                        cfg.category_field: picker.pick(cfg.category_field, idx, attempt=0),
                        "risk_type": cfg.risk_type_map[lbl],
                        "severity": sev,
                        "reason": f"存量{idx}",
                    },
                )

            mock_llm = _FaultInjectingMockLLM(
                cfg,
                picker,
                existing_sample=_unique_harmless_text("uniq", 1),
                cross_cat_sample=_unique_harmless_text("other", 1),
            )
            try:
                with (
                    mock.patch.object(core, "PROJECT_ROOT", root),
                    mock.patch.object(core, "_build_sampling_params", _mock_sampling_params),
                ):
                    # 目标 40 条且启用 ensure_unique_total：需生成 21..40 (20条) + 补偿 41..42 (2条) = 22 条
                    stats = core.run_generation(
                        cfg,
                        target_total=40,
                        ensure_unique_total=True,
                        dedup_scope="category",
                        dedup_db=db_path,
                        llm=mock_llm,
                    )
                self.assertEqual(stats["success"], 22)
                self.assertEqual(stats["total_after"], 42)
                _, unique_cnt = core._count_unique_canonical(out)
                self.assertEqual(unique_cnt, 40)
            finally:
                logger = logging.getLogger("demo_unique")
                for h in list(logger.handlers):
                    h.close()
                    logger.removeHandler(h)


if __name__ == "__main__":
    unittest.main()
