# -*- coding: utf-8 -*-
"""文本规范化、多层去重判定与 SQLite 持久化去重索引。

设计要点
--------
1. 三层去重判定：
   - Layer 1 (clean_storage_text): 统一存储清洗，折叠换行/制表符/连续空格，去除零宽字符。
   - Layer 2 (canonical_text + sha256): NFKC 归一化 + 小写 + 去空白与标点（保留字母/数字/汉字），
     彻底拦截完全重复与仅标点/空格/全半角差异的重复变体。
   - Layer 3 (NearDupIndex): 类内字符 3-gram Jaccard + 步进 4-gram 倒排候选索引，
     拦截高度相似的模板化微调改写（默认阈值 0.85，设为 0 可关闭）。

2. 持久化与多进程安全：
   - 使用标准库 sqlite3（零额外依赖），开启 WAL 模式与 busy_timeout=30000，
     支持服务器单卡续跑与多卡按类别并行写入同一去重库。
   - 每次启动自动从已有 JSONL 文件做冷启动自愈同步（INSERT OR IGNORE），
     即使数据库文件缺失或重建，也可在数秒内由现有数据无损恢复。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------
# 文本清洗与规范化
# --------------------------------------------------------------------------

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_WS_CONTROL = re.compile(r"[\r\n\t]+")
_MULTI_SPACE = re.compile(r"\s{2,}")

# 文件名过滤：失败清单与历史备份文件
_FAIL_NAME = re.compile(r"fail", re.I)
_BACKUP_NAME = re.compile(r"(?:^|_)(?:old|bak|backup)(?:[_\-.]|$)", re.I)


def is_ignored_jsonl(path: Path, *, include_old: bool = False) -> bool:
    """判断某个 .jsonl 文件是否应被常规扫描跳过。"""
    name = path.name
    if name.endswith(".tmp") or _FAIL_NAME.search(name):
        return True
    if not include_old and _BACKUP_NAME.search(path.stem):
        return True
    return False


def clean_storage_text(text: object) -> str:
    """存储级清洗：去除零宽字符，折叠换行/制表符与连续空格。

    与 split_dataset.py 保持完全一致的口径，保证 JSONL 落盘与下游按行导出一致。
    """
    if not isinstance(text, str):
        return ""
    s = _ZERO_WIDTH.sub("", text)
    s = _WS_CONTROL.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip()


def canonical_text(text: object) -> str:
    """强规范化文本：NFKC 全半角归一 + 小写 + 仅保留汉字/字母/数字。"""
    s = clean_storage_text(text)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if ch.isalnum())


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _char_ngrams(text: str, n: int = 3) -> frozenset[str]:
    if not text:
        return frozenset()
    if len(text) <= n:
        return frozenset((text,))
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


def _anchor_grams(text: str) -> list[str]:
    """提取用于近似检索的倒排锚点。

    长度 >= 8 时按步长 2 提取 4-gram（含末尾 4-gram），兼顾高召回与低内存；
    更短文本直接用全部 3-gram 作为锚点。
    """
    n = len(text)
    if n == 0:
        return []
    if n < 8:
        return list(_char_ngrams(text, 3))
    anchors = [text[i : i + 4] for i in range(0, n - 3, 2)]
    tail = text[n - 4 :]
    if not anchors or anchors[-1] != tail:
        anchors.append(tail)
    return anchors


# --------------------------------------------------------------------------
# 重复异常定义
# --------------------------------------------------------------------------


class DuplicateTextError(ValueError):
    """样本命中去重拦截时抛出的异常（消息中不包含正文内容）。"""

    def __init__(
        self,
        dup_type: str,
        matched_category: str,
        matched_id: int | None = None,
        similarity: float = 1.0,
    ) -> None:
        self.dup_type = dup_type
        self.matched_category = matched_category
        self.matched_id = matched_id
        self.similarity = similarity
        id_part = f"#id={matched_id}" if matched_id is not None else ""
        super().__init__(
            f"文本重复({dup_type})：命中 {matched_category}{id_part}（相似度={similarity:.3f}）"
        )


# --------------------------------------------------------------------------
# 类内近似重复倒排索引
# --------------------------------------------------------------------------


class NearDupIndex:
    """基于字符 3-gram Jaccard 的轻量级近似重复索引。"""

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = float(threshold)
        self._items: list[tuple[int | None, frozenset[str]]] = []
        self._inverted: dict[str, list[int]] = defaultdict(list)

    @property
    def enabled(self) -> bool:
        return 0.0 < self.threshold <= 1.0

    def __len__(self) -> int:
        return len(self._items)

    def find_match(self, canon: str) -> tuple[int | None, float] | None:
        """查询是否存在 Jaccard >= threshold 的近似文本。"""
        if not self.enabled or not canon:
            return None
        g1 = _char_ngrams(canon, 3)
        if not g1:
            return None
        len1 = len(g1)

        candidates: set[int] = set()
        for anchor in _anchor_grams(canon):
            posting = self._inverted.get(anchor)
            if posting:
                candidates.update(posting)

        best_id: int | None = None
        best_sim = 0.0
        for idx in candidates:
            rec_id, g2 = self._items[idx]
            len2 = len(g2)
            if not len2:
                continue
            # 长度比上界剪枝：min/max < threshold 则 Jaccard 必然 < threshold
            if (len1 if len1 < len2 else len2) / (len1 if len1 > len2 else len2) < self.threshold:
                continue
            inter = len(g1 & g2)
            if inter == 0:
                continue
            sim = inter / (len1 + len2 - inter)
            if sim >= self.threshold and sim > best_sim:
                best_sim = sim
                best_id = rec_id

        if best_sim >= self.threshold:
            return best_id, best_sim
        return None

    def add(self, canon: str, record_id: int | None = None) -> None:
        if not self.enabled or not canon:
            return
        g = _char_ngrams(canon, 3)
        if not g:
            return
        slot = len(self._items)
        self._items.append((record_id, g))
        for anchor in set(_anchor_grams(canon)):
            self._inverted[anchor].append(slot)


# --------------------------------------------------------------------------
# SQLite 持久化去重器
# --------------------------------------------------------------------------


def _iter_jsonl_raw(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PersistentDeduplicator:
    """SQLite + 内存缓存 + 近似索引组成的持久化去重器。

    - 支持 global（全类别精确去重 + 类内近似去重）与 category（仅类内）两种作用域。
    - 采用 WAL + BEGIN IMMEDIATE 事务，确保崩溃一致与多进程并发安全。
    """

    def __init__(
        self,
        db_path: Path,
        category: str,
        *,
        scope: str = "global",
        near_dup_threshold: float = 0.85,
    ) -> None:
        if scope not in ("global", "category"):
            raise ValueError(f"不支持的去重作用域：{scope}（仅允许 global / category）")

        self.db_path = Path(db_path)
        self.category = category
        self.scope = scope
        self.near_index = NearDupIndex(threshold=near_dup_threshold)

        # 内存热缓存：canon_hash -> (category, record_id)，clean_hash 集合用于区分 exact/canonical
        self._seen_canon: dict[str, tuple[str, int]] = {}
        self._seen_clean: set[str] = set()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None 启用显式事务控制，便于 BEGIN IMMEDIATE / ROLLBACK
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=30000;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_texts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT    NOT NULL,
                record_id   INTEGER NOT NULL,
                label       TEXT    NOT NULL,
                clean_hash  TEXT    NOT NULL,
                canon_hash  TEXT    NOT NULL,
                source_file TEXT    NOT NULL,
                origin      TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                UNIQUE(category, canon_hash)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_seen_canon_hash ON seen_texts(canon_hash);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_seen_cat_rec ON seen_texts(category, record_id);"
        )
        cur.close()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "PersistentDeduplicator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def sync_from_jsonl(
        self,
        jsonl_path: Path,
        category: str,
        *,
        origin: str = "existing",
        load_near_index: bool = False,
    ) -> dict[str, int]:
        """从已有 JSONL 同步指纹到 SQLite 与内存缓存（幂等，重复调用安全）。"""
        stats = {
            "scanned": 0,
            "unique_in_file": 0,
            "internal_dups": 0,
            "inserted_db": 0,
            "empty": 0,
        }
        if not jsonl_path.exists():
            return stats

        now = _utc_now()
        rows_to_insert: list[tuple[Any, ...]] = []
        file_seen: set[str] = set()
        src_str = str(jsonl_path)

        for item in _iter_jsonl_raw(jsonl_path):
            stats["scanned"] += 1
            clean = clean_storage_text(item.get("text"))
            canon = canonical_text(clean)
            if not canon:
                stats["empty"] += 1
                continue

            rec_id = _safe_int(item.get("id"), -1)
            label = str(item.get("label") or "").strip()
            clean_h = text_sha256(clean)
            canon_h = text_sha256(canon)

            if canon_h in file_seen:
                stats["internal_dups"] += 1
            else:
                file_seen.add(canon_h)
                stats["unique_in_file"] += 1
                rows_to_insert.append(
                    (category, rec_id, label, clean_h, canon_h, src_str, origin, now)
                )
                if load_near_index and self.near_index.enabled:
                    self.near_index.add(canon, rec_id)

            # 更新内存缓存（全局模式全收录，类内模式仅收录当前类别）
            if self.scope == "global" or category == self.category:
                self._seen_canon.setdefault(canon_h, (category, rec_id))
                self._seen_clean.add(clean_h)

        if rows_to_insert:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                before = self._conn.total_changes
                cur.executemany(
                    """
                    INSERT OR IGNORE INTO seen_texts
                        (category, record_id, label, clean_hash, canon_hash, source_file, origin, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    rows_to_insert,
                )
                stats["inserted_db"] = self._conn.total_changes - before
                cur.execute("COMMIT;")
            except Exception:
                cur.execute("ROLLBACK;")
                raise
            finally:
                cur.close()

        return stats

    def load_memory_cache(self) -> int:
        """从 SQLite 加载已有指纹到内存热缓存，返回加载条数。"""
        cur = self._conn.cursor()
        try:
            if self.scope == "global":
                cur.execute("SELECT category, record_id, clean_hash, canon_hash FROM seen_texts;")
            else:
                cur.execute(
                    "SELECT category, record_id, clean_hash, canon_hash FROM seen_texts WHERE category = ?;",
                    (self.category,),
                )
            loaded = 0
            for cat, rec_id, clean_h, canon_h in cur:
                self._seen_canon.setdefault(canon_h, (cat, rec_id))
                self._seen_clean.add(clean_h)
                loaded += 1
            return loaded
        finally:
            cur.close()

    def count_in_db(self, category: str | None = None) -> int:
        cur = self._conn.cursor()
        try:
            if category is None:
                cur.execute("SELECT COUNT(*) FROM seen_texts;")
            else:
                cur.execute("SELECT COUNT(*) FROM seen_texts WHERE category = ?;", (category,))
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            cur.close()

    def check_and_persist(
        self,
        record: dict[str, Any],
        output_file: Path,
        append_fn: Callable[[Path, dict[str, Any]], None],
    ) -> None:
        """对单条待落库记录执行多层去重判定并原子写入 SQLite 与 JSONL。

        校验失败抛 ValueError，命中重复抛 DuplicateTextError。
        若写盘异常则自动回滚 SQLite 事务，保持两侧严格一致。
        """
        clean = clean_storage_text(record.get("text"))
        canon = canonical_text(clean)
        if not canon:
            raise ValueError("text 规范化后为空（仅含空白或标点符号）")

        clean_h = text_sha256(clean)
        canon_h = text_sha256(canon)
        rec_id = _safe_int(record.get("id"), -1)
        label = str(record.get("label") or "").strip()

        # 1. 内存热缓存 O(1) 快速拦截
        cached = self._seen_canon.get(canon_h)
        if cached is not None:
            m_cat, m_id = cached
            dup_type = "exact" if clean_h in self._seen_clean else "canonical"
            raise DuplicateTextError(dup_type, m_cat, m_id, 1.0)

        # 2. 类内近似重复拦截（3-gram Jaccard）
        if self.near_index.enabled:
            near_hit = self.near_index.find_match(canon)
            if near_hit is not None:
                m_id, sim = near_hit
                raise DuplicateTextError("near", self.category, m_id, sim)

        # 3. SQLite BEGIN IMMEDIATE 原子检查 + 占用 + JSONL 落盘 + 提交
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE;")
        committed = False
        try:
            if self.scope == "global":
                cur.execute(
                    "SELECT category, record_id, clean_hash FROM seen_texts WHERE canon_hash = ? LIMIT 1;",
                    (canon_h,),
                )
            else:
                cur.execute(
                    "SELECT category, record_id, clean_hash FROM seen_texts WHERE category = ? AND canon_hash = ? LIMIT 1;",
                    (self.category, canon_h),
                )
            row = cur.fetchone()
            if row is not None:
                m_cat, m_id, m_clean_h = row[0], row[1], row[2]
                cur.execute("ROLLBACK;")
                self._seen_canon.setdefault(canon_h, (m_cat, m_id))
                self._seen_clean.add(m_clean_h)
                dup_type = "exact" if m_clean_h == clean_h else "canonical"
                raise DuplicateTextError(dup_type, m_cat, m_id, 1.0)

            cur.execute(
                """
                INSERT INTO seen_texts
                    (category, record_id, label, clean_hash, canon_hash, source_file, origin, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'generated', ?);
                """,
                (self.category, rec_id, label, clean_h, canon_h, str(output_file), _utc_now()),
            )
            # 先写盘再提交事务；若写盘报错则回滚 SQLite
            append_fn(output_file, record)
            cur.execute("COMMIT;")
            committed = True
        except DuplicateTextError:
            raise
        except Exception:
            if not committed:
                try:
                    cur.execute("ROLLBACK;")
                except sqlite3.Error:
                    pass
            raise
        finally:
            cur.close()

        # 4. 事务提交成功后更新内存热缓存与近似索引
        self._seen_canon[canon_h] = (self.category, rec_id)
        self._seen_clean.add(clean_h)
        if self.near_index.enabled:
            self.near_index.add(canon, rec_id)
