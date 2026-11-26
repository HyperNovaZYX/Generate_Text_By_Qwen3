# -*- coding: utf-8 -*-
"""
有害文本分类样本生成 —— 公共引擎。

设计目标
--------
18 个业务类别（暴恐 / 政治 / 色情 / 违禁品 / 诈骗……）此前各自维护一份
完整脚本，95% 代码是复制粘贴，导致同一处缺陷只在一两个文件里被修掉。
本模块把「与业务无关的流水线」全部收拢到这里，业务类别只保留配置：

    core.py                 LLM 调用 / 提示词组装 / 解析 / 校验 / 落库 / 续跑
    configs/<name>.py       主题、标签表、各维度词表、提示词正文
    run.py                  统一命令行入口
    <name>.py               兼容旧用法的薄壳

关键修复（对应审查报告编号）
--------------------------
1.2  max_tokens 由 800 提到 2048，并剥离 Qwen3 的 <think> 思维链；
     未闭合的 <think> 视为截断，直接判失败——而不是把半截推理当成正文。
1.1  语义维度取值不再用 (idx*k-1)%n —— 该式在 gcd(k,n)>1 时只能取到
     n/gcd 个值（实测 USER_INTENTS 只能取 4/12、LANGUAGE_STYLES 只能取 2/10）。
     改为「逆序打乱 + 独立偏移的循环」，保证每个维度在周期内跑满全部取值。
2.3  JSON 提取改为括号配对扫描，替换贪婪正则 `\\{.*\\}`（贪婪匹配会从第一个
     `{` 一直吞到最后一个 `}`，遇到正文里带花括号时整条解析报废）。
2.2  失败样本进入 FAILED_FILE 后可以 --retry-failed 重新生成，成功后自动
     从失败清单中移除，失败率才能真正收敛。
2.5  模型路径与输出路径可被环境变量覆盖，不再硬编码。
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from dedup import (
    DuplicateTextError,
    PersistentDeduplicator,
    canonical_text,
    clean_storage_text,
    is_ignored_jsonl,
)

# --------------------------------------------------------------------------
# 路径与默认参数
# --------------------------------------------------------------------------

# 项目根目录 = 本文件所在目录。所有输出路径都以它为基准，避免受 cwd 影响。
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B-AWQ"

DEFAULT_TOTAL = 12000
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEDUP_DB = PROJECT_ROOT / "data" / "dedup_index.sqlite"
DEFAULT_DEDUP_SCOPE = "global"
DEFAULT_NEAR_DUP_THRESHOLD = 0.85
DEFAULT_MAX_ROUNDS = 5

# 文本长度闸门。提示词里一直写着「不超过 120 字」，但旧校验只判空，
# 结果实际数据里既有 5 字样本也有 192 字样本。这里让约束真正生效。
MIN_TEXT_LEN = 6
MAX_TEXT_LEN = 120
MAX_REASON_LEN = 80

# 模型单次输出的 token 预算。Qwen3 默认先输出 <think> 推理再给 JSON，
# 800 远远不够，实测导致 15%~26% 样本因推理未写完就被截断而失败。
MAX_NEW_TOKENS = 2048

# 维度取值的循环周期。取 12 的倍数，能被下列常见词表长度整除或接近：
# 为了让 (idx + offset) % n 在任意 n 下都均匀覆盖，这里取一个足够大的
# 公倍数区间，实际取值由 _pick 内部处理，周期只影响「组合重复的间隔」。
DIVERSITY_CYCLE = 240


# --------------------------------------------------------------------------
# 配置数据结构
# --------------------------------------------------------------------------


@dataclass
class LabelSpec:
    """标签在 20 周期中的配额。

    month_slots 是 idx%20 的取值集合，用于精确保留原来的标签配比
    （例如 violent_event 的 5/5/4/3/2/1 分布）。
    """

    name: str
    slots: tuple[int, ...]


@dataclass
class GeneratorConfig:
    """单个业务类别的全部生成配置。

    只在这里放「业务内容」，不放任何流程控制。
    """

    name: str                       # 类别标识，同时是目录名，如 violence_event
    theme: str                      # 一句话主题，写入提示词
    labels: Sequence[str]           # 允许的标签全集
    label_slots: dict[str, tuple[int, ...]]   # 标签 -> idx%20 的槽位
    severities: Sequence[str]
    # 主分类维度：键是提示词中的占位符名，值是候选取值表。
    # 例如 {"scene_category": [...], "text_style": [...], ...}
    dimensions: dict[str, Sequence[str]]
    risk_type_map: dict[str, str]    # 标签 -> 中文风险类型
    prompt_body: str                 # 提示词正文（{占位符} 由引擎填充）
    category_field: str = "scene_category"   # 落库字段名，history/event 用 event_category
    total: int = DEFAULT_TOTAL
    output_file: str = ""            # 相对 PROJECT_ROOT；空则用默认约定
    failed_file: str = ""
    reference_file: str = ""         # 可选：few-shot 参考样本池（纯文本，每行一条）
    reference_count: int = 2
    extra_validators: list = field(default_factory=list)

    # ---- 派生属性 ----------------------------------------------------

    @property
    def label_names(self) -> list[str]:
        return list(self.labels)

    @property
    def harmful_labels(self) -> list[str]:
        """除 borderline / safe 之外的标签都视为有害。"""
        return [x for x in self.labels if x not in ("borderline", "safe")]

    def default_output(self) -> Path:
        if self.output_file:
            p = Path(self.output_file)
            return p if p.is_absolute() else PROJECT_ROOT / p
        return PROJECT_ROOT / "text" / self.name / f"{self.name}.jsonl"

    def default_failed(self) -> Path:
        if self.failed_file:
            p = Path(self.failed_file)
            return p if p.is_absolute() else PROJECT_ROOT / p
        return PROJECT_ROOT / "text" / self.name / f"{self.name}_failed.jsonl"

    def label_at(self, idx: int) -> str:
        """按 20 周期配额取标签。

        原脚本用 if/elif 硬编码 `r in [1,2,3,4,5]`，这里改成查表，
        配置里写 slots 即可，不必改代码逻辑。
        """
        r = idx % 20
        for name, slots in self.label_slots.items():
            if r in slots:
                return name
        # 兜底：最后一个标签（原逻辑的 else 分支就是 safe）
        return self.labels[-1]

    def validate(self) -> None:
        """配置自检。任何一项不满足都在加载阶段直接报错，不留到生成途中。"""
        if not self.theme:
            raise ValueError(f"[{self.name}] THEME 为空")
        if not self.labels:
            raise ValueError(f"[{self.name}] LABELS 为空")

        unknown = set(self.label_slots) - set(self.labels)
        if unknown:
            raise ValueError(f"[{self.name}] LABEL_SLOTS 含未定义标签：{sorted(unknown)}")

        covered: list[int] = []
        for slots in self.label_slots.values():
            covered.extend(slots)
        dup = {x for x in covered if covered.count(x) > 1}
        if dup:
            raise ValueError(f"[{self.name}] 槽位重复分配：{sorted(dup)}")
        missing = set(range(20)) - set(covered)
        if missing:
            raise ValueError(f"[{self.name}] 槽位未铺满，缺少 {sorted(missing)}")

        if self.category_field not in self.dimensions:
            raise ValueError(f"[{self.name}] 主分类字段 {self.category_field} 不在 DIMENSIONS 中")

        for key, values in self.dimensions.items():
            if not values:
                raise ValueError(f"[{self.name}] 维度 {key} 为空")

        placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", self.prompt_body))
        known = set(self.dimensions) | {"idx", "target_label", "risk_type", "theme",
                                        "ref_samples_block"}
        unknown_ph = placeholders - known
        if unknown_ph:
            raise ValueError(f"[{self.name}] 提示词含未知占位符：{sorted(unknown_ph)}")

        missing_label = [x for x in self.labels if x not in self.risk_type_map]
        if missing_label:
            raise ValueError(f"[{self.name}] RISK_TYPE_MAP 缺少标签：{missing_label}")


# --------------------------------------------------------------------------
# 配置加载
# --------------------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "configs"


def list_configs() -> list[str]:
    """返回 configs/ 下所有可用类别名（按字母序）。"""
    if not CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.py") if not p.stem.startswith("_"))


def load_config(name: str) -> GeneratorConfig:
    """按类别名加载配置对象。

    使用 importlib 而非 exec，是为了让配置文件本身能被 IDE 与静态检查识别。
    """
    if name not in list_configs():
        raise FileNotFoundError(f"未找到配置：configs/{name}.py")
    module = importlib.import_module(f"configs.{name}")
    cfg = getattr(module, "CONFIG", None)
    if not isinstance(cfg, GeneratorConfig):
        raise TypeError(f"configs/{name}.py 未导出合法的 CONFIG 对象")
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------


def setup_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """同时输出到控制台与 logs/<name>.log。

    旧脚本全部用 print，没有时间戳也没有级别，配合 tqdm 会把进度条打乱。
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # 避免重复添加 handler
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    target_dir = log_dir or (PROJECT_ROOT / "logs")
    target_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(target_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# 语义维度取值（修复 1.1）
# --------------------------------------------------------------------------

# 每个维度一个固定偏移，保证不同维度在同一个 idx 上错开，不会整齐对齐。
_DIMENSION_OFFSETS = {
    "scene_category": 0,
    "event_category": 0,
    "role_category": 1,
    "figure_category": 1,
    "text_style": 2,
    "user_intent": 3,
    "language_style": 4,
    "resource_hint": 5,
    "risk_terms": 5,
    "speaker_context": 6,
}

# 跨轮次错位步长：每完整走完一轮 len(pool) 后按不同步长旋转，
# 既保证单轮内 100% 覆盖全部取值，又打破 10/12/15 词表 60 条即锁死循环的问题。
_DIMENSION_ROUND_STRIDES = {
    "scene_category": 0,
    "event_category": 0,
    "role_category": 1,
    "figure_category": 1,
    "text_style": 2,
    "user_intent": 5,
    "language_style": 3,
    "resource_hint": 7,
    "risk_terms": 7,
    "speaker_context": 4,
}


class DimensionPicker:
    """保证每个维度在其周期内跑满全部取值，并在大样本扩容时避免组合锁死。

    旧实现 `lst[(idx*3-1) % len(lst)]` 的问题：
        取值集合 = {(3idx-1) mod n}，当 gcd(3, n)=3 时只能得到 n/3 个值。
        USER_INTENTS 长度 12、步长 3 -> 只剩 4 个取值；
        LANGUAGE_STYLES 长度 10、步长 5 -> 只剩 2 个取值（严格交替）。

    扩容增强实现：
        对每个维度按固定种子打乱一次，单轮内按 (idx + offset + round_num * stride) % n
        遍历全部取值；不同维度具有不同 stride，跨轮次自动错位重组，
        将 12,000 条规模下的有效提示词组合空间从数十种提升到数万种。
        重试时可对非主分类辅助维度传入 attempt > 0 进行扰动，避免重复生成。
    """

    def __init__(self, dimensions: dict[str, Sequence[str]], seed: int = 20261102):
        self._tables: dict[str, list[str]] = {}
        for key, values in dimensions.items():
            pool = list(dict.fromkeys(str(v) for v in values))  # 去重且保持稳定
            rng = random.Random(f"{seed}:{key}")
            rng.shuffle(pool)
            self._tables[key] = pool

    def names(self) -> list[str]:
        return list(self._tables)

    def length(self, key: str) -> int:
        return len(self._tables[key])

    def pick(self, key: str, idx: int, attempt: int = 0) -> str:
        pool = self._tables[key]
        if not pool:
            return ""
        n = len(pool)
        offset = _DIMENSION_OFFSETS.get(key, 0)
        stride = _DIMENSION_ROUND_STRIDES.get(key, 1)
        round_num = max(0, idx - 1) // n
        retry_shift = attempt * (stride or 1)
        return pool[(idx + offset + round_num * stride + retry_shift) % n]

    def coverage(self) -> dict[str, int]:
        """返回每个维度在 DIVERSITY_CYCLE 条内能覆盖到的取值数。"""
        result = {}
        for key, pool in self._tables.items():
            if not pool:
                result[key] = 0
                continue
            seen = {self.pick(key, i) for i in range(1, DIVERSITY_CYCLE + 1)}
            result[key] = len(seen)
        return result


# --------------------------------------------------------------------------
# 输出清洗与 JSON 提取（修复 1.2 / 2.3）
# --------------------------------------------------------------------------

_FENCE_HEAD = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_TAIL = re.compile(r"\s*```$")
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)


class TruncatedOutput(ValueError):
    """模型输出被截断（<think> 未闭合），需要以更大预算重试。"""


def clean_model_text(text: str) -> str:
    """去掉思维链与代码块围栏。

    Qwen3 会先输出 `<think>...</think>` 再给答案。旧版只处理 ``` 围栏，
    于是 <think> 整段被当作正文送进 JSON 解析，必然失败。
    """
    text = (text or "").replace("\u200b", "").strip()

    # 已正常闭合的 think 块直接删除
    text = _THINK_BLOCK.sub("", text).strip()

    # 仍未闭合 -> 输出在推理中途被截断，JSON 根本不存在
    if "<think" in text.lower():
        raise TruncatedOutput(
            "输出在 <think> 推理中途被截断，未产生 JSON；"
            f"原始尾部：{text[-200:]!r}"
        )

    text = _FENCE_HEAD.sub("", text)
    text = _FENCE_TAIL.sub("", text)
    return text.strip()


def _iter_balanced_objects(text: str) -> Iterable[str]:
    """按花括号配对扫描，产出所有顶层 {...} 片段（正确跳过字符串内的括号）。

    旧实现 `re.search(r"\\{.*\\}", text, re.S)` 是贪婪匹配，会从第一个 `{`
    一直吞到最后一个 `}`。当模型在 JSON 前后各写一段带花括号的说明时，
    整体就会变成非法 JSON。
    """
    depth = 0
    start = -1
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取第一个可解析的 JSON 对象。"""
    text = clean_model_text(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    candidates = list(_iter_balanced_objects(text))
    for chunk in candidates:
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"未找到可解析的 JSON 对象，原始输出：{text[:500]}")


# --------------------------------------------------------------------------
# 提示词组装
# --------------------------------------------------------------------------


_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def build_prompt(cfg: GeneratorConfig, picker: DimensionPicker, idx: int,
                 ref_samples: Sequence[str] | None = None,
                 attempt: int = 0) -> str:
    """把配置里的正文与当前 idx 的维度取值拼成完整提示词。

    注意不能直接用 str.format：提示词末尾的 JSON 骨架本身就是花括号，
    { 会被当成占位符起始，导致 KeyError / 花括号被吞。这里只替换
    白名单内的占位符，其余花括号原样保留。
    重试时（attempt > 0）仅对非主分类辅助维度施加扰动，主分类与标签严格绑定 idx。
    """
    target_label = cfg.label_at(idx)

    slots: dict[str, str] = {
        "idx": str(idx),
        "target_label": target_label,
        "risk_type": cfg.risk_type_map.get(target_label, "未分类风险"),
        "theme": cfg.theme,
    }
    for key in picker.names():
        key_attempt = 0 if key == cfg.category_field else attempt
        slots[key] = picker.pick(key, idx, attempt=key_attempt)

    # 主分类字段在提示词与落库中使用同一个名字
    slots.setdefault(cfg.category_field, picker.pick(cfg.category_field, idx, attempt=0))

    if "ref_samples_block" in cfg.prompt_body:
        chosen = ref_samples or []
        if chosen:
            slots["ref_samples_block"] = "\n".join(
                f'- 历史样本 {i + 1}："原文：{t}"' for i, t in enumerate(chosen)
            )
        else:
            slots["ref_samples_block"] = "暂无历史真实参考样本。"

    def repl(m: re.Match) -> str:
        key = m.group(1)
        return slots.get(key, m.group(0))

    return _PLACEHOLDER.sub(repl, cfg.prompt_body).strip()


# --------------------------------------------------------------------------
# 校验（补长度闸门与统一清洗，修复 4.4）
# --------------------------------------------------------------------------


def validate_item(cfg: GeneratorConfig, picker: DimensionPicker,
                  data: dict[str, Any], idx: int) -> dict[str, Any]:
    """校验模型返回的字段，返回可直接落库的干净记录。"""
    expected_label = cfg.label_at(idx)
    expected_category = picker.pick(cfg.category_field, idx, attempt=0)

    text = clean_storage_text(data.get("text"))
    label = str(data.get("label", "") or "").strip()
    severity = str(data.get("severity", "") or "").strip()
    reason = clean_storage_text(data.get("reason"))
    category = str(data.get(cfg.category_field, "") or "").strip()
    risk_type = str(data.get("risk_type", "") or "").strip()

    if label not in cfg.labels:
        raise ValueError(f"label 不合法：{label}")

    # 先判语义冲突，再判与目标是否一致——否则模型输出错误标签时，
    # 报出来的永远是「与目标不一致」，看不到真正的矛盾点。
    if label in ("safe", "borderline") and severity == "high":
        raise ValueError(f"语义冲突：label={label} 不应为 high")

    if label != expected_label:
        raise ValueError(f"label 与目标不一致：期望 {expected_label}，实际 {label}")

    if category and category != expected_category:
        raise ValueError(
            f"{cfg.category_field} 与目标不一致：期望 {expected_category}，实际 {category}"
        )

    if severity not in cfg.severities:
        raise ValueError(f"severity 不合法：{severity}")

    if not text:
        raise ValueError("text 为空")
    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        raise ValueError(f"text 长度越界：{len(text)}（允许 {MIN_TEXT_LEN}~{MAX_TEXT_LEN}）")
    if not canonical_text(text):
        raise ValueError("text 仅含空白或符号，无有效内容")

    if not reason:
        raise ValueError("reason 为空")
    if len(reason) > MAX_REASON_LEN:
        raise ValueError(f"reason 过长：{len(reason)}（上限 {MAX_REASON_LEN}）")

    if not risk_type:
        risk_type = cfg.risk_type_map.get(label, "未分类风险")

    record = {
        "id": idx,
        "text": text,
        "label": label,
        cfg.category_field: category or expected_category,
        "risk_type": risk_type,
        "severity": severity,
        "reason": reason,
    }

    for fn in cfg.extra_validators:
        fn(record)

    return record


# --------------------------------------------------------------------------
# 读写工具
# --------------------------------------------------------------------------


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_existing_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    for item in iter_jsonl(path):
        try:
            ids.add(int(item["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def load_failed_ids(path: Path) -> set[int]:
    return load_existing_ids(path)


def rewrite_jsonl(path: Path, items: Sequence[dict[str, Any]]) -> None:
    """原子写回（先写临时文件再替换），避免中途崩溃损坏数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# 参考样本池（few-shot）
# --------------------------------------------------------------------------


def load_reference_pool(path: Path, fallback: Sequence[str] | None = None) -> list[str]:
    """按行读取参考样本。文件不存在时用 fallback 生成，保证可复现。"""
    if not path.exists():
        samples = list(fallback or [])
        if samples:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(s + "\n")
        return samples

    with open(path, "r", encoding="utf-8-sig") as f:
        return [ln.strip() for ln in f if ln.strip()]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def _init_llm(model_path: str, max_model_len: int, gpu_util: float):
    """延迟导入 vLLM，便于在无 GPU 环境下做 dry-run 与单测。"""
    from vllm import LLM

    return LLM(
        model=model_path,
        trust_remote_code=True,
        quantization="awq",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        dtype="float16",
    )


def _build_sampling_params(
    *,
    attempt: int = 0,
    truncated_retry: bool = False,
    max_model_len: int = 3072,
    seed: int | None = None,
):
    from vllm import SamplingParams

    temps = (0.85, 0.92, 0.98, 1.02, 1.05)
    rep_penalties = (1.0, 1.03, 1.06, 1.08, 1.10)
    idx = min(max(0, attempt), len(temps) - 1)

    max_tokens = MAX_NEW_TOKENS
    if truncated_retry:
        # 截断样本重试时适当放宽输出预算，同时给 prompt 留出安全余量
        upper = max(1024, max_model_len - 512)
        max_tokens = min(max(MAX_NEW_TOKENS + 512, 2560), upper)

    kwargs: dict[str, Any] = {
        "temperature": temps[idx],
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>", "<|im_start|>"],
        "skip_special_tokens": True,
    }

    try:
        sig = inspect.signature(SamplingParams.__init__)
        params = sig.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        params = {}
        accepts_kwargs = True

    if accepts_kwargs or "repetition_penalty" in params:
        kwargs["repetition_penalty"] = rep_penalties[idx]
    if seed is not None and (accepts_kwargs or "seed" in params):
        kwargs["seed"] = seed

    return SamplingParams(**kwargs)


def _infer_category_from_path(path: Path, fallback: str) -> str:
    text_root = PROJECT_ROOT / "text"
    try:
        rel = path.resolve().relative_to(text_root.resolve())
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass
    return path.parent.name or fallback


def _collect_dedup_sources(
    cfg: GeneratorConfig,
    output_file: Path,
    dedup_scope: str,
    extra_existing: Sequence[str | Path] | None = None,
) -> list[tuple[Path, str, bool]]:
    """收集去重基准 JSONL 文件列表，返回 [(path, category, is_active_category), ...]。"""
    ordered: dict[Path, tuple[Path, str, bool]] = {}

    def _add(p: Path, cat: str, is_active: bool) -> None:
        if not p.exists():
            return
        resolved = p.resolve()
        prev = ordered.get(resolved)
        if prev is None:
            ordered[resolved] = (p, cat, is_active)
        elif is_active and not prev[2]:
            ordered[resolved] = (p, cat, True)

    _add(output_file, cfg.name, True)

    if dedup_scope == "global":
        text_root = PROJECT_ROOT / "text"
        if text_root.exists():
            for p in sorted(text_root.rglob("*.jsonl")):
                if is_ignored_jsonl(p):
                    continue
                cat = _infer_category_from_path(p, cfg.name)
                _add(p, cat, cat == cfg.name)

    for raw_extra in extra_existing or ():
        ep = Path(raw_extra)
        if not ep.is_absolute():
            ep = PROJECT_ROOT / ep
        if not ep.exists():
            continue
        if ep.is_dir():
            for p in sorted(ep.rglob("*.jsonl")):
                if is_ignored_jsonl(p):
                    continue
                cat = _infer_category_from_path(p, cfg.name)
                _add(p, cat, cat == cfg.name)
        else:
            cat = _infer_category_from_path(ep, cfg.name)
            _add(ep, cat, cat == cfg.name)

    return list(ordered.values())


def _count_unique_canonical(path: Path) -> tuple[int, int]:
    """统计单个 JSONL 文件的有效行数与唯一强规范化文本数（只读）。"""
    rows = 0
    seen: set[str] = set()
    for item in iter_jsonl(path):
        rows += 1
        canon = canonical_text(item.get("text"))
        if canon:
            seen.add(canon)
    return rows, len(seen)


def _compact_failed_file(failed_file: Path, output_file: Path) -> int:
    """清理失败清单：移除已成功 id，并对同一 id 仅保留最新失败记录。"""
    if not failed_file.exists():
        return 0
    done_ids = load_existing_ids(output_file)
    latest_by_id: dict[int, dict[str, Any]] = {}
    for item in iter_jsonl(failed_file):
        rec_id = _safe_int(item.get("id"))
        if rec_id is None or rec_id in done_ids:
            continue
        latest_by_id[rec_id] = item
    kept = [latest_by_id[k] for k in sorted(latest_by_id)]
    rewrite_jsonl(failed_file, kept)
    return len(kept)


def run_generation(
    cfg: GeneratorConfig,
    *,
    retry_failed: bool = False,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_path: str | None = None,
    max_model_len: int = 3072,
    gpu_util: float = 0.85,
    seed: int = 20261102,
    dry_run: bool = False,
    target_total: int | None = None,
    add_count: int | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    dedup_scope: str = DEFAULT_DEDUP_SCOPE,
    dedup_db: str | Path | None = None,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    extra_existing: Sequence[str | Path] | None = None,
    ensure_unique_total: bool = False,
    llm: Any = None,
) -> dict[str, int]:
    """按配置生成样本（内置持久化去重与多轮自动重试收敛）。

    参数
    ----
    retry_failed        : 仅重试 FAILED_FILE 中未成功的 id，成功后从失败清单移除。
    limit               : 仅处理前 N 条（用于小样本试跑）。
    dry_run             : 只组装提示词并打印统计，不加载模型、不写任何文件/数据库。
    target_total        : 覆盖本次目标总条数（默认使用 cfg.total，即 12000）。
    add_count           : 显式增量模式，从 max(existing_ids)+1 起新增指定条数。
    max_rounds          : 含主轮次在内的最大运行轮数（失败或重复样本自动重试）。
    dedup_scope         : 'global'（跨全类别精确去重+类内近似去重）或 'category'。
    dedup_db            : SQLite 持久化去重库路径（默认 data/dedup_index.sqlite）。
    near_dup_threshold  : 类内 3-gram Jaccard 近似重复阈值（<=0 表示关闭）。
    extra_existing      : 额外纳入去重基准的已有 JSONL 文件或目录。
    ensure_unique_total : 若存量存在历史重复，额外追加新 id 使唯一文本总数 >= target_total。
    llm                 : 可选的外部共享 vLLM 实例，避免 --all 模式下重复加载模型。
    """
    logger = setup_logger(cfg.name)
    random.seed(seed)

    output_file = cfg.default_output()
    failed_file = cfg.default_failed()

    picker = DimensionPicker(cfg.dimensions, seed=seed)
    _assert_dimension_coverage(cfg, picker, logger)

    existing_ids = load_existing_ids(output_file)
    effective_total = target_total if target_total is not None else cfg.total

    if retry_failed:
        raw_failed_ids = load_failed_ids(failed_file)
        failed_ids = raw_failed_ids - existing_ids
        todo = sorted(failed_ids)
        logger.info(
            "重试模式：失败清单 %d 条（其中已成功 %d 条已排除），待处理 %d 条。",
            len(raw_failed_ids),
            len(raw_failed_ids & existing_ids),
            len(todo),
        )
    elif add_count is not None:
        if add_count < 0:
            raise ValueError(f"add_count 不能为负数：{add_count}")
        start_id = (max(existing_ids) + 1) if existing_ids else 1
        todo = list(range(start_id, start_id + add_count))
        logger.info(
            "增量扩容模式：已有 %d 条，计划新增 %d 条（id=%d..%d）。",
            len(existing_ids),
            len(todo),
            start_id,
            start_id + add_count - 1 if add_count else start_id,
        )
    else:
        todo = [i for i in range(1, effective_total + 1) if i not in existing_ids]
        extra_unique_needed = 0
        if ensure_unique_total and output_file.exists():
            _, existing_unique = _count_unique_canonical(output_file)
            projected_unique = existing_unique + len(todo)
            if projected_unique < effective_total:
                extra_unique_needed = effective_total - projected_unique
                next_id = max(effective_total, max(existing_ids, default=0)) + 1
                todo.extend(range(next_id, next_id + extra_unique_needed))
        logger.info(
            "扩容续跑模式：已有 %d 条，目标 %d 条，待生成 %d 条%s。",
            len(existing_ids),
            effective_total,
            len(todo),
            f"（含补齐历史重复 {extra_unique_needed} 条）" if extra_unique_needed else "",
        )

    if limit is not None:
        todo = todo[:limit]
    if not todo:
        logger.info("没有待生成的样本，退出。")
        return {
            "success": 0,
            "failed": 0,
            "skipped": len(existing_ids),
            "total_after": len(existing_ids),
            "dup_blocked": 0,
            "rounds_used": 0,
        }

    if dry_run:
        _dry_run_report(cfg, picker, todo, logger)
        return {
            "success": 0,
            "failed": 0,
            "skipped": len(existing_ids),
            "total_after": len(existing_ids),
            "dup_blocked": 0,
            "rounds_used": 0,
        }

    reference_pool: list[str] = []
    if cfg.reference_file:
        ref_path = Path(cfg.reference_file)
        if not ref_path.is_absolute():
            ref_path = PROJECT_ROOT / ref_path
        reference_pool = load_reference_pool(ref_path, getattr(cfg, "reference_fallback", None))
        logger.info("载入参考样本 %d 条。", len(reference_pool))

    # 初始化持久化去重器并从已有数据冷启动同步
    db_path_raw = dedup_db or os.environ.get("GEN_DEDUP_DB") or DEFAULT_DEDUP_DB
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    dedup = PersistentDeduplicator(
        db_path=db_path,
        category=cfg.name,
        scope=dedup_scope,
        near_dup_threshold=near_dup_threshold,
    )

    try:
        sources = _collect_dedup_sources(cfg, output_file, dedup_scope, extra_existing)
        synced_rows = 0
        synced_inserted = 0
        for src_path, src_cat, is_active in sources:
            st = dedup.sync_from_jsonl(
                src_path,
                src_cat,
                origin="existing",
                load_near_index=is_active,
            )
            synced_rows += st["scanned"]
            synced_inserted += st["inserted_db"]
        cached_cnt = dedup.load_memory_cache()
        logger.info(
            "持久化去重已就绪（scope=%s, near_threshold=%.2f, db=%s）："
            "扫描已有文件 %d 个（%d 行，新增入库 %d），内存索引 %d 条，类内近似索引 %d 条。",
            dedup_scope,
            near_dup_threshold,
            db_path,
            len(sources),
            synced_rows,
            synced_inserted,
            cached_cnt,
            len(dedup.near_index),
        )

        if llm is None:
            llm = _init_llm(
                model_path or os.environ.get("GEN_MODEL_PATH", DEFAULT_MODEL_PATH),
                max_model_len,
                gpu_util,
            )

        from tqdm import tqdm

        success = 0
        dup_counts = {"exact": 0, "canonical": 0, "near": 0}
        pending = list(todo)
        last_error_type: dict[int, str] = {}
        total_rounds = max(1, int(max_rounds))
        rounds_used = 0

        for round_idx in range(total_rounds):
            if not pending:
                break
            rounds_used = round_idx + 1
            if round_idx > 0:
                logger.info(
                    "第 %d/%d 轮自动重试：待补齐 %d 条。",
                    round_idx + 1,
                    total_rounds,
                    len(pending),
                )

            # 将上次因截断失败的样本单独分组，给予更高 max_tokens 预算
            if round_idx == 0:
                groups = [(False, pending)]
            else:
                trunc_ids = [i for i in pending if last_error_type.get(i) == "truncated"]
                other_ids = [i for i in pending if last_error_type.get(i) != "truncated"]
                groups = []
                if other_ids:
                    groups.append((False, other_ids))
                if trunc_ids:
                    groups.append((True, trunc_ids))

            next_pending: list[int] = []

            for is_trunc_group, group_ids in groups:
                round_seed = seed + round_idx * 100003 + (50021 if is_trunc_group else 0)
                sampling_params = _build_sampling_params(
                    attempt=round_idx,
                    truncated_retry=is_trunc_group,
                    max_model_len=max_model_len,
                    seed=round_seed,
                )

                desc = (
                    f"{cfg.name}"
                    if round_idx == 0
                    else f"{cfg.name} [retry {round_idx}/{total_rounds - 1}]"
                )

                for start in tqdm(range(0, len(group_ids), batch_size), desc=desc):
                    batch_ids = group_ids[start : start + batch_size]
                    prompts = [
                        build_prompt(
                            cfg,
                            picker,
                            idx,
                            _sample_reference(
                                reference_pool,
                                cfg.reference_count,
                                rng=random.Random(round_seed + idx),
                            ),
                            attempt=round_idx,
                        )
                        for idx in batch_ids
                    ]

                    batch_pairs: list[tuple[int, str | None, str | None]] = []
                    try:
                        outputs = llm.generate(prompts, sampling_params)
                        for idx, output in zip(batch_ids, outputs):
                            raw = (output.outputs[0].text if output.outputs else "").strip()
                            batch_pairs.append((idx, raw, None))
                    except Exception as exc:
                        logger.warning(
                            "批次生成异常（ids=%s），退化为逐条重试：%s",
                            batch_ids[:1] + batch_ids[-1:],
                            exc,
                        )
                        for idx, prompt in zip(batch_ids, prompts):
                            try:
                                single_out = llm.generate([prompt], sampling_params)
                                raw = (
                                    single_out[0].outputs[0].text
                                    if single_out and single_out[0].outputs
                                    else ""
                                ).strip()
                                batch_pairs.append((idx, raw, None))
                            except Exception as single_exc:
                                batch_pairs.append((idx, None, f"单条推理失败：{single_exc}"))

                    for idx, raw, infer_err in batch_pairs:
                        if infer_err is not None or raw is None:
                            logger.warning("id=%d %s", idx, infer_err)
                            _record_failure(
                                cfg,
                                picker,
                                failed_file,
                                idx,
                                infer_err or "推理返回为空",
                                error_type="inference",
                                attempt=round_idx + 1,
                            )
                            last_error_type[idx] = "inference"
                            next_pending.append(idx)
                            continue

                        try:
                            data = extract_json_object(raw)
                            record = validate_item(cfg, picker, data, idx)
                            dedup.check_and_persist(record, output_file, append_jsonl)
                            success += 1
                            last_error_type.pop(idx, None)
                        except DuplicateTextError as exc:
                            dup_counts[exc.dup_type] = dup_counts.get(exc.dup_type, 0) + 1
                            logger.warning("id=%d %s", idx, exc)
                            _record_failure(
                                cfg,
                                picker,
                                failed_file,
                                idx,
                                str(exc),
                                error_type=f"duplicate_{exc.dup_type}",
                                attempt=round_idx + 1,
                            )
                            last_error_type[idx] = "duplicate"
                            next_pending.append(idx)
                        except TruncatedOutput as exc:
                            logger.warning("id=%d 输出被截断：%s", idx, exc)
                            _record_failure(
                                cfg,
                                picker,
                                failed_file,
                                idx,
                                str(exc),
                                raw,
                                error_type="truncated",
                                attempt=round_idx + 1,
                            )
                            last_error_type[idx] = "truncated"
                            next_pending.append(idx)
                        except Exception as exc:
                            logger.warning("id=%d 校验失败：%s", idx, exc)
                            _record_failure(
                                cfg,
                                picker,
                                failed_file,
                                idx,
                                str(exc),
                                raw,
                                error_type="validation",
                                attempt=round_idx + 1,
                            )
                            last_error_type[idx] = "validation"
                            next_pending.append(idx)

            # 每轮结束后整理失败清单，剔除已成功 id 并按 id 去重
            remaining_failed = _compact_failed_file(failed_file, output_file)
            pending = next_pending
            logger.info(
                "第 %d 轮结束：累计新增成功 %d 条，本轮待重试 %d 条，失败清单剩余 %d 条。",
                round_idx + 1,
                success,
                len(pending),
                remaining_failed,
            )

        total_after = len(load_existing_ids(output_file))
        dup_total = sum(dup_counts.values())
        logger.info(
            "运行结束：新增成功 %d 条，最终未收敛 %d 条，当前文件总计 %d 条，"
            "拦截重复 %d 次（exact=%d, canonical=%d, near=%d），执行轮次 %d。",
            success,
            len(pending),
            total_after,
            dup_total,
            dup_counts["exact"],
            dup_counts["canonical"],
            dup_counts["near"],
            rounds_used,
        )
        logger.info("数据文件：%s", output_file)
        logger.info("失败清单：%s", failed_file)
        logger.info("去重索引：%s", db_path)
        return {
            "success": success,
            "failed": len(pending),
            "skipped": len(existing_ids),
            "total_after": total_after,
            "dup_blocked": dup_total,
            "rounds_used": rounds_used,
        }
    finally:
        dedup.close()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sample_reference(
    pool: Sequence[str],
    count: int,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """从参考池中抽样，每条 prompt 与重试轮次可复现且互不相同。"""
    if not pool or count <= 0:
        return []
    chooser = rng if rng is not None else random
    return chooser.sample(list(pool), min(len(pool), count))


def _record_failure(
    cfg: GeneratorConfig,
    picker: DimensionPicker,
    path: Path,
    idx: int,
    error: str,
    raw: str = "",
    *,
    error_type: str = "error",
    attempt: int = 1,
) -> None:
    item: dict[str, Any] = {
        "id": idx,
        "label": cfg.label_at(idx),
        cfg.category_field: picker.pick(cfg.category_field, idx, attempt=0),
        "error_type": error_type,
        "attempt": attempt,
        "error": error,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if raw:
        item["raw"] = raw[:1000]
    append_jsonl(path, item)


def _assert_dimension_coverage(cfg: GeneratorConfig, picker: DimensionPicker,
                               logger: logging.Logger) -> None:
    """自检：每个维度都必须能覆盖到全部取值。不满足直接报错，避免静默退化。"""
    cov = picker.coverage()
    bad = []
    for key, n_seen in cov.items():
        total = picker.length(key)
        if total and n_seen < total:
            bad.append(f"{key} 仅覆盖 {n_seen}/{total}")
    if bad:
        raise RuntimeError("维度取值退化：" + "；".join(bad))
    logger.info("维度覆盖自检通过：%s",
                {k: f"{v}/{picker.length(k)}" for k, v in cov.items()})


def _dry_run_report(cfg: GeneratorConfig, picker: DimensionPicker,
                    todo: Sequence[int], logger: logging.Logger) -> None:
    """不调用模型，打印标签与维度分布，并展示首条提示词。"""
    from collections import Counter

    dist = Counter(cfg.label_at(i) for i in todo)
    logger.info("dry-run：待生成 %d 条", len(todo))
    for label, n in dist.most_common():
        logger.info("    %5d  %5.1f%%  %s", n, n / len(todo) * 100, label)

    cov = picker.coverage()
    logger.info("维度覆盖：%s", {k: f"{v}/{picker.length(k)}" for k, v in cov.items()})

    if todo:
        sample = build_prompt(cfg, picker, todo[0])
        print("\n----- 首条提示词（id=%d）-----" % todo[0])
        print(sample)
        print("----- 提示词结束 -----\n")
