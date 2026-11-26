# Qwen3 有害文本分类数据集生成流水线

面向内容安全分类模型训练与评测的多类别文本数据生成、持久化去重、断点续跑与自动化验收工具链（基于 Qwen3 + vLLM）。

> **合规与安全声明**
> 本项目仅用于内容安全风控、合规审核分类模型的训练与评测数据工程。仓库中**不包含**任何业务提示词配置（`configs/*.py`，仅保留脱敏模板 `configs/_example.py`）、已生成的样本数据（`text/`、`final/`）、运行日志（`logs/`）及业务历史参考样本（`data/reference/*.txt`），相关路径均已默认加入 `.gitignore`。

---

## 目录结构

```text
.
├── run.py                           # 统一生成命令行入口（支持单类/多类/全量与共享单 vLLM 实例）
├── core.py                          # 公共生成引擎（提示词组装、维度轮转、JSON 提取、校验、自动重试）
├── dedup.py                         # 三层去重引擎（清洗、NFKC 规范化 SHA-256、3-gram Jaccard、SQLite WAL）
├── configs/                         # 业务类别配置目录（业务 configs/*.py 已被 .gitignore 屏蔽）
│   ├── __init__.py
│   └── _example.py                  # 脱敏配置模板（复制重命名后填入实际业务配置）
├── scripts/                         # 数据后处理、导出与只读验收工具
│   ├── __init__.py
│   ├── verify_dataset.py            # 数量、ID、20 槽位配比、7 字段与多维去重只读验收脚本
│   ├── split_dataset.py             # 全局去重并切分导出 final/<类别>.txt 与 final/normal.txt
│   └── export_by_label.py           # 将单个或多个 JSONL 按 safe / borderline / other 导出纯文本
├── tests/                           # 无 GPU 离线单元测试（Mock LLM + 无害占位数据）
│   ├── __init__.py
│   └── test_expansion_pipeline.py
├── .env.example                     # 服务器环境变量配置模板
├── .gitignore                       # Git 忽略规则（屏蔽业务配置、数据、日志、SQLite、缓存等）
├── requirements.txt                 # Python 依赖声明
└── README.md
```

---

## 核心特性

1. **配置与引擎解耦**：18 个业务类别仅在 `configs/<category>.py` 中声明主题、标签、20 槽位配比（`LABEL_SLOTS`）、语义维度词表与提示词模板，公共流水线集中在 `core.py`。
2. **三层持久化去重（`dedup.py`）**：
   - **存储清洗**：去除零宽/BOM 字符，折叠换行/制表符与连续空白，确保生成端与导出端口径一致；
   - **强规范化指纹**：`NFKC` 归一 + 小写 + 过滤空白与标点后计算 `SHA-256`，支持跨 18 类全局去重（`--dedup-scope global`）或类内去重（`category`）；
   - **类内近似去重**：字符 3-gram Jaccard + 4-gram 倒排候选索引（默认阈值 `0.85`，`--near-dup-threshold 0` 可关闭）；
   - **SQLite WAL 持久化**：支持断点续跑、冷启动从既有 JSONL 无损重建索引以及多 GPU 进程并发安全写入。
3. **维度多样性与自动重试闭环**：
   - `DimensionPicker` 采用跨轮次错位轮转，避免大样本扩容（如每类 12,000 条）时提示词组合锁死；
   - 内置 `--max-rounds` 自动重试（默认 `5` 轮）：对重复/校验失败样本逐轮微调采样温度、重复惩罚与辅助维度，对 `<think>` 截断样本自动增大 `max_tokens` 预算，批次异常自动降级逐条重试，每轮自动收敛失败清单。
4. **严格输出与只读验收**：
   - 每条记录输出统一 7 字段 JSONL（`id`, `text`, `label`, `scene_category`/`event_category`, `risk_type`, `severity`, `reason`）；
   - `scripts/verify_dataset.py` 提供只读门禁校验（仅打印聚合统计，不输出正文内容）。

---

## 环境准备

```bash
# 1. 创建环境并安装依赖（服务端 GPU 环境）
pip install -r requirements.txt

# 2. 配置模型路径与运行参数（参考 .env.example）
export GEN_MODEL_PATH=/path/to/Qwen3-4B-AWQ
export GEN_MAX_MODEL_LEN=3072
export GEN_GPU_UTIL=0.55
```

> 若仅在本机执行离线单元测试或数据验收/切分脚本，仅需 Python 3.9+ 标准库即可运行，无需 GPU 或 `vllm`。

---

## 使用指南

### 1. 查看类别与当前进度
```bash
python run.py --list
```

### 2. 预检（不加载模型、不写文件）
```bash
python run.py --all --total 12000 --dry-run
```

### 3. 单类试跑与全量扩容
```bash
# 小样本试跑（20 条）
python run.py gamble --total 12000 --limit 20

# 单卡串行运行全部 18 类（自动共享单个 vLLM 实例）
CUDA_VISIBLE_DEVICES=0 nohup python run.py --all --total 12000 --max-rounds 5 > logs/expand_all.out 2>&1 &

# 多卡分组并行（每卡仅加载一次模型，共享 SQLite WAL 去重索引）
CUDA_VISIBLE_DEVICES=0 python run.py contraband_made contraband_used --total 12000 --max-rounds 5
```

常用参数说明：
- `--total N`：指定目标总条数（默认 `12000`，自动补齐 `1..N` 中缺失的 `id`）；
- `--add N`：显式增量模式，从当前最大 `id + 1` 起新增 `N` 条；
- `--max-rounds K`：含主轮次在内的最大自动重试轮数（默认 `5`）；
- `--dedup-scope {global,category}`：去重作用域（默认 `global`）；
- `--near-dup-threshold T`：类内近似重复 Jaccard 阈值（默认 `0.85`，设为 `0` 关闭）；
- `--extra-existing PATH`：额外纳入去重基准的历史 JSONL 文件或目录（可重复指定）；
- `--ensure-unique-total`：若存量存在历史重复，额外追加新 `id` 使唯一文本总数不低于目标值；
- `--retry-failed`：仅针对失败清单 `*_failed.jsonl` 中尚未成功的 `id` 进行补跑。

### 4. 全量只读验收校验
```bash
python scripts/verify_dataset.py \
  --expected-total 12000 \
  --baseline-max-id 8000 \
  --require-db-sync \
  --json-report logs/verify_12000.json
```
全部 18 类达标时返回退出码 `0`，否则返回 `1` 并列出异常统计项（不打印任何样本正文）。

### 5. 下游训练数据切分与导出
```bash
# 预览全局去重与切分统计（默认自动跳过 *_failed.jsonl 与 *_old.jsonl）
python scripts/split_dataset.py --dry-run

# 正式导出至 final/<类别>.txt 与 final/normal.txt
python scripts/split_dataset.py

# 按 safe / borderline / other 单独导出指定 JSONL
python scripts/export_by_label.py text/gamble/gamble.jsonl
```

---

## 离线单元测试

```bash
python -m unittest discover -s tests -v
```
测试覆盖文本清洗与规范化、近似去重索引、SQLite 事务回滚与删库重建、多线程并发写互斥、全部 18 个配置的维度覆盖自检、故障注入自动重试闭环以及 `scripts/verify_dataset.py` / `scripts/export_by_label.py` 验证。
