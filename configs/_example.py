# -*- coding: utf-8 -*-
"""类别配置脱敏示例模板（文件名以 '_' 开头，不会被 run.py --all 自动加载）。

使用方式
--------
1. 复制本文件为 configs/<your_category>.py（去掉开头下划线，例如 configs/my_category.py）。
2. 按业务需要填写 THEME / LABEL_SLOTS / LABELS / DIMENSIONS / RISK_TYPE_MAP / PROMPT_BODY。
3. 运行 `python run.py my_category --dry-run` 验证配置与 20 槽位覆盖是否正确。
"""

from core import GeneratorConfig

THEME = "示例内容安全分类任务主题（请替换为实际业务主题说明）"

# 必须完整铺满 0..19 共 20 个槽位，决定各类标签的精确配比
LABEL_SLOTS = {
    "harmful_type_a": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),   # 50%
    "harmful_type_b": (11, 12, 13, 14, 15, 16),          # 30%
    "borderline": (17, 18, 19),                          # 15%
    "safe": (0,),                                        # 5%
}

LABELS = [
    "harmful_type_a",
    "harmful_type_b",
    "borderline",
    "safe",
]

SEVERITIES = ["low", "medium", "high"]

# 语义维度词表：键名对应 PROMPT_BODY 中的 {占位符}
DIMENSIONS = {
    "scene_category": [
        "场景类别A",
        "场景类别B",
        "场景类别C",
        "场景类别D",
        "场景类别E",
    ],
    "text_style": [
        "短句评论",
        "论坛发帖",
        "站内私信",
        "客服工单",
        "平台审核备注",
        "用户举报描述",
    ],
    "user_intent": [
        "意图维度1",
        "意图维度2",
        "意图维度3",
        "意图维度4",
    ],
    "language_style": [
        "日常口语",
        "书面正式",
        "简短摘要",
    ],
}

RISK_TYPE_MAP = {
    "harmful_type_a": "风险类型A",
    "harmful_type_b": "风险类型B",
    "borderline": "边界讨论/风控分析",
    "safe": "正常合规/正向科普",
}

PROMPT_BODY = """
<|im_start|>system
你是文本分类训练数据生成助手，请严格只输出合法 JSON。<|im_end|>
<|im_start|>user
请生成第 {idx} 条分类训练样本。

主题：{theme}
目标标签：{target_label}
场景类别：{scene_category}
文本形态：{text_style}
用户意图：{user_intent}
语言风格：{language_style}

输出要求：只输出严格 JSON，包含 id, text, label, scene_category, risk_type, severity, reason 七个字段：
{
  "id": {idx},
  "text": "",
  "label": "{target_label}",
  "scene_category": "{scene_category}",
  "risk_type": "",
  "severity": "",
  "reason": ""
}
<|im_end|>
<|im_start|>assistant
"""

CONFIG = GeneratorConfig(
    name="_example",
    theme=THEME,
    labels=LABELS,
    label_slots=LABEL_SLOTS,
    severities=SEVERITIES,
    dimensions=DIMENSIONS,
    risk_type_map=RISK_TYPE_MAP,
    prompt_body=PROMPT_BODY,
    category_field="scene_category",
)
