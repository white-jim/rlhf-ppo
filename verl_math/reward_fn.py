"""
GSM8K 规则型奖励函数 — 提取模型回复中的最终答案，与 ground truth 比较。

verl 通过 `reward.custom_reward_function.path` 和 `name` 配置加载此函数。
data_source 为 "openai/gsm8k" 时走此路由。

函数签名必须为:
    def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict,
        **kwargs
    ) -> float | dict
"""

import re

_SOLUTION_CLIP_CHARS = 300


def _extract_solution(solution_str: str, method: str = "strict") -> str | None:
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall(r"####\s*(\-?[\d\.\,]+)", solution_str)
        if len(solutions) == 0:
            return None
        return solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answers = re.findall(r"(\-?[\d\.\,]+)", solution_str)
        invalid = {"", "."}
        for ans in reversed(answers):
            if ans not in invalid:
                return ans
        return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
) -> float:
    """GSM8K 规则型打分：答案完全匹配得 1.0，格式正确但答案错误得 0.1，否则 0。"""
    answer = _extract_solution(solution_str, method="strict")
    if answer is None:
        # 尝试 flexible 提取
        answer = _extract_solution(solution_str, method="flexible")
        if answer is None:
            return 0.0
        return 1.0 if answer == ground_truth else 0.1
    return 1.0 if answer == ground_truth else 0.1