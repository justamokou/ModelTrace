from __future__ import annotations

import json
import math
import random
import re
import secrets
from pathlib import Path
from typing import Iterable

import numpy as np


VALUE_MIN = 1
VALUE_MAX = 355
DIMENSION = VALUE_MAX - VALUE_MIN + 1
ALPHA = 0.5
ORDERED_BLOCK_WEIGHT = 0.25
DEFAULT_BANK = Path(__file__).with_name("data") / "gpt_bank.json"
FAMILY_DISPLAY_NAMES = {"gpt": "GPT", "claude": "Claude"}
FAMILY_GATE_CALIBRATION = {
    "1": {"beta": 3.448920538621421, "cv_accuracy": 0.9629629629629629},
    "2": {"beta": 6.216642555777372, "cv_accuracy": 0.9861111111111112},
    "3": {"beta": 12.0, "cv_accuracy": 1.0},
}


def parse_numbers(text: str) -> list[int]:
    runs: list[list[int]] = []
    current: list[int] = []
    previous_end = 0
    for match in re.finditer(r"\d+", text):
        separator = text[previous_end : match.start()]
        value = int(match.group())
        if current and any(character.isalpha() for character in separator):
            runs.append(current)
            current = []
        if VALUE_MIN <= value <= VALUE_MAX:
            current.append(value)
        previous_end = match.end()
    if current:
        runs.append(current)
    return max(runs, key=len) if runs else []


def count_numbers(numbers: Iterable[int]) -> list[int]:
    counts = [0] * DIMENSION
    for number in numbers:
        counts[number - VALUE_MIN] += 1
    return counts


def standardize(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = max(math.sqrt(variance), 1e-12)
    return [(value - mean) / scale for value in values]


def hellinger_feature(counts: list[int]) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64) + ALPHA
    return np.sqrt(values / values.sum())


def ordered_block_feature(numbers: list[int]) -> np.ndarray:
    values = np.asarray(numbers, dtype=np.float64)
    pieces = []
    for chunk in np.array_split(values, 4):
        counts, _ = np.histogram(chunk, bins=16, range=(1.0, 356.0))
        smoothed = counts.astype(np.float64) + 0.5
        pieces.append(np.sqrt(smoothed / smoothed.sum()))
    last_digits = np.bincount(values.astype(int) % 10, minlength=10).astype(np.float64) + 0.5
    pieces.append(np.sqrt(last_digits / last_digits.sum()))
    return np.concatenate(pieces)


def robust_score_counts(counts: list[int], bank: dict) -> dict[str, list[float]]:
    robust = bank["robust"]
    model_count = len(robust["model_order"])

    hellinger = robust["hellinger"]
    feature = hellinger_feature(counts)
    mean = np.asarray(hellinger["feature_mean"], dtype=np.float64)
    scale = np.asarray(hellinger["feature_scale"], dtype=np.float64)
    projected = (feature - mean) / scale
    basis = np.asarray(hellinger["nuisance_basis"], dtype=np.float64)
    if basis.size:
        projected -= (projected @ basis.T) @ basis
    norm = max(float(np.linalg.norm(projected)), 1e-12)
    centroids = np.asarray(hellinger["centroids"], dtype=np.float64)
    nuisance = (projected / norm) @ centroids.T
    nuisance = np.asarray(standardize(nuisance.tolist()))

    fused = standardize(nuisance.tolist())
    assert len(fused) == model_count
    return {
        "fused": fused,
        "nuisance": nuisance.tolist(),
    }


def ordered_block_scores(numbers: list[int], bank: dict) -> list[float]:
    artifact = bank["robust"]["ordered_blocks"]
    feature = ordered_block_feature(numbers)
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    scale = np.asarray(artifact["feature_scale"], dtype=np.float64)
    standardized_feature = (feature - mean) / scale

    normalized = standardized_feature / max(float(np.linalg.norm(standardized_feature)), 1e-12)
    templates = np.asarray(artifact["environment_centroids"], dtype=np.float64)
    environment_scores = np.stack([normalized @ centroids.T for centroids in templates])
    template = np.max(environment_scores, axis=0)
    template = np.asarray(standardize(template.tolist()))

    basis = np.asarray(artifact["nuisance_basis"], dtype=np.float64)
    projected = standardized_feature.copy()
    if basis.size:
        projected -= (projected @ basis.T) @ basis
    projected /= max(float(np.linalg.norm(projected)), 1e-12)
    centroids = np.asarray(artifact["centroids"], dtype=np.float64)
    nuisance = projected @ centroids.T
    nuisance = np.asarray(standardize(nuisance.tolist()))
    return standardize((0.5 * template + 0.5 * nuisance).tolist())


def robust_score_numbers(numbers: list[int], bank: dict) -> dict[str, list[float]]:
    marginal = robust_score_counts(count_numbers(numbers), bank)
    artifact = bank["robust"].get("ordered_blocks")
    ordered_weight = float(artifact.get("weight", 0.0)) if artifact else 0.0
    if not artifact or ordered_weight == 0.0:
        return {**marginal, "marginal_fused": marginal["fused"], "ordered": marginal["fused"]}
    ordered = ordered_block_scores(numbers, bank)
    fused = [
        (1.0 - ordered_weight) * marginal_score + ordered_weight * ordered_score
        for marginal_score, ordered_score in zip(marginal["fused"], ordered)
    ]
    return {**marginal, "fused": fused, "marginal_fused": marginal["fused"], "ordered": ordered}


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def js_similarity(left: list[int], right: list[int]) -> float:
    left_total = sum(left)
    right_total = sum(right) + ALPHA * DIMENSION
    p = [value / left_total for value in left]
    q = [(value + ALPHA) / right_total for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]

    def divergence(values: list[float], middle: list[float]) -> float:
        return sum(value * math.log(value / target) for value, target in zip(values, middle) if value)

    js = (divergence(p, midpoint) + divergence(q, midpoint)) / 2.0
    return 1.0 - math.sqrt(js / math.log(2.0))


def load_bank(path: Path = DEFAULT_BANK) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_family_gate(rows_by_family: dict[str, list[dict]]) -> dict:
    family_order = list(rows_by_family)
    rows = [row for family_id in family_order for row in rows_by_family[family_id]]
    model_order = list(dict.fromkeys(row["source"] for row in rows))
    labels = np.asarray([row["source"] for row in rows])
    model_families = [
        family_id
        for model_id in model_order
        for family_id in family_order
        if any(row["source"] == model_id for row in rows_by_family[family_id])
    ]
    features = np.stack([hellinger_feature(row["counts"]) for row in rows])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    centroids = np.stack(
        [standardized[labels == model_id].mean(axis=0) for model_id in model_order]
    )
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    normalized = standardized / np.maximum(
        np.linalg.norm(standardized, axis=1, keepdims=True), 1e-12
    )
    model_scores = normalized @ centroids.T
    family_scores = np.stack(
        [
            model_scores[:, np.asarray([item == family_id for item in model_families])].max(axis=1)
            for family_id in family_order
        ],
        axis=1,
    )
    if len(family_order) == 2:
        family_score_scale = max(
            float(np.std(family_scores[:, 0] - family_scores[:, 1])), 1e-12
        )
    else:
        centered = family_scores - family_scores.mean(axis=1, keepdims=True)
        family_score_scale = max(float(np.std(centered)), 1e-12)
    return {
        "family_order": family_order,
        "model_order": model_order,
        "model_families": model_families,
        "aggregation": "maximum model-centroid similarity within each family",
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "centroids": centroids.tolist(),
        "family_score_scale": family_score_scale,
        "calibration": FAMILY_GATE_CALIBRATION,
    }


def family_score_numbers(numbers: list[int], gate: dict) -> list[float]:
    feature = hellinger_feature(count_numbers(numbers))
    mean = np.asarray(gate["feature_mean"], dtype=np.float64)
    scale = np.asarray(gate["feature_scale"], dtype=np.float64)
    standardized = (feature - mean) / scale
    normalized = standardized / max(float(np.linalg.norm(standardized)), 1e-12)
    model_scores = normalized @ np.asarray(gate["centroids"], dtype=np.float64).T
    family_scores = [
        max(
            model_scores[index]
            for index, family_id in enumerate(gate["model_families"])
            if family_id == target_family
        )
        for target_family in gate["family_order"]
    ]
    scale = gate["family_score_scale"]
    if len(family_scores) == 2:
        margin = float((family_scores[0] - family_scores[1]) / scale)
        return [margin, -margin]
    center = sum(family_scores) / len(family_scores)
    return [float((score - center) / scale) for score in family_scores]


def analyze_unified_outputs(outputs: list[dict], banks: dict[str, dict], family_gate: dict) -> dict:
    family_order = family_gate["family_order"]
    conditional = {
        family_id: analyze_outputs(outputs, banks[family_id])
        for family_id in family_order
    }
    diagnostics = conditional[family_order[0]]["diagnostics"]
    valid_numbers = [
        parse_numbers(str(item.get("text", "")))
        for item, diagnostic in zip(outputs, diagnostics)
        if diagnostic["accepted"]
    ]
    family_scores = [family_score_numbers(numbers, family_gate) for numbers in valid_numbers]
    combined_family_scores = [
        sum(scores[index] for scores in family_scores) / len(family_scores)
        for index in range(len(family_order))
    ]
    calibration_key = str(min(len(valid_numbers), 3))
    family_calibration = family_gate["calibration"][calibration_key]
    family_probabilities = softmax(
        [family_calibration["beta"] * score for score in combined_family_scores]
    )

    results = []
    for family_index, family_id in enumerate(family_order):
        for item in conditional[family_id]["results"]:
            results.append(
                {
                    **item,
                    "family": family_id,
                    "family_name": banks[family_id].get(
                        "family_name", FAMILY_DISPLAY_NAMES.get(family_id, family_id)
                    ),
                    "conditional_probability": item["probability"],
                    "probability": family_probabilities[family_index] * item["probability"],
                }
            )
    results.sort(key=lambda item: item["probability"], reverse=True)
    winning_family_index = max(
        range(len(family_order)), key=lambda index: family_probabilities[index]
    )
    winning_family = family_order[winning_family_index]
    return {
        "prediction": results[0]["model"],
        "prediction_name": results[0]["display_name"],
        "probability": results[0]["probability"],
        "used_outputs": len(valid_numbers),
        "results": results,
        "diagnostics": diagnostics,
        "family_prediction": winning_family,
        "family_prediction_name": banks[winning_family].get(
            "family_name", FAMILY_DISPLAY_NAMES.get(winning_family, winning_family)
        ),
        "family_probability": family_probabilities[winning_family_index],
        "family_probabilities": [
            {
                "family": family_id,
                "display_name": banks[family_id].get(
                    "family_name", FAMILY_DISPLAY_NAMES.get(family_id, family_id)
                ),
                "probability": family_probabilities[index],
            }
            for index, family_id in enumerate(family_order)
        ],
        "calibration": {
            "queries": calibration_key,
            "family_beta": family_calibration["beta"],
            "family_cv_accuracy": family_calibration["cv_accuracy"],
        },
        "conditional_calibration": {
            family_id: conditional[family_id]["calibration"]
            for family_id in family_order
        },
        "method": "统一模型质心家族门控 + 家族内稳健数字指纹",
    }


def analyze_outputs(outputs: list[dict], bank: dict) -> dict:
    model_ids = [model["id"] for model in bank["models"]]
    valid = []
    diagnostics = []
    for index, item in enumerate(outputs):
        text = str(item.get("text", ""))
        expected = int(item.get("expected_count") or 0)
        numbers = parse_numbers(text)
        minimum = max(80, math.ceil(expected * 0.55)) if expected else 80
        accepted = len(numbers) >= minimum
        diagnostics.append(
            {
                "index": index,
                "parsed_numbers": len(numbers),
                "minimum_numbers": minimum,
                "accepted": accepted,
            }
        )
        if accepted:
            counts = count_numbers(numbers)
            components = robust_score_numbers(numbers, bank)
            valid.append({"counts": counts, "scores": components["fused"], **components})

    if not valid:
        raise ValueError("没有可用回答：请粘贴完整数字序列；拒答或严重截断的回答不会计入。")

    combined_scores = [
        sum(item["scores"][index] for item in valid) / len(valid)
        for index in range(len(model_ids))
    ]
    combined_nuisance = [
        sum(item.get("nuisance", item["scores"])[index] for item in valid) / len(valid)
        for index in range(len(model_ids))
    ]
    calibration_key = str(min(len(valid), 3))
    beta = float(bank["calibration"][calibration_key]["beta"])
    probabilities = softmax([beta * value for value in combined_scores])
    pooled_counts = [sum(item["counts"][index] for item in valid) for index in range(DIMENSION)]
    bank_models = {model["id"]: model for model in bank["models"]}
    results = [
        {
            "model": model_id,
            "display_name": bank_models[model_id]["display_name"],
            "probability": probabilities[index],
            "profile_similarity": js_similarity(pooled_counts, bank_models[model_id]["counts"]),
            "score": combined_scores[index],
            "nuisance_score": combined_nuisance[index],
        }
        for index, model_id in enumerate(model_ids)
    ]
    results.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "prediction": results[0]["model"],
        "prediction_name": results[0]["display_name"],
        "probability": results[0]["probability"],
        "used_outputs": len(valid),
        "results": results,
        "diagnostics": diagnostics,
        "calibration": {
            "queries": calibration_key,
            "beta": beta,
            "cv_accuracy": bank["calibration"][calibration_key]["cv_accuracy"],
        },
        "method": bank.get("method", {}).get("name", "Ordered-block + nuisance-Hellinger"),
    }


def generate_challenges(count: int = 3) -> list[dict]:
    rng = random.SystemRandom()
    lengths = rng.sample(range(292, 333), count)
    openings = [
        "这是一次独立的数值流采样",
        "请完成下面的无语义整数流任务",
        "执行一次直觉随机取值记录",
        "生成一段不承载语义的整数序列",
        "进行一轮快速数字采样",
    ]
    actions = [
        "凭第一直觉依次给出",
        "连续写出",
        "不做计算地生成",
        "按自然生成顺序输出",
        "直接产生",
    ]
    endings = [
        "不要排序；允许重复；保留完整结果。",
        "允许同一数字多次出现，不要去重或重新排列。",
        "按产生顺序保留重复项，不需要解释这些数字。",
        "不要总结、筛选或修正序列，重复值是有效的。",
        "无需赋予数字任何含义，也不要对结果做排序。",
    ]
    separator_hints = [
        "数字之间用逗号或空格分隔均可。",
        "使用一种一致的常见分隔符即可。",
        "可以用逗号、空格或换行分隔。",
        "只要每个整数边界清楚，格式可自行选择。",
    ]
    challenges = []
    for index, length in enumerate(lengths):
        prompt = (
            f"{rng.choice(openings)}。{rng.choice(actions)} {length} 个 1 到 355（含端点）的整数。"
            "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、"
            "计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
            f"{rng.choice(endings)}{rng.choice(separator_hints)}"
            "直接从第一个取值开始输出，不要在序列前重复数量、范围或任务说明。"
        )
        challenge_id = secrets.token_hex(7)
        challenges.append(
            {
                "id": f"probe-{index + 1}-{challenge_id}",
                "expected_count": length,
                "prompt": prompt,
            }
        )
    return challenges
