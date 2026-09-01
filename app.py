from __future__ import annotations

import json
import math
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from cryptography.fernet import Fernet

from enrollment import bank_summary, enroll_automatic, request_completion, test_automatic
from fingerprint import analyze_global_outputs, generate_challenges, load_bank, parse_numbers
from bank_builder import build_bank, read_rows


app = Flask(__name__)
PROJECT = Path(__file__).resolve().parent
CUSTOM_BANKS_FILE = PROJECT / "data" / "custom_banks.json"
UNIFIED_BANK_FILE = PROJECT / "data" / "unified_bank.json"
PRESETS_FILE = PROJECT / "data" / "presets.json"
HISTORY_FILE = PROJECT / "data" / "detection_history.json"
PRESET_KEY_FILE = Path.home() / ".modelrace_key"
_ENCRYPTED_PREFIX = "enc1:"
DEFAULT_BANK_ID = "claude"
HISTORY_PASS_THRESHOLD = 0.9


def _load_fernet() -> Fernet | None:
    """读取或创建设置加密密钥（位于仓库外的用户主目录）。

    密钥文件与项目仓库分离，因此 commit/push 不会带上它；即使预设文件被误
    提交，其中的 API Key 也只是密文。若 cryptography 缺失或密钥文件不可用，
    返回 None，预设功能退化为不加密明文（并仍受 .gitignore 保护）。
    """
    try:
        if PRESET_KEY_FILE.exists():
            key = PRESET_KEY_FILE.read_bytes().strip()
            if key:
                return Fernet(key)
        key = Fernet.generate_key()
        PRESET_KEY_FILE.write_bytes(key + b"\n")
        return Fernet(key)
    except Exception:
        return None


def _encrypt_value(fernet: Fernet | None, plaintext: str) -> str:
    if not plaintext or fernet is None:
        return plaintext
    return _ENCRYPTED_PREFIX + fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")


def _decrypt_value(fernet: Fernet | None, stored: str) -> str:
    if not stored or not stored.startswith(_ENCRYPTED_PREFIX):
        return stored  # 未加密（旧数据或无法加密环境），原样返回
    try:
        if fernet is None:
            return ""
        token = stored[len(_ENCRYPTED_PREFIX):]
        return fernet.decrypt(token.encode("ascii")).decode("utf-8")
    except Exception:
        return ""  # 密钥不匹配等：不暴露密文，视为空


def load_presets() -> dict[str, dict]:
    """读取本地保存的 URL/Key 预设（读取时解密 API Key）。返回 {name: preset}。"""
    if not PRESETS_FILE.exists():
        return {}
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    presets = data.get("presets") if isinstance(data, dict) else data
    if not isinstance(presets, list):
        return {}
    fernet = _load_fernet()
    return {
        str(item["name"]).strip(): {
            "name": str(item["name"]).strip(),
            "base_url": str(item.get("base_url", "")),
            "api_key": _decrypt_value(fernet, str(item.get("api_key", ""))),
            "model": str(item.get("model", "")),
            "temperature": str(item.get("temperature", "")),
        }
        for item in presets
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    }


def save_presets(presets: dict[str, dict]) -> None:
    """把 {name: preset} 写入本地文件（API Key 加密后落盘）。"""
    PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fernet = _load_fernet()
    data = {
        "presets": [
            {
                **presets[name],
                "api_key": _encrypt_value(fernet, presets[name].get("api_key", "")),
            }
            for name in preserved_order(presets)
        ]
    }
    PRESETS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preserved_order(presets: dict[str, dict]) -> list[str]:
    return list(presets.keys())


def _history_text(value: object, maximum: int = 180) -> str:
    """规范化会展示或落盘的历史文本，避免保存原始回答等大块内容。"""
    return " ".join(str(value or "").split())[:maximum]


def _history_probability(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"历史记录中的 {field} 无效") from error
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"历史记录中的 {field} 应介于 0 和 1 之间")
    return number


def _history_output_count(value: object) -> int:
    try:
        return max(0, min(3, int(value)))
    except (TypeError, ValueError):
        return 0


def load_history_records() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        payload = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    records = payload.get("records") if isinstance(payload, dict) else payload
    return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []


def save_history_records(records: list[dict]) -> None:
    """原子写入本地历史，避免程序中断留下半份 JSON 文件。"""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = HISTORY_FILE.with_suffix(".json.tmp")
    data = {"version": 1, "records": records}
    temporary_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_file.replace(HISTORY_FILE)


def _history_model_key(value: object) -> str:
    """比较模型名时忽略大小写、分隔符和常见供应商前缀。"""
    text = _history_text(value, 180).casefold()
    text = text.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "", text)


def history_status(record: dict) -> dict:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    try:
        probability = float(result.get("probability", 0))
    except (TypeError, ValueError):
        probability = 0
    probability = probability if math.isfinite(probability) else 0
    prediction_name = _history_text(result.get("prediction_name"), 180)
    expected_model = _history_text(record.get("api_model"), 180)
    probability_passed = probability >= HISTORY_PASS_THRESHOLD
    prediction_key = _history_model_key(prediction_name)
    expected_key = _history_model_key(expected_model)
    model_matches = not expected_model or (
        bool(prediction_name)
        and bool(prediction_key)
        and prediction_key == expected_key
    )
    passed = probability_passed and model_matches

    if expected_model and not model_matches:
        reason = f"目标模型为 {expected_model}，实际预测为 {prediction_name or '未知模型'}"
    elif probability_passed:
        reason = f"归因概率 {probability * 100:.1f}% 已达到 90% 阈值"
    else:
        reason = f"归因概率 {probability * 100:.1f}% 未达到 90% 阈值"
    return {
        "status": "passed" if passed else "failed",
        "status_label": "检测通过" if passed else "检测未通过",
        "status_reason": reason,
        "status_threshold": HISTORY_PASS_THRESHOLD,
    }


def history_summary(record: dict) -> dict:
    result = record.get("result", {})
    return {
        "id": record.get("id", ""),
        "label": record.get("label", "未命名检测"),
        "test_type": record.get("test_type", "manual"),
        "source_name": record.get("source_name", ""),
        "api_model": record.get("api_model", ""),
        "saved_at": record.get("saved_at", ""),
        "prediction_name": result.get("prediction_name", ""),
        "probability": result.get("probability", 0),
        "family_prediction_name": result.get("family_prediction_name", ""),
        "family_probability": result.get("family_probability", 0),
        "used_outputs": result.get("used_outputs", 0),
        **history_status(record),
    }


def make_history_record(payload: dict) -> dict:
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("没有可保存的检测结果")
    test_type = _history_text(payload.get("test_type"), 20)
    if test_type not in {"manual", "api", "batch"}:
        raise ValueError("未知检测类型")
    raw_candidates = result.get("results")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("检测结果缺少候选模型")

    candidates = []
    for item in raw_candidates[:10]:
        if not isinstance(item, dict):
            continue
        display_name = _history_text(item.get("display_name"), 180)
        if not display_name:
            continue
        candidates.append(
            {
                "display_name": display_name,
                "family_name": _history_text(item.get("family_name"), 100),
                "probability": _history_probability(item.get("probability"), "候选模型概率"),
                "profile_similarity": _history_probability(item.get("profile_similarity"), "分布相似度"),
            }
        )
    if not candidates:
        raise ValueError("检测结果不包含可保存的候选模型")

    diagnostics = []
    for item in result.get("diagnostics", [])[:3]:
        if not isinstance(item, dict):
            continue
        try:
            parsed_numbers = max(0, int(item.get("parsed_numbers", 0)))
        except (TypeError, ValueError):
            parsed_numbers = 0
        diagnostics.append({"parsed_numbers": parsed_numbers, "accepted": bool(item.get("accepted"))})

    saved_at = datetime.now(timezone.utc).astimezone()
    method_names = {"manual": "手动检测", "api": "API 单次检测", "batch": "批量检测"}
    source_name = _history_text(payload.get("source_name"), 180)
    label_source = source_name or _history_text(payload.get("api_model"), 180) or method_names[test_type]
    label = f"{saved_at:%Y-%m-%d %H:%M:%S} · {label_source}"
    prediction_name = _history_text(result.get("prediction_name"), 180) or candidates[0]["display_name"]
    return {
        "id": secrets.token_urlsafe(9),
        "label": label,
        "test_type": test_type,
        "source_name": source_name,
        "api_model": _history_text(payload.get("api_model"), 180),
        "saved_at": saved_at.isoformat(),
        "result": {
            "prediction_name": prediction_name,
            "probability": _history_probability(result.get("probability"), "归因概率"),
            "family_prediction_name": _history_text(result.get("family_prediction_name"), 100),
            "family_probability": _history_probability(result.get("family_probability"), "家族概率"),
            "used_outputs": _history_output_count(result.get("used_outputs")),
            "diagnostics": diagnostics,
            "results": candidates,
        },
    }


def builtin_configs() -> dict[str, dict]:
    return {
        "gpt": {
            "label": "GPT",
            "bank_file": PROJECT / "data" / "gpt_bank.json",
            "data_file": PROJECT / "data" / "gpt_reference.jsonl",
        },
        "claude": {
            "label": "Claude",
            "bank_file": PROJECT / "data" / "claude_bank.json",
            "data_file": PROJECT / "data" / "claude_reference.jsonl",
        },
    }


def load_configs() -> dict[str, dict]:
    configs = builtin_configs()
    if CUSTOM_BANKS_FILE.exists():
        for item in json.loads(CUSTOM_BANKS_FILE.read_text(encoding="utf-8")):
            bank_id = item["id"]
            configs[bank_id] = {
                "label": item["label"],
                "bank_file": PROJECT / "data" / f"{bank_id}_bank.json",
                "data_file": PROJECT / "data" / f"{bank_id}_reference.jsonl",
                "custom": True,
            }
    return configs


def save_custom_configs() -> None:
    items = [
        {"id": bank_id, "label": config["label"]}
        for bank_id, config in BANK_CONFIGS.items()
        if config.get("custom")
    ]
    CUSTOM_BANKS_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_configured_bank(bank_id: str) -> dict | None:
    config = BANK_CONFIGS[bank_id]
    if not config["bank_file"].exists():
        return None
    bank = load_bank(config["bank_file"])
    bank["family_name"] = config["label"]
    return bank


BANK_CONFIGS = load_configs()
banks = {bank_id: load_configured_bank(bank_id) for bank_id in BANK_CONFIGS}


def active_banks() -> dict[str, dict]:
    return {
        bank_id: bank
        for bank_id, bank in banks.items()
        if bank is not None and bank.get("models")
    }


def global_reference_rows() -> list[dict]:
    rows = []
    for bank_id in active_banks():
        config = BANK_CONFIGS[bank_id]
        rows.extend(
            {
                **row,
                "family_id": bank_id,
                "family_name": config["label"],
            }
            for row in read_rows(config["data_file"])
        )
    return rows


def rebuild_global_bank() -> dict:
    global unified_bank
    unified_bank = build_bank(global_reference_rows())
    UNIFIED_BANK_FILE.write_text(
        json.dumps(unified_bank, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return unified_bank


unified_bank = load_bank(UNIFIED_BANK_FILE) if UNIFIED_BANK_FILE.exists() else None
if unified_bank is None:
    rebuild_global_bank()


def requested_bank_id(payload: dict | None = None) -> str:
    bank_id = (payload or {}).get("bank_id") or request.args.get("bank_id") or DEFAULT_BANK_ID
    if bank_id not in BANK_CONFIGS:
        raise ValueError(f"未知指纹库：{bank_id}")
    return bank_id


def requested_temperature(payload: dict) -> float | None:
    value = payload.get("temperature")
    return None if value in (None, "") else float(value)


def summarized_bank(bank_id: str) -> dict:
    config = BANK_CONFIGS[bank_id]
    bank = banks.get(bank_id)
    summary = bank_summary(bank) if bank is not None else {
        "model_count": 0,
        "response_count": 0,
        "number_count": 0,
        "models": [],
        "calibration": {},
    }
    summary.update({"id": bank_id, "label": config["label"]})
    return summary


def summarized_unified_bank() -> dict:
    summaries = {bank_id: summarized_bank(bank_id) for bank_id in active_banks()}
    return {
        "id": "unified",
        "label": "全部指纹",
        "model_count": sum(item["model_count"] for item in summaries.values()),
        "response_count": sum(item["response_count"] for item in summaries.values()),
        "family_count": len(summaries),
        "families": [
            {"id": bank_id, "label": item["label"], "model_count": item["model_count"]}
            for bank_id, item in summaries.items()
        ],
    }


def replace_bank(bank_id: str) -> None:
    banks[bank_id] = load_configured_bank(bank_id)
    rebuild_global_bank()


@app.get("/")
def index():
    summaries = {bank_id: summarized_bank(bank_id) for bank_id in BANK_CONFIGS}
    return render_template(
        "index.html",
        banks=summaries,
        bank=summaries[DEFAULT_BANK_ID],
        unified=summarized_unified_bank(),
        default_bank_id=DEFAULT_BANK_ID,
    )


@app.get("/api/challenges")
def challenges():
    return jsonify({"challenges": generate_challenges(3)})


@app.post("/api/analyze")
def analyze():
    try:
        payload = request.get_json()
        result = analyze_global_outputs(payload["outputs"], unified_bank)
        result["bank"] = summarized_unified_bank()
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/test/auto")
def automatic_test():
    payload = request.get_json()
    try:
        result = test_automatic(
            base_url=payload["base_url"].strip(),
            api_key=payload["api_key"],
            api_model=payload["api_model"].strip(),
            temperature=requested_temperature(payload),
            bank=unified_bank,
            api_format="auto",
        )
        result["bank"] = summarized_unified_bank()
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/test/probe")
def automatic_test_probe():
    payload = request.get_json()
    try:
        text = request_completion(
            base_url=payload["base_url"].strip(),
            api_key=payload["api_key"],
            api_model=payload["api_model"].strip(),
            prompt=payload["prompt"],
            temperature=requested_temperature(payload),
            api_format="auto",
        )
        expected_count = int(payload["expected_count"])
        parsed_numbers = len(parse_numbers(text))
        minimum_numbers = max(80, math.ceil(expected_count * 0.55))
        return jsonify(
            {
                "text": text,
                "parsed_numbers": parsed_numbers,
                "minimum_numbers": minimum_numbers,
                "accepted": parsed_numbers >= minimum_numbers,
            }
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 502


@app.get("/api/bank")
def get_bank():
    try:
        return jsonify(summarized_bank(requested_bank_id()))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/banks")
def get_banks():
    return jsonify({bank_id: summarized_bank(bank_id) for bank_id in BANK_CONFIGS})


@app.post("/api/banks")
def create_bank():
    payload = request.get_json()
    label = payload["label"].strip()
    if not label:
        return jsonify({"error": "请输入指纹库名称"}), 400
    bank_id = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or f"bank-{secrets.token_hex(3)}"
    if bank_id in BANK_CONFIGS:
        return jsonify({"error": "同名指纹库已存在"}), 400
    config = {
        "label": label,
        "bank_file": PROJECT / "data" / f"{bank_id}_bank.json",
        "data_file": PROJECT / "data" / f"{bank_id}_reference.jsonl",
        "custom": True,
    }
    config["data_file"].parent.mkdir(parents=True, exist_ok=True)
    config["data_file"].touch()
    BANK_CONFIGS[bank_id] = config
    banks[bank_id] = None
    save_custom_configs()
    return jsonify(
        {
            "bank": summarized_bank(bank_id),
            "banks": {item: summarized_bank(item) for item in BANK_CONFIGS},
            "unified": summarized_unified_bank(),
        }
    )


@app.post("/api/enroll/auto")
def automatic_enrollment():
    payload = request.get_json()
    try:
        bank_id = requested_bank_id(payload)
        config = BANK_CONFIGS[bank_id]
        result = enroll_automatic(
            base_url=payload["base_url"].strip(),
            api_key=payload["api_key"],
            api_model=payload["api_model"].strip(),
            model_label=payload["model_label"].strip(),
            sample_count=int(payload.get("sample_count", 36)),
            temperature=requested_temperature(payload),
            api_format="auto",
            data_file=config["data_file"],
            bank_file=config["bank_file"],
            bank_id=bank_id,
            provider="api",
        )
        replace_bank(bank_id)
        result["bank"] = summarized_bank(bank_id)
        result["unified"] = summarized_unified_bank()
        return jsonify(result)
    except (RuntimeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/presets")
def get_presets():
    return jsonify({"presets": list(load_presets().values())})


@app.post("/api/presets")
def save_preset():
    payload = request.get_json() or {}
    name = str(payload.get("name", "")).strip()
    if not name:
        return jsonify({"error": "请输入预设名称"}), 400
    base_url = str(payload.get("base_url", "")).strip()
    if not base_url:
        return jsonify({"error": "请输入 Base URL"}), 400
    preset = {
        "name": name,
        "base_url": base_url,
        "api_key": str(payload.get("api_key", "")),
        "model": str(payload.get("model", "")).strip(),
        "temperature": "" if payload.get("temperature") in (None, "") else str(payload["temperature"]),
    }
    presets = load_presets()
    presets[name] = preset
    save_presets(presets)
    return jsonify({"presets": list(load_presets().values())})


@app.delete("/api/presets")
def delete_preset():
    name = str((request.get_json() or {}).get("name", "")).strip()
    presets = load_presets()
    if name in presets:
        del presets[name]
        save_presets(presets)
    return jsonify({"presets": list(load_presets().values())})


@app.post("/api/presets/rename")
def rename_preset():
    """重命名预设：保持原有排序位置，校验重名冲突。"""
    payload = request.get_json() or {}
    name = str(payload.get("name", "")).strip()
    new_name = str(payload.get("new_name", "")).strip()
    if not name or not new_name:
        return jsonify({"error": "请提供预设名称和新名称"}), 400
    presets = load_presets()
    if name not in presets:
        return jsonify({"error": f"预设「{name}」不存在"}), 404
    if new_name == name:
        return jsonify({"presets": list(presets.values())})
    if new_name in presets:
        return jsonify({"error": f"预设「{new_name}」已存在，请换一个名称"}), 400
    renamed: dict[str, dict] = {}
    for key, preset in presets.items():
        if key == name:
            renamed[new_name] = {**preset, "name": new_name}
        else:
            renamed[key] = preset
    save_presets(renamed)
    return jsonify({"presets": list(load_presets().values())})


@app.post("/api/presets/reorder")
def reorder_presets():
    """按给定名称顺序重排预设；未提及的名称忽略，原有预设一律保留（追加在末尾）。"""
    payload = request.get_json() or {}
    names = payload.get("names")
    if not isinstance(names, list):
        return jsonify({"error": "names 必须是预设名称列表"}), 400
    presets = load_presets()
    ordered: dict[str, dict] = {}
    for raw in names:
        name = str(raw).strip()
        if name in presets:
            ordered[name] = presets[name]
    for name, preset in presets.items():
        ordered.setdefault(name, preset)
    save_presets(ordered)
    return jsonify({"presets": list(load_presets().values())})


@app.get("/api/history")
def get_history():
    records = load_history_records()
    records.sort(key=lambda item: str(item.get("saved_at", "")), reverse=True)
    return jsonify({"records": [history_summary(record) for record in records]})


@app.post("/api/history")
def save_history():
    try:
        record = make_history_record(request.get_json() or {})
        records = load_history_records()
        records.append(record)
        save_history_records(records)
        return jsonify({"record": history_summary(record)}), 201
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/history/<record_id>")
def get_history_record(record_id: str):
    record = next((item for item in load_history_records() if item.get("id") == record_id), None)
    if record is None:
        return jsonify({"error": "检测历史不存在或已删除"}), 404
    return jsonify({"record": {**record, **history_status(record)}})


@app.delete("/api/history")
def delete_history_records():
    payload = request.get_json() or {}
    identifiers = payload.get("ids")
    if not isinstance(identifiers, list):
        return jsonify({"error": "ids 必须是历史记录 ID 列表"}), 400
    ids = {str(item) for item in identifiers if str(item)}
    if not ids:
        return jsonify({"error": "请至少选择一条检测历史"}), 400
    records = load_history_records()
    kept_records = [record for record in records if record.get("id") not in ids]
    deleted = len(records) - len(kept_records)
    if deleted:
        save_history_records(kept_records)
    return jsonify({"deleted": deleted})


@app.delete("/api/history/<record_id>")
def delete_history_record(record_id: str):
    records = load_history_records()
    kept_records = [record for record in records if record.get("id") != record_id]
    if len(kept_records) == len(records):
        return jsonify({"error": "检测历史不存在或已删除"}), 404
    save_history_records(kept_records)
    return jsonify({"deleted": 1})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
