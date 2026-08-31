from __future__ import annotations

import json
import math
import re
import secrets
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
PRESET_KEY_FILE = Path.home() / ".modelrace_key"
_ENCRYPTED_PREFIX = "enc1:"
DEFAULT_BANK_ID = "claude"


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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
