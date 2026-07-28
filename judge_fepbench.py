import os
import json
import time
import base64
import traceback
import re
import csv
import statistics
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI



# =========================================================
# 1. API 配置（OpenAI-compatible SDK）
# =========================================================
API_BASE_URL = "your url"
API_KEY = 'your api key'
MODEL_NAME = "gpt-5.4"
MLLM_SYSTEM_PROMPT = "You are a helpful assistant."

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# 为了尽量兼容原脚本中使用的变量名
MLLM_MODEL = MODEL_NAME

# =========================================================
# 1.1 网络 / 重试配置
# =========================================================
REQUEST_TIMEOUT_SECONDS = 300

MAX_RETRIES = 8
BASE_RETRY_SLEEP_SECONDS = 5
MAX_RETRY_SLEEP_SECONDS = 90

TEMPERATURE = 0
SKIP_EXISTING = True





# =========================================================
# 2. 路径配置
# =========================================================
GENERATED_IMAGE_DIR = ""
GOLD_GRAPH_DIR = ""
PROMPT_JSON_DIR = None
OUTPUT_DIR = ""

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# Public/anonymized result files should not contain absolute paths or tracebacks.
# Set to True only for private local debugging.
EXPOSE_PRIVATE_PATHS = False
SAVE_FULL_TRACEBACK = False

# =========================================================
# 2.1 调用次数控制
# =========================================================
# 按你的要求：每张图中，文本实体核验一次、视觉实体核验一次、关系核验一次。
# 如果某类原子过多导致上下文超限，再临时改回分批策略。
SINGLE_CALL_PER_STAGE = True

# =========================================================
# 3. Prompt
# =========================================================
MLLM_SYSTEM_PROMPT = """
You are a strict verifier for generated scientific figures.
Return valid JSON only. Use double quotes for all object keys and string values.
Do not use markdown fences, comments, Python dict syntax, or explanations.
""".strip()

GOLD_GRAPH_VERIFY_PROMPT_TEMPLATE = r"""
You are a strict verifier for generated scientific figures.

You are given:
1. a generated scientific figure image
2. compact gold graph atoms from final_annotation_json:
   - text entities
   - visual entities
   - relations
   - layout constraints

Your task is to verify the whole gold graph against the generated image in ONE pass.

For each gold text entity, output only:
- entity_id
- presence_status: "present" | "absent"
- exact_match: 1 | 0
- readable: 1 | 0
- attachment_match: 1 | 0

For each gold visual entity, output only:
- entity_id
- presence_status: "present" | "absent"
- count_match: 1 | 0
- coarse_location_match: 1 | 0

For each gold relation, output only:
- relation_id
- relation_status: "satisfied" | "violated"
- type_match: 1 | 0
- direction_match: 1 | 0

For layout, output only:
- panel_structure_match: 1 | 0
- panel_count_match: 1 | 0
- global_layout_match: 1 | 0

Rules:
- Verify only against the generated image and the compact gold graph.
- Do not use or infer anything from the original generation prompt.
- Do not output importance, confidence, notes, or explanations.
- Return exactly one check item for every input text entity, visual entity, and relation.
- For text: exact_match = 1 only if the target text appears exactly as surface_text.
- For text: if exact text is absent, set exact_match = 0, readable = 0, attachment_match = 0.
- For visual: presence_status is "present" only when the gold visual atom is clearly realized.
- For visual: if absent, set count_match = 0 and coarse_location_match = 0.
- For relation: relation_status is "satisfied" only if source entity, target entity, and relation are clearly realized.
- For relation: if either endpoint is absent, the relation should normally be violated.
- For direction_match: use 1 for symmetric/non-directional relations if the relation is otherwise correct.
- Be conservative when uncertain.
- Return valid JSON only; do not wrap it in markdown fences.

Compact gold graph:
<<<GOLD_GRAPH_START>>>
{gold_graph_json}
<<<GOLD_GRAPH_END>>>

Return JSON only:
{{
  "text_checks": [
    {{
      "entity_id": "t1",
      "presence_status": "present",
      "exact_match": 1,
      "readable": 1,
      "attachment_match": 1
    }}
  ],
  "visual_checks": [
    {{
      "entity_id": "v1",
      "presence_status": "present",
      "count_match": 1,
      "coarse_location_match": 1
    }}
  ],
  "relation_checks": [
    {{
      "relation_id": "r1",
      "relation_status": "satisfied",
      "type_match": 1,
      "direction_match": 1
    }}
  ],
  "layout_check": {{
    "panel_structure_match": 1,
    "panel_count_match": 1,
    "global_layout_match": 1
  }}
}}
""".strip()

REDUNDANCY_CHECK_PROMPT_TEMPLATE = r"""
You are a strict verifier for semantic precision of generated scientific figures.

You are given:
1. a generated scientific figure image
2. compact allowed atoms from the gold graph:
   - required/optional allowed texts
   - allowed visual entities
   - allowed entity lookup for relation endpoints
   - allowed relations
   - gold visual entity count limits

Inspect the generated image directly and identify unsupported scientific content in ONE pass.

Output:
1. supported_texts: realized scientific texts that align to required or optional gold text atoms
2. unsupported_texts: realized scientific texts that align to neither required nor optional gold text atoms
3. unsupported_visual_entities: salient generated visual entities with scientific meaning that are not allowed by gold atoms
4. unsupported_relations: generated scientific relations that are not allowed by gold relations
5. generated_visual_entity_counts: generated counts for supported visual entity kinds

Rules:
- Do not use or infer anything from the original generation prompt.
- Report only content with clear scientific meaning.
- Ignore harmless decoration, layout fillers, watermark-like noise, and unreadable artifacts.
- Do not output importance, confidence, notes, or explanations.
- Be conservative when uncertain.
- Return valid JSON only; do not wrap it in markdown fences.

Allowed atoms:
<<<ALLOWED_ATOMS_START>>>
{allowed_atoms_json}
<<<ALLOWED_ATOMS_END>>>

Return JSON only:
{{
  "supported_texts": [
    {{"surface_text": "ACE2"}}
  ],
  "unsupported_texts": [
    {{"surface_text": "Random label"}}
  ],
  "unsupported_visual_entities": [
    {{
      "name": "",
      "category": "",
      "panel": "single_panel"
    }}
  ],
  "unsupported_relations": [
    {{
      "relation_type": "",
      "source_name": "",
      "target_name": "",
      "panel": "single_panel"
    }}
  ],
  "generated_visual_entity_counts": [
    {{
      "name": "",
      "category": "",
      "panel": "single_panel",
      "generated_count": 0
    }}
  ]
}}
""".strip()


# =========================================================
# 4. 工具函数
# =========================================================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_data_url(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".bmp":
        mime = "image/bmp"
    else:
        mime = "image/jpeg"
    b64 = encode_image(image_path)
    return f"data:{mime};base64,{b64}"


def extract_model_text(response: Any) -> str:
    """
    兼容 OpenAI SDK 对象返回、dict 返回和部分兼容接口的变体返回。
    特别处理某些兼容接口把最终 JSON 放在 message.reasoning / reasoning_content 中、content 为 None 的情况。
    """
    def _extract_from_message_obj(msg: Any) -> Optional[str]:
        for attr in ["content", "reasoning", "reasoning_content", "reasoning_text", "analysis", "output_text"]:
            try:
                value = getattr(msg, attr)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif hasattr(item, "text") and isinstance(item.text, str):
                        parts.append(item.text)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts)
        return None

    def _extract_from_message_dict(msg: Dict[str, Any]) -> Optional[str]:
        for key in ["content", "reasoning", "reasoning_content", "reasoning_text", "analysis", "output_text"]:
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts)
        return None

    try:
        msg = response.choices[0].message
        found = _extract_from_message_obj(msg)
        if found:
            return found
    except Exception:
        pass

    if isinstance(response, dict):
        try:
            msg = response["choices"][0]["message"]
            if isinstance(msg, dict):
                found = _extract_from_message_dict(msg)
                if found:
                    return found
        except Exception:
            pass

    try:
        if isinstance(response.output_text, str) and response.output_text.strip():
            return response.output_text
    except Exception:
        pass

    return str(response)


def extract_usage_from_response(response: Any) -> Dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage = None
    try:
        usage = response.usage
    except Exception:
        pass

    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if usage is not None:
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif not isinstance(usage, dict):
            try:
                usage = dict(usage)
            except Exception:
                usage = {}

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def try_parse_json(text: Any):
    """
    尽量从模型返回中提取真正的 JSON。
    兼容：严格 JSON、markdown fence、前后解释文字、Python dict 风格、完整 chat completion response 字符串。
    """
    if isinstance(text, dict):
        # 如果误传入完整 chat completion dict，先尝试取 message 内的真实输出。
        try:
            msg = text["choices"][0]["message"]
            for key in ["content", "reasoning", "reasoning_content", "reasoning_text", "analysis", "output_text"]:
                value = msg.get(key)
                if isinstance(value, str) and value.strip():
                    return try_parse_json(value)
        except Exception:
            pass
        return text

    stripped = strip_code_fence(str(text)).strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "choices" in parsed:
            return try_parse_json(parsed)
        return parsed
    except Exception:
        pass

    # 截取第一个 { 到最后一个 }，避免模型前后夹解释文本。
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start:end + 1].strip()

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "choices" in parsed:
                return try_parse_json(parsed)
            return parsed
        except Exception:
            pass

        # 兼容 Python dict 字面量，例如 {'text_checks': ...}
        try:
            import ast
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict) and "choices" in parsed:
                return try_parse_json(parsed)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

        # 轻量修复 Python/JSON 常见差异
        repaired = candidate
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict) and "choices" in parsed:
                return try_parse_json(parsed)
            return parsed
        except Exception:
            pass

        raise ValueError("无法解析模型返回 JSON。候选内容前 1200 字符如下:\n" + candidate[:1200])

    raise ValueError("模型返回中没有找到 JSON 对象。raw 前 1200 字符如下:\n" + stripped[:1200])


def compact_json(obj: Any) -> str:
    """尽量压缩发送给 MLLM 的 JSON，减少输入 token。"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def coerce_int01(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except Exception:
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "present", "satisfied"}:
            return 1
        return 0


def fill_missing_text_checks(
    text_entities: List[Dict[str, Any]],
    checks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    check_map = {str(x.get("entity_id")): x for x in checks if x.get("entity_id") is not None}
    out = []
    for ent in text_entities:
        eid = ent.get("entity_id")
        c = dict(check_map.get(str(eid), {}))
        c["entity_id"] = eid
        c["importance"] = c.get("importance", ent.get("importance", "required"))
        status = str(c.get("presence_status", "absent")).strip().lower()
        c["presence_status"] = "present" if status == "present" else "absent"
        c["exact_match"] = coerce_int01(c.get("exact_match", 0))
        c["readable"] = coerce_int01(c.get("readable", 0))
        c["attachment_match"] = coerce_int01(c.get("attachment_match", 0))
        if c["presence_status"] == "absent":
            c["exact_match"] = 0
            c["readable"] = 0
            c["attachment_match"] = 0
        out.append(c)
    return out


def fill_missing_visual_checks(
    visual_entities: List[Dict[str, Any]],
    checks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    check_map = {str(x.get("entity_id")): x for x in checks if x.get("entity_id") is not None}
    out = []
    for ent in visual_entities:
        eid = ent.get("entity_id")
        c = dict(check_map.get(str(eid), {}))
        c["entity_id"] = eid
        c["importance"] = c.get("importance", ent.get("importance", "required"))
        status = str(c.get("presence_status", "absent")).strip().lower()
        c["presence_status"] = "present" if status == "present" else "absent"
        c["count_match"] = coerce_int01(c.get("count_match", 0))
        c["coarse_location_match"] = coerce_int01(c.get("coarse_location_match", 0))
        if c["presence_status"] == "absent":
            c["count_match"] = 0
            c["coarse_location_match"] = 0
        out.append(c)
    return out


def fill_missing_relation_checks(
    relations: List[Dict[str, Any]],
    checks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    check_map = {str(x.get("relation_id")): x for x in checks if x.get("relation_id") is not None}
    out = []
    for rel in relations:
        rid = rel.get("relation_id")
        c = dict(check_map.get(str(rid), {}))
        c["relation_id"] = rid
        c["importance"] = c.get("importance", rel.get("importance", "required"))
        status = str(c.get("relation_status", "violated")).strip().lower()
        c["relation_status"] = "satisfied" if status == "satisfied" else "violated"
        c["type_match"] = coerce_int01(c.get("type_match", 0))
        c["direction_match"] = coerce_int01(c.get("direction_match", 0))
        out.append(c)
    return out



def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def output_path_value(path: str) -> str:
    """Return a publishable path value without leaking local directory structure."""
    if EXPOSE_PRIVATE_PATHS:
        return path
    return os.path.basename(path)


def checked_importance(value: Any, *, atom_id: Any, atom_kind: str) -> str:
    """Validate the required/optional split used by the paper."""
    importance = "required" if value is None else str(value).strip().lower()
    if importance not in {"required", "optional"}:
        raise ValueError(
            f"Invalid importance={value!r} for {atom_kind} atom {atom_id!r}; "
            "expected 'required' or 'optional'."
        )
    return importance


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gold_graph_path_for_image(image_path: str) -> str:
    stem = Path(image_path).stem
    return os.path.join(GOLD_GRAPH_DIR, f"{stem}.json")


def prompt_json_path_for_image(image_path: str) -> Optional[str]:
    if PROMPT_JSON_DIR is None:
        return None
    stem = Path(image_path).stem
    return os.path.join(PROMPT_JSON_DIR, f"{stem}.json")


def is_meaningful_prompt_text(text: Optional[str]) -> bool:
    if text is None or not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if s in {".", "null", "None", "N/A", "na"}:
        return False
    return True


def find_prompt_in_json_obj(obj: Any) -> Optional[str]:
    priority_keys = [
        "generation_prompt",
        "prompt",
        "text_prompt",
        "final_prompt",
        "revised_prompt",
        "instruction",
        "caption",
        "description",
        "model_output_raw",
    ]

    if isinstance(obj, dict):
        for key in priority_keys:
            if key in obj and is_meaningful_prompt_text(obj[key]):
                return obj[key].strip()

        if "model_output_json" in obj:
            found = find_prompt_in_json_obj(obj["model_output_json"])
            if found:
                return found

        for _, value in obj.items():
            found = find_prompt_in_json_obj(value)
            if found:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_prompt_in_json_obj(item)
            if found:
                return found

    elif isinstance(obj, str):
        if is_meaningful_prompt_text(obj):
            return obj.strip()

    return None


def load_generation_prompt(image_path: str, graph: Dict[str, Any]) -> str:
    gp = graph.get("generation_prompt")
    if is_meaningful_prompt_text(gp):
        return gp.strip()

    path = prompt_json_path_for_image(image_path)
    if path and os.path.exists(path):
        obj = load_json(path)
        found = find_prompt_in_json_obj(obj)
        if found:
            return found

    raise ValueError(f"未找到 generation_prompt: {image_path}")


def merge_token_usage(*usages: Dict[str, int]) -> Dict[str, int]:
    merged = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for u in usages:
        if not u:
            continue
        merged["input_tokens"] += int(u.get("input_tokens", 0))
        merged["output_tokens"] += int(u.get("output_tokens", 0))
        merged["total_tokens"] += int(u.get("total_tokens", 0))
    return merged


def chunks(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_retryable_status_code(exc: Exception) -> Optional[int]:
    visited = set()
    cur = exc
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))

        response = getattr(cur, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                return int(status_code)

        status_code = getattr(cur, "status_code", None)
        if status_code is not None:
            return int(status_code)

        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return None


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        msg = str(exc).lower()
        if "json" in msg or "解析" in msg or "parse" in msg:
            return True
    visited = set()
    cur = exc

    retryable_error_names = {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
    }

    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))

        if type(cur).__name__ in retryable_error_names:
            return True

        status_code = extract_retryable_status_code(cur)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True

        msg = str(cur).lower()
        retryable_signals = [
            "connection reset by peer",
            "connection aborted",
            "temporary failure",
            "timed out",
            "timeout",
            "connection error",
            "server disconnected",
            "502",
            "503",
            "504",
            "429",
        ]
        if any(x in msg for x in retryable_signals):
            return True

        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)

    return False


def compute_retry_sleep_seconds(attempt: int) -> float:
    base = min(MAX_RETRY_SLEEP_SECONDS, BASE_RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)))
    jitter = random.uniform(0, min(3.0, base * 0.2))
    return base + jitter


def run_stage_with_retry(stage_name: str, fn):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"    [{stage_name}] 第 {attempt}/{MAX_RETRIES} 次尝试")
            return fn()

        except KeyboardInterrupt:
            print(f"    [{stage_name}] 用户中断，停止重试。")
            raise

        except Exception as e:
            last_error = e
            retryable = is_retryable_exception(e)

            print(f"    [{stage_name}] 失败: {type(e).__name__}: {repr(e)}")
            traceback.print_exc()

            if attempt >= MAX_RETRIES or not retryable:
                print(f"    [{stage_name}] 不再重试。retryable={retryable}, attempt={attempt}")
                break

            sleep_s = compute_retry_sleep_seconds(attempt)
            print(f"    [{stage_name}] 将在 {sleep_s:.1f} 秒后重试...")
            time.sleep(sleep_s)

    raise last_error


def call_model_for_image(
    prompt_text: str,
    image_path: str,
) -> Tuple[Any, str, Dict[str, int]]:
    """
    使用 OpenAI-compatible SDK 调用多模态 chat.completions。
    输入顺序保持为：先图像，后文本 prompt。
    """
    image_data_url = build_data_url(image_path)

    response = client.chat.completions.create(
        model=MLLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": MLLM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            },
        ],
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    text = extract_model_text(response)
    usage = extract_usage_from_response(response)
    return response, text, usage


# =========================================================
# 5. gold graph 解析
# =========================================================
def load_gold_graph(image_path: str) -> Dict[str, Any]:
    path = gold_graph_path_for_image(image_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 gold graph: {path}")

    raw = load_json(path)
    if isinstance(raw, dict) and "final_annotation_json" in raw and isinstance(raw["final_annotation_json"], dict):
        graph = raw["final_annotation_json"]
    else:
        graph = raw

    if "entities" not in graph:
        raise ValueError(f"gold graph 缺少 entities 字段: {path}")

    if "relations" not in graph:
        graph["relations"] = []

    if "figure_summary" not in graph:
        graph["figure_summary"] = {}

    return graph


def build_visual_count_limits(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter = {}
    for ent in entities:
        if ent.get("entity_type") != "visual":
            continue
        if ent.get("importance") not in {"required", "optional"}:
            continue

        key = (
            (ent.get("name") or "").strip(),
            (ent.get("category") or "").strip(),
            (ent.get("panel") or "single_panel").strip(),
        )
        count_val = ent.get("count", 1)
        try:
            count_val = int(count_val)
        except Exception:
            count_val = 1
        if count_val < 0:
            count_val = 0

        counter[key] = counter.get(key, 0) + count_val

    out = []
    for (name, category, panel), gold_count in counter.items():
        out.append({
            "name": name,
            "category": category,
            "panel": panel,
            "gold_count": gold_count,
        })
    out.sort(key=lambda x: (x["category"], x["name"], x["panel"]))
    return out


def split_gold_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    entities = graph.get("entities", [])
    relations = graph.get("relations", [])
    figure_summary = graph.get("figure_summary", {})

    text_entities: List[Dict[str, Any]] = []
    visual_entities: List[Dict[str, Any]] = []
    entity_lookup: Dict[str, Dict[str, Any]] = {}

    for ent in entities:
        entity_id = ent.get("entity_id")
        importance = checked_importance(
            ent.get("importance"), atom_id=entity_id, atom_kind="entity"
        )
        compact = {
            "entity_id": entity_id,
            "importance": importance,
            "entity_type": ent.get("entity_type"),
            "name": ent.get("name", ""),
            "category": ent.get("category", ""),
            # Preserve the annotated surface form exactly; no text equivalence mapping.
            "surface_text": ent.get("surface_text"),
            "panel": ent.get("panel", "single_panel"),
            "bbox_approx": ent.get("bbox_approx"),
            "count": ent.get("count", 1),
            "key_attributes": ent.get("key_attributes", []),
        }
        entity_lookup[str(entity_id)] = compact
        if ent.get("entity_type") == "text":
            text_entities.append(compact)
        elif ent.get("entity_type") == "visual":
            visual_entities.append(compact)

    rels: List[Dict[str, Any]] = []
    allowed_relations: List[Dict[str, Any]] = []
    for rel in relations:
        relation_id = rel.get("relation_id")
        importance = checked_importance(
            rel.get("importance"), atom_id=relation_id, atom_kind="relation"
        )
        source_id = rel.get("source_entity_id")
        target_id = rel.get("target_entity_id")
        compact_rel = {
            "relation_id": relation_id,
            "importance": importance,
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "relation_type": rel.get("relation_type", ""),
            "panel_scope": rel.get("panel_scope", "same_panel"),
        }
        rels.append(compact_rel)

        source = entity_lookup.get(str(source_id), {})
        target = entity_lookup.get(str(target_id), {})
        allowed_relations.append({
            "relation_id": relation_id,
            "relation_type": rel.get("relation_type", ""),
            "source_entity_id": source_id,
            "source_name": source.get("name") or source.get("surface_text") or "",
            "target_entity_id": target_id,
            "target_name": target.get("name") or target.get("surface_text") or "",
            "panel_scope": rel.get("panel_scope", "same_panel"),
        })

    layout_atoms = [
        {"atom_id": "panel_structure_match", "label": "panel_structure_match"},
        {"atom_id": "panel_count_match", "label": "panel_count_match"},
        {"atom_id": "global_layout_match", "label": "global_layout_match"},
    ]

    # Include every available field needed to judge the three binary layout atoms.
    layout = {
        "layout_type": figure_summary.get("layout_type", ""),
        "panel_structure": figure_summary.get("panel_structure", ""),
        "panel_count": figure_summary.get("panel_count"),
        "global_layout": figure_summary.get("global_layout", ""),
        "relative_panel_positions": figure_summary.get("relative_panel_positions", []),
        "figure_type": figure_summary.get("figure_type", ""),
        "domain": figure_summary.get("domain", ""),
    }

    allowed_entities: List[Dict[str, Any]] = []
    required_texts: List[Dict[str, Any]] = []
    optional_texts: List[Dict[str, Any]] = []
    allowed_entity_lookup: List[Dict[str, Any]] = []

    for ent in text_entities + visual_entities:
        allowed_entity_lookup.append({
            "entity_id": ent.get("entity_id"),
            "entity_type": ent.get("entity_type"),
            "name": ent.get("name", ""),
            "surface_text": ent.get("surface_text"),
            "category": ent.get("category", ""),
            "panel": ent.get("panel", "single_panel"),
        })
        if ent.get("entity_type") == "visual":
            allowed_entities.append({
                "entity_id": ent.get("entity_id"),
                "name": ent.get("name", ""),
                "category": ent.get("category", ""),
                "panel": ent.get("panel", "single_panel"),
            })
        elif ent.get("entity_type") == "text":
            item = {
                "entity_id": ent.get("entity_id"),
                "surface_text": ent.get("surface_text", ""),
                "category": ent.get("category", ""),
                "panel": ent.get("panel", "single_panel"),
            }
            if ent.get("importance") == "required":
                required_texts.append(item)
            else:
                optional_texts.append(item)

    visual_count_limits = build_visual_count_limits(text_entities + visual_entities)

    return {
        "text_entities": text_entities,
        "visual_entities": visual_entities,
        "relations": rels,
        "layout": layout,
        "layout_atoms": layout_atoms,
        "required_texts": required_texts,
        "optional_texts": optional_texts,
        "allowed_entities": allowed_entities,
        "allowed_entity_lookup": allowed_entity_lookup,
        "allowed_relations": allowed_relations,
        "visual_count_limits": visual_count_limits,
    }


def validate_nonempty_atom_categories(parts: Dict[str, Any]) -> None:
    """Enforce the paper's non-zero atom-category construction requirement."""
    groups = {
        "required_visual": [x for x in parts["visual_entities"] if x["importance"] == "required"],
        "required_text": [x for x in parts["text_entities"] if x["importance"] == "required"],
        "required_relation": [x for x in parts["relations"] if x["importance"] == "required"],
        "optional_visual": [x for x in parts["visual_entities"] if x["importance"] == "optional"],
        "optional_text": [x for x in parts["text_entities"] if x["importance"] == "optional"],
        "optional_relation": [x for x in parts["relations"] if x["importance"] == "optional"],
    }
    empty = [name for name, atoms in groups.items() if not atoms]
    if empty:
        raise ValueError(
            "Gold graph violates the paper's non-zero atom-category requirement: "
            + ", ".join(empty)
        )

def build_allowed_sets(parts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "required_texts": parts["required_texts"],
        "optional_texts": parts["optional_texts"],
        "allowed_visual_entities": parts["allowed_entities"],
        "allowed_entity_lookup": parts["allowed_entity_lookup"],
        "allowed_relations": parts["allowed_relations"],
        "gold_visual_entity_count_limits": parts["visual_count_limits"],
    }


# =========================================================
# 6. Prompt 构造
# =========================================================
def compact_entity_for_prompt(ent: Dict[str, Any]) -> Dict[str, Any]:
    """发给 MLLM 的 entity 版本：只保留关键信息，不包含 importance / bbox / notes。"""
    return {
        "entity_id": ent.get("entity_id"),
        "entity_type": ent.get("entity_type"),
        "name": ent.get("name", ""),
        "category": ent.get("category", ""),
        "surface_text": ent.get("surface_text"),
        "panel": ent.get("panel", "single_panel"),
        "count": ent.get("count", 1),
        "key_attributes": ent.get("key_attributes", []),
    }


def compact_relation_for_prompt(rel: Dict[str, Any]) -> Dict[str, Any]:
    """发给 MLLM 的 relation 版本：只保留关键信息，不包含 importance。"""
    return {
        "relation_id": rel.get("relation_id"),
        "source_entity_id": rel.get("source_entity_id"),
        "target_entity_id": rel.get("target_entity_id"),
        "relation_type": rel.get("relation_type", ""),
        "panel_scope": rel.get("panel_scope", "same_panel"),
    }


def compact_entities_for_prompt(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [compact_entity_for_prompt(x) for x in entities]


def compact_relations_for_prompt(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [compact_relation_for_prompt(x) for x in relations]


def build_gold_graph_verify_prompt(parts: Dict[str, Any]) -> str:
    gold_graph_json = {
        "text_entities": compact_entities_for_prompt(parts.get("text_entities", [])),
        "visual_entities": compact_entities_for_prompt(parts.get("visual_entities", [])),
        "relations": compact_relations_for_prompt(parts.get("relations", [])),
        "layout": parts.get("layout", {}),
    }
    return GOLD_GRAPH_VERIFY_PROMPT_TEMPLATE.format(
        gold_graph_json=compact_json(gold_graph_json),
    )


def build_redundancy_check_prompt(allowed_sets: Dict[str, Any]) -> str:
    allowed_atoms = {
        "required_texts": allowed_sets.get("required_texts", []),
        "optional_texts": allowed_sets.get("optional_texts", []),
        "allowed_visual_entities": allowed_sets.get("allowed_visual_entities", []),
        "allowed_entity_lookup": allowed_sets.get("allowed_entity_lookup", []),
        "allowed_relations": allowed_sets.get("allowed_relations", []),
        "gold_visual_entity_count_limits": allowed_sets.get("gold_visual_entity_count_limits", []),
    }
    return REDUNDANCY_CHECK_PROMPT_TEMPLATE.format(
        allowed_atoms_json=compact_json(allowed_atoms),
    )


# =========================================================
# 7. 两次 API 执行阶段
# =========================================================
def run_gold_graph_verification(image_path: str, parts: Dict[str, Any]) -> Dict[str, Any]:
    """
    每张图第一次 API 调用：同时核验 text / visual / relation / layout。
    MLLM 输入不包含 generation_prompt / importance。
    """
    prompt = build_gold_graph_verify_prompt(parts)

    def call_and_parse():
        _, raw, usage = call_model_for_image(
            prompt_text=prompt,
            image_path=image_path,
        )
        parsed = try_parse_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"GOLD_GRAPH_VERIFY 返回不是 JSON object: {type(parsed).__name__}")
        return raw, parsed, usage

    raw, parsed, usage = run_stage_with_retry("GOLD_GRAPH_VERIFY", call_and_parse)

    text_checks = fill_missing_text_checks(
        parts.get("text_entities", []),
        parsed.get("text_checks", []) or [],
    )
    visual_checks = fill_missing_visual_checks(
        parts.get("visual_entities", []),
        parsed.get("visual_checks", []) or [],
    )
    relation_checks = fill_missing_relation_checks(
        parts.get("relations", []),
        parsed.get("relation_checks", []) or [],
    )
    layout_check = parsed.get("layout_check", {}) or {}

    # 为了兼容原先输出结构，仍然构造四个 stage，但它们共享同一次 API 的 raw/json。
    # token 只记在 text_verification_stage，其他三个置零，避免 total 重复计算。
    zero_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    text_stage = {
        "raw_batches": [{
            "batch_id": 1,
            "batch_size": len(parts.get("text_entities", [])),
            "batch": compact_entities_for_prompt(parts.get("text_entities", [])),
            "raw": raw,
            "json": {"text_checks": parsed.get("text_checks", []) or []},
            "token_usage": usage,
            "source_api_call": "gold_graph_verification",
        }],
        "results": text_checks,
        "token_usage": usage,
    }
    visual_stage = {
        "raw_batches": [{
            "batch_id": 1,
            "batch_size": len(parts.get("visual_entities", [])),
            "batch": compact_entities_for_prompt(parts.get("visual_entities", [])),
            "raw": raw,
            "json": {"visual_checks": parsed.get("visual_checks", []) or []},
            "token_usage": zero_usage,
            "source_api_call": "gold_graph_verification",
        }],
        "results": visual_checks,
        "token_usage": zero_usage,
    }
    relation_stage = {
        "raw_batches": [{
            "batch_id": 1,
            "batch_size": len(parts.get("relations", [])),
            "batch": compact_relations_for_prompt(parts.get("relations", [])),
            "raw": raw,
            "json": {"relation_checks": parsed.get("relation_checks", []) or []},
            "token_usage": zero_usage,
            "source_api_call": "gold_graph_verification",
        }],
        "results": relation_checks,
        "token_usage": zero_usage,
    }
    layout_stage = {
        "raw": raw,
        "json": {"layout_check": layout_check},
        "token_usage": zero_usage,
        "source_api_call": "gold_graph_verification",
    }

    return {
        "raw": raw,
        "json": parsed,
        "token_usage": usage,
        "text_verification_stage": text_stage,
        "visual_verification_stage": visual_stage,
        "relation_verification_stage": relation_stage,
        "layout_verification_stage": layout_stage,
    }


def run_redundancy_check(image_path: str, allowed_sets: Dict[str, Any]) -> Dict[str, Any]:
    """
    每张图第二次 API 调用：同时做文本冗余、视觉冗余、关系冗余和支持实体计数。
    """
    prompt = build_redundancy_check_prompt(allowed_sets)

    def call_and_parse():
        _, raw, usage = call_model_for_image(
            prompt_text=prompt,
            image_path=image_path,
        )
        parsed = try_parse_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"REDUNDANCY_CHECK 返回不是 JSON object: {type(parsed).__name__}")
        return raw, parsed, usage

    raw, parsed, usage = run_stage_with_retry("REDUNDANCY_CHECK", call_and_parse)

    text_red_json = {
        "supported_texts": parsed.get("supported_texts", []) or [],
        "unsupported_texts": parsed.get("unsupported_texts", []) or [],
    }
    visrel_red_json = {
        "unsupported_visual_entities": parsed.get("unsupported_visual_entities", []) or [],
        "unsupported_relations": parsed.get("unsupported_relations", []) or [],
        "generated_visual_entity_counts": parsed.get("generated_visual_entity_counts", []) or [],
    }

    zero_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # 兼容原先输出结构，但只把 token 记到 text_redundancy_stage，避免总 token 重复计算。
    text_red_stage = {
        "raw": raw,
        "json": text_red_json,
        "token_usage": usage,
        "source_api_call": "redundancy_check",
    }
    visrel_red_stage = {
        "raw": raw,
        "json": visrel_red_json,
        "token_usage": zero_usage,
        "source_api_call": "redundancy_check",
    }

    return {
        "raw": raw,
        "json": parsed,
        "token_usage": usage,
        "text_redundancy_stage": text_red_stage,
        "visual_relation_redundancy_stage": visrel_red_stage,
    }


# =========================================================
# 8. 本地打分
# =========================================================
def binary_presence_to_num(status: str) -> float:
    return 1.0 if (status or "").strip().lower() == "present" else 0.0


def binary_relation_status_to_num(status: str) -> float:
    return 1.0 if (status or "").strip().lower() == "satisfied" else 0.0


def compute_text_scores(
    gold_text_entities: List[Dict[str, Any]],
    text_checks: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    check_map = {x["entity_id"]: x for x in text_checks}
    req_scores = []
    opt_scores = []

    for ent in gold_text_entities:
        c = check_map.get(ent["entity_id"], {})
        exact = int(c.get("exact_match", 0))
        readable = int(c.get("readable", 0))
        attach = int(c.get("attachment_match", 0))
        s = exact * (0.5 * readable + 0.5 * attach)

        if ent.get("importance") == "required":
            req_scores.append(s)
        elif ent.get("importance") == "optional":
            opt_scores.append(s)

    return {
        "S_t_req": safe_mean(req_scores),
        "n_text_req": len(req_scores),
    }, {
        "S_t_opt": safe_mean(opt_scores),
        "n_text_opt": len(opt_scores),
    }


def compute_visual_scores(
    gold_visual_entities: List[Dict[str, Any]],
    visual_checks: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    check_map = {x["entity_id"]: x for x in visual_checks}
    req_scores = []
    opt_scores = []

    for ent in gold_visual_entities:
        c = check_map.get(ent["entity_id"], {})
        pres = binary_presence_to_num(c.get("presence_status", "absent"))
        cnt = int(c.get("count_match", 0))
        loc = int(c.get("coarse_location_match", 0))
        s = pres * (0.5 * cnt + 0.5 * loc)

        if ent.get("importance") == "required":
            req_scores.append(s)
        elif ent.get("importance") == "optional":
            opt_scores.append(s)

    return {
        "S_v_req": safe_mean(req_scores),
        "n_visual_req": len(req_scores),
    }, {
        "S_v_opt": safe_mean(opt_scores),
        "n_visual_opt": len(opt_scores),
    }


def build_entity_presence_map(
    gold_text_entities: List[Dict[str, Any]],
    gold_visual_entities: List[Dict[str, Any]],
    text_checks: List[Dict[str, Any]],
    visual_checks: List[Dict[str, Any]],
) -> Dict[str, int]:
    presence_map = {}

    for x in text_checks:
        presence_map[x["entity_id"]] = 1 if x.get("presence_status") == "present" else 0

    for x in visual_checks:
        presence_map[x["entity_id"]] = 1 if x.get("presence_status") == "present" else 0

    for ent in gold_text_entities + gold_visual_entities:
        if ent["entity_id"] not in presence_map:
            presence_map[ent["entity_id"]] = 0

    return presence_map


def compute_relation_scores(
    gold_relations: List[Dict[str, Any]],
    relation_checks: List[Dict[str, Any]],
    entity_presence_map: Dict[str, int],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    check_map = {x["relation_id"]: x for x in relation_checks}
    req_scores = []
    opt_scores = []

    for rel in gold_relations:
        c = check_map.get(rel["relation_id"], {})
        stat = binary_relation_status_to_num(c.get("relation_status", "violated"))
        source_present = entity_presence_map.get(rel.get("source_entity_id"), 0)
        target_present = entity_presence_map.get(rel.get("target_entity_id"), 0)
        if not (source_present and target_present):
            stat = 0.0
        type_match = coerce_int01(c.get("type_match", 0))
        direction = coerce_int01(c.get("direction_match", 0))
        s = stat * (0.5 * type_match + 0.5 * direction)

        if rel.get("importance") == "required":
            req_scores.append(s)
        elif rel.get("importance") == "optional":
            opt_scores.append(s)

    return {
        "S_r_req": safe_mean(req_scores),
        "n_rel_req": len(req_scores),
    }, {
        "S_r_opt": safe_mean(opt_scores),
        "n_rel_opt": len(opt_scores),
    }


def compute_layout_score(layout_check: Dict[str, Any]) -> float:
    vals = [
        coerce_int01(layout_check.get("panel_structure_match", 0)),
        coerce_int01(layout_check.get("panel_count_match", 0)),
        coerce_int01(layout_check.get("global_layout_match", 0)),
    ]
    return safe_mean(vals)


def aggregate_scores(
    S_v_req: float,
    S_t_req: float,
    S_r_req: float,
    S_l: float,
    S_v_opt: float,
    S_t_opt: float,
    S_r_opt: float,
    SP: float,
) -> Dict[str, float]:
    IF = (S_v_req + S_t_req + S_r_req + S_l) / 4.0
    RE = (S_v_opt + S_t_opt + S_r_opt) / 3.0

    return {
        "IF": IF,
        "RE": RE,
        "SP": SP,
    }


def normalize_visual_kind_key(name: str, category: str, panel: str) -> Tuple[str, str, str]:
    return (
        (name or "").strip().lower(),
        (category or "").strip().lower(),
        (panel or "single_panel").strip().lower(),
    )


def compute_overgenerated_visuals(
    generated_visual_entity_counts: List[Dict[str, Any]],
    gold_visual_count_limits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gold_map = {}
    for item in gold_visual_count_limits:
        key = normalize_visual_kind_key(
            item.get("name", ""),
            item.get("category", ""),
            item.get("panel", "single_panel"),
        )
        try:
            gold_count = int(item.get("gold_count", 0))
        except Exception:
            gold_count = 0
        if gold_count < 0:
            gold_count = 0
        gold_map[key] = gold_count

    generated_map = {}
    for item in generated_visual_entity_counts:
        key = normalize_visual_kind_key(
            item.get("name", ""),
            item.get("category", ""),
            item.get("panel", "single_panel"),
        )
        try:
            gen_count = int(item.get("generated_count", 0))
        except Exception:
            gen_count = 0
        if gen_count < 0:
            gen_count = 0
        generated_map[key] = max(generated_map.get(key, 0), gen_count)

    overgenerated_items = []
    total_extra = 0
    supported_visual_count = 0

    for key, gen_count in generated_map.items():
        gold_count = gold_map.get(key, 0)
        supported_here = min(gen_count, gold_count)
        extra = max(0, gen_count - gold_count)
        supported_visual_count += supported_here

        if extra > 0:
            name, category, panel = key
            overgenerated_items.append({
                "name": name,
                "category": category,
                "panel": panel,
                "gold_count": gold_count,
                "generated_count": gen_count,
                "extra_count": extra,
            })
            total_extra += extra

    return {
        "overgenerated_visual_entities": overgenerated_items,
        "num_overgenerated_visual_instances": total_extra,
        "num_supported_visual_instances": supported_visual_count,
    }


def _exact_surface_text_set(items: List[Dict[str, Any]]) -> set:
    """Deduplicate text atoms by exact surface string only; no semantic normalization."""
    values = set()
    for item in items:
        value = item.get("surface_text")
        if value is None:
            continue
        value = str(value).strip()
        if value:
            values.add(value)
    return values


def _unsupported_visual_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    return normalize_visual_kind_key(
        item.get("name", ""),
        item.get("category", ""),
        item.get("panel", "single_panel"),
    )


def _unsupported_relation_key(item: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(item.get("relation_type", "")).strip(),
        str(item.get("source_name", "")).strip(),
        str(item.get("target_name", "")).strip(),
        str(item.get("panel", item.get("panel_scope", "same_panel"))).strip(),
    )


def semantic_precision_from_counts(supported: int, unsupported: int) -> float:
    """Paper Eq. (8), with the explicit empty-realization convention SP_x=1."""
    total = supported + unsupported
    if total == 0:
        return 1.0
    return supported / total


def compute_semantic_precision(
    text_red_json: Dict[str, Any],
    visrel_red_json: Dict[str, Any],
    relation_checks: List[Dict[str, Any]],
    gold_visual_count_limits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    supported_texts = text_red_json.get("supported_texts", []) or []
    unsupported_texts = text_red_json.get("unsupported_texts", []) or []

    unsupported_visual = visrel_red_json.get("unsupported_visual_entities", []) or []
    unsupported_relations = visrel_red_json.get("unsupported_relations", []) or []
    generated_visual_entity_counts = visrel_red_json.get("generated_visual_entity_counts", []) or []

    overgen_info = compute_overgenerated_visuals(
        generated_visual_entity_counts=generated_visual_entity_counts,
        gold_visual_count_limits=gold_visual_count_limits,
    )

    # Eq. (8) uses set cardinality. Text is deduplicated by exact surface form only.
    supported_text_set = _exact_surface_text_set(supported_texts)
    unsupported_text_set = _exact_surface_text_set(unsupported_texts) - supported_text_set
    n_supported_text = len(supported_text_set)
    n_unsupported_text = len(unsupported_text_set)

    # Generated visual counts instantiate realized visual entities. Extra instances beyond
    # the gold count limits are unsupported/excessive realizations, as specified by the
    # precision-verifier input. Explicit unsupported entities are deduplicated by atom key.
    n_supported_visual = overgen_info["num_supported_visual_instances"]
    unsupported_visual_keys = {_unsupported_visual_key(x) for x in unsupported_visual}
    unsupported_visual_keys.discard(("", "", "single_panel"))
    n_unsupported_visual_explicit = len(unsupported_visual_keys)
    n_unsupported_visual_overgenerated = overgen_info["num_overgenerated_visual_instances"]
    n_unsupported_visual = n_unsupported_visual_explicit + n_unsupported_visual_overgenerated

    supported_relation_ids = {
        str(x.get("relation_id"))
        for x in relation_checks
        if x.get("relation_id") is not None
        and str(x.get("relation_status", "")).strip().lower() == "satisfied"
    }
    unsupported_relation_keys = {_unsupported_relation_key(x) for x in unsupported_relations}
    unsupported_relation_keys.discard(("", "", "", "single_panel"))
    unsupported_relation_keys.discard(("", "", "", "same_panel"))
    n_supported_relation = len(supported_relation_ids)
    n_unsupported_relation = len(unsupported_relation_keys)

    SP_text = semantic_precision_from_counts(n_supported_text, n_unsupported_text)
    SP_visual = semantic_precision_from_counts(n_supported_visual, n_unsupported_visual)
    SP_relation = semantic_precision_from_counts(n_supported_relation, n_unsupported_relation)
    SP = (SP_text + SP_visual + SP_relation) / 3.0

    text_red_ratio = 1.0 - SP_text
    visual_red_ratio = 1.0 - SP_visual
    relation_red_ratio = 1.0 - SP_relation

    return {
        "SP_text": SP_text,
        "SP_visual": SP_visual,
        "SP_relation": SP_relation,
        "SP": SP,
        "text_red_ratio": text_red_ratio,
        "visual_red_ratio": visual_red_ratio,
        "relation_red_ratio": relation_red_ratio,
        "n_supported_text": n_supported_text,
        "n_unsupported_text": n_unsupported_text,
        "n_supported_visual": n_supported_visual,
        "n_unsupported_visual": n_unsupported_visual,
        "n_unsupported_visual_explicit": n_unsupported_visual_explicit,
        "n_unsupported_visual_overgenerated": n_unsupported_visual_overgenerated,
        "n_supported_relation": n_supported_relation,
        "n_unsupported_relation": n_unsupported_relation,
        "overgenerated_visual_info": overgen_info,
    }


# =========================================================
# 9. 单图评测
# =========================================================
def evaluate_one_image(image_path: str) -> Dict[str, Any]:
    graph = load_gold_graph(image_path)
    parts = split_gold_graph(graph)
    validate_nonempty_atom_categories(parts)

    # 第一次 API：gold graph 全量核验，返回 text / visual / relation / layout。
    gold_verify_stage = run_gold_graph_verification(
        image_path=image_path,
        parts=parts,
    )

    text_stage = gold_verify_stage["text_verification_stage"]
    visual_stage = gold_verify_stage["visual_verification_stage"]
    relation_stage = gold_verify_stage["relation_verification_stage"]
    layout_stage = gold_verify_stage["layout_verification_stage"]

    text_checks = text_stage["results"]
    visual_checks = visual_stage["results"]
    relation_checks = relation_stage["results"]
    layout_check = layout_stage["json"].get("layout_check", {})

    allowed_sets = build_allowed_sets(parts)

    # 第二次 API：统一冗余检查，返回文本/视觉/关系冗余与支持计数。
    redundancy_stage = run_redundancy_check(
        image_path=image_path,
        allowed_sets=allowed_sets,
    )

    text_red_stage = redundancy_stage["text_redundancy_stage"]
    visrel_red_stage = redundancy_stage["visual_relation_redundancy_stage"]

    req_text_scores, opt_text_scores = compute_text_scores(
        gold_text_entities=parts["text_entities"],
        text_checks=text_checks,
    )

    req_visual_scores, opt_visual_scores = compute_visual_scores(
        gold_visual_entities=parts["visual_entities"],
        visual_checks=visual_checks,
    )

    entity_presence_map = build_entity_presence_map(
        gold_text_entities=parts["text_entities"],
        gold_visual_entities=parts["visual_entities"],
        text_checks=text_checks,
        visual_checks=visual_checks,
    )

    req_rel_scores, opt_rel_scores = compute_relation_scores(
        gold_relations=parts["relations"],
        relation_checks=relation_checks,
        entity_presence_map=entity_presence_map,
    )

    S_l = compute_layout_score(layout_check)

    precision_scores = compute_semantic_precision(
        text_red_json=text_red_stage.get("json", {}),
        visrel_red_json=visrel_red_stage.get("json", {}),
        relation_checks=relation_checks,
        gold_visual_count_limits=parts["visual_count_limits"],
    )

    final_scores = aggregate_scores(
        S_v_req=req_visual_scores["S_v_req"],
        S_t_req=req_text_scores["S_t_req"],
        S_r_req=req_rel_scores["S_r_req"],
        S_l=S_l,
        S_v_opt=opt_visual_scores["S_v_opt"],
        S_t_opt=opt_text_scores["S_t_opt"],
        S_r_opt=opt_rel_scores["S_r_opt"],
        SP=precision_scores["SP"],
    )

    intermediate_scores = {
        **req_visual_scores,
        **opt_visual_scores,
        **req_text_scores,
        **opt_text_scores,
        **req_rel_scores,
        **opt_rel_scores,
        "S_l": S_l,
        "SP_text": precision_scores["SP_text"],
        "SP_visual": precision_scores["SP_visual"],
        "SP_relation": precision_scores["SP_relation"],
        "text_red_ratio": precision_scores["text_red_ratio"],
        "visual_red_ratio": precision_scores["visual_red_ratio"],
        "relation_red_ratio": precision_scores["relation_red_ratio"],
        "n_supported_text": precision_scores["n_supported_text"],
        "n_unsupported_text": precision_scores["n_unsupported_text"],
        "n_supported_visual": precision_scores["n_supported_visual"],
        "n_unsupported_visual": precision_scores["n_unsupported_visual"],
        "n_unsupported_visual_explicit": precision_scores["n_unsupported_visual_explicit"],
        "n_unsupported_visual_overgenerated": precision_scores["n_unsupported_visual_overgenerated"],
        "n_supported_relation": precision_scores["n_supported_relation"],
        "n_unsupported_relation": precision_scores["n_unsupported_relation"],
    }

    token_usage = {
        "gold_graph_verification_stage": gold_verify_stage.get("token_usage", {}),
        "redundancy_check_stage": redundancy_stage.get("token_usage", {}),

        # 以下字段保留给旧的 summary/export 逻辑；其中重复来源 stage 的 token 已置零。
        "text_verification_stage": text_stage.get("token_usage", {}),
        "visual_verification_stage": visual_stage.get("token_usage", {}),
        "relation_verification_stage": relation_stage.get("token_usage", {}),
        "layout_verification_stage": layout_stage.get("token_usage", {}),
        "text_redundancy_stage": text_red_stage.get("token_usage", {}),
        "visual_relation_redundancy_stage": visrel_red_stage.get("token_usage", {}),
    }
    token_usage["total"] = merge_token_usage(
        token_usage["gold_graph_verification_stage"],
        token_usage["redundancy_check_stage"],
    )

    return {
        "image_name": os.path.basename(image_path),
        "image_path": output_path_value(image_path),
        "gold_graph_path": output_path_value(gold_graph_path_for_image(image_path)),
        "num_api_calls": 2,

        "gold_graph_verification_stage": {
            "raw": gold_verify_stage.get("raw", ""),
            "json": gold_verify_stage.get("json", {}),
            "token_usage": gold_verify_stage.get("token_usage", {}),
        },
        "redundancy_check_stage": {
            "raw": redundancy_stage.get("raw", ""),
            "json": redundancy_stage.get("json", {}),
            "token_usage": redundancy_stage.get("token_usage", {}),
        },

        # 保留原先字段，方便旧的 summary/export 代码兼容。
        "text_verification_stage": text_stage,
        "visual_verification_stage": visual_stage,
        "relation_verification_stage": relation_stage,
        "layout_verification_stage": layout_stage,
        "text_redundancy_stage": text_red_stage,
        "visual_relation_redundancy_stage": visrel_red_stage,

        "gold_visual_count_limits": parts["visual_count_limits"],
        "overgenerated_visual_info": precision_scores["overgenerated_visual_info"],

        "intermediate_scores": intermediate_scores,
        "final_scores": final_scores,
        "token_usage": token_usage,
    }


# =========================================================
# 10. 汇总与导出
# =========================================================
def build_summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    fs = result.get("final_scores", {})
    ims = result.get("intermediate_scores", {})
    tus = result.get("token_usage", {}).get("total", {})

    return {
        "image_name": result.get("image_name", ""),
        "image_path": result.get("image_path", ""),
        "S_v_req": ims.get("S_v_req", 0.0),
        "S_t_req": ims.get("S_t_req", 0.0),
        "S_r_req": ims.get("S_r_req", 0.0),
        "S_l": ims.get("S_l", 0.0),
        "S_v_opt": ims.get("S_v_opt", 0.0),
        "S_t_opt": ims.get("S_t_opt", 0.0),
        "S_r_opt": ims.get("S_r_opt", 0.0),
        "SP_text": ims.get("SP_text", 0.0),
        "SP_visual": ims.get("SP_visual", 0.0),
        "SP_relation": ims.get("SP_relation", 0.0),
        "text_red_ratio": ims.get("text_red_ratio", 0.0),
        "visual_red_ratio": ims.get("visual_red_ratio", 0.0),
        "relation_red_ratio": ims.get("relation_red_ratio", 0.0),
        "n_supported_text": ims.get("n_supported_text", 0),
        "n_unsupported_text": ims.get("n_unsupported_text", 0),
        "n_supported_visual": ims.get("n_supported_visual", 0),
        "n_unsupported_visual": ims.get("n_unsupported_visual", 0),
        "n_unsupported_visual_explicit": ims.get("n_unsupported_visual_explicit", 0),
        "n_unsupported_visual_overgenerated": ims.get("n_unsupported_visual_overgenerated", 0),
        "n_supported_relation": ims.get("n_supported_relation", 0),
        "n_unsupported_relation": ims.get("n_unsupported_relation", 0),
        "IF": fs.get("IF", fs.get("PF", 0.0)),
        "RE": fs.get("RE", 0.0),
        "SP": fs.get("SP", 0.0),
        "token_input_total": tus.get("input_tokens", 0),
        "token_output_total": tus.get("output_tokens", 0),
        "token_total": tus.get("total_tokens", 0),
    }


def build_overgenerated_detail_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    image_name = result.get("image_name", "")
    image_path = output_path_value(result.get("image_path", ""))

    overgen_info = result.get("overgenerated_visual_info", {}) or {}
    items = overgen_info.get("overgenerated_visual_entities", []) or []

    for item in items:
        rows.append({
            "image_name": image_name,
            "image_path": image_path,
            "name": item.get("name", ""),
            "category": item.get("category", ""),
            "panel": item.get("panel", "single_panel"),
            "gold_count": item.get("gold_count", 0),
            "generated_count": item.get("generated_count", 0),
            "extra_count": item.get("extra_count", 0),
        })

    return rows


def save_summary_csv(rows: List[Dict[str, Any]], path: str):
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            pass
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_overgenerated_detail_csv(rows: List[Dict[str, Any]], path: str):
    fieldnames = [
        "image_name",
        "image_path",
        "name",
        "category",
        "panel",
        "gold_count",
        "generated_count",
        "extra_count",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_overgenerated_detail_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg = {}

    for row in rows:
        key = (
            row.get("name", ""),
            row.get("category", ""),
            row.get("panel", "single_panel"),
        )

        if key not in agg:
            agg[key] = {
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "panel": row.get("panel", "single_panel"),
                "num_images": 0,
                "sum_gold_count": 0,
                "sum_generated_count": 0,
                "sum_extra_count": 0,
            }

        agg[key]["num_images"] += 1
        agg[key]["sum_gold_count"] += int(row.get("gold_count", 0))
        agg[key]["sum_generated_count"] += int(row.get("generated_count", 0))
        agg[key]["sum_extra_count"] += int(row.get("extra_count", 0))

    out = list(agg.values())
    out.sort(key=lambda x: (-x["sum_extra_count"], x["category"], x["name"], x["panel"]))
    return out


def save_overgenerated_aggregate_csv(rows: List[Dict[str, Any]], path: str):
    fieldnames = [
        "name",
        "category",
        "panel",
        "num_images",
        "sum_gold_count",
        "sum_generated_count",
        "sum_extra_count",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_dataset_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    numeric_keys = [
        "S_v_req", "S_t_req", "S_r_req", "S_l",
        "S_v_opt", "S_t_opt", "S_r_opt",
        "SP_text", "SP_visual", "SP_relation",
        "text_red_ratio", "visual_red_ratio", "relation_red_ratio",
        "n_supported_text", "n_unsupported_text",
        "n_supported_visual", "n_unsupported_visual", "n_unsupported_visual_explicit",
        "n_unsupported_visual_overgenerated", "n_supported_relation", "n_unsupported_relation",
        "IF", "RE", "SP",
        "token_input_total", "token_output_total", "token_total",
    ]

    metrics = {
        "num_samples": len(rows),
        "metrics": {}
    }

    for key in numeric_keys:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except Exception:
                pass
        if values:
            metrics["metrics"][key] = {
                "mean": safe_mean(values),
                "std": safe_std(values),
                "min": min(values),
                "max": max(values),
            }

    return metrics


def compute_dataset_token_usage(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    total_input = 0
    total_output = 0
    total = 0

    for row in rows:
        total_input += int(row.get("token_input_total", 0))
        total_output += int(row.get("token_output_total", 0))
        total += int(row.get("token_total", 0))

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total,
    }


# =========================================================
# 11. 主程序
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = []
    for name in os.listdir(GENERATED_IMAGE_DIR):
        full_path = os.path.join(GENERATED_IMAGE_DIR, name)
        if os.path.isfile(full_path) and Path(name).suffix.lower() in IMAGE_EXTS:
            image_files.append(full_path)

    image_files.sort()
    print(f"共检测到 {len(image_files)} 张生成图片")

    summary_rows = []
    overgenerated_detail_rows = []

    for idx, image_path in enumerate(image_files, start=1):
        image_name = os.path.basename(image_path)
        stem = Path(image_name).stem
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.json")

        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[{idx}/{len(image_files)}] 已存在，跳过: {image_name}")
            try:
                existing = load_json(out_path)
                summary_rows.append(build_summary_row(existing))
                overgenerated_detail_rows.extend(build_overgenerated_detail_rows(existing))
            except Exception:
                pass
            continue

        print(f"[{idx}/{len(image_files)}] 开始评测: {image_name}")

        try:
            result = evaluate_one_image(image_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"  -> 成功保存: {out_path}")
            summary_rows.append(build_summary_row(result))
            overgenerated_detail_rows.extend(build_overgenerated_detail_rows(result))

        except Exception as e:
            fail_path = os.path.join(OUTPUT_DIR, f"{stem}_error.json")
            error_data = {
                "image_name": image_name,
                "image_path": output_path_value(image_path),
                "gold_graph_path": output_path_value(gold_graph_path_for_image(image_path)),
                "error_type": type(e).__name__,
                "error": repr(e),
                "retryable": is_retryable_exception(e),
                "traceback": traceback.format_exc() if SAVE_FULL_TRACEBACK else ""
            }
            with open(fail_path, "w", encoding="utf-8") as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            print(f"  -> 失败，错误信息已保存到: {fail_path}")

    summary_json_path = os.path.join(OUTPUT_DIR, "summary_scores.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    summary_csv_path = os.path.join(OUTPUT_DIR, "summary_scores.csv")
    save_summary_csv(summary_rows, summary_csv_path)

    dataset_metrics = compute_dataset_metrics(summary_rows)
    dataset_token_usage = compute_dataset_token_usage(summary_rows)

    dataset_metrics_path = os.path.join(OUTPUT_DIR, "dataset_metrics.json")
    with open(dataset_metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            **dataset_metrics,
            "dataset_token_usage": dataset_token_usage,
        }, f, ensure_ascii=False, indent=2)

    dataset_token_usage_path = os.path.join(OUTPUT_DIR, "dataset_token_usage.json")
    with open(dataset_token_usage_path, "w", encoding="utf-8") as f:
        json.dump(dataset_token_usage, f, ensure_ascii=False, indent=2)

    overgenerated_detail_csv_path = os.path.join(OUTPUT_DIR, "overgenerated_visual_entities.csv")
    save_overgenerated_detail_csv(overgenerated_detail_rows, overgenerated_detail_csv_path)

    overgenerated_aggregate_rows = aggregate_overgenerated_detail_rows(overgenerated_detail_rows)
    overgenerated_aggregate_csv_path = os.path.join(OUTPUT_DIR, "overgenerated_visual_entities_aggregate.csv")
    save_overgenerated_aggregate_csv(overgenerated_aggregate_rows, overgenerated_aggregate_csv_path)

    print("评测完成。")
    print(f"summary json: {summary_json_path}")
    print(f"summary csv : {summary_csv_path}")
    print(f"dataset stats: {dataset_metrics_path}")
    print(f"dataset token: {dataset_token_usage_path}")
    print(f"overgenerated detail: {overgenerated_detail_csv_path}")
    print(f"overgenerated agg   : {overgenerated_aggregate_csv_path}")


if __name__ == "__main__":
    main()