import os
import base64
import json
import re
import time
from io import BytesIO

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__)


REFERENCE_SHEET_PROMPT_TEMPLATE = """
Create a highly compressed exam reference sheet from the analysis below.

Constraints:
- Cover every topic and skill mentioned in the analysis.
- Output plain text only (no markdown fences).
- Keep formatting aggressively compact with short headers, bullets, formulas, quick examples, and minimal spacing.
- Maximize information density for exactly two letter-sized pages at 6pt font.
- Use approximately 2200-3200 words.
- Avoid long introductions and avoid filler wording.
- Use very short bullet lines and compact notation where possible.
- Keep blank lines to a minimum.
- Include:
  1) Topic map
  2) Core rules/formulas/facts
  3) Common traps/mistakes
  4) Fast solving patterns
  5) Mini worked examples
  6) Last-minute checklist

Extra requirements:
{requirements_block}

Analysis to transform:
{analysis_text}
""".strip()


def _split_csv_or_lines(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    parts = re.split(r"[,\n;]+", value)
    return [item.strip() for item in parts if item.strip()]


def _normalize_requirements(raw):
    raw = raw or {}
    sections = _split_csv_or_lines(raw.get("allowed_sections"))
    required_topics = _split_csv_or_lines(raw.get("required_topics"))
    banned_topics = _split_csv_or_lines(raw.get("banned_topics"))
    target_length_words = raw.get("target_length_words")
    try:
        target_length_words = int(target_length_words) if target_length_words is not None else 2600
    except (TypeError, ValueError):
        target_length_words = 2600
    target_length_words = max(1200, min(4200, target_length_words))

    difficulty = str(raw.get("difficulty", "advanced")).strip().lower()
    if difficulty not in {"beginner", "intermediate", "advanced"}:
        difficulty = "advanced"

    event_name = str(raw.get("event_name", "Science Olympiad event")).strip() or "Science Olympiad event"
    division = str(raw.get("division", "B/C")).strip() or "B/C"
    notes = str(raw.get("notes", "")).strip()

    if not sections:
        sections = [
            "Topic map",
            "Core rules/formulas/facts",
            "Common traps/mistakes",
            "Fast solving patterns",
            "Mini worked examples",
            "Last-minute checklist",
        ]

    return {
        "event_name": event_name[:120],
        "division": division[:60],
        "difficulty": difficulty,
        "target_length_words": target_length_words,
        "allowed_sections": sections[:12],
        "required_topics": required_topics[:40],
        "banned_topics": banned_topics[:40],
        "notes": notes[:1200],
    }


def _requirements_to_block(requirements):
    lines = [
        f"- Event: {requirements['event_name']}",
        f"- Division: {requirements['division']}",
        f"- Difficulty: {requirements['difficulty']}",
        f"- Target length (words): {requirements['target_length_words']}",
        f"- Required sections: {', '.join(requirements['allowed_sections'])}",
        f"- Required topics: {', '.join(requirements['required_topics']) if requirements['required_topics'] else 'None explicitly provided'}",
        f"- Disallowed topics: {', '.join(requirements['banned_topics']) if requirements['banned_topics'] else 'None'}",
        f"- Additional notes: {requirements['notes'] or 'None'}",
    ]
    return "\n".join(lines)


def build_reference_sheet_prompt(analysis_text, requirements):
    return REFERENCE_SHEET_PROMPT_TEMPLATE.format(
        analysis_text=analysis_text,
        requirements_block=_requirements_to_block(requirements),
    )


def generate_reference_sheet_local(analysis_text, requirements):
    from training.local_generator import generate_reference_sheet_with_local_model

    prompt = build_reference_sheet_prompt(analysis_text, requirements)
    return generate_reference_sheet_with_local_model(prompt)


def _safe_parse_json_object(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _hf_text(api_key, prompt, primary_model, fallback_models_raw, system_prompt=None):
    return call_hf_with_fallback(
        api_key,
        prompt,
        primary_model=primary_model,
        fallback_models_raw=fallback_models_raw,
        system_prompt=system_prompt,
    )


def _parse_model_list(models_raw):
    if not models_raw:
        return []
    return [model.strip() for model in models_raw.split(",") if model.strip()]


def _coerce_score(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _judge_sheet_quality(api_key, analysis_text, requirements, sheet_text, judge_model, judge_fallback):
    judge_prompt = f"""
Score this Science Olympiad reference sheet and return ONLY JSON.
Schema:
{{
  "score": 0-100,
  "coverage": 0-100,
  "accuracy_risk": 0-100,
  "density": 0-100,
  "requirements_fit": 0-100,
  "issues": ["short bullet", "..."]
}}

Higher is better except `accuracy_risk` where higher = higher risk.

Requirements:
{_requirements_to_block(requirements)}

Analysis:
{analysis_text}

Sheet:
{sheet_text}
""".strip()
    judge_result = _hf_text(api_key, judge_prompt, judge_model, judge_fallback)
    quality = {"score": None, "coverage": None, "accuracy_risk": None, "density": None, "requirements_fit": None, "issues": []}
    if judge_result["ok"]:
        parsed = _safe_parse_json_object(judge_result["text"])
        if isinstance(parsed, dict):
            quality.update(
                {
                    "score": _coerce_score(parsed.get("score")),
                    "coverage": _coerce_score(parsed.get("coverage")),
                    "accuracy_risk": _coerce_score(parsed.get("accuracy_risk")),
                    "density": _coerce_score(parsed.get("density")),
                    "requirements_fit": _coerce_score(parsed.get("requirements_fit")),
                    "issues": parsed.get("issues") if isinstance(parsed.get("issues"), list) else [],
                }
            )
    return judge_result, quality


def _generate_reference_sheet_hf_high_quality(api_key, analysis_text, requirements):
    gen_model = os.getenv("HF_REFERENCE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    gen_fallback = os.getenv(
        "HF_REFERENCE_FALLBACK_MODELS",
        "Qwen/Qwen3.5-35B-A3B,Qwen/Qwen3.5-27B",
    )
    candidate_models = _parse_model_list(
        os.getenv(
            "HF_REFERENCE_CANDIDATE_MODELS",
            (
                "Qwen/Qwen2.5-72B-Instruct,"
                "Qwen/Qwen3.5-35B-A3B,"
                "Qwen/Qwen3.5-27B"
            ),
        )
    )
    if not candidate_models:
        candidate_models = [gen_model]
    max_candidates = max(1, min(5, int(os.getenv("HF_REFERENCE_MAX_CANDIDATES", "3"))))

    critique_model = os.getenv("HF_CRITIQUE_MODEL", "Qwen/Qwen3.5-27B")
    critique_fallback = os.getenv("HF_CRITIQUE_FALLBACK_MODELS", "Qwen/Qwen3.5-35B-A3B")
    judge_model = os.getenv("HF_JUDGE_MODEL", critique_model)
    judge_fallback = os.getenv("HF_JUDGE_FALLBACK_MODELS", critique_fallback)

    base_prompt = build_reference_sheet_prompt(analysis_text, requirements)
    draft_candidates = []
    for model in candidate_models[:max_candidates]:
        candidate_result = _hf_text(api_key, base_prompt, model, gen_fallback)
        if candidate_result["ok"]:
            draft_candidates.append(candidate_result)
    if not draft_candidates:
        return {
            "ok": False,
            "error": {"error": "All candidate generation models failed."},
            "models_tried": candidate_models[:max_candidates],
            "failed_models": candidate_models[:max_candidates],
        }

    scored_candidates = []
    for candidate in draft_candidates:
        judge_result, quality = _judge_sheet_quality(
            api_key,
            analysis_text,
            requirements,
            candidate["text"],
            judge_model,
            judge_fallback,
        )
        score = quality.get("score") if quality.get("score") is not None else 0.0
        accuracy_risk = quality.get("accuracy_risk") if quality.get("accuracy_risk") is not None else 50.0
        blended_score = score - (0.25 * accuracy_risk)
        scored_candidates.append(
            {
                "draft_result": candidate,
                "judge_result": judge_result,
                "quality": quality,
                "blended_score": blended_score,
            }
        )

    scored_candidates.sort(key=lambda item: item["blended_score"], reverse=True)
    best_candidate = scored_candidates[0]
    draft_result = best_candidate["draft_result"]
    draft = draft_result["text"]

    critique_prompt = f"""
You are a strict quality reviewer for Science Olympiad reference sheets.

Return concise bullets:
1) Coverage gaps (missing required topics or sections)
2) Potential factual/logic risks
3) Density/format issues (too verbose or too sparse)
4) Specific rewrites to improve competitive usefulness

Reference requirements:
{_requirements_to_block(requirements)}

Analysis source:
{analysis_text}

Draft sheet:
{draft}
""".strip()
    critique_result = _hf_text(api_key, critique_prompt, critique_model, critique_fallback)
    critique = critique_result["text"] if critique_result["ok"] else "No critique available."

    revise_prompt = f"""
Revise the draft into a significantly better final reference sheet.

Rules:
- Keep plain text only.
- Keep extremely high information density.
- Strictly satisfy every requirement.
- Remove weak or generic lines.
- Prefer concrete, test-usable compact content.
- Output only the final revised sheet.

Requirements:
{_requirements_to_block(requirements)}

Analysis:
{analysis_text}

Draft:
{draft}

Critique to address:
{critique}
""".strip()
    final_result = _hf_text(api_key, revise_prompt, gen_model, gen_fallback)
    if not final_result["ok"]:
        return {
            "ok": True,
            "text": draft,
            "model_used": draft_result["model_used"],
            "quality": {**best_candidate["quality"], "notes": ["Revision stage failed; using best draft output."]},
            "candidate_quality": [
                {
                    "model_used": item["draft_result"]["model_used"],
                    "score": item["quality"].get("score"),
                    "accuracy_risk": item["quality"].get("accuracy_risk"),
                    "blended_score": item["blended_score"],
                }
                for item in scored_candidates
            ],
        }
    final_sheet = final_result["text"]

    judge_result, quality = _judge_sheet_quality(
        api_key,
        analysis_text,
        requirements,
        final_sheet,
        judge_model,
        judge_fallback,
    )

    return {
        "ok": True,
        "text": final_sheet,
        "model_used": final_result["model_used"],
        "quality": quality,
        "candidate_quality": [
            {
                "model_used": item["draft_result"]["model_used"],
                "score": item["quality"].get("score"),
                "accuracy_risk": item["quality"].get("accuracy_risk"),
                "blended_score": item["blended_score"],
            }
            for item in scored_candidates
        ],
        "pipeline_models": {
            "draft_model": draft_result["model_used"],
            "critique_model": critique_result["model_used"] if critique_result["ok"] else None,
            "final_model": final_result["model_used"],
            "judge_model": judge_result["model_used"] if judge_result["ok"] else None,
        },
    }


def call_hf_with_fallback(api_key, prompt, primary_model=None, fallback_models_raw=None, system_prompt=None):
    primary_model = primary_model or os.getenv("HF_MODEL", "Qwen/Qwen3.5-27B")
    fallback_models_raw = fallback_models_raw if fallback_models_raw is not None else os.getenv(
        "HF_FALLBACK_MODELS",
        "Qwen/Qwen3.5-35B-A3B,Qwen/Qwen2.5-72B-Instruct",
    )
    models_to_try = [primary_model]
    for fallback_model in fallback_models_raw.split(","):
        fallback_model = fallback_model.strip()
        if fallback_model and fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

    failed_models = []
    final_error = {"error": "Hugging Face request failed.", "details": "No models attempted."}

    for model in models_to_try:
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages, "max_tokens": 3500, "temperature": 0.2}

        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code in {408, 424, 429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue

        if response is None:
            failed_models.append(model)
            final_error = {"error": f"Hugging Face request failed for model '{model}'.", "details": str(last_error)}
            continue

        if not response.ok:
            status_code = response.status_code
            try:
                details = response.json()
            except ValueError:
                details = (response.text or "")[:500]

            failed_models.append(model)
            final_error = {
                "error": f"Hugging Face request failed with status {status_code}.",
                "details": details,
                "model": model,
            }

            if status_code in {401, 402, 404, 408, 424, 429, 500, 502, 503, 504}:
                continue
            break

        body = response.json()
        text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if text:
            return {
                "ok": True,
                "text": text,
                "model_used": model,
            }

        failed_models.append(model)
        final_error = {
            "error": f"Hugging Face returned no text response for model '{model}'.",
            "details": body,
        }

    return {
        "ok": False,
        "error": final_error,
        "models_tried": models_to_try,
        "failed_models": failed_models,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze-practice")
def analyze_practice():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Upload at least one text file."}), 400

    api_key = os.getenv("HF_API_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    if not api_key:
        return jsonify({"error": "Missing HF_API_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in environment."}), 400

    max_files = 20
    max_chars_per_file = 12000
    max_total_chars = 90000
    max_image_bytes_per_file = 8 * 1024 * 1024
    max_total_image_bytes = 24 * 1024 * 1024
    allowed_extensions = {".txt", ".md", ".csv", ".rtf", ".pdf", ".png", ".jpg", ".jpeg"}

    if len(files) > max_files:
        return jsonify({"error": f"Too many files. Max allowed is {max_files}."}), 400

    parsed_files = []
    image_files = []
    total_chars = 0
    total_image_bytes = 0

    for uploaded_file in files:
        filename = (uploaded_file.filename or "").strip()
        if not filename:
            continue

        _, ext = os.path.splitext(filename.lower())
        if ext and ext not in allowed_extensions:
            return jsonify(
                {
                    "error": (
                        f"Unsupported file extension for '{filename}'. "
                        "Allowed: .txt, .md, .csv, .rtf, .pdf, .png, .jpg, .jpeg"
                    )
                }
            ), 400

        raw = uploaded_file.read()
        if not raw:
            continue

        if ext in {".png", ".jpg", ".jpeg"}:
            image_size = len(raw)
            if image_size > max_image_bytes_per_file:
                return jsonify(
                    {"error": f"'{filename}' exceeds {max_image_bytes_per_file // (1024 * 1024)}MB image limit."}
                ), 400
            total_image_bytes += image_size
            if total_image_bytes > max_total_image_bytes:
                return jsonify(
                    {"error": f"Total image upload size exceeds {max_total_image_bytes // (1024 * 1024)}MB."}
                ), 400

            mime = "image/png" if ext == ".png" else "image/jpeg"
            data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"
            image_files.append({"filename": filename, "data_uri": data_uri})
            continue

        if ext == ".pdf":
            try:
                reader = PdfReader(BytesIO(raw))
                text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
            except Exception:
                return jsonify({"error": f"Could not read text from PDF '{filename}'."}), 400
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = raw.decode("latin-1")
                except UnicodeDecodeError:
                    return jsonify({"error": f"Could not decode '{filename}' as text."}), 400

        text = (text or "").strip()
        if not text:
            continue

        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file]

        total_chars += len(text)
        if total_chars > max_total_chars:
            return jsonify(
                {
                    "error": (
                        "Uploaded text content is too large. "
                        f"Keep total text characters under {max_total_chars}."
                    )
                }
            ), 400

        parsed_files.append({"filename": filename, "text": text})

    if not parsed_files and not image_files:
        return jsonify({"error": "No readable text content found in uploaded files."}), 400

    file_blocks = "\n\n".join(
        [
            f"FILE: {item['filename']}\n---\n{item['text']}"
            for item in parsed_files
        ]
    )
    if image_files:
        return jsonify({"error": "Image analysis is currently disabled for Hugging Face backend. Upload text/PDF files only."}), 400

    analysis_prompt = f"""
You are analyzing multiple student practice materials.

Goal:
Tell me exactly what is being covered across these files and images.

Output format:
1) Covered topics:
- A bullet list of concrete topics that appear in the material.
2) Skills practiced:
- A bullet list of skills/question types being practiced.
3) Frequency map:
- For each topic, estimate how often it appears (high/medium/low) with brief evidence from file names.
4) Missing or weak areas:
- Mention important adjacent topics that are absent or lightly covered.
5) 5-point summary:
- Give exactly five concise bullets a teacher can scan quickly.

Use clear headings and keep it concise but specific.

File contents:
{file_blocks or "None"}
""".strip()
    analysis_model = os.getenv("HF_ANALYSIS_MODEL", "Qwen/Qwen3.5-27B")
    analysis_fallback = os.getenv(
        "HF_ANALYSIS_FALLBACK_MODELS",
        "Qwen/Qwen3.5-35B-A3B,Qwen/Qwen2.5-72B-Instruct",
    )
    result = _hf_text(
        api_key,
        analysis_prompt,
        analysis_model,
        analysis_fallback,
        system_prompt="You produce precise, structured educational analysis.",
    )
    if not result["ok"]:
        return jsonify({**result["error"], "models_tried": result["models_tried"], "failed_models": result["failed_models"]}), 502
    text = result["text"]

    return jsonify(
        {
            "response": text,
            "files_analyzed": [item["filename"] for item in parsed_files] + [item["filename"] for item in image_files],
            "total_files": len(parsed_files) + len(image_files),
            "model_used": result["model_used"],
        }
    )


@app.post("/api/generate-reference-sheet")
def generate_reference_sheet():
    data = request.get_json(silent=True) or {}
    analysis_text = (data.get("analysis") or "").strip()
    if not analysis_text:
        return jsonify({"error": "Analysis text is required."}), 400
    requirements = _normalize_requirements(data.get("requirements") or {})

    generation_mode = os.getenv("REFERENCE_GENERATION_MODE", "huggingface").strip().lower()
    if generation_mode not in {"huggingface", "local", "auto"}:
        return jsonify({"error": "REFERENCE_GENERATION_MODE must be one of: huggingface, local, auto"}), 400

    quality_mode = os.getenv("REFERENCE_QUALITY_MODE", "high").strip().lower()
    if quality_mode not in {"standard", "high"}:
        return jsonify({"error": "REFERENCE_QUALITY_MODE must be one of: standard, high"}), 400

    if generation_mode in {"local", "auto"}:
        try:
            local_result = generate_reference_sheet_local(analysis_text, requirements)
            local_result["requirements"] = requirements
            local_result["quality_mode"] = "local_single_pass"
            return jsonify(local_result)
        except Exception as exc:
            if generation_mode == "local":
                return jsonify({"error": f"Local generation failed: {exc}"}), 500

    api_key = os.getenv("HF_API_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    if not api_key:
        return jsonify({"error": "Missing HF_API_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in environment."}), 400

    if quality_mode == "high":
        result = _generate_reference_sheet_hf_high_quality(api_key, analysis_text, requirements)
    else:
        prompt = build_reference_sheet_prompt(analysis_text, requirements)
        reference_model = os.getenv("HF_REFERENCE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        reference_fallback_models = os.getenv(
            "HF_REFERENCE_FALLBACK_MODELS",
            "Qwen/Qwen3.5-35B-A3B,Qwen/Qwen3.5-27B",
        )
        result = _hf_text(api_key, prompt, reference_model, reference_fallback_models)
    if not result["ok"]:
        return jsonify({**result["error"], "models_tried": result["models_tried"], "failed_models": result["failed_models"]}), 502

    return jsonify(
        {
            "reference_sheet": result["text"],
            "model_used": result["model_used"],
            "quality": result.get("quality"),
            "pipeline_models": result.get("pipeline_models"),
            "requirements": requirements,
            "quality_mode": quality_mode,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
