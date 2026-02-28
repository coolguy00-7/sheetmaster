import os
import base64
import time
from io import BytesIO

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__)


def call_openrouter_with_fallback(api_key, user_content):
    primary_model = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free")
    fallback_models_raw = os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "google/gemma-3-12b-it:free,google/gemma-3-4b-it:free",
    )
    models_to_try = [primary_model]
    for fallback_model in fallback_models_raw.split(","):
        fallback_model = fallback_model.strip()
        if fallback_model and fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    failed_models = []
    final_error = {"error": "OpenRouter request failed.", "details": "No models attempted."}

    for model in models_to_try:
        # Some providers (e.g., Gemma via Google AI Studio) reject system/developer instructions.
        if model.startswith("google/gemma-"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You produce precise, structured educational analysis.",
                        },
                        *user_content,
                    ],
                }
            ]
        else:
            messages = [
                {"role": "system", "content": "You produce precise, structured educational analysis."},
                {"role": "user", "content": user_content},
            ]

        payload = {
            "model": model,
            "messages": messages,
        }

        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
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
            final_error = {"error": f"OpenRouter request failed for model '{model}'.", "details": str(last_error)}
            continue

        if not response.ok:
            status_code = response.status_code
            try:
                details = response.json()
            except ValueError:
                details = (response.text or "")[:500]

            failed_models.append(model)
            final_error = {
                "error": f"OpenRouter request failed with status {status_code}.",
                "details": details,
                "model": model,
            }

            if status_code in {402, 404, 429, 500, 502, 503, 504}:
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
            "error": f"OpenRouter returned no text response for model '{model}'.",
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

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "Missing OPENROUTER_API_KEY in environment."}), 400

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
    image_names = ", ".join([item["filename"] for item in image_files]) or "None"
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

Image files:
{image_names}

File contents:
{file_blocks or "None"}
""".strip()

    user_content = [{"type": "text", "text": analysis_prompt}]
    for image in image_files:
        user_content.append({"type": "text", "text": f"Image file: {image['filename']}"})
        user_content.append({"type": "image_url", "image_url": {"url": image["data_uri"]}})

    result = call_openrouter_with_fallback(api_key, user_content)
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

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "Missing OPENROUTER_API_KEY in environment."}), 400

    prompt = f"""
Create a highly compressed exam reference sheet from the analysis below.

Constraints:
- Cover every topic and skill mentioned in the analysis.
- Output plain text only (no markdown fences).
- Keep formatting compact with short headers, bullets, formulas, and quick examples.
- Target print density for two letter-sized pages at 6pt font.
- Use approximately 1400-2000 words.
- Include:
  1) Topic map
  2) Core rules/formulas/facts
  3) Common traps/mistakes
  4) Fast solving patterns
  5) Mini worked examples
  6) Last-minute checklist

Analysis to transform:
{analysis_text}
""".strip()

    user_content = [{"type": "text", "text": prompt}]
    result = call_openrouter_with_fallback(api_key, user_content)
    if not result["ok"]:
        return jsonify({**result["error"], "models_tried": result["models_tried"], "failed_models": result["failed_models"]}), 502

    return jsonify({"reference_sheet": result["text"], "model_used": result["model_used"]})


if __name__ == "__main__":
    app.run(debug=True)
