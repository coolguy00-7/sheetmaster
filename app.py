import os
import base64

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze-practice")
def analyze_practice():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Upload at least one text file."}), 400

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "Missing GEMINI_API_KEY in environment."}), 400

    max_files = 20
    max_chars_per_file = 12000
    max_total_chars = 90000
    max_binary_bytes_per_file = 8 * 1024 * 1024
    max_total_binary_bytes = 30 * 1024 * 1024
    allowed_extensions = {".txt", ".md", ".csv", ".rtf", ".pdf", ".png", ".jpg", ".jpeg"}

    if len(files) > max_files:
        return jsonify({"error": f"Too many files. Max allowed is {max_files}."}), 400

    text_files = []
    binary_parts = []
    total_chars = 0
    total_binary_bytes = 0

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
                        "Allowed: .txt, .md, .csv, .rtf"
                    )
                }
            ), 400

        raw = uploaded_file.read()
        if not raw:
            continue

        if ext in {".pdf", ".png", ".jpg", ".jpeg"}:
            file_size = len(raw)
            if file_size > max_binary_bytes_per_file:
                return jsonify(
                    {
                        "error": (
                            f"'{filename}' is too large. "
                            f"Max size per PDF/image is {max_binary_bytes_per_file // (1024 * 1024)}MB."
                        )
                    }
                ), 400

            total_binary_bytes += file_size
            if total_binary_bytes > max_total_binary_bytes:
                return jsonify(
                    {
                        "error": (
                            "Total PDF/image upload size is too large. "
                            f"Keep combined binary files under {max_total_binary_bytes // (1024 * 1024)}MB."
                        )
                    }
                ), 400

            mime_type = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
            }[ext]
            encoded = base64.b64encode(raw).decode("ascii")
            binary_parts.append(
                {
                    "filename": filename,
                    "part": {"inline_data": {"mime_type": mime_type, "data": encoded}},
                }
            )
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
            except UnicodeDecodeError:
                return jsonify({"error": f"Could not decode '{filename}' as text."}), 400

        text = text.strip()
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

        text_files.append({"filename": filename, "text": text})

    if not text_files and not binary_parts:
        return jsonify({"error": "No readable text content found in uploaded files."}), 400

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    file_blocks = "\n\n".join(
        [
            f"FILE: {item['filename']}\n---\n{item['text']}"
            for item in text_files
        ]
    )
    binary_file_list = ", ".join(item["filename"] for item in binary_parts) or "None"
    analysis_prompt = f"""
You are analyzing multiple student practice materials.

Goal:
Tell me exactly what is being covered across these files (including text files, PDFs, and images).

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

PDF/Image files included:
{binary_file_list}

Text file contents:
{file_blocks or "None"}
""".strip()

    parts = [{"text": analysis_prompt}]
    for item in binary_parts:
        parts.append({"text": f"FILE: {item['filename']}"})
        parts.append(item["part"])

    payload = {"contents": [{"parts": parts}]}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        return jsonify({"error": f"Gemini request failed: {exc}"}), 502

    text = ""
    candidates = body.get("candidates") or []
    if candidates:
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        if parts:
            text = parts[0].get("text", "")

    if not text:
        return jsonify({"error": "Gemini returned no text response.", "raw": body}), 502

    return jsonify(
        {
            "response": text,
            "files_analyzed": [
                *[item["filename"] for item in text_files],
                *[item["filename"] for item in binary_parts],
            ],
            "total_files": len(text_files) + len(binary_parts),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
