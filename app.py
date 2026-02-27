import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/gemini")
def gemini_chat():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "Missing GEMINI_API_KEY in environment."}), 400

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

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

    return jsonify({"response": text})


if __name__ == "__main__":
    app.run(debug=True)
