# Training an LLM for Science Olympiad Reference Sheets

This project now supports two generation backends:
- `huggingface` (default hosted generation)
- `local` (your fine-tuned local model + optional LoRA adapter)

Set with `REFERENCE_GENERATION_MODE` in `.env`:
- `huggingface`
- `local`
- `auto` (try local first, then Hugging Face)

For highest quality on free models, use:

```dotenv
REFERENCE_GENERATION_MODE=huggingface
REFERENCE_QUALITY_MODE=high
HF_API_TOKEN=your_hf_token
HF_ANALYSIS_MODEL=Qwen/Qwen3.5-27B
HF_ANALYSIS_FALLBACK_MODELS=Qwen/Qwen3.5-35B-A3B,Qwen/Qwen2.5-72B-Instruct
HF_REFERENCE_MODEL=Qwen/Qwen2.5-72B-Instruct
HF_REFERENCE_FALLBACK_MODELS=Qwen/Qwen3.5-35B-A3B,Qwen/Qwen3.5-27B
HF_REFERENCE_CANDIDATE_MODELS=Qwen/Qwen2.5-72B-Instruct,Qwen/Qwen3.5-35B-A3B,Qwen/Qwen3.5-27B
HF_REFERENCE_MAX_CANDIDATES=3
HF_CRITIQUE_MODEL=Qwen/Qwen3.5-27B
HF_CRITIQUE_FALLBACK_MODELS=Qwen/Qwen3.5-35B-A3B
HF_JUDGE_MODEL=Qwen/Qwen3.5-27B
HF_JUDGE_FALLBACK_MODELS=Qwen/Qwen3.5-35B-A3B
```

## 1) Build your dataset

Raw format (`.jsonl`, one object per line):
- `analysis` (required)
- `reference_sheet` (required)
- `event`, `division`, `source` (optional metadata)

A sample file is included:
- `training/data/reference_sheet_samples.jsonl`

Prepare training file:

```bash
python3 training/prepare_dataset.py \
  --input training/data/reference_sheet_samples.jsonl \
  --output training/data/train_prepared.jsonl
```

## 1a) Scrape online reference material (optional)

You can collect candidate source pages with:

```bash
python3 training/scrape_reference_sheets.py \
  --seed-file training/data/seed_urls.txt \
  --output training/data/scraped_pages.jsonl \
  --max-pages 250 \
  --delay-seconds 1.0
```

Tips:
- Use only sources where crawling/reuse is permitted.
- Respect `robots.txt` and site terms.
- Add trusted domains with `--allowed-domains`.
- Use `--same-domain-only` for stricter crawling boundaries.

Note: scraped pages are not directly train pairs. You still need to convert them into
`analysis -> reference_sheet` examples (manually or with a curation pipeline).

## 2) Install training dependencies

```bash
python3 -m pip install -r requirements-train.txt
```

## 3) Train a LoRA adapter

Example:

```bash
python3 training/train_lora.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --train-file training/data/train_prepared.jsonl \
  --output-dir training/checkpoints/mistral-so-refsheet-lora \
  --num-epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-seq-len 4096
```

For real quality, use a much larger dataset (hundreds or thousands of examples).

## 4) Plug trained model into app

Add to `.env`:

```dotenv
REFERENCE_GENERATION_MODE=local
LOCAL_REFERENCE_MODEL=mistralai/Mistral-7B-Instruct-v0.3
LOCAL_REFERENCE_ADAPTER=training/checkpoints/mistral-so-refsheet-lora
LOCAL_REFERENCE_MAX_NEW_TOKENS=2800
LOCAL_REFERENCE_TEMPERATURE=0.2
LOCAL_REFERENCE_TOP_P=0.95
```

Then run Flask app as usual:

```bash
python3 app.py
```

The `/api/generate-reference-sheet` endpoint will use your local tuned model.

## 5) Quick held-out eval

```bash
python3 training/evaluate.py \
  --input training/data/reference_sheet_samples.jsonl \
  --output training/data/eval_outputs.jsonl
```

This writes generated outputs for manual review against gold sheets.
