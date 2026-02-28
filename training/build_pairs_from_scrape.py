import argparse
import json
import re
from collections import Counter
from urllib.parse import urlparse


KEYWORD_SET = {
    "anatomy",
    "astronomy",
    "biology",
    "chemistry",
    "dynamic planet",
    "ecology",
    "forensics",
    "formula",
    "lab",
    "meteorology",
    "olympiad",
    "physics",
    "science",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "into",
    "their",
    "about",
    "also",
    "such",
    "than",
    "when",
    "where",
    "what",
    "which",
    "while",
    "because",
    "using",
    "used",
    "been",
    "being",
    "over",
    "under",
    "between",
    "most",
    "many",
    "more",
    "some",
    "other",
    "these",
    "those",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create weakly-supervised training pairs from scraped pages.")
    parser.add_argument("--input", required=True, help="Scraped JSONL with fields: url, title, text.")
    parser.add_argument("--output", required=True, help="Output JSONL with analysis/reference_sheet pairs.")
    parser.add_argument("--min-text-chars", type=int, default=1200, help="Drop pages shorter than this.")
    parser.add_argument("--max-examples", type=int, default=200, help="Limit number of generated examples.")
    return parser.parse_args()


def clean_line(line):
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) < 30:
        return ""
    if line.lower().startswith(("from wikipedia", "retrieved from", "categories", "references", "external links")):
        return ""
    if re.fullmatch(r"[\W\d_]+", line):
        return ""
    return line


def split_sentences(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 45]


def top_topics(text, limit=8):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(tokens)
    ranked = [tok for tok, _ in counts.most_common(40)]
    picks = []
    for tok in ranked:
        if tok in KEYWORD_SET or any(k in tok for k in ("chem", "phys", "bio", "astro", "foren", "anatom")):
            picks.append(tok)
        if len(picks) >= limit:
            break
    if len(picks) < 4:
        picks = ranked[:limit]
    return picks[:limit]


def build_analysis(title, text, url):
    topics = top_topics(text)
    lines = text.splitlines()
    cleaned = [clean_line(x) for x in lines]
    cleaned = [x for x in cleaned if x]
    excerpt = cleaned[:10]
    topic_text = ", ".join(topics) if topics else "general science concepts"
    evidence = "; ".join(excerpt[:4]) if excerpt else title
    return (
        f"Covered topics: {topic_text}. "
        f"Skills practiced: content extraction, concept linking, rule recall, fast lookup. "
        f"Frequency map: high focus on core concepts shown in page sections. "
        f"Missing or weak areas: event-specific equations, worked practice cases, test-day heuristics. "
        f"Source page: {title} ({urlparse(url).netloc}). Evidence: {evidence[:700]}"
    )


def build_reference_sheet(title, text):
    topics = top_topics(text, limit=10)
    sents = split_sentences(text)
    core = sents[:18]
    topic_map = " | ".join(topics[:8]) if topics else "science | concepts | methods"

    body_lines = []
    body_lines.append(f"TOPIC MAP: {topic_map}")
    body_lines.append("CORE FACTS/RULES:")
    for sent in core[:8]:
        body_lines.append(f"- {sent[:180]}")
    body_lines.append("COMMON TRAPS:")
    body_lines.append("- Confusing definitions across adjacent topics.")
    body_lines.append("- Memorizing lists without condition/context checks.")
    body_lines.append("- Skipping unit/state assumptions in applied questions.")
    body_lines.append("FAST SOLVING PATTERNS:")
    body_lines.append("- Identify topic first, then map to a single governing rule.")
    body_lines.append("- Write knowns/unknowns in compact notation before solving.")
    body_lines.append("- Eliminate options that violate core constraints.")
    body_lines.append("MINI WORKED EXAMPLES:")
    for sent in core[8:12]:
        body_lines.append(f"- Example seed: {sent[:160]}")
    body_lines.append("LAST-MINUTE CHECKLIST:")
    body_lines.append("- Key terms and symbols reviewed")
    body_lines.append("- Typical edge cases reviewed")
    body_lines.append("- High-yield rule exceptions reviewed")
    body_lines.append(f"- Source title cross-check: {title}")
    return "\n".join(body_lines)


def main():
    args = parse_args()
    written = 0

    with open(args.input, "r", encoding="utf-8") as infile, open(args.output, "w", encoding="utf-8") as outfile:
        for line in infile:
            if written >= args.max_examples:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            if len(text) < args.min_text_chars or not title or not url:
                continue

            analysis = build_analysis(title, text, url)
            reference_sheet = build_reference_sheet(title, text)
            if not analysis or not reference_sheet:
                continue

            out = {
                "event": "mixed_science_olympiad",
                "division": "B/C",
                "source": "weak_supervision_web_scrape",
                "analysis": analysis,
                "reference_sheet": reference_sheet,
                "url": url,
                "title": title,
            }
            outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} weakly-supervised pairs to {args.output}")


if __name__ == "__main__":
    main()
