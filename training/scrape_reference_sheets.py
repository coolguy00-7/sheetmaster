import argparse
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


USER_AGENT = "SheetmasterReferenceScraper/1.0 (+https://github.com/coolguy00-7/sheetmaster)"
HTML_TYPES = ("text/html", "application/xhtml+xml")
RELEVANCE_KEYWORDS = {
    "science olympiad",
    "reference sheet",
    "cheat sheet",
    "formula sheet",
    "study guide",
    "event notes",
    "forensics",
    "chem lab",
    "anatomy",
    "dynamic planet",
    "astronomy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Crawl and scrape Science Olympiad reference sheets into JSONL.")
    parser.add_argument("--seed-file", required=True, help="Text file with one seed URL per line.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--max-pages", type=int, default=250, help="Maximum number of pages to scrape.")
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Delay between requests.")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    parser.add_argument("--min-chars", type=int, default=900, help="Minimum extracted text length to keep.")
    parser.add_argument(
        "--allowed-domains",
        default="",
        help="Comma-separated domain allowlist. If empty, domains from seed URLs are used.",
    )
    parser.add_argument(
        "--same-domain-only",
        action="store_true",
        help="Restrict crawling to the exact domain(s) in the allowlist (no subdomains).",
    )
    return parser.parse_args()


def normalize_url(url):
    url = urldefrag(url.strip())[0]
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        return ""
    if parsed.scheme not in {"http", "https"}:
        return ""
    return url


def base_domain(hostname):
    if not hostname:
        return ""
    parts = hostname.lower().split(".")
    if len(parts) < 2:
        return hostname.lower()
    return ".".join(parts[-2:])


def load_seeds(path):
    seeds = []
    with open(path, "r", encoding="utf-8") as infile:
        for raw in infile:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            url = normalize_url(line)
            if url:
                seeds.append(url)
    return seeds


def is_allowed(url, allowed_domains, same_domain_only):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if same_domain_only:
        return host in allowed_domains
    return any(host == dom or host.endswith(f".{dom}") for dom in allowed_domains)


def get_robot_parser(url, cache):
    parsed = urlparse(url)
    host = f"{parsed.scheme}://{parsed.netloc}"
    if host in cache:
        return cache[host]
    rp = RobotFileParser()
    robots_url = urljoin(host, "/robots.txt")
    rp.set_url(robots_url)
    try:
        response = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if response.status_code >= 400:
            cache[host] = None
            return None
        rp.parse(response.text.splitlines())
    except Exception:
        cache[host] = None
        return None
    cache[host] = rp
    return rp


def text_is_relevant(title, text):
    haystack = f"{title}\n{text}".lower()
    return any(keyword in haystack for keyword in RELEVANCE_KEYWORDS)


def extract_text_and_links(html, current_url):
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""

    for tag_name in ["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    links = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(current_url, anchor["href"])
        href = normalize_url(href)
        if href:
            links.append(href)
    return title, text, links


def crawl(args):
    seeds = load_seeds(args.seed_file)
    if not seeds:
        raise ValueError("No valid seed URLs found.")

    if args.allowed_domains:
        allowed_domains = {d.strip().lower() for d in args.allowed_domains.split(",") if d.strip()}
    else:
        if args.same_domain_only:
            allowed_domains = {(urlparse(seed).hostname or "").lower() for seed in seeds}
        else:
            allowed_domains = {base_domain(urlparse(seed).hostname) for seed in seeds}
    allowed_domains = {d for d in allowed_domains if d}
    if not allowed_domains:
        raise ValueError("Could not derive allowed domains.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    queue = deque(seeds)
    visited = set()
    robots_cache = {}
    kept = 0

    with open(args.output, "w", encoding="utf-8") as outfile:
        while queue and kept < args.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if not is_allowed(url, allowed_domains, args.same_domain_only):
                continue

            rp = get_robot_parser(url, robots_cache)
            if rp is not None and not rp.can_fetch(USER_AGENT, url):
                continue

            try:
                response = session.get(url, timeout=args.timeout, allow_redirects=True)
            except requests.RequestException:
                time.sleep(args.delay_seconds)
                continue

            content_type = (response.headers.get("Content-Type") or "").lower()
            if response.status_code != 200 or not any(t in content_type for t in HTML_TYPES):
                time.sleep(args.delay_seconds)
                continue

            final_url = normalize_url(response.url)
            if not final_url:
                time.sleep(args.delay_seconds)
                continue
            if final_url != url and final_url in visited:
                time.sleep(args.delay_seconds)
                continue
            visited.add(final_url)

            try:
                title, text, links = extract_text_and_links(response.text, final_url)
            except Exception:
                time.sleep(args.delay_seconds)
                continue

            for link in links:
                if link not in visited:
                    queue.append(link)

            if len(text) < args.min_chars or not text_is_relevant(title, text):
                time.sleep(args.delay_seconds)
                continue

            row = {
                "url": final_url,
                "title": title,
                "text": text,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "source": "web_scrape",
            }
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            time.sleep(args.delay_seconds)

    return kept


def main():
    args = parse_args()
    kept = crawl(args)
    print(f"Wrote {kept} scraped pages to {args.output}")


if __name__ == "__main__":
    main()
