#!/usr/bin/env python3
"""Weekly pain point miner.

Fetches public discussions, extracts corroborated pain points with OpenAI,
updates canonical markdown database, and stores dedupe state.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "sources.json"
STATE_PATH = ROOT / "data" / "state.json"
DB_PATH = ROOT / "data" / "pain_points.md"

SOURCE_WEIGHTS = {
    "reddit": 1.0,
    "x": 1.0,
    "hackernews": 1.0,
    "indiehackers": 1.0,
    "producthunt": 1.0,
    "forum": 0.9,
    "rss": 0.8,
}
CROSS_SOURCE_BONUS = 2.0
MAX_PER_RUN_SCORE_INCREASE = 15.0


@dataclass
class Discussion:
    source: str
    url: str
    title: str
    text: str
    author: str | None
    published_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PainPoint:
    pid: str
    title: str
    personas: list[str]
    tags: list[str]
    summary: str
    score: float
    score_breakdown: str
    evidence: list[dict[str, str]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_run": None, "seen_urls": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def http_get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    with urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_iso8601(raw: str) -> datetime:
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def filter_by_lookback(items: list[Discussion], lookback_days: int) -> list[Discussion]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[Discussion] = []
    for d in items:
        try:
            if parse_iso8601(d.published_at) >= cutoff:
                out.append(d)
        except ValueError:
            logging.warning("Skipping item with invalid date: %s", d.url)
    return out


def fetch_hn(cfg: dict[str, Any], lookback_days: int) -> list[Discussion]:
    discussions: list[Discussion] = []
    for query in cfg.get("queries", []):
        params = urlencode({"query": query, "tags": "story", "hitsPerPage": 40})
        url = f"https://hn.algolia.com/api/v1/search_by_date?{params}"
        payload = http_get_json(url)
        for hit in payload.get("hits", []):
            created = hit.get("created_at")
            story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            discussions.append(
                Discussion(
                    source="hackernews",
                    url=story_url,
                    title=hit.get("title") or "",
                    text=(hit.get("story_text") or "")[:4000],
                    author=hit.get("author"),
                    published_at=created,
                    metadata={"hn_id": hit.get("objectID")},
                )
            )
    return filter_by_lookback(discussions, lookback_days)


def fetch_rss_items(source: str, feed_url: str) -> list[Discussion]:
    req = Request(feed_url, headers={"User-Agent": "pain-point-miner/1.0"})
    with urlopen(req, timeout=20) as resp:
        raw = resp.read()
    root = ET.fromstring(raw)
    items: list[Discussion] = []
    channel_items = root.findall(".//item")
    if not channel_items:
        channel_items = root.findall("{http://www.w3.org/2005/Atom}entry")

    for item in channel_items:
        title = (item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link = item.findtext("link")
        if not link:
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            if atom_link is not None:
                link = atom_link.attrib.get("href")
        pub = (
            item.findtext("pubDate")
            or item.findtext("{http://www.w3.org/2005/Atom}updated")
            or item.findtext("{http://www.w3.org/2005/Atom}published")
        )
        if not pub:
            continue
        try:
            dt = datetime.strptime(pub[:31], "%a, %d %b %Y %H:%M:%S %z") if "," in pub else parse_iso8601(pub)
        except ValueError:
            try:
                dt = parse_iso8601(pub)
            except ValueError:
                continue
        desc = item.findtext("description") or item.findtext("{http://www.w3.org/2005/Atom}summary") or ""
        items.append(
            Discussion(
                source=source,
                url=(link or "").strip(),
                title=title,
                text=re.sub("<[^>]+>", " ", desc)[:4000],
                author=None,
                published_at=dt.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
            )
        )
    return items


def fetch_reddit(cfg: dict[str, Any], lookback_days: int) -> list[Discussion]:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "pain-point-miner/1.0")
    if not client_id or not client_secret:
        logging.warning("Reddit credentials missing; skipping reddit fetch")
        return []

    auth = Request(
        "https://www.reddit.com/api/v1/access_token",
        data=urlencode({"grant_type": "client_credentials"}).encode(),
        headers={
            "Authorization": "Basic " + __import__("base64").b64encode(f"{client_id}:{client_secret}".encode()).decode(),
            "User-Agent": user_agent,
        },
        method="POST",
    )
    with urlopen(auth, timeout=20) as resp:
        token = json.loads(resp.read().decode("utf-8"))["access_token"]

    headers = {"Authorization": f"Bearer {token}", "User-Agent": user_agent}
    discussions: list[Discussion] = []
    for sub in cfg.get("subreddits", []):
        url = f"https://oauth.reddit.com/r/{sub}/new?limit=50"
        payload = http_get_json(url, headers=headers)
        for child in payload.get("data", {}).get("children", []):
            d = child.get("data", {})
            discussions.append(
                Discussion(
                    source="reddit",
                    url=f"https://www.reddit.com{d.get('permalink','')}",
                    title=d.get("title") or "",
                    text=d.get("selftext") or "",
                    author=d.get("author"),
                    published_at=datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc).isoformat(),
                    metadata={"subreddit": sub, "score": d.get("score", 0)},
                )
            )
    return filter_by_lookback(discussions, lookback_days)


def fetch_x(cfg: dict[str, Any], lookback_days: int) -> list[Discussion]:
    bearer = os.getenv("X_BEARER_TOKEN")
    if not bearer:
        logging.warning("X bearer token missing; skipping X fetch")
        return []

    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    discussions: list[Discussion] = []
    for query in cfg.get("queries", []):
        params = urlencode(
            {
                "query": query,
                "max_results": 25,
                "tweet.fields": "created_at,author_id,text",
                "start_time": start,
            }
        )
        url = f"https://api.x.com/2/tweets/search/recent?{params}"
        payload = http_get_json(url, headers={"Authorization": f"Bearer {bearer}"})
        for tw in payload.get("data", []):
            discussions.append(
                Discussion(
                    source="x",
                    url=f"https://x.com/i/web/status/{tw['id']}",
                    title=(tw.get("text", "")[:90] + "...") if len(tw.get("text", "")) > 90 else tw.get("text", ""),
                    text=tw.get("text") or "",
                    author=tw.get("author_id"),
                    published_at=tw.get("created_at"),
                )
            )
    return discussions


def fetch_producthunt(cfg: dict[str, Any], lookback_days: int) -> list[Discussion]:
    token = os.getenv("PRODUCTHUNT_TOKEN")
    if not token:
        logging.warning("Product Hunt token missing; skipping Product Hunt fetch")
        return []

    discussions: list[Discussion] = []
    for topic in cfg.get("topics", []):
        query = {
            "query": """
            query($topic: String!) {
              posts(first: 20, topic: $topic) {
                edges {
                  node {
                    name
                    tagline
                    url
                    createdAt
                    comments(first: 5) { edges { node { body createdAt } } }
                  }
                }
              }
            }
            """,
            "variables": {"topic": topic},
        }
        payload = http_post_json(
            "https://api.producthunt.com/v2/api/graphql",
            query,
            headers={"Authorization": f"Bearer {token}"},
        )
        edges = payload.get("data", {}).get("posts", {}).get("edges", [])
        for edge in edges:
            node = edge.get("node", {})
            comments = "\n".join(c.get("node", {}).get("body", "") for c in node.get("comments", {}).get("edges", []))
            discussions.append(
                Discussion(
                    source="producthunt",
                    url=node.get("url") or "",
                    title=node.get("name") or "",
                    text=f"{node.get('tagline','')}\n{comments}"[:4000],
                    author=None,
                    published_at=node.get("createdAt") or utc_now_iso(),
                    metadata={"topic": topic},
                )
            )
    return filter_by_lookback(discussions, lookback_days)


def fetch_rss_bundle(source: str, feeds: list[str], lookback_days: int) -> list[Discussion]:
    out: list[Discussion] = []
    for feed in feeds:
        try:
            out.extend(fetch_rss_items(source, feed))
        except Exception as exc:
            logging.warning("Failed RSS feed %s (%s): %s", source, feed, exc)
    return filter_by_lookback(out, lookback_days)


def load_existing_db(path: Path) -> list[PainPoint]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=## PP-\d{4} - )", content)
    points: list[PainPoint] = []
    for block in blocks:
        if not block.startswith("## PP-"):
            continue
        lines = block.strip().splitlines()
        m = re.match(r"## (PP-\d{4}) - (.+)", lines[0].strip())
        if not m:
            continue
        pid, title = m.group(1), m.group(2)

        def field_list(name: str) -> list[str]:
            for line in lines:
                if line.startswith(f"- {name}: "):
                    raw = line.split(":", 1)[1].strip()
                    return [x.strip() for x in raw.strip("[]").split(",") if x.strip()]
            return []

        def field_text(name: str) -> str:
            for line in lines:
                if line.startswith(f"- {name}: "):
                    return line.split(":", 1)[1].strip()
            return ""

        evidence: list[dict[str, str]] = []
        for line in lines:
            em = re.match(r"\s*- \[(\d{4}-\d{2}-\d{2})\] \(([^)]+)\) (https?://\S+)", line)
            if em:
                evidence.append({"date": em.group(1), "source": em.group(2), "url": em.group(3)})

        try:
            score = float(field_text("score") or 0)
        except ValueError:
            score = 0.0

        points.append(
            PainPoint(
                pid=pid,
                title=title,
                personas=field_list("personas"),
                tags=field_list("tags"),
                summary=field_text("summary"),
                score=score,
                score_breakdown=field_text("score_breakdown"),
                evidence=evidence,
            )
        )
    return points


def write_db(path: Path, points: list[PainPoint]) -> None:
    points = sorted(points, key=lambda p: p.pid)
    lines: list[str] = [
        "# Pain Points Database",
        "",
        "Schema per entry:",
        "- ID: immutable identifier (PP-XXXX)",
        "- title: concise pain point name",
        "- personas: comma-separated list in brackets",
        "- tags: comma-separated list in brackets",
        "- summary: specific workflow breakdown",
        "- score: cumulative pain score",
        "- score_breakdown: human-readable scoring rationale",
        "- evidence: list of dated source URLs; each URL appears only once across DB",
        "",
    ]
    for p in points:
        lines.extend(
            [
                f"## {p.pid} - {p.title}",
                f"- personas: [{', '.join(p.personas)}]",
                f"- tags: [{', '.join(p.tags)}]",
                f"- summary: {p.summary}",
                f"- score: {p.score:.2f}",
                f"- score_breakdown: {p.score_breakdown}",
                "- evidence:",
            ]
        )
        for ev in sorted(p.evidence, key=lambda x: (x["date"], x["url"])):
            lines.append(f"  - [{ev['date']}] ({ev['source']}) {ev['url']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_llm_input(new_discussions: list[Discussion], points: list[PainPoint]) -> dict[str, Any]:
    return {
        "existing_pain_points": [
            {
                "id": p.pid,
                "title": p.title,
                "summary": p.summary,
                "personas": p.personas,
                "tags": p.tags,
                "evidence_urls": [e["url"] for e in p.evidence],
            }
            for p in points
        ],
        "discussions": [
            {
                "source": d.source,
                "url": d.url,
                "title": d.title,
                "text": d.text[:3000],
                "published_at": d.published_at,
            }
            for d in new_discussions
        ],
    }


def call_openai_extract(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "new_pain_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "personas": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                        "evidence_urls": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "personas", "tags", "summary", "evidence_urls"],
                },
            },
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "additional_evidence_urls": {"type": "array", "items": {"type": "string"}},
                        "summary_patch": {"type": "string"},
                    },
                    "required": ["id", "additional_evidence_urls", "summary_patch"],
                },
            },
            "skipped_urls": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["new_pain_points", "updates", "skipped_urls"],
    }

    prompt = (
        "Extract only evidence-backed recurring pain points from discussions. "
        "Skip spam, promo, bots, vague complaints. Prefer specific workflow pain. "
        "Do not invent URLs. Reuse existing pain points when semantically similar."
    )

    response = http_post_json(
        "https://api.openai.com/v1/responses",
        {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload)}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "pain_point_updates",
                    "strict": True,
                    "schema": schema,
                }
            },
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    text = response.get("output", [{}])[0].get("content", [{}])[0].get("text")
    if not text:
        raise RuntimeError("OpenAI response missing structured text")
    return json.loads(text)


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def next_id(points: list[PainPoint]) -> str:
    if not points:
        return "PP-0001"
    n = max(int(p.pid.split("-")[1]) for p in points) + 1
    return f"PP-{n:04d}"


def compute_score_delta(new_evidence: list[dict[str, str]]) -> tuple[float, str]:
    if not new_evidence:
        return 0.0, "no new corroboration"
    unique_urls = {normalize_url(e["url"]) for e in new_evidence}
    unique_sources = {e["source"] for e in new_evidence}
    base = sum(SOURCE_WEIGHTS.get(src, 0.8) for src in unique_sources)
    delta = base + len(unique_urls) * 0.5
    if len(unique_sources) > 1:
        delta += CROSS_SOURCE_BONUS
    delta = min(delta, MAX_PER_RUN_SCORE_INCREASE)
    return round(delta, 2), f"+{delta:.2f} ({len(unique_urls)} unique URLs, {len(unique_sources)} sources)"


def apply_updates(
    points: list[PainPoint],
    llm_out: dict[str, Any],
    discussions_by_url: dict[str, Discussion],
) -> tuple[list[PainPoint], bool]:
    changed = False
    by_id = {p.pid: p for p in points}
    urls_in_db = {normalize_url(e["url"]) for p in points for e in p.evidence}

    for upd in llm_out.get("updates", []):
        pid = upd["id"]
        if pid not in by_id:
            continue
        point = by_id[pid]
        additions: list[dict[str, str]] = []
        for url in upd.get("additional_evidence_urls", []):
            nu = normalize_url(url)
            disc = discussions_by_url.get(nu)
            if not disc or nu in urls_in_db:
                continue
            additions.append({"date": disc.published_at[:10], "source": disc.source, "url": disc.url})
            urls_in_db.add(nu)
        if additions:
            point.evidence.extend(additions)
            delta, breakdown = compute_score_delta(additions)
            point.score = round(point.score + delta, 2)
            point.score_breakdown = breakdown
            changed = True
        patch = (upd.get("summary_patch") or "").strip()
        if patch and patch.lower() != "none":
            point.summary = f"{point.summary} {patch}".strip()
            changed = True

    for np in llm_out.get("new_pain_points", []):
        evidence: list[dict[str, str]] = []
        for url in np.get("evidence_urls", []):
            nu = normalize_url(url)
            disc = discussions_by_url.get(nu)
            if not disc or nu in urls_in_db:
                continue
            evidence.append({"date": disc.published_at[:10], "source": disc.source, "url": disc.url})
            urls_in_db.add(nu)
        if not evidence:
            continue
        delta, breakdown = compute_score_delta(evidence)
        points.append(
            PainPoint(
                pid=next_id(points),
                title=np["title"].strip(),
                personas=sorted(set(np.get("personas", []))),
                tags=sorted(set(np.get("tags", []))),
                summary=np["summary"].strip(),
                score=delta,
                score_breakdown=breakdown,
                evidence=evidence,
            )
        )
        changed = True

    return points, changed


def collect_discussions(config: dict[str, Any], seen_urls: set[str]) -> tuple[list[Discussion], dict[str, int]]:
    lookback = int(config.get("lookback_days", 7))
    source_cfg = config.get("sources", {})
    out: list[Discussion] = []
    stats: dict[str, int] = {}

    jobs = [
        ("hackernews", lambda: fetch_hn(source_cfg.get("hackernews", {}), lookback)),
        ("reddit", lambda: fetch_reddit(source_cfg.get("reddit", {}), lookback)),
        ("x", lambda: fetch_x(source_cfg.get("x", {}), lookback)),
        ("producthunt", lambda: fetch_producthunt(source_cfg.get("producthunt", {}), lookback)),
        ("indiehackers", lambda: fetch_rss_bundle("indiehackers", source_cfg.get("indiehackers", {}).get("feeds", []), lookback)),
        ("forum", lambda: fetch_rss_bundle("forum", source_cfg.get("forums", {}).get("feeds", []), lookback)),
    ]

    for name, fn in jobs:
        try:
            items = fn()
            unique_new = []
            for d in items:
                nu = normalize_url(d.url)
                if not nu or nu in seen_urls:
                    continue
                unique_new.append(d)
            out.extend(unique_new)
            stats[name] = len(unique_new)
            logging.info("Fetched %d new discussions from %s", len(unique_new), name)
        except Exception as exc:
            stats[name] = 0
            logging.warning("Source failure (%s): %s", name, exc)

    dedup: dict[str, Discussion] = {}
    for d in out:
        dedup[normalize_url(d.url)] = d
    return list(dedup.values()), stats


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(message)s")

    config = load_config(CONFIG_PATH)
    state = load_state(STATE_PATH)
    seen_urls = {normalize_url(u) for u in state.get("seen_urls", [])}

    points = load_existing_db(DB_PATH)
    new_discussions, stats = collect_discussions(config, seen_urls)

    if not new_discussions:
        logging.info("No new discussions found; exiting without file changes")
        return 0

    discussions_by_url = {normalize_url(d.url): d for d in new_discussions}
    llm_in = build_llm_input(new_discussions, points)

    try:
        llm_out = call_openai_extract(llm_in)
    except Exception as exc:
        logging.error("OpenAI extraction failed: %s", exc)
        return 1

    points, changed = apply_updates(points, llm_out, discussions_by_url)

    if not changed:
        logging.info("LLM returned no applicable updates after validation")
        return 0

    write_db(DB_PATH, points)

    new_seen = sorted(seen_urls.union(set(discussions_by_url.keys())))
    state["seen_urls"] = new_seen
    state["last_run"] = utc_now_iso()
    state["last_source_counts"] = stats
    save_state(STATE_PATH, state)

    logging.info("Updated DB with %d pain points", len(points))
    return 0


if __name__ == "__main__":
    sys.exit(main())
