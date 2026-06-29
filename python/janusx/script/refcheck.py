from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog


@dataclass
class RisEntry:
    index: int
    start_line: int
    raw: str
    fields: dict[str, list[str]]

    @property
    def title(self) -> str:
        return (self.fields.get("TI") or [""])[0]

    @property
    def journal(self) -> str:
        return (self.fields.get("T2") or [""])[0]

    @property
    def year(self) -> str:
        return (self.fields.get("PY") or [""])[0]

    @property
    def volume(self) -> str:
        return (self.fields.get("VL") or [""])[0]

    @property
    def sp(self) -> str:
        return (self.fields.get("SP") or [""])[0]

    @property
    def doi(self) -> str:
        raw = (self.fields.get("DO") or [""])[0].strip()
        return raw.removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    @property
    def authors(self) -> list[str]:
        return list(self.fields.get("AU") or [])


def _parse_ris(path: Path) -> list[RisEntry]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    entries: list[RisEntry] = []
    start = None
    buf: list[str] = []
    idx = 0
    for lineno, line in enumerate(lines, start=1):
        if line.startswith("TY  - "):
            if buf and start is not None:
                idx += 1
                entries.append(RisEntry(idx, start, "\n".join(buf), _parse_fields(buf)))
            start = lineno
            buf = [line]
        elif buf:
            buf.append(line)
    if buf and start is not None:
        idx += 1
        entries.append(RisEntry(idx, start, "\n".join(buf), _parse_fields(buf)))
    return entries


def _parse_fields(lines: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    last_tag: str | None = None
    for line in lines:
        m = re.match(r"^([A-Z0-9]{2})  - (.*)$", line)
        if m:
            tag, value = m.groups()
            out.setdefault(tag, []).append(value)
            last_tag = tag
        elif last_tag is not None:
            out[last_tag][-1] += " " + line.strip()
    return out


def _normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", text)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("—", "-").replace("–", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 JanusX refcheck"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _score_match(query_title: str, candidate: dict) -> int:
    cand_title = candidate.get("title") or candidate.get("display_name") or ""
    a = _normalize_text(query_title)
    b = _normalize_text(cand_title)
    score = 0
    if a == b:
        score += 1000
    if a in b or b in a:
        score += 200
    sa = set(a.split())
    sb = set(b.split())
    if sa and sb:
        score += int(100 * len(sa & sb) / max(len(sa), len(sb)))
    score += int(candidate.get("relevance_score") or 0)
    return score


def _fetch_openalex_record(entry: RisEntry) -> dict | None:
    try:
        if entry.doi:
            url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(
                entry.doi, safe="/:.-"
            )
            return _fetch_json(url)
        url = "https://api.openalex.org/works?search=" + urllib.parse.quote(entry.title) + "&per-page=5"
        payload = _fetch_json(url)
        results = payload.get("results") or []
        if not results:
            return None
        return max(results, key=lambda cand: _score_match(entry.title, cand))
    except Exception:
        return None


def _format_pages(first: str | None, last: str | None) -> str:
    if first and last and first != last:
        return f"{first}–{last}"
    if first:
        return first
    return ""


def _local_issues(entry: RisEntry) -> list[str]:
    issues: list[str] = []
    authors = entry.authors
    if not authors:
        issues.append("missing authors")
    if any(a.strip().lower() == "others" for a in authors):
        issues.append("contains literal `others` author")
    if any(a.strip() in {"Manuscript Writing Group", "UK Biobank", "FinnGen"} for a in authors) and len(authors) <= 2:
        issues.append("group author only; likely incomplete author list")
    if "SP" not in entry.fields:
        issues.append("missing page/article number")
    if "N1" in entry.fields:
        issues.append("contains leftover note/encoding field `N1`")
    if any("\\&" in v for v in entry.fields.get("T2", [])):
        issues.append("journal contains escaped `\\&`")
    norm_seen: set[str] = set()
    for author in authors:
        key = _normalize_text(author)
        if key in norm_seen:
            issues.append(f"duplicate/near-duplicate author `{author}`")
            break
        norm_seen.add(key)
    return issues


def _online_issues(entry: RisEntry, record: dict | None) -> list[str]:
    if record is None:
        return ["could not resolve external metadata"]
    issues: list[str] = []
    journal = ((record.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    bib = record.get("biblio") or {}
    pages = _format_pages(bib.get("first_page"), bib.get("last_page"))
    if entry.year and str(record.get("publication_year") or "") and entry.year != str(record.get("publication_year")):
        issues.append(f"year differs: RIS `{entry.year}` vs external `{record.get('publication_year')}`")
    if entry.journal and journal:
        if _normalize_text(entry.journal) != _normalize_text(journal):
            issues.append(f"journal differs: RIS `{entry.journal}` vs external `{journal}`")
    if entry.sp and pages:
        if _normalize_text(entry.sp) != _normalize_text(pages):
            issues.append(f"pages/article number differs: RIS `{entry.sp}` vs external `{pages}`")
    ext_authors = [(a.get("author") or {}).get("display_name", "") for a in (record.get("authorships") or [])]
    if ext_authors:
        ris_norm = [_normalize_text(a) for a in entry.authors]
        ext_norm = [_normalize_text(a) for a in ext_authors]
        if len(entry.authors) < len(ext_authors) and (
            len(entry.authors) <= 2 or any(a in {"manuscript writing group", "uk biobank", "finngen"} for a in ris_norm)
        ):
            issues.append(f"author list appears truncated: RIS {len(entry.authors)} vs external {len(ext_authors)}")
        elif ris_norm and ext_norm:
            mismatch = sum(1 for a, b in zip(ris_norm, ext_norm) if a != b)
            if mismatch > 0 and mismatch >= max(1, min(len(ris_norm), len(ext_norm)) // 3):
                issues.append("author order/content differs materially from external metadata")
    return issues


def _render_markdown(path: Path, entries: list[RisEntry], issues: dict[int, list[str]]) -> str:
    lines: list[str] = []
    lines.append(f"# RIS Reference Check")
    lines.append("")
    lines.append(f"- File: `{path}`")
    lines.append(f"- Entries: {len(entries)}")
    lines.append(f"- Problem entries: {len(issues)}")
    lines.append("")
    if not issues:
        lines.append("No issues found.")
        return "\n".join(lines)
    for entry in entries:
        entry_issues = issues.get(entry.index)
        if not entry_issues:
            continue
        lines.append(f"## #{entry.index} {entry.title}")
        lines.append("")
        lines.append(f"- Line: {entry.start_line}")
        lines.append(f"- Authors: {', '.join(entry.authors) if entry.authors else '(none)'}")
        if entry.doi:
            lines.append(f"- DOI: `{entry.doi}`")
        for item in entry_issues:
            lines.append(f"- Issue: {item}")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx refcheck",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx refcheck -i test.ris",
                "jx refcheck -i refs.ris --online",
            ]
        ),
        description="Check RIS references and print a Markdown summary of suspicious entries.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input RIS file.")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Also compare each entry against DOI/OpenAlex metadata.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    path = Path(args.input).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RIS file not found: {path}")

    entries = _parse_ris(path)
    issues: dict[int, list[str]] = {}
    for entry in entries:
        found = _local_issues(entry)
        if args.online and found:
            found.extend(_online_issues(entry, _fetch_openalex_record(entry)))
        if found:
            seen: set[str] = set()
            uniq: list[str] = []
            for item in found:
                if item not in seen:
                    seen.add(item)
                    uniq.append(item)
            issues[entry.index] = uniq

    sys.stdout.write(_render_markdown(path, entries, issues))
    sys.stdout.write("\n")
    return 1 if issues else 0


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    raise SystemExit(main())
