from __future__ import annotations

import fcntl
import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

from pralph.models import IterationResult, PhaseState, Story, StoryStatus

# Map file patterns to domain names for auto-detection.
# Each entry: (glob_pattern, domain_name)
_DOMAIN_DETECTION_RULES: list[tuple[str, str]] = [
    # Languages
    ("*.swift", "swift-ios"),
    ("Package.swift", "swift-ios"),
    ("*.xcodeproj", "swift-ios"),
    ("*.xcworkspace", "swift-ios"),
    ("Cargo.toml", "rust"),
    ("*.rs", "rust"),
    ("*.go", "go"),
    ("go.mod", "go"),
    ("*.py", "python"),
    ("pyproject.toml", "python"),
    ("*.ts", "typescript"),
    ("*.tsx", "typescript"),
    ("package.json", "typescript"),
    ("*.js", "javascript"),
    ("*.jsx", "javascript"),
    ("*.java", "java"),
    ("*.kt", "kotlin"),
    ("*.cs", "csharp"),
    ("*.rb", "ruby"),
    ("Gemfile", "ruby"),
    ("*.ex", "elixir"),
    ("*.exs", "elixir"),
    ("*.zig", "zig"),
    ("*.cpp", "cpp"),
    ("*.c", "c"),
    ("CMakeLists.txt", "cmake"),
    # Platforms / tools
    ("Dockerfile", "docker"),
    ("docker-compose.yml", "docker"),
    ("docker-compose.yaml", "docker"),
    ("terraform/*.tf", "terraform"),
    ("*.tf", "terraform"),
    ("serverless.yml", "serverless"),
    ("template.yaml", "cloudformation"),
    ("cdk.json", "aws-cdk"),
    ("pulumi.yaml", "pulumi"),
    ("k8s/*.yaml", "kubernetes"),
    ("*.proto", "protobuf"),
    ("flutter/pubspec.yaml", "flutter"),
    ("pubspec.yaml", "flutter"),
    ("*.dart", "flutter"),
    ("android/build.gradle", "android"),
    ("Podfile", "cocoapods"),
]

# Secondary tag-to-domain associations for common keywords that aren't exact
# domain names but strongly imply one.
_TAG_DOMAIN_HINTS: dict[str, str] = {
    "container": "docker",
    "dockerfile": "docker",
    "docker-compose": "docker",
    "pip": "python",
    "pytest": "python",
    "pyproject": "python",
    "npm": "typescript",
    "yarn": "typescript",
    "node": "typescript",
    "cargo": "rust",
    "crate": "rust",
    "helm": "kubernetes",
    "k8s": "kubernetes",
    "pod": "kubernetes",
    "gradle": "java",
    "maven": "java",
    "cocoapods": "cocoapods",
    "xcode": "swift-ios",
    "swift-package": "swift-ios",
    "flutter": "flutter",
    "dart": "flutter",
    "tf": "terraform",
    "hcl": "terraform",
}

# Known error prefixes/substrings mapped to domains.
_ERROR_DOMAIN_HINTS: list[tuple[str, str]] = [
    ("ModuleNotFoundError", "python"),
    ("ImportError", "python"),
    ("SyntaxError", "python"),
    ("IndentationError", "python"),
    ("cannot find module", "typescript"),
    ("Module not found", "typescript"),
    ("TS2305", "typescript"),
    ("TS2307", "typescript"),
    ("ReferenceError", "javascript"),
    ("cargo build", "rust"),
    ("rustc", "rust"),
    ("go build", "go"),
    ("undefined reference", "cpp"),
]


def _infer_solution_domains(
    related_files: list[str],
    tags: list[str],
    error_signature: str,
    available_domains: list[str],
) -> set[str]:
    """Infer which domain(s) a solution applies to from its metadata.

    Returns the subset of *available_domains* that match.  Returns an empty set
    if nothing could be inferred (caller should fall back to all domains).
    """
    import fnmatch as _fnmatch

    all_domains_set = set(available_domains)
    matched: set[str] = set()

    # A) File-extension matching via _DOMAIN_DETECTION_RULES
    for fpath in related_files:
        fname = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath
        for pattern, domain in _DOMAIN_DETECTION_RULES:
            if "/" in pattern:
                continue  # skip path-based rules
            if _fnmatch.fnmatch(fname, pattern):
                matched.add(domain)

    # B) Tag matching — exact domain name or hint lookup
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in all_domains_set:
            matched.add(tag_lower)
        if tag_lower in _TAG_DOMAIN_HINTS:
            matched.add(_TAG_DOMAIN_HINTS[tag_lower])

    # C) Error signature matching
    if error_signature:
        sig_lower = error_signature.lower()
        for pattern, domain in _ERROR_DOMAIN_HINTS:
            if pattern.lower() in sig_lower:
                matched.add(domain)

    # D) Intersect with available domains
    return matched & all_domains_set


def _infer_domains_llm(
    *,
    content: str,
    title: str,
    category: str,
    tags: list[str],
    error_signature: str,
    available_domains: list[str],
) -> set[str]:
    """Ask Haiku to infer domain(s) from solution content.

    Returns the subset of *available_domains* that match, or an empty set on
    failure (caller should fall back to all domains).
    """
    from pralph.parser import extract_json_from_text
    from pralph.prompts.compact import INFER_DOMAIN_PROMPT
    from pralph.runner import run_with_retry

    prompt = INFER_DOMAIN_PROMPT
    prompt = prompt.replace("{{available_domains}}", ", ".join(available_domains))
    prompt = prompt.replace("{{title}}", title)
    prompt = prompt.replace("{{category}}", category)
    prompt = prompt.replace("{{tags}}", ", ".join(tags))
    prompt = prompt.replace("{{error_signature}}", error_signature or "(none)")
    # Cap content to avoid blowing context for a cheap inference call
    truncated = content[:4000] if len(content) > 4000 else content
    prompt = prompt.replace("{{content}}", truncated)

    result = run_with_retry(prompt, model="haiku", timeout=30, max_retries=1)
    if not result.success:
        return set()

    parsed = extract_json_from_text(result.result)
    if not isinstance(parsed, dict) or "domains" not in parsed:
        return set()

    domains_set = set(available_domains)
    return {d for d in parsed["domains"] if isinstance(d, str) and d in domains_set}


def _safe_resolve(base: Path, untrusted: str) -> Path | None:
    """Resolve *untrusted* relative to *base*, returning None if it escapes."""
    resolved = (base / untrusted).resolve()
    if not resolved.is_relative_to(base.resolve()):
        return None
    return resolved


@contextmanager
def _index_lock(index_path: Path):
    """Acquire an exclusive file lock next to *index_path*.

    Both solution saves and index compaction use this lock so that compaction
    blocks new writes and vice-versa.  The lock is blocking (waits until
    released).
    """
    lock_path = index_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = lock_path.open("w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


class _NegStr:
    """Wrap a string so it sorts in reverse (newest first for ISO timestamps)."""

    __slots__ = ("s",)

    def __init__(self, s: str) -> None:
        self.s = s

    def __lt__(self, other: "_NegStr") -> bool:
        return self.s > other.s

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _NegStr) and self.s == other.s

    def __hash__(self) -> int:
        return hash(self.s)


# USD per million tokens. Conservative estimates; refine if pricing shifts.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "haiku": (1.00, 5.00),
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
}

# Compaction tunables.
_COMPACT_CHUNK_SIZE = 30
_COMPACT_BODY_CAP = 3000  # per-file body chars sent to LLM


def _estimate_compaction_cost(
    entries: list[dict],
    solutions_base: Path,
    *,
    model: str = "haiku",
    chunk_size: int = _COMPACT_CHUNK_SIZE,
    marker: str | None = None,
) -> dict:
    """Rough pre-flight estimate for ``_compact_index`` over *entries*.

    Estimates one LLM call per batch ``_plan_compaction`` would send: one per
    multi-entry cluster (clusters > ``chunk_size`` are split), minus the
    clusters an incremental run skips because nothing in them is newer than
    *marker*. Token estimate is ``chars // 4`` for input; output is capped
    at 8000 tokens per call. Returns ``{calls, input_tokens, output_tokens, usd}``.
    """
    if not entries:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}

    in_per_mtok, out_per_mtok = _MODEL_PRICING.get(model, _MODEL_PRICING["haiku"])

    def _entry_chars(entry: dict) -> int:
        chars = len(json.dumps(entry))
        filename = entry.get("filename", "")
        if filename:
            path = _safe_resolve(solutions_base, filename)
            if path is not None and path.exists():
                try:
                    chars += min(path.stat().st_size, _COMPACT_BODY_CAP)
                except OSError:
                    pass
        return chars

    _, batches = _plan_compaction(entries, marker=marker, chunk_size=chunk_size)
    total_input_tokens = 0
    total_output_tokens = 0
    calls = 0
    for batch in batches:
        if len(batch) <= 1:
            continue  # singletons skip the LLM
        batch_chars = sum(_entry_chars(e) for e in batch)
        in_tok = batch_chars // 4
        out_tok = min(8000, max(500, in_tok // 2))
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        calls += 1

    usd = (
        (total_input_tokens / 1_000_000) * in_per_mtok
        + (total_output_tokens / 1_000_000) * out_per_mtok
    )
    return {
        "calls": calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "usd": round(usd, 4),
    }


def _read_index_entries(index_path: Path) -> list[dict]:
    """Read all valid JSON dict entries from a .jsonl index file."""
    if not index_path.exists():
        return []
    entries: list[dict] = []
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


def _atomic_write_index(index_path: Path, entries: list[dict]) -> None:
    """Atomic JSONL rewrite via tmp + os.replace."""
    tmp_path = index_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    os.replace(tmp_path, index_path)


def _compact_state_path(index_path: Path) -> Path:
    return index_path.parent / ".compact_state.json"


def _read_compact_state(index_path: Path) -> dict:
    p = _compact_state_path(index_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_compact_state(index_path: Path, state: dict) -> None:
    p = _compact_state_path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, p)


def _slugify(text: str) -> str:
    import re as _re

    slug = text.lower().strip()
    slug = _re.sub(r"[^\w\s-]", "", slug)
    slug = _re.sub(r"[\s_]+", "-", slug)
    slug = _re.sub(r"-+", "-", slug)
    return slug[:80].strip("-")


def _is_stable_filename(filename: str) -> bool:
    """True if ``filename`` already follows the ``category/{8hex}-slug.md`` shape."""
    import re as _re

    # category may contain - and word chars; require a hex prefix on the basename.
    return bool(_re.match(r"^[\w./-]+?/[0-9a-f]{8}-", filename or ""))


def _stable_filename(category: str, error_signature: str, title: str) -> str:
    """Filename whose name is stable across compactions.

    Built from a hash of (error_signature or title) plus a short readable slug.
    Two solutions describing the same incident produce the same filename, so a
    re-merge overwrites cleanly instead of orphaning files.
    """
    import hashlib

    cat = category or "misc"
    key = (error_signature or title or "untitled").lower().strip()
    if not key:
        key = "untitled"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    slug = _slugify(title or "untitled")[:40].strip("-") or "untitled"
    return f"{cat}/{h}-{slug}.md"


def _pre_prune_entries(
    entries: list[dict], solutions_base: Path
) -> tuple[list[dict], int, int]:
    """Return (valid, exact_dupes, pre_orphans).

    Drops exact filename duplicates (keep last) and entries whose file is
    missing on disk.
    """
    seen: dict[str, int] = {}
    for i, entry in enumerate(entries):
        seen[entry.get("filename", "")] = i
    deduped = [entries[i] for i in sorted(seen.values())]
    exact_dupes = len(entries) - len(deduped)

    valid: list[dict] = []
    pre_orphans = 0
    for entry in deduped:
        filename = entry.get("filename", "")
        if not filename:
            pre_orphans += 1
            continue
        path = _safe_resolve(solutions_base, filename)
        if path is not None and path.exists():
            valid.append(entry)
        else:
            pre_orphans += 1
    return valid, exact_dupes, pre_orphans


def _normalize_error_signature(s: str) -> str:
    import re as _re

    return _re.sub(r"\s+", " ", (s or "").strip().lower())


def _cluster_entries(entries: list[dict]) -> list[list[dict]]:
    """Group entries that likely describe the same incident.

    Uses a union-find over indices, joining pairs that share *any* of:
      * exact normalized error_signature
      * same category + tag Jaccard ≥ 0.6
      * same category + same story_id + same source_project
      * same category + same first 5 title tokens (lowercased)

    Returns multi-entry groups first (largest first), then singletons.
    """
    n = len(entries)
    if n <= 1:
        return [list(entries)] if entries else []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Index entries by deterministic keys. Hits → unions.
    by_sig: dict[str, list[int]] = defaultdict(list)
    by_cat_story_proj: dict[tuple, list[int]] = defaultdict(list)
    by_cat_title_prefix: dict[tuple, list[int]] = defaultdict(list)
    cat_to_indices: dict[str, list[int]] = defaultdict(list)
    tag_sets: list[set[str]] = []

    for i, e in enumerate(entries):
        cat = e.get("category", "") or ""
        sig = _normalize_error_signature(e.get("error_signature", ""))
        story = e.get("story_id", "") or ""
        proj = e.get("source_project", "") or ""
        title = (e.get("title", "") or "").lower()
        title_tokens = tuple(title.split()[:5])
        tags = {t.lower() for t in (e.get("tags") or []) if isinstance(t, str)}
        tag_sets.append(tags)
        cat_to_indices[cat].append(i)
        if sig:
            by_sig[sig].append(i)
        if cat and story and proj:
            by_cat_story_proj[(cat, story, proj)].append(i)
        if cat and title_tokens:
            by_cat_title_prefix[(cat, title_tokens)].append(i)

    for indices in by_sig.values():
        for j in indices[1:]:
            union(indices[0], j)
    for indices in by_cat_story_proj.values():
        for j in indices[1:]:
            union(indices[0], j)
    for indices in by_cat_title_prefix.values():
        for j in indices[1:]:
            union(indices[0], j)

    # Tag Jaccard ≥ 0.6, scoped to same category.
    for indices in cat_to_indices.values():
        if len(indices) < 2:
            continue
        # Skip if no tags exist to consider.
        for a_pos, i in enumerate(indices):
            tags_i = tag_sets[i]
            if not tags_i:
                continue
            for j in indices[a_pos + 1:]:
                tags_j = tag_sets[j]
                if not tags_j:
                    continue
                union_size = len(tags_i | tags_j)
                if union_size == 0:
                    continue
                jacc = len(tags_i & tags_j) / union_size
                if jacc >= 0.6:
                    union(i, j)

    groups: dict[int, list[dict]] = defaultdict(list)
    for i, entry in enumerate(entries):
        groups[find(i)].append(entry)

    multi = [g for g in groups.values() if len(g) > 1]
    singles = [g for g in groups.values() if len(g) == 1]
    multi.sort(key=lambda g: -len(g))
    return multi + singles


def _plan_compaction(
    entries: list[dict],
    *,
    marker: str | None = None,
    chunk_size: int = _COMPACT_CHUNK_SIZE,
) -> tuple[list[dict], list[list[dict]]]:
    """Decide what a compaction run sends to the LLM.

    Returns ``(skipped, batches)``. ``batches`` are the clusters to process,
    split into ``chunk_size`` pieces, multi-entry batches first (largest
    first). ``skipped`` are the entries of clusters an incremental run leaves
    alone: with a *marker* (the start time of the last fully successful run),
    a cluster is only worth another LLM call if it gained an entry since —
    everything else was already judged, merged or left distinct, back then.
    Entries without a ``created`` timestamp count as new. ``marker=None``
    (first run, or ``--full``) processes every cluster.
    """
    def _is_new(e: dict) -> bool:
        created = e.get("created") or ""
        return not marker or not created or created > marker

    def _sort_key(e: dict) -> tuple:
        return (e.get("category", ""), _NegStr(e.get("created") or ""), e.get("filename", ""))

    skipped: list[dict] = []
    batches: list[list[dict]] = []
    for cluster in _cluster_entries(entries):
        if not any(_is_new(e) for e in cluster):
            skipped.extend(cluster)
            continue
        cluster = sorted(cluster, key=_sort_key)
        for i in range(0, len(cluster), chunk_size):
            batches.append(cluster[i : i + chunk_size])
    batches.sort(key=lambda b: (len(b) <= 1, -len(b)))
    return skipped, batches


def _compact_chunk(
    chunk: list[dict],
    solutions_base: Path,
    *,
    model: str,
    verbose: bool,
    dangerously_skip_permissions: bool,
) -> tuple[list[dict], set[str], int, int, float, bool]:
    """Run one LLM compaction pass on *chunk*.

    Returns ``(new_entries, files_to_delete, merges_count, removed_count,
    cost, success)``. On failure, ``new_entries`` is the chunk unchanged,
    ``files_to_delete`` is empty, counts are zero, and ``success=False``.
    """
    from pralph.parser import extract_json_from_text
    from pralph.prompts.compact import COMPACT_INDEX_PROMPT
    from pralph.runner import run_with_retry

    if len(chunk) <= 1:
        return list(chunk), set(), 0, 0, 0.0, True

    index_json = json.dumps(chunk, indent=2)
    contents_parts: list[str] = []
    for entry in chunk:
        filename = entry.get("filename", "")
        path = _safe_resolve(solutions_base, filename)
        if path is None or not path.exists():
            continue
        body = path.read_text().strip()
        if len(body) > _COMPACT_BODY_CAP:
            body = body[:_COMPACT_BODY_CAP] + "\n\n(truncated)"
        contents_parts.append(f"### {filename}\n\n{body}")
    contents_text = "\n\n---\n\n".join(contents_parts)

    prompt = COMPACT_INDEX_PROMPT
    prompt = prompt.replace("{{index_entries}}", index_json)
    prompt = prompt.replace("{{solution_contents}}", contents_text)

    result = run_with_retry(
        prompt,
        model=model,
        timeout=120,
        verbose=verbose,
        dangerously_skip_permissions=dangerously_skip_permissions,
    )
    cost = result.cost_usd

    if not result.success:
        return list(chunk), set(), 0, 0, cost, False

    parsed = extract_json_from_text(result.result)
    if not isinstance(parsed, dict) or "entries" not in parsed:
        return list(chunk), set(), 0, 0, cost, False

    new_entries_raw = parsed["entries"]
    merges = parsed.get("merges", []) or []
    removed = parsed.get("removed", []) or []

    failed = (list(chunk), set(), 0, 0, cost, False)
    if not isinstance(new_entries_raw, list):
        return failed

    sources: dict[str, dict] = {e["filename"]: e for e in chunk if e.get("filename")}

    # Pass 1 — decide every destination before touching the disk, so a
    # response we can't apply safely leaves the chunk exactly as it was.
    planned: list[tuple[dict, Path, str]] = []
    new_filenames: set[str] = set()
    for entry in new_entries_raw:
        if not isinstance(entry, dict):
            continue
        content = entry.pop("content", "")
        if not isinstance(content, str):
            content = ""
        filename = entry.get("filename", "")
        if not isinstance(filename, str) or filename not in sources:
            # A new entry (the product of a merge): name it by the stable hash
            # so merging the same incident again lands on the same file.
            filename = _stable_filename(
                entry.get("category", "") or "misc",
                entry.get("error_signature", "") or "",
                entry.get("title", "") or "untitled",
            )
        # An entry that keeps the name of a file in this chunk is never
        # renamed: the LLM may have reworded its title or signature, and a
        # recomputed name would point the index at a file that doesn't exist.
        resolved = _safe_resolve(solutions_base, filename)
        if resolved is None:
            return failed
        if not content and not resolved.exists():
            return failed  # index entry without a file
        entry["filename"] = filename
        planned.append((entry, resolved, content))
        new_filenames.add(filename)

    merged_sources: set[str] = set()
    for merge in merges:
        if isinstance(merge, dict):
            for src in merge.get("sources", []) or []:
                if isinstance(src, str):
                    merged_sources.add(src)
    removed_filenames: set[str] = {
        rem.get("filename", "") for rem in removed if isinstance(rem, dict)
    }
    # Only files of this chunk can be superseded — never trust a filename the
    # LLM made up to pick what gets deleted.
    accounted = (merged_sources | removed_filenames) & set(sources)
    files_to_delete = accounted - new_filenames

    # Pass 2 — write.
    final_entries: list[dict] = []
    for entry, resolved, content in planned:
        if content:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
        final_entries.append(entry)

    # Entries the LLM neither returned nor declared merged/removed stay as they are.
    for filename, source in sources.items():
        if filename not in new_filenames and filename not in accounted:
            final_entries.append(source)

    return final_entries, files_to_delete, len(merges), len(removed), cost, True


def _maybe_merge_on_save(
    new_entry: dict,
    index_path: Path,
    solutions_base: Path,
    *,
    model: str = "haiku",
    verbose: bool = False,
) -> dict | None:
    """Best-effort merge of *new_entry* against existing candidates.

    Gated by ``PRALPH_COMPACT_ON_SAVE=1``. Reads the index, finds entries that
    share a deterministic dedupe signal with the just-saved one, runs Haiku
    against just that small group, and rewrites the index if a merge happens.
    Returns the LLM batch result dict, or ``None`` if no work happened.

    All errors are swallowed: this is a maintenance hook, not a save guarantee.
    """
    if os.environ.get("PRALPH_COMPACT_ON_SAVE") != "1":
        return None
    new_fn = new_entry.get("filename", "")
    if not new_fn:
        return None

    try:
        with _index_lock(index_path):
            entries = _read_index_entries(index_path)
        if len(entries) <= 1:
            return None

        clusters = _cluster_entries(entries)
        target_group: list[dict] | None = None
        for g in clusters:
            if any(e.get("filename") == new_fn for e in g):
                target_group = g
                break
        if target_group is None or len(target_group) <= 1:
            return None
        if len(target_group) > _COMPACT_CHUNK_SIZE:
            target_group = target_group[:_COMPACT_CHUNK_SIZE]

        new_entries, files_to_delete, merges_n, removed_n, cost, success = _compact_chunk(
            target_group,
            solutions_base,
            model=model,
            verbose=verbose,
            dangerously_skip_permissions=False,
        )
        if not success or not new_entries:
            return None

        with _index_lock(index_path):
            on_disk = _read_index_entries(index_path)
            target_filenames = {e.get("filename", "") for e in target_group}
            kept = [e for e in on_disk if e.get("filename", "") not in target_filenames]
            kept.extend(new_entries)
            _atomic_write_index(index_path, kept)
            for old_fn in files_to_delete:
                old_path = _safe_resolve(solutions_base, old_fn)
                if old_path is not None and old_path.exists():
                    try:
                        old_path.unlink()
                    except OSError:
                        pass

        return {
            "merged": merges_n,
            "removed": removed_n,
            "cost": cost,
            "candidates": len(target_group),
        }
    except Exception:
        # Compact-on-save is best-effort — never let it break a save.
        return None


class StateManager:
    def __init__(self, project_dir: str, *, domains: list[str] | None = None) -> None:
        self.project_dir = Path(project_dir)
        self.state_dir = self.project_dir / ".pralph"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._domain_override = domains
        self._detected_domains: list[str] | None = None
        # Callbacks fired when the backlog changes — see add_story_listener().
        self._story_listeners: list[Callable[[str, Story | None, dict], None]] = []

    # -- config --

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def get_config(self, key: str, default=None):
        """Resolve a config value: project .pralph/config.json > ~/.pralph/config.json > default."""
        project_cfg = self._load_config(self.state_dir / "config.json")
        if key in project_cfg:
            return project_cfg[key]
        home_cfg = self._load_config(self.home_dir / "config.json")
        if key in home_cfg:
            return home_cfg[key]
        return default

    @property
    def global_compound(self) -> bool:
        """Whether compound learnings should be saved globally."""
        return bool(self.get_config("global_compound", False))

    # -- file paths --

    @property
    def design_doc_path(self) -> Path:
        return self.state_dir / "design-doc.md"

    @property
    def guardrails_path(self) -> Path:
        return self.state_dir / "guardrails.md"

    @property
    def stories_path(self) -> Path:
        return self.state_dir / "stories.jsonl"

    @property
    def status_path(self) -> Path:
        return self.state_dir / "status.jsonl"

    @property
    def run_log_path(self) -> Path:
        return self.state_dir / "run-log.jsonl"

    @property
    def phase_state_path(self) -> Path:
        return self.state_dir / "phase-state.json"

    @property
    def phase1_analysis_path(self) -> Path:
        return self.state_dir / "phase1-analysis.json"

    @property
    def research_notes_path(self) -> Path:
        return self.state_dir / "research-notes.md"

    @property
    def ideas_path(self) -> Path:
        return self.state_dir / "ideas.md"

    def phase_prompt_path(self, phase: str) -> Path:
        return self.state_dir / f"{phase}-prompt.md"

    # -- spec-driven development --

    @property
    def spec_dir(self) -> Path:
        """Directory inside the repo that holds the specs (config ``spec_dir``, default ``specs/``)."""
        rel = self.get_config("spec_dir", "specs")
        if not isinstance(rel, str) or not rel.strip():
            rel = "specs"
        # The spec must live inside the project — ignore paths that escape it.
        return _safe_resolve(self.project_dir, rel) or (self.project_dir / "specs")

    @property
    def spec_baseline_path(self) -> Path:
        return self.state_dir / "spec-baseline.json"

    @property
    def spec_mode(self) -> bool:
        """Whether the project is spec-driven.

        Config ``spec_driven`` wins when set; otherwise the mode is on as soon
        as the repo contains a spec (``<spec_dir>/<name>/spec.md``).
        """
        configured = self.get_config("spec_driven", None)
        if configured is not None:
            return bool(configured)
        spec_dir = self.spec_dir
        return spec_dir.is_dir() and any(p.is_file() for p in spec_dir.glob("*/spec.md"))

    # -- linear --

    @property
    def linear_link_path(self) -> Path:
        return self.state_dir / "linear.json"

    @property
    def home_dir(self) -> Path:
        return Path.home() / ".pralph"

    @property
    def domains_path(self) -> Path:
        return self.state_dir / "domains.txt"

    # -- domain detection --

    def detect_domains(self) -> list[str]:
        """Return the active domains for this project.

        Resolution order: CLI override > .pralph/domains.txt > auto-detect from files.
        Results are cached for the lifetime of this StateManager instance.
        """
        if self._detected_domains is not None:
            return self._detected_domains

        if self._domain_override:
            self._detected_domains = self._domain_override
            return self._detected_domains

        # Project-level explicit config
        if self.domains_path.exists():
            raw = self.domains_path.read_text().strip()
            if raw:
                self._detected_domains = [d.strip() for d in raw.splitlines() if d.strip() and not d.strip().startswith("#")]
                return self._detected_domains

        # Auto-detect from project files (only scan top two levels for speed)
        import fnmatch as _fnmatch

        found: set[str] = set()
        try:
            entries = list(self.project_dir.iterdir())
        except OSError:
            self._detected_domains = []
            return self._detected_domains

        # Level 0 (project root)
        names_l0 = [e.name for e in entries]
        for pattern, domain in _DOMAIN_DETECTION_RULES:
            if "/" in pattern:
                continue  # skip nested patterns for level 0
            for name in names_l0:
                if _fnmatch.fnmatch(name, pattern):
                    found.add(domain)
                    break

        # Level 1 (immediate subdirs) — only check dirs, skip hidden/vendor
        _SKIP_DIRS = {".git", ".pralph", "node_modules", ".build", "build", "target", "vendor", "__pycache__", ".venv", "venv"}
        for entry in entries:
            if not entry.is_dir() or entry.name in _SKIP_DIRS or entry.name.startswith("."):
                continue
            try:
                subnames = [e.name for e in entry.iterdir()]
            except OSError:
                continue
            for pattern, domain in _DOMAIN_DETECTION_RULES:
                if "/" in pattern:
                    # e.g. "terraform/*.tf" — check if entry.name matches prefix
                    parts = pattern.split("/", 1)
                    if _fnmatch.fnmatch(entry.name, parts[0]):
                        for sn in subnames:
                            if _fnmatch.fnmatch(sn, parts[1]):
                                found.add(domain)
                                break
                else:
                    for sn in subnames:
                        if _fnmatch.fnmatch(sn, pattern):
                            found.add(domain)
                            break

        self._detected_domains = sorted(found)
        return self._detected_domains

    def read_phase_prompt(self, phase: str) -> str:
        # Project-level overrides home-level
        path = self.phase_prompt_path(phase)
        if path.exists():
            return path.read_text().strip()
        home_path = self.home_dir / f"{phase}-prompt.md"
        if home_path.exists():
            return home_path.read_text().strip()
        return ""

    def resolve_prompt_template(self, name: str, default: str) -> str:
        """Resolve a prompt template: project prompts/ > home prompts/ > built-in default."""
        project_path = self.state_dir / "prompts" / f"{name}.md"
        if project_path.exists():
            return project_path.read_text()
        home_path = self.home_dir / "prompts" / f"{name}.md"
        if home_path.exists():
            return home_path.read_text()
        return default

    @property
    def extra_tools_path(self) -> Path:
        return self.state_dir / "extra-tools.txt"

    def read_extra_tools(self) -> str:
        """Read project-level extra tools (one per line or comma-separated)."""
        if not self.extra_tools_path.exists():
            return ""
        raw = self.extra_tools_path.read_text().strip()
        # Normalize: support one-per-line or comma-separated
        tools = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
        return ",".join(tools)

    # -- review feedback --

    @property
    def review_feedback_dir(self) -> Path:
        return self.state_dir / "review-feedback"

    def review_feedback_path(self, story_id: str) -> Path:
        return self.review_feedback_dir / f"{story_id}.md"

    def write_review_feedback(self, story_id: str, feedback: str) -> None:
        self.review_feedback_dir.mkdir(parents=True, exist_ok=True)
        self.review_feedback_path(story_id).write_text(feedback)

    def read_review_feedback(self, story_id: str) -> str:
        path = self.review_feedback_path(story_id)
        if path.exists():
            return path.read_text().strip()
        return ""

    def clear_review_feedback(self, story_id: str) -> None:
        self.review_feedback_path(story_id).unlink(missing_ok=True)

    # -- claude session validation --

    def claude_session_exists(self, session_id: str) -> bool:
        """Check if a Claude session file exists on disk."""
        encoded = str(self.project_dir).replace("/", "-")
        path = Path.home() / ".claude" / "projects" / encoded / f"{session_id}.jsonl"
        return path.exists()

    # -- design doc --

    def read_design_doc(self) -> str:
        if self.design_doc_path.exists():
            return self.design_doc_path.read_text()
        return ""

    def write_design_doc(self, content: str) -> None:
        self.design_doc_path.write_text(content)

    def has_design_doc(self) -> bool:
        return self.design_doc_path.exists() and self.design_doc_path.stat().st_size > 0

    # -- guardrails --

    def read_guardrails(self) -> str:
        if self.guardrails_path.exists():
            return self.guardrails_path.read_text()
        return ""

    # -- stories --

    def load_stories(self) -> list[Story]:
        stories: list[Story] = []
        if not self.stories_path.exists():
            return stories
        for line in self.stories_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                stories.append(Story.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
        return stories

    def append_stories(self, stories: list[Story]) -> None:
        with open(self.stories_path, "a") as f:
            for s in stories:
                f.write(json.dumps(s.to_dict()) + "\n")
        for s in stories:
            self._notify("added", s)

    # -- story listeners --

    def add_story_listener(self, fn: Callable[[str, Story | None, dict], None]) -> None:
        """Register ``fn(event, story, info)`` to be called when the backlog changes.

        Events:
          - ``added``   — a story was appended to the backlog
          - ``status``  — a story changed status; info has ``previous``, ``summary``,
                          ``extra``, ``error_reason`` and ``review_pending``
          - ``reviewed`` — an implemented story passed review
          - ``updated`` — story fields/metadata were changed via update_story()
          - ``synced``  — the backlog was replaced wholesale (story is None)

        Integrations (tasks.md in spec mode, Linear) hang off this. Listener
        failures are reported but never propagate: a broken integration must
        not take the implementation loop down with it.
        """
        if fn not in self._story_listeners:
            self._story_listeners.append(fn)

    def _notify(self, event: str, story: Story | None, info: dict | None = None) -> None:
        for fn in list(self._story_listeners):
            try:
                fn(event, story, info or {})
            except Exception as e:  # noqa: BLE001 — see add_story_listener
                target = story.id if story is not None else "backlog"
                print(f"  [warn] {event} hook failed for {target}: {e}", file=sys.stderr)

    def update_story(
        self,
        story_id: str,
        mutate: Callable[[Story], None],
        *,
        notify: bool = True,
    ) -> Story | None:
        """Apply ``mutate`` to one story and persist it. Returns the story, or None if unknown."""
        stories = self.load_stories()
        target = next((s for s in stories if s.id == story_id), None)
        if target is None:
            return None
        mutate(target)
        self._rewrite_stories(stories)
        if notify:
            self._notify("updated", target)
        return target

    def replace_stories(self, stories: list[Story], status_entries: list[dict] | None = None) -> None:
        """Replace the whole backlog (used when syncing from an external tracker).

        ``status_entries`` are appended to the status history so externally
        driven status changes show up in the timeline like any other.
        """
        self._rewrite_stories(stories)
        if status_entries:
            with open(self.status_path, "a") as f:
                for entry in status_entries:
                    f.write(json.dumps(entry) + "\n")
        self._notify("synced", None)

    def get_pending_stories(self) -> list[Story]:
        return [s for s in self.load_stories() if s.status == StoryStatus.pending]

    def get_actionable_stories(self) -> list[Story]:
        """Return stories that are pending or need rework (rework first)."""
        stories = self.load_stories()
        rework = [s for s in stories if s.status == StoryStatus.rework]
        pending = [s for s in stories if s.status == StoryStatus.pending]
        return rework + pending

    def reset_error_stories(self) -> list[Story]:
        """Find stories with error status and reset them to pending."""
        stories = self.load_stories()
        reset: list[Story] = []

        for s in stories:
            if s.status == StoryStatus.error:
                s.status = StoryStatus.pending
                s.metadata.pop("error_reason", None)
                s.metadata.pop("error_output", None)
                s.metadata.pop("error_at", None)
                reset.append(s)

        if reset:
            self._rewrite_stories(stories)
            with open(self.status_path, "a") as f:
                for s in reset:
                    entry = {
                        "story_id": s.id,
                        "status": "pending",
                        "summary": "Reset from error status",
                    }
                    f.write(json.dumps(entry) + "\n")
            for s in reset:
                self._notify("status", s, {
                    "previous": StoryStatus.error,
                    "summary": "Reset from error status",
                })

        return reset

    def recover_orphaned_stories(self) -> list[Story]:
        """Find in_progress stories (orphans from crashes) and reset to pending."""
        stories = self.load_stories()
        recovered: list[Story] = []

        for s in stories:
            if s.status == StoryStatus.in_progress:
                s.status = StoryStatus.pending
                s.metadata["previous_attempt"] = {
                    "was_in_progress": True,
                    "recovered_at": datetime.now().isoformat(),
                }
                recovered.append(s)

        if recovered:
            self._rewrite_stories(stories)
            with open(self.status_path, "a") as f:
                for s in recovered:
                    entry = {
                        "story_id": s.id,
                        "status": "pending",
                        "summary": "Recovered from crash (was in_progress)",
                        "recovery": True,
                    }
                    f.write(json.dumps(entry) + "\n")
            for s in recovered:
                self._notify("status", s, {
                    "previous": StoryStatus.in_progress,
                    "summary": "Recovered from crash (was in_progress)",
                })

        return recovered

    def get_story_ids(self) -> set[str]:
        return {s.id for s in self.load_stories()}

    def get_category_stats(self) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "next_id": 1})
        for story in self.load_stories():
            cat = story.category.upper()
            if not cat:
                continue
            stats[cat]["count"] += 1
            # Parse numeric suffix from id like "AUTH-043"
            parts = story.id.rsplit("-", 1)
            if len(parts) == 2:
                try:
                    num = int(parts[1])
                    stats[cat]["next_id"] = max(stats[cat]["next_id"], num + 1)
                except ValueError:
                    pass
        return dict(stats)

    def format_existing_stories_context(self) -> str:
        stories = self.load_stories()
        if not stories:
            return "(none yet)"
        lines: list[str] = []
        for s in stories:
            deps = ", ".join(s.dependencies) if s.dependencies else "none"
            lines.append(f"- {s.id}: {s.title} [priority={s.priority}, status={s.status.value}, deps={deps}]")
        return "\n".join(lines)

    def format_category_stats(self) -> str:
        stats = self.get_category_stats()
        if not stats:
            return "(no categories yet — start IDs at CATEGORY-001)"
        lines: list[str] = []
        for cat, info in sorted(stats.items()):
            lines.append(f"- {cat}: count={info['count']}, next_id={info['next_id']:03d}")
        return "\n".join(lines)

    # -- story status --

    def mark_story_status(
        self,
        story_id: str,
        status: StoryStatus,
        summary: str = "",
        extra: dict | None = None,
        error_reason: str = "",
        error_output: str = "",
        review_pending: bool = False,
    ) -> None:
        """Record a status change.

        ``review_pending`` tells listeners that an ``implemented`` status is
        still subject to review — see notify_review_passed().
        """
        entry = {
            "story_id": story_id,
            "status": status.value,
            "summary": summary,
            **(extra or {}),
        }
        if error_reason:
            entry["error_reason"] = error_reason
        with open(self.status_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Also rewrite stories.jsonl with updated status
        stories = self.load_stories()
        target: Story | None = None
        previous: StoryStatus | None = None
        for s in stories:
            if s.id == story_id:
                previous = s.status
                s.status = status
                if status == StoryStatus.error:
                    s.metadata["error_reason"] = error_reason or summary
                    if error_output:
                        # Keep last 2000 chars of output for context
                        s.metadata["error_output"] = error_output[-2000:]
                    s.metadata["error_at"] = datetime.now().isoformat()
                target = s
                break
        self._rewrite_stories(stories)
        if target is not None:
            self._notify("status", target, {
                "previous": previous,
                "summary": summary,
                "extra": extra or {},
                "error_reason": error_reason,
                "review_pending": review_pending,
            })

    def notify_review_passed(self, story_id: str, feedback: str = "") -> None:
        """Tell listeners a story passed review, i.e. its ``implemented`` status is final."""
        story = next((s for s in self.load_stories() if s.id == story_id), None)
        if story is not None:
            self._notify("reviewed", story, {"feedback": feedback})

    def _rewrite_stories(self, stories: list[Story]) -> None:
        with open(self.stories_path, "w") as f:
            for s in stories:
                f.write(json.dumps(s.to_dict()) + "\n")

    def get_implemented_summary(self) -> str:
        if not self.status_path.exists():
            return ""
        count = 0
        for raw in self.status_path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                json.loads(raw)
                count += 1
            except json.JSONDecodeError:
                continue
        if count == 0:
            return ""
        return f"## Previously Implemented Stories\n\n{count} stories tracked in {self.status_path}"

    # -- phase state --

    def load_phase_state(self, phase: str) -> PhaseState:
        if self.phase_state_path.exists():
            try:
                data = json.loads(self.phase_state_path.read_text())
                if data.get("phase") == phase:
                    return PhaseState.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass
        return PhaseState(phase=phase)

    def save_phase_state(self, state: PhaseState) -> None:
        self.phase_state_path.write_text(json.dumps(state.to_dict(), indent=2) + "\n")

    # -- run log --

    def log_iteration(self, result: IterationResult) -> None:
        with open(self.run_log_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def get_story_tokens(self) -> dict[str, dict[str, int]]:
        """Aggregate tokens per story from run-log.jsonl."""
        totals: dict[str, dict[str, int]] = {}
        if not self.run_log_path.exists():
            return totals
        for line in self.run_log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            story_id = entry.get("story_id")
            if not story_id:
                continue
            if story_id not in totals:
                totals[story_id] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            totals[story_id]["input_tokens"] += entry.get("input_tokens", 0)
            totals[story_id]["output_tokens"] += entry.get("output_tokens", 0)
            totals[story_id]["cache_read_input_tokens"] += entry.get("cache_read_input_tokens", 0)
            totals[story_id]["cache_creation_input_tokens"] += entry.get("cache_creation_input_tokens", 0)
        return totals

    # -- solutions (compound learning) --

    @property
    def solutions_dir(self) -> Path:
        return self.state_dir / "solutions"

    @property
    def solutions_index_path(self) -> Path:
        return self.solutions_dir / "index.jsonl"

    def has_solutions(self) -> bool:
        return self.solutions_index_path.exists() and self.solutions_index_path.stat().st_size > 0

    @staticmethod
    def make_stable_filename(category: str, error_signature: str, title: str) -> str:
        """Public wrapper around the stable hash-based filename helper.

        Callers that produce solution entries can use this to pick a filename
        whose name doesn't drift across compactions.
        """
        return _stable_filename(category, error_signature, title)

    def save_solution(
        self,
        category: str,
        filename: str,
        content: str,
        index_entry: dict,
    ) -> Path:
        """Write a solution markdown file and append to index."""
        solution_path = _safe_resolve(self.solutions_dir, f"{category}/{filename}")
        if solution_path is None:
            raise ValueError(f"Invalid solution path: {category}/{filename}")

        with _index_lock(self.solutions_index_path):
            solution_path.parent.mkdir(parents=True, exist_ok=True)
            solution_path.write_text(content)

            # Append index entry
            with open(self.solutions_index_path, "a") as f:
                f.write(json.dumps(index_entry) + "\n")

        # Best-effort merge of the just-saved entry against existing candidates.
        # No-op unless PRALPH_COMPACT_ON_SAVE=1 is set.
        _maybe_merge_on_save(index_entry, self.solutions_index_path, self.solutions_dir)

        return solution_path

    def load_solutions_index(self) -> list[dict]:
        """Read all index entries."""
        entries: list[dict] = []
        if not self.solutions_index_path.exists():
            return entries
        for line in self.solutions_index_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    entries.append(parsed)
            except json.JSONDecodeError:
                continue
        return entries

    def read_solution(self, filename: str) -> str:
        """Read a specific solution file."""
        path = _safe_resolve(self.solutions_dir, filename)
        if path is not None and path.exists():
            return path.read_text()
        return ""

    def get_solutions_summary(self, max_chars: int = 4000) -> str:
        """Compact summary of all solutions for prompt context."""
        entries = self.load_solutions_index()
        if not entries:
            return ""

        lines: list[str] = []
        total_len = 0
        for entry in entries:
            title = entry.get("title", "?")
            category = entry.get("category", "?")
            tags = ", ".join(entry.get("tags", []))
            line = f"- [{category}] {title} (tags: {tags})"
            if total_len + len(line) + 1 > max_chars:
                lines.append(f"(... and {len(entries) - len(lines)} more solutions)")
                break
            lines.append(line)
            total_len += len(line) + 1

        return "\n".join(lines)

    # -- global solutions (cross-project compound learning) --

    def _global_domain_solutions_dir(self, domain: str) -> Path:
        return self.home_dir / "solutions" / domain

    def _global_domain_index_path(self, domain: str) -> Path:
        return self._global_domain_solutions_dir(domain) / "index.jsonl"

    def save_solution_global(
        self,
        category: str,
        filename: str,
        content: str,
        index_entry: dict,
    ) -> list[Path]:
        """Save a solution to ~/.pralph/solutions/{domain}/ for relevant detected domains.

        Uses heuristics (related_files, tags, error_signature) to infer which
        domain(s) a solution applies to.  Falls back to Haiku LLM inference when
        heuristics return nothing.

        The index_entry is augmented with source_project for traceability.
        Returns list of paths written.
        """
        all_domains = self.detect_domains()
        if not all_domains:
            return []

        inferred = _infer_solution_domains(
            index_entry.get("related_files", []),
            index_entry.get("tags", []),
            index_entry.get("error_signature", ""),
            all_domains,
        )
        if not inferred:
            inferred = _infer_domains_llm(
                content=content,
                title=index_entry.get("title", ""),
                category=index_entry.get("category", ""),
                tags=index_entry.get("tags", []),
                error_signature=index_entry.get("error_signature", ""),
                available_domains=all_domains,
            )
        domains = inferred if inferred else set(all_domains)

        entry = {
            **index_entry,
            "source_project": str(self.project_dir),
        }

        paths: list[Path] = []
        for domain in domains:
            domain_dir = self._global_domain_solutions_dir(domain)
            solution_path = _safe_resolve(domain_dir, f"{category}/{filename}")
            if solution_path is None:
                continue

            idx_path = self._global_domain_index_path(domain)
            with _index_lock(idx_path):
                solution_path.parent.mkdir(parents=True, exist_ok=True)
                solution_path.write_text(content)

                with open(idx_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            # Best-effort merge against existing candidates in this domain.
            _maybe_merge_on_save(entry, idx_path, domain_dir)
            paths.append(solution_path)

        return paths

    def has_global_solutions(self) -> bool:
        """Check if any global solutions exist for the project's detected domains."""
        for domain in self.detect_domains():
            idx = self._global_domain_index_path(domain)
            if idx.exists() and idx.stat().st_size > 0:
                return True
        return False

    def load_global_solutions_index(self) -> list[dict]:
        """Load index entries from all detected domains in ~/.pralph/solutions/."""
        seen_filenames: set[str] = set()
        entries: list[dict] = []
        for domain in self.detect_domains():
            idx = self._global_domain_index_path(domain)
            if not idx.exists():
                continue
            for line in idx.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        continue
                    # Deduplicate: same filename within a domain
                    key = f"{domain}/{parsed.get('filename', '')}"
                    if key in seen_filenames:
                        continue
                    seen_filenames.add(key)
                    parsed["_domain"] = domain
                    entries.append(parsed)
                except json.JSONDecodeError:
                    continue
        return entries

    def read_global_solution(self, domain: str, filename: str) -> str:
        """Read a specific solution file from global domain store."""
        path = _safe_resolve(self._global_domain_solutions_dir(domain), filename)
        if path is not None and path.exists():
            return path.read_text()
        return ""

    def get_global_solutions_summary(self, max_chars: int = 3000) -> str:
        """Compact summary of global solutions for the detected domains."""
        entries = self.load_global_solutions_index()
        if not entries:
            return ""

        lines: list[str] = []
        total_len = 0
        for entry in entries:
            title = entry.get("title", "?")
            category = entry.get("category", "?")
            domain = entry.get("_domain", "?")
            tags = ", ".join(entry.get("tags", []))
            line = f"- [{domain}/{category}] {title} (tags: {tags})"
            if total_len + len(line) + 1 > max_chars:
                lines.append(f"(... and {len(entries) - len(lines)} more global solutions)")
                break
            lines.append(line)
            total_len += len(line) + 1

        return "\n".join(lines)

    def search_all_solutions(self, query: str, max_results: int = 5) -> list[dict]:
        """Search both project-local and global solutions, merged by relevance.

        Project-local results are prioritised (score boosted) over global ones.
        """
        keywords = query.lower().split()

        def _score(entry: dict) -> int:
            score = 0
            title = (entry.get("title") or "").lower()
            tags = [t.lower() for t in (entry.get("tags") or [])]
            error_sig = (entry.get("error_signature") or "").lower()
            category = (entry.get("category") or "").lower()
            for kw in keywords:
                if kw in title:
                    score += 3
                if any(kw in tag for tag in tags):
                    score += 2
                if kw in error_sig:
                    score += 2
                if kw in category:
                    score += 1
            return score

        scored: list[tuple[int, dict]] = []
        seen_titles: set[str] = set()

        # Project-local (boosted by +5)
        for entry in self.load_solutions_index():
            if not isinstance(entry, dict):
                continue
            s = _score(entry)
            if s > 0:
                entry["_source"] = "local"
                scored.append((s + 5, entry))
                seen_titles.add((entry.get("title") or "").lower())

        # Global (no boost, skip duplicates by title)
        for entry in self.load_global_solutions_index():
            title_key = (entry.get("title") or "").lower()
            if title_key in seen_titles:
                continue
            s = _score(entry)
            if s > 0:
                entry["_source"] = "global"
                scored.append((s, entry))
                seen_titles.add(title_key)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def read_any_solution(self, entry: dict) -> str:
        """Read a solution file, handling both local and global entries."""
        filename = entry.get("filename", "")
        if entry.get("_source") == "global":
            domain = entry.get("_domain", "")
            if domain:
                return self.read_global_solution(domain, filename)
        return self.read_solution(filename)

    # -- index compaction --

    @staticmethod
    def _estimate_index(index_path: Path, solutions_base: Path, *, model: str, full: bool) -> dict:
        """Estimate for one index, over the same entries and marker the run will use."""
        entries = _read_index_entries(index_path)
        valid, _, _ = _pre_prune_entries(entries, solutions_base)
        marker = None if full else _read_compact_state(index_path).get("last_compacted_at")
        est = _estimate_compaction_cost(valid, solutions_base, model=model, marker=marker)
        est["entries"] = len(entries)
        return est

    def estimate_local_compaction(self, *, model: str = "haiku", full: bool = False) -> dict:
        """Pre-flight cost estimate for the local solutions index."""
        return self._estimate_index(self.solutions_index_path, self.solutions_dir, model=model, full=full)

    def estimate_global_compactions(self, *, model: str = "haiku", full: bool = False) -> list[dict]:
        """Pre-flight cost estimate for every global domain index."""
        out: list[dict] = []
        solutions_root = self.home_dir / "solutions"
        if not solutions_root.exists():
            return out
        for domain_dir in sorted(solutions_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            idx_path = domain_dir / "index.jsonl"
            if not idx_path.exists():
                continue
            est = self._estimate_index(idx_path, domain_dir, model=model, full=full)
            est["domain"] = domain_dir.name
            out.append(est)
        return out

    def compact_local_index(
        self,
        *,
        model: str = "haiku",
        verbose: bool = False,
        dangerously_skip_permissions: bool = False,
        full: bool = False,
    ) -> dict:
        """Compact the project-local solutions index using an LLM to merge duplicates.

        Returns stats dict with original, kept, merged, removed counts and cost.
        """
        return self._compact_index(
            self.solutions_index_path,
            self.solutions_dir,
            model=model,
            verbose=verbose,
            dangerously_skip_permissions=dangerously_skip_permissions,
            full=full,
        )

    def compact_global_indexes(
        self,
        *,
        model: str = "haiku",
        verbose: bool = False,
        dangerously_skip_permissions: bool = False,
        full: bool = False,
    ) -> list[dict]:
        """Compact every global solution index that exists under ~/.pralph/solutions/.

        Global compaction is independent of the current project's detected
        domains — `--global-only` from any cwd compacts all global domains.
        Returns a list of stats dicts, one per domain.
        """
        results = []
        solutions_root = self.home_dir / "solutions"
        if not solutions_root.exists():
            return results
        for domain_dir in sorted(solutions_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            idx_path = domain_dir / "index.jsonl"
            if not idx_path.exists():
                continue
            stats = self._compact_index(
                idx_path,
                domain_dir,
                model=model,
                verbose=verbose,
                dangerously_skip_permissions=dangerously_skip_permissions,
                full=full,
            )
            stats["domain"] = domain_dir.name
            results.append(stats)
        return results

    @staticmethod
    def _compact_index(
        index_path: Path,
        solutions_base: Path,
        *,
        model: str = "haiku",
        verbose: bool = False,
        dangerously_skip_permissions: bool = False,
        full: bool = False,
    ) -> dict:
        """Use an LLM to semantically merge duplicate solutions and prune orphans.

        Clusters the valid entries (see ``_cluster_entries``) and sends each
        multi-entry cluster to the LLM, in chunks of ``_COMPACT_CHUNK_SIZE``.
        The index is rewritten atomically after every chunk so a mid-run
        failure preserves earlier chunks' merges.

        Runs are incremental: clusters that gained no entry since the last
        fully successful run are not sent again. ``full=True`` sends them all.
        """
        # Taken before the index is read: entries saved while we run are not
        # looked at, so they have to count as new for the next run.
        started_at = datetime.now().isoformat()

        if not index_path.exists():
            return {
                "original": 0, "kept": 0, "merged": 0, "removed": 0,
                "cost": 0.0, "chunks": 0, "failed_chunks": 0,
            }

        # Initial read + clustering: short critical section.
        with _index_lock(index_path):
            entries = _read_index_entries(index_path)
            if not entries:
                return {
                    "original": 0, "kept": 0, "merged": 0, "removed": 0,
                    "cost": 0.0, "chunks": 0, "failed_chunks": 0,
                }

            original = len(entries)
            valid, exact_dupes, pre_orphans = _pre_prune_entries(entries, solutions_base)

            if len(valid) <= 1:
                _atomic_write_index(index_path, valid)
                return {
                    "original": original,
                    "kept": len(valid),
                    "merged": 0,
                    "removed": exact_dupes + pre_orphans,
                    "cost": 0.0,
                    "chunks": 0,
                    "failed_chunks": 0,
                }

        # Cluster + batch work happens OUTSIDE the lock — it's pure CPU and
        # frees concurrent saves to land while we're computing batches.
        marker = None if full else _read_compact_state(index_path).get("last_compacted_at")
        skipped, batches = _plan_compaction(valid, marker=marker)

        # Filenames we are responsible for — anything else seen on disk during
        # processing is a concurrent append and gets carried through unchanged.
        # That includes the entries pre-pruning dropped (orphans, exact
        # duplicates): they are still in the file until the first rewrite and
        # must not be mistaken for appends.
        claimed: set[str] = {e.get("filename", "") for e in entries if e.get("filename")}
        # Skipped clusters go straight into the output unchanged.
        processed: list[dict] = list(skipped)
        remaining_flat = [e for b in batches for e in b]
        total_cost = 0.0
        total_merged = 0
        total_removed = 0
        failed_chunks = 0
        llm_batches = sum(1 for b in batches if len(b) > 1)
        llm_done = 0

        for batch in batches:
            if len(batch) <= 1:
                processed.extend(batch)
                remaining_flat = remaining_flat[len(batch):]
                continue

            llm_done += 1
            if verbose:
                print(
                    f"  [compact] batch {llm_done}/{llm_batches} ({len(batch)} entries)",
                    file=sys.stderr,
                )

            # LLM call — lock NOT held. Saves can land in parallel.
            new_entries, files_to_delete, merges_n, removed_n, cost, success = _compact_chunk(
                batch,
                solutions_base,
                model=model,
                verbose=verbose,
                dangerously_skip_permissions=dangerously_skip_permissions,
            )
            total_cost += cost

            # Brief lock to apply the batch and absorb any concurrent appends.
            with _index_lock(index_path):
                on_disk = _read_index_entries(index_path)
                # Pick up entries that landed since we started. We carry them
                # through unchanged; they'll be candidates next compaction run.
                new_appends = [
                    e for e in on_disk if e.get("filename", "") and e.get("filename", "") not in claimed
                ]
                for e in new_appends:
                    claimed.add(e.get("filename", ""))

                if success:
                    for old_fn in files_to_delete:
                        old_path = _safe_resolve(solutions_base, old_fn)
                        if old_path is not None and old_path.exists():
                            try:
                                old_path.unlink()
                            except OSError:
                                pass
                    total_merged += merges_n
                    total_removed += removed_n
                    batch_result = new_entries
                else:
                    failed_chunks += 1
                    if verbose:
                        print(f"  [compact] batch {llm_done} failed — keeping unchanged", file=sys.stderr)
                    batch_result = batch

                for e in batch_result:
                    fn = e.get("filename", "")
                    if fn:
                        claimed.add(fn)

                processed.extend(batch_result)
                processed.extend(new_appends)
                remaining_flat = remaining_flat[len(batch):]
                _atomic_write_index(index_path, processed + remaining_flat)

        chunks_count = len(batches)

        # Final writeback — also picks up any last-minute concurrent appends.
        with _index_lock(index_path):
            on_disk = _read_index_entries(index_path)
            new_appends = [
                e for e in on_disk if e.get("filename", "") and e.get("filename", "") not in claimed
            ]
            if new_appends:
                processed.extend(new_appends)
            _atomic_write_index(index_path, processed)

        # If every batch we attempted succeeded, advance the incremental
        # marker so the next run can skip the clusters that didn't change.
        # If any batch failed, leave the marker alone — its entries are then
        # still "new" and get retried.
        if failed_chunks == 0:
            _write_compact_state(
                index_path,
                {
                    "last_compacted_at": started_at,
                    "last_count": len(processed),
                },
            )
        return {
            "original": original,
            "kept": len(processed),
            "merged": total_merged,
            "removed": exact_dupes + pre_orphans + total_removed,
            "cost": total_cost,
            "chunks": chunks_count,
            "failed_chunks": failed_chunks,
        }
