"""Tests for the chunked, clustered, incremental ``_compact_index``.

These cover the building blocks (``_cluster_entries``, ``_stable_filename``,
``_estimate_compaction_cost``) plus the end-to-end driver under a stubbed LLM,
and the ``_maybe_merge_on_save`` hook.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import pytest

from pralph import state as S
from pralph.runner import ClaudeResult, _is_context_overflow, _parse_result_event, run_with_retry


# ── _is_context_overflow / context detection in run_with_retry ────────


class TestContextOverflowDetection:
    def test_marker_detection(self) -> None:
        assert _is_context_overflow("prompt is too long: 250000 tokens")
        assert _is_context_overflow("Error: context_length_exceeded")
        assert _is_context_overflow("input is too long for the model")
        assert not _is_context_overflow("rate limit reached")
        assert not _is_context_overflow("")

    def test_parse_result_event_sets_flag(self) -> None:
        r = _parse_result_event(
            {"subtype": "error", "error": "prompt is too long", "usage": {}}
        )
        assert r.is_context_overflow is True
        assert r.success is False

    def test_run_with_retry_does_not_retry_on_overflow(self) -> None:
        calls = {"n": 0}

        def fake_run_claude(prompt: str, **kw):
            calls["n"] += 1
            return ClaudeResult(
                success=False,
                error="prompt is too long",
                is_context_overflow=True,
            )

        with patch("pralph.runner.run_claude", side_effect=fake_run_claude):
            r = run_with_retry("anything", max_retries=3)
        assert calls["n"] == 1, "should give up immediately on overflow"
        assert r.is_context_overflow


# ── _cluster_entries ──────────────────────────────────────────────────


def _multi_groups(groups: list[list[dict]]) -> list[list[dict]]:
    return [g for g in groups if len(g) > 1]


class TestClusterEntries:
    def test_empty(self) -> None:
        assert S._cluster_entries([]) == []

    def test_singleton(self) -> None:
        groups = S._cluster_entries([{"filename": "a.md", "category": "c", "title": "t"}])
        assert len(groups) == 1 and len(groups[0]) == 1

    def test_group_by_error_signature(self) -> None:
        e = [
            {"category": "a", "tags": [], "error_signature": "NPE at foo:1", "title": "x"},
            {"category": "b", "tags": [], "error_signature": "NPE at foo:1", "title": "y"},
            {"category": "c", "tags": [], "error_signature": "IndexError", "title": "z"},
        ]
        m = _multi_groups(S._cluster_entries(e))
        assert len(m) == 1 and len(m[0]) == 2

    def test_group_by_tag_jaccard(self) -> None:
        # Same category required; Jaccard 3/5 = 0.6 → grouped.
        e = [
            {"category": "c", "tags": ["a", "b", "c", "d"], "title": "t1"},
            {"category": "c", "tags": ["a", "b", "c", "e"], "title": "t2"},
            {"category": "c", "tags": ["x", "y", "z"], "title": "t3"},
        ]
        m = _multi_groups(S._cluster_entries(e))
        assert len(m) == 1 and len(m[0]) == 2

    def test_no_cross_category_grouping_by_tags(self) -> None:
        e = [
            {"category": "a", "tags": ["x", "y", "z"], "title": "t1"},
            {"category": "b", "tags": ["x", "y", "z"], "title": "t2"},
        ]
        assert _multi_groups(S._cluster_entries(e)) == []

    def test_group_by_category_story_project(self) -> None:
        e = [
            {"category": "c", "story_id": "S1", "source_project": "/p", "tags": [], "title": "t1"},
            {"category": "c", "story_id": "S1", "source_project": "/p", "tags": [], "title": "t2"},
            {"category": "c", "story_id": "S2", "source_project": "/p", "tags": [], "title": "t3"},
        ]
        m = _multi_groups(S._cluster_entries(e))
        assert len(m) == 1 and len(m[0]) == 2

    def test_group_by_title_prefix(self) -> None:
        e = [
            {"category": "c", "tags": [], "title": "fix the broken login flow today"},
            {"category": "c", "tags": [], "title": "fix the broken login flow tomorrow"},
            {"category": "c", "tags": [], "title": "completely unrelated"},
        ]
        m = _multi_groups(S._cluster_entries(e))
        assert len(m) == 1 and len(m[0]) == 2

    def test_transitive_union(self) -> None:
        # A~B by error_signature, B~C by tag jaccard → A~B~C.
        e = [
            {"category": "c", "tags": ["x", "y", "z", "w"], "error_signature": "E1", "title": "a"},
            {"category": "c", "tags": ["x", "y", "z", "w"], "error_signature": "E1", "title": "b"},
            {"category": "c", "tags": ["x", "y", "z", "q"], "title": "c"},
        ]
        m = _multi_groups(S._cluster_entries(e))
        assert len(m) == 1 and len(m[0]) == 3

    def test_100_unrelated_singletons(self) -> None:
        e = [
            {"category": f"c{i}", "tags": [f"t{i}"], "title": f"u{i}"}
            for i in range(100)
        ]
        groups = S._cluster_entries(e)
        assert len(groups) == 100
        assert _multi_groups(groups) == []


# ── _stable_filename ─────────────────────────────────────────────────


class TestStableFilename:
    def test_idempotent(self) -> None:
        a = S._stable_filename("c", "sig", "title")
        b = S._stable_filename("c", "sig", "title")
        assert a == b

    def test_shape(self) -> None:
        fn = S._stable_filename("build-errors", "Foo", "My Title")
        assert re.match(r"^build-errors/[0-9a-f]{8}-", fn)
        assert fn.endswith(".md")

    def test_same_signature_same_hash(self) -> None:
        a = S._stable_filename("c", "shared-sig", "title-one")
        b = S._stable_filename("c", "shared-sig", "title-two")
        assert a.split("/")[1].split("-")[0] == b.split("/")[1].split("-")[0]

    def test_fallback_when_empty(self) -> None:
        fn = S._stable_filename("", "", "")
        assert fn  # never crashes
        assert "/" in fn

    def test_is_stable_filename(self) -> None:
        assert S._is_stable_filename("build-errors/abc12345-foo.md")
        assert not S._is_stable_filename("build-errors/foo.md")
        assert not S._is_stable_filename("")

    def test_public_wrapper(self) -> None:
        assert S.StateManager.make_stable_filename("c", "s", "t") == S._stable_filename("c", "s", "t")


# ── _estimate_compaction_cost ────────────────────────────────────────


class TestEstimator:
    def test_empty_input(self, tmp_path: Path) -> None:
        est = S._estimate_compaction_cost([], tmp_path)
        assert est["calls"] == 0 and est["usd"] == 0.0

    def test_singletons_cost_nothing(self, tmp_path: Path) -> None:
        # All singletons: nothing for the LLM to do.
        entries = [
            {"category": f"c{i}", "tags": [f"t{i}"], "title": f"u{i}"}
            for i in range(50)
        ]
        est = S._estimate_compaction_cost(entries, tmp_path)
        assert est["calls"] == 0
        assert est["usd"] == 0.0

    def test_multi_entry_cluster_costs_one_call(self, tmp_path: Path) -> None:
        entries = [
            {"category": "c", "tags": ["x"], "error_signature": "E", "title": "a", "filename": "c/a.md"},
            {"category": "c", "tags": ["x"], "error_signature": "E", "title": "b", "filename": "c/b.md"},
        ]
        est = S._estimate_compaction_cost(entries, tmp_path)
        assert est["calls"] == 1
        assert est["usd"] > 0


    def test_marker_skips_clusters_without_new_entries(self, tmp_path: Path) -> None:
        old = [
            {"category": "c", "error_signature": "E", "title": "a", "filename": "c/a.md", "created": "2026-04-01T00:00:00"},
            {"category": "c", "error_signature": "E", "title": "b", "filename": "c/b.md", "created": "2026-04-02T00:00:00"},
        ]
        marker = "2026-05-01T00:00:00"
        assert S._estimate_compaction_cost(old, tmp_path, marker=marker)["calls"] == 0
        # A new entry joining the cluster makes it worth a call again.
        new = {"category": "c", "error_signature": "E", "title": "n", "filename": "c/n.md", "created": "2026-06-01T00:00:00"}
        assert S._estimate_compaction_cost(old + [new], tmp_path, marker=marker)["calls"] == 1
        # No marker (first run / --full): everything is processed.
        assert S._estimate_compaction_cost(old, tmp_path)["calls"] == 1


# ── End-to-end _compact_index with stubbed LLM ───────────────────────


def _write_index(idx_path: Path, entries: Iterable[dict]) -> None:
    with open(idx_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _make_solution_files(base: Path, entries: list[dict], body: str = "body") -> None:
    for e in entries:
        p = base / e["filename"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def _llm_no_op_response() -> str:
    return "```json\n" + json.dumps(
        {"entries": [], "merges": [], "removed": []}
    ) + "\n```"


def _llm_passthrough_response(prompt: str, fn_pattern: str) -> str:
    """Return all entries from the prompt unchanged."""
    fns = re.findall(fn_pattern, prompt)
    seen: set[str] = set()
    uniq: list[str] = []
    for fn in fns:
        if fn not in seen:
            seen.add(fn)
            uniq.append(fn)
    out = [
        {"filename": fn, "title": fn, "category": fn.split("/")[0], "tags": []}
        for fn in uniq
    ]
    return "```json\n" + json.dumps(
        {"entries": out, "merges": [], "removed": []}
    ) + "\n```"


class TestCompactIndexEndToEnd:
    def _index(self, tmp_path: Path) -> tuple[Path, Path]:
        base = tmp_path / "solutions"
        base.mkdir()
        return base / "index.jsonl", base

    def test_singletons_skip_llm(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        entries = [
            {
                "filename": f"c{i}/file.md",
                "category": f"c{i}",
                "title": f"u{i}",
                "tags": [f"t{i}"],
                "created": "2026-04-01T00:00:00",
            }
            for i in range(20)
        ]
        _make_solution_files(base, entries)
        _write_index(idx, entries)

        calls = {"n": 0}

        def fake_run(prompt: str, **kw):
            calls["n"] += 1
            return ClaudeResult(success=True, result=_llm_no_op_response(), cost_usd=0.0)

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            stats = S.StateManager._compact_index(idx, base)

        assert calls["n"] == 0
        assert stats["original"] == 20
        assert stats["kept"] == 20
        assert stats["failed_chunks"] == 0

    def test_merge_produces_stable_filename(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        entries = [
            {
                "filename": "c/a.md",
                "category": "c",
                "title": "shared",
                "tags": ["x", "y", "z", "w"],
                "error_signature": "NullPointer",
            },
            {
                "filename": "c/b.md",
                "category": "c",
                "title": "shared",
                "tags": ["x", "y", "z", "w"],
                "error_signature": "NullPointer",
            },
        ]
        _make_solution_files(base, entries)
        _write_index(idx, entries)

        def fake_run(prompt: str, **kw):
            merged = {
                "category": "c",
                "title": "shared",
                "error_signature": "NullPointer",
                "tags": ["x", "y", "z", "w"],
                "content": "# merged\n\nbody",
            }
            body = json.dumps(
                {
                    "entries": [merged],
                    "merges": [{"merged_into": "_", "sources": ["c/a.md", "c/b.md"]}],
                    "removed": [],
                }
            )
            return ClaudeResult(success=True, result=f"```json\n{body}\n```", cost_usd=0.001)

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            stats = S.StateManager._compact_index(idx, base)

        assert stats["merged"] == 1
        assert stats["kept"] == 1
        final = [json.loads(line) for line in open(idx)]
        assert len(final) == 1
        assert S._is_stable_filename(final[0]["filename"])
        assert not (base / "c/a.md").exists()
        assert not (base / "c/b.md").exists()

    def test_failed_chunk_preserves_entries(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        # Two clusters of 2 (each is its own LLM batch).
        entries = [
            {"filename": "c/a.md", "category": "c", "title": "p", "tags": ["x", "y", "z", "w"], "error_signature": "E1"},
            {"filename": "c/b.md", "category": "c", "title": "q", "tags": ["x", "y", "z", "w"], "error_signature": "E1"},
            {"filename": "c/c.md", "category": "c", "title": "r", "tags": ["p", "q", "r", "s"], "error_signature": "E2"},
            {"filename": "c/d.md", "category": "c", "title": "s", "tags": ["p", "q", "r", "s"], "error_signature": "E2"},
        ]
        _make_solution_files(base, entries)
        _write_index(idx, entries)

        seen = {"n": 0}

        def fake_run(prompt: str, **kw):
            seen["n"] += 1
            if seen["n"] == 1:
                return ClaudeResult(
                    success=False,
                    error="prompt is too long",
                    is_context_overflow=True,
                    cost_usd=0.001,
                )
            return ClaudeResult(
                success=True,
                result=_llm_passthrough_response(prompt, r"c/[a-d]\.md"),
                cost_usd=0.001,
            )

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            stats = S.StateManager._compact_index(idx, base)

        assert stats["failed_chunks"] == 1
        # All 4 entries still present (failed chunk passed through).
        final = [json.loads(line) for line in open(idx)]
        assert len(final) == 4
        # Marker NOT advanced because a chunk failed.
        assert not S._compact_state_path(idx).exists()

    def test_incremental_marker_skips_clean_singletons(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        entries = [
            {
                "filename": f"c{i}/file.md",
                "category": f"c{i}",
                "title": f"u{i}",
                "tags": [f"t{i}"],
                "created": "2026-04-01T00:00:00",
            }
            for i in range(5)
        ]
        _make_solution_files(base, entries)
        _write_index(idx, entries)

        calls = {"n": 0}

        def fake_run(prompt: str, **kw):
            calls["n"] += 1
            return ClaudeResult(success=True, result=_llm_no_op_response(), cost_usd=0.0)

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            S.StateManager._compact_index(idx, base)
        # Marker should be present after a clean run.
        assert S._compact_state_path(idx).exists()
        first_calls = calls["n"]

        # Second run: every entry is "old", every cluster is a singleton.
        # Driver should skip everything → zero work.
        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            stats = S.StateManager._compact_index(idx, base)
        assert calls["n"] == first_calls
        assert stats["chunks"] == 0

    def test_full_flag_ignores_marker(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        # All singletons (each in its own category), all marked "old".
        entries = [
            {
                "filename": f"c{i}/file.md",
                "category": f"c{i}",
                "title": f"u{i}",
                "tags": [f"t{i}"],
                "created": "2026-04-01T00:00:00",
            }
            for i in range(5)
        ]
        _make_solution_files(base, entries)
        _write_index(idx, entries)
        # Pretend a previous successful compaction happened after these entries.
        S._write_compact_state(idx, {"last_compacted_at": "2026-05-01T00:00:00", "last_count": 5})

        def fake_run(prompt: str, **kw):
            return ClaudeResult(success=True, result=_llm_no_op_response(), cost_usd=0.0)

        # Without --full: every entry is an old singleton → all skipped.
        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            stats = S.StateManager._compact_index(idx, base, full=False)
        assert stats["chunks"] == 0

        # With --full: marker ignored → every entry is "active" again.
        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            stats = S.StateManager._compact_index(idx, base, full=True)
        assert stats["chunks"] == 5


    def _cluster(self, base: Path, idx: Path, extra: list[dict] | None = None) -> list[dict]:
        """Two entries the clustering groups together, plus *extra*, written to disk."""
        entries = [
            {"filename": "c/a.md", "category": "c", "title": "p", "error_signature": "E1", "created": "2026-04-01T00:00:00"},
            {"filename": "c/b.md", "category": "c", "title": "q", "error_signature": "E1", "created": "2026-04-02T00:00:00"},
        ]
        _make_solution_files(base, entries)
        _write_index(idx, entries + (extra or []))
        return entries

    def test_orphans_are_pruned_alongside_llm_batches(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        orphan = {"filename": "c/gone.md", "category": "c", "title": "z", "error_signature": "E9"}
        self._cluster(base, idx, extra=[orphan])

        def fake_run(prompt: str, **kw):
            return ClaudeResult(success=True, result=_llm_passthrough_response(prompt, r"c/[ab]\.md"))

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            stats = S.StateManager._compact_index(idx, base)

        final = [json.loads(line)["filename"] for line in open(idx)]
        assert sorted(final) == ["c/a.md", "c/b.md"]
        assert stats["removed"] == 1 and stats["kept"] == 2

    def test_unchanged_cluster_is_not_sent_again(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        known = self._cluster(base, idx)
        new = {"filename": "c/n.md", "category": "c", "title": "n", "error_signature": "E1", "created": "2999-01-01T00:00:00"}
        calls = {"n": 0}

        def fake_run(prompt: str, **kw):
            # Judges the entries distinct: returns what it was shown, unchanged.
            calls["n"] += 1
            shown = [e for e in known + [new] if e["filename"] in prompt]
            body = json.dumps({"entries": shown, "merges": [], "removed": []})
            return ClaudeResult(success=True, result=f"```json\n{body}\n```")

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            S.StateManager._compact_index(idx, base)
            assert calls["n"] == 1
            # Nothing new since the last run: the cluster was already judged.
            S.StateManager._compact_index(idx, base)
            assert calls["n"] == 1
            # --full looks again.
            S.StateManager._compact_index(idx, base, full=True)
            assert calls["n"] == 2
            # So does a new entry joining the cluster.
            _make_solution_files(base, [new])
            with open(idx, "a") as f:
                f.write(json.dumps(new) + "\n")
            S.StateManager._compact_index(idx, base)
            assert calls["n"] == 3
        assert len(open(idx).readlines()) == 3

    def test_passthrough_entry_keeps_its_filename(self, tmp_path: Path) -> None:
        """A reworded title must not point the index at a file that doesn't exist."""
        idx, base = self._index(tmp_path)
        entries = [
            {"filename": "c/0a1b2c3d-first.md", "category": "c", "title": "first", "error_signature": "E1"},
            {"filename": "c/b.md", "category": "c", "title": "second", "error_signature": "E1"},
        ]
        _make_solution_files(base, entries)
        _write_index(idx, entries)

        def fake_run(prompt: str, **kw):
            out = [dict(e, title=e["title"] + " (reworded)", error_signature="E1: detail") for e in entries]
            body = json.dumps({"entries": out, "merges": [], "removed": []})
            return ClaudeResult(success=True, result=f"```json\n{body}\n```")

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            S.StateManager._compact_index(idx, base)

        final = [json.loads(line) for line in open(idx)]
        assert sorted(e["filename"] for e in final) == ["c/0a1b2c3d-first.md", "c/b.md"]
        assert all((base / e["filename"]).exists() for e in final)
        assert {e["title"] for e in final} == {"first (reworded)", "second (reworded)"}

    def test_merge_without_content_fails_the_chunk(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        self._cluster(base, idx)

        def fake_run(prompt: str, **kw):
            merged = {"category": "c", "title": "merged", "error_signature": "E1"}  # no content, new name
            body = json.dumps({
                "entries": [merged],
                "merges": [{"merged_into": "_", "sources": ["c/a.md", "c/b.md"]}],
                "removed": [],
            })
            return ClaudeResult(success=True, result=f"```json\n{body}\n```")

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            stats = S.StateManager._compact_index(idx, base)

        assert stats["failed_chunks"] == 1
        assert sorted(json.loads(line)["filename"] for line in open(idx)) == ["c/a.md", "c/b.md"]
        assert (base / "c/a.md").exists() and (base / "c/b.md").exists()

    def test_llm_cannot_drop_or_delete_outside_its_chunk(self, tmp_path: Path) -> None:
        idx, base = self._index(tmp_path)
        other = {"filename": "other/keep.md", "category": "other", "title": "unrelated", "created": "2026-04-01T00:00:00"}
        _make_solution_files(base, [other])
        self._cluster(base, idx, extra=[other])

        def fake_run(prompt: str, **kw):
            # Returns only one of the two entries, and "removes" a file it was never shown.
            body = json.dumps({
                "entries": [{"filename": "c/a.md", "category": "c", "title": "p"}],
                "merges": [],
                "removed": [{"filename": "other/keep.md", "reason": "made up"}],
            })
            return ClaudeResult(success=True, result=f"```json\n{body}\n```")

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            S.StateManager._compact_index(idx, base)

        assert sorted(json.loads(line)["filename"] for line in open(idx)) == ["c/a.md", "c/b.md", "other/keep.md"]
        assert (base / "other/keep.md").exists() and (base / "c/b.md").exists()


# ── _maybe_merge_on_save ─────────────────────────────────────────────


class TestMaybeMergeOnSave:
    def test_env_off_skips(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("PRALPH_COMPACT_ON_SAVE", raising=False)
        idx = tmp_path / "index.jsonl"
        idx.write_text("")
        calls = {"n": 0}

        def fake_run(prompt: str, **kw):
            calls["n"] += 1
            return ClaudeResult(success=True, result="{}", cost_usd=0.0)

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            r = S._maybe_merge_on_save({"filename": "c/a.md"}, idx, tmp_path)
        assert r is None
        assert calls["n"] == 0

    def test_no_candidates_skips_llm(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("PRALPH_COMPACT_ON_SAVE", "1")
        idx = tmp_path / "index.jsonl"
        (tmp_path / "ca").mkdir()
        (tmp_path / "ca/a.md").write_text("a")
        e1 = {"filename": "ca/a.md", "category": "ca", "title": "thing one", "tags": ["p"], "error_signature": "E1"}
        e2_other_cat = {"filename": "cb/b.md", "category": "cb", "title": "thing two", "tags": ["q"], "error_signature": "E2"}
        (tmp_path / "cb").mkdir()
        (tmp_path / "cb/b.md").write_text("b")
        with open(idx, "w") as f:
            f.write(json.dumps(e1) + "\n")
            f.write(json.dumps(e2_other_cat) + "\n")

        calls = {"n": 0}

        def fake_run(prompt: str, **kw):
            calls["n"] += 1
            return ClaudeResult(success=True, result="{}", cost_usd=0.0)

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            r = S._maybe_merge_on_save(e2_other_cat, idx, tmp_path)
        assert r is None
        assert calls["n"] == 0

    def test_candidate_triggers_merge(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("PRALPH_COMPACT_ON_SAVE", "1")
        idx = tmp_path / "index.jsonl"
        (tmp_path / "c").mkdir()
        (tmp_path / "c/a.md").write_text("first")
        (tmp_path / "c/b.md").write_text("second")
        e1 = {"filename": "c/a.md", "category": "c", "title": "shared", "tags": ["x", "y", "z", "w"], "error_signature": "NPE"}
        e2 = {"filename": "c/b.md", "category": "c", "title": "shared", "tags": ["x", "y", "z", "w"], "error_signature": "NPE"}
        with open(idx, "w") as f:
            f.write(json.dumps(e1) + "\n")
            f.write(json.dumps(e2) + "\n")

        def fake_run(prompt: str, **kw):
            merged = {
                "category": "c",
                "title": "shared",
                "error_signature": "NPE",
                "tags": ["x", "y", "z", "w"],
                "content": "# merged\nbody",
            }
            body = json.dumps(
                {
                    "entries": [merged],
                    "merges": [{"merged_into": "_", "sources": ["c/a.md", "c/b.md"]}],
                    "removed": [],
                }
            )
            return ClaudeResult(success=True, result=f"```json\n{body}\n```", cost_usd=0.001)

        with patch("pralph.runner.run_with_retry", side_effect=fake_run):
            r = S._maybe_merge_on_save(e2, idx, tmp_path)
        assert r is not None
        assert r["merged"] == 1
        final = [json.loads(line) for line in open(idx)]
        assert len(final) == 1
        assert S._is_stable_filename(final[0]["filename"])

    def test_swallows_exceptions(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("PRALPH_COMPACT_ON_SAVE", "1")
        idx = tmp_path / "index.jsonl"
        idx.write_text("")  # no entries

        def boom(prompt: str, **kw):
            raise RuntimeError("network down")

        # Even with broken LLM, this must not raise.
        with patch("pralph.runner.run_with_retry", side_effect=boom):
            r = S._maybe_merge_on_save({"filename": "x/y.md"}, idx, tmp_path)
        assert r is None
