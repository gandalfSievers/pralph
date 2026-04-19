# pralph GUI — Full Redesign

The current viewer is a single-page story browser served from a Python HTTP server. This document proposes evolving it into a full project management GUI for pralph — a local desktop-grade web app that replaces most CLI interactions.

## Design principles

- **Local-first** — runs on localhost, reads/writes the same DuckDB + filesystem state as the CLI
- **Non-blocking** — always uses readonly snapshots for reads; transient connections for writes
- **Real-time** — polls for updates while phases are running so you can watch progress live
- **CLI parity** — anything you can do from the CLI, you can do from the GUI (and vice versa)

## Layout

Top-level navigation as a sidebar or top bar with these sections:

- **Dashboard** — overview of the active project
- **Stories** — current story browser (upgraded)
- **Timeline** — current dependency Gantt view (upgraded)
- **Terminal** — embedded terminal with tabs
- **Files** — file browser and editor
- **Settings** — model selection, global config, project config
- **Solutions** — compound learning knowledge base

A persistent **status bar** at the bottom shows: active project, current phase, active story, elapsed time, cumulative cost, model in use.

## Sections

### 1. Dashboard

The landing page when you open the GUI. Replaces `pralph query --report`.

- **Phase progress** — visual pipeline showing plan → stories → webgen → implement, with each phase's status (not started / running / completed), iteration count, and cost
- **Story summary** — donut or bar chart of stories by status (pending, in_progress, implemented, error, skipped)
- **Cost breakdown** — cost by phase, cost per story, grand total; bar chart + table
- **Cost projection** — based on average cost per story so far, estimate remaining cost to complete the backlog
- **Time estimates** — average duration per story, estimated time to complete remaining stories
- **Activity feed** — live stream of recent status_log and run_log entries (story started, story completed, error occurred, phase finished)
- **Active work** — what's currently running: phase, story ID, iteration number, elapsed time, cost so far. If parallel > 1, show all active workers

### 2. Stories (upgraded)

Extends the current story viewer.

- Everything the current viewer does (sidebar, detail panel, filtering, editing, badges)
- **Bulk actions** — multi-select stories to change status, priority, or category in batch
- **Drag-and-drop priority** — reorder stories by dragging
- **Inline add** — quick-add a story without leaving the page (calls the `add` logic)
- **Review feedback** — show/edit review feedback markdown for each story (from `.pralph/review-feedback/`)
- **Error details** — for stories in error status, show error_reason, error_output, and error_at from metadata; button to reset to pending
- **Token usage** — show per-story token breakdown (input, output, cache read, cache creation) from run_log
- **Status history** — expandable timeline of status transitions from status_log

### 3. Timeline (upgraded)

Extends the current Gantt/dependency view.

- **Status colors** — color cards by current status with a legend
- **Click-to-detail** — clicking a card opens the story detail panel (same as Stories tab)
- **Critical path** — highlight the longest dependency chain
- **Progress overlay** — shade implemented stories differently from pending ones to show how far through the dependency graph you are
- **Zoom/pan** — for large backlogs, allow zooming and panning the timeline

### 4. Terminal

Embedded terminal with tab support. This is where you run pralph commands and git operations without leaving the GUI.

- **Multiple tabs** — open several terminal sessions side by side or in tabs
- **Prebuilt actions** — toolbar buttons for common operations:
  - Run phase: plan, stories, webgen, implement (with option dialogs for flags)
  - Git: pull, push, rebase, status, log, diff
  - pralph: reset-errors, export-solutions, query
- **Output streaming** — show Claude's streaming output in real-time when running phases
- **Interactive takeover** — support the ESC-to-takeover flow that the CLI already has
- **History** — command history persists across sessions

### 5. Files

A lightweight file browser and editor for the project directory. Not a full IDE, but enough to view and make small edits without switching to another tool.

- **Tree view** — collapsible directory tree of the project, with `.pralph/` highlighted
- **File viewer** — syntax-highlighted read-only view for code files
- **Markdown preview** — rendered preview for `.md` files (design doc, guardrails, review feedback, solutions)
- **Quick edit** — open a file in a simple editor for small changes (design-doc.md, guardrails.md, prompt overrides, ideas.md, extra-tools.txt)
- **Design doc** — dedicated view for `.pralph/design-doc.md` with rendered markdown
- **Guardrails** — dedicated view for `.pralph/guardrails.md`
- **Diff view** — show git diff for modified files

### 6. Settings

Configuration panel that reads from and writes to the actual config files.

**Claude settings** (from `~/.claude/settings.json`):

- Model selection — dropdown showing available models (opus, sonnet, haiku) with their ARNs from the settings file
- Default model override per project

**pralph global options:**

- Max iterations
- Max budget USD
- Cooldown seconds
- Verbose mode
- Dangerously skip permissions toggle
- Extra tools (global)

**Project config:**

- Project name / ID (read-only, from `.pralph/project.json`)
- Working directory (read-only)
- Extra tools (project-level, from `.pralph/extra-tools.txt`)
- Prompt overrides — list which prompt templates have project-level or home-level overrides, with links to edit them in the Files section

### 7. Solutions

Browse and manage the compound learning knowledge base. Replaces `pralph export-solutions` for browsing, and makes the knowledge base more accessible.

- **Index view** — searchable, filterable table of all solutions (title, category, tags, story ID, created date)
- **Detail view** — rendered markdown of the full solution content
- **Search** — keyword search across title, tags, and error signatures (same as the existing `search_solutions` logic)
- **Export** — export all or filtered solutions as markdown or JSON (same as `export-solutions` CLI command)
- **Cross-project** — dropdown to switch between projects and browse solutions from other projects for reference

### 8. Project switcher

The GUI should support switching between projects without restarting.

- **Project list** — from the `projects` DuckDB table, show all registered projects with name, creation date, and story counts
- **Switch project** — clicking a project changes the active context (project_id, project_dir, state_dir)
- **Working directory** — display the resolved project directory; allow changing it via a directory picker or text input
- **Worktree support** — if the project is a git repo, show available worktrees and allow switching between them

## API design

The current viewer has three GET endpoints and one PUT. The full GUI needs a broader API.

### Read endpoints (all use readonly snapshots)

- `GET /api/stories` — all stories for active project
- `GET /api/stories/:id` — single story detail
- `GET /api/status` — status log entries
- `GET /api/tokens` — per-story token usage
- `GET /api/report` — full report data (same as `_gather_report_data`)
- `GET /api/phases` — phase state for all phases
- `GET /api/solutions` — solutions index
- `GET /api/solutions/:filename` — solution content
- `GET /api/run-log` — recent run log entries (with pagination)
- `GET /api/projects` — all registered projects
- `GET /api/settings` — current settings (model, max_iterations, etc.)
- `GET /api/files/:path` — read a file from the project directory
- `GET /api/files` — directory tree listing
- `GET /api/git/status` — git status of the project directory

### Write endpoints (use transient connections with retry)

- `PUT /api/stories/:id` — update a story (existing)
- `POST /api/stories` — create a new story (wraps `add` logic)
- `POST /api/stories/bulk` — bulk status/priority update
- `PUT /api/settings` — update settings
- `PUT /api/files/:path` — write a file
- `POST /api/reset-errors` — reset error stories to pending
- `POST /api/export-solutions` — trigger export, return content

### Action endpoints (spawn CLI subprocesses)

- `POST /api/run/plan` — start `pralph plan` with provided options
- `POST /api/run/stories` — start `pralph stories`
- `POST /api/run/webgen` — start `pralph webgen`
- `POST /api/run/implement` — start `pralph implement` with options
- `POST /api/run/compound` — start `pralph compound`
- `POST /api/run/refine` — start `pralph refine`
- `DELETE /api/run` — kill running subprocess
- `GET /api/run/output` — stream subprocess output via SSE

### WebSocket (alternative to polling)

- `ws://localhost:PORT/ws` — push events for status changes, phase progress, new run_log entries. Falls back to polling if WebSocket isn't available.

## Technology choices

The current viewer is a single Python file with inline HTML/CSS/JS served from `http.server`. For the full GUI, options to consider:

**Option A: Keep it simple — enhanced inline SPA**

- Same approach as today: Python `http.server` backend, single HTML file with vanilla JS
- Add more API endpoints, more JS for the new sections
- Pros: zero dependencies, single file, no build step
- Cons: gets unwieldy at this scope, no component model, hard to maintain

**Option B: Lightweight frontend framework + Python backend**

- Backend: FastAPI or Flask for the API server
- Frontend: Preact/htmx/Alpine.js or similar lightweight framework, bundled as static assets
- Terminal: xterm.js for the embedded terminal
- Pros: maintainable at scale, component model, good terminal emulation
- Cons: adds frontend build step and dependencies

**Option C: Electron/Tauri wrapper**

- Package the web app as a desktop application
- Pros: native feel, file system access, system tray
- Cons: heavy dependency, overkill for a local tool

Recommendation: **Option B** — the scope of this redesign warrants a real frontend framework, but keep it lightweight. The Python backend already exists and just needs more endpoints. xterm.js is well-proven for browser terminals.

## Migration path

This doesn't need to be built all at once. Incremental phases:

1. **API expansion** — add the read endpoints to the existing Python server, keep the current viewer working
2. **Dashboard** — add the dashboard as a new tab in the existing viewer, consuming the new API
3. **Settings + project switcher** — add settings panel and multi-project support
4. **Terminal** — integrate xterm.js with a pty backend
5. **Files** — add the file browser and editor
6. **Solutions** — add the solutions browser
7. **Stories/Timeline upgrades** — bulk actions, drag-and-drop, enhanced timeline

Each phase is independently useful and ships as a standalone improvement.

## Open questions

- Should the GUI auto-start when running `pralph implement`, or remain a separate `pralph viewer` command?
- Should the terminal be a full PTY (can run any shell command) or restricted to pralph/git commands?
- Should settings changes write back to `~/.claude/settings.json` directly, or maintain a separate pralph config?
- Is there value in a notification system (desktop notifications when a story completes or errors)?
- Should the GUI support viewing Claude's conversation history (from `.claude/projects/` session JSONL files)?
