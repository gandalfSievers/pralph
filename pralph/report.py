"""Shared reporting logic used by both the CLI and the viewer."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from pralph import db


BUILTIN_QUERIES = {
    "progress": (
        "Story progress by status",
        "SELECT status, COUNT(*) as count FROM stories WHERE project_id = ? GROUP BY status ORDER BY count DESC",
    ),
    "cost": (
        "Cost breakdown by phase",
        """SELECT phase,
                  COUNT(*) as iterations,
                  ROUND(SUM(cost_usd), 4) as total_cost,
                  SUM(input_tokens) as input_tokens,
                  SUM(output_tokens) as output_tokens
           FROM run_log WHERE project_id = ?
           GROUP BY phase ORDER BY total_cost DESC""",
    ),
    "stories": (
        "All stories",
        "SELECT id, title, status, priority, category, complexity FROM stories WHERE project_id = ? ORDER BY priority, id",
    ),
    "cost-per-story": (
        "Cost per story",
        """SELECT story_id, COUNT(*) as iterations,
                  ROUND(SUM(cost_usd), 4) as total_cost,
                  SUM(input_tokens) as input_tokens,
                  SUM(output_tokens) as output_tokens
           FROM run_log WHERE project_id = ? AND story_id != ''
           GROUP BY story_id ORDER BY total_cost DESC""",
    ),
    "errors": (
        "Recent errors",
        """SELECT iteration, phase, story_id, error, ROUND(duration, 1) as duration_s
           FROM run_log WHERE project_id = ? AND success = false AND error != ''
           ORDER BY logged_at DESC LIMIT 20""",
    ),
    "timeline": (
        "Implementation timeline",
        """SELECT story_id, phase, success, ROUND(cost_usd, 4) as cost, ROUND(duration, 1) as duration_s, logged_at
           FROM run_log WHERE project_id = ? AND story_id != ''
           ORDER BY logged_at""",
    ),
    "projects": (
        "All registered projects",
        "SELECT project_id, name, created_at FROM projects ORDER BY created_at DESC",
    ),
}


def read_project_id(project_dir: str) -> str:
    """Read project_id from .pralph/project.json without opening a write connection."""
    config = Path(project_dir) / ".pralph" / "project.json"
    if not config.exists():
        raise FileNotFoundError(
            f"Project not initialized. Run 'pralph plan --name <project-name>' first.\n"
            f"  directory: {project_dir}"
        )
    data = json.loads(config.read_text())
    pid = data.get("project_id", "")
    if not pid:
        raise ValueError(f"project_id not set in {config}")
    return pid


def gather_report_data(project_id: str) -> dict:
    """Gather all data needed for the progress report from DuckDB (read-only).

    Returns a dict consumable by both the CLI printer and the viewer API.
    """
    conn = db.get_readonly_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM phase_state WHERE project_id = ? ORDER BY phase", [project_id]
        )
        cols = [d[0] for d in rows.description]
        phase_states = [dict(zip(cols, r)) for r in rows.fetchall()]

        current_phase = {}
        for ps in phase_states:
            if not ps.get("completed", False):
                current_phase = ps
                break
        if not current_phase and phase_states:
            current_phase = phase_states[-1]

        rows = conn.execute(
            "SELECT id, title, status, priority, category, complexity FROM stories WHERE project_id = ? ORDER BY priority, id",
            [project_id],
        )
        cols = [d[0] for d in rows.description]
        stories = {r[0]: dict(zip(cols, r)) for r in rows.fetchall()}

        rows = conn.execute(
            "SELECT status, COUNT(*) FROM stories WHERE project_id = ? GROUP BY status", [project_id]
        )
        status_counts = {r[0]: r[1] for r in rows.fetchall()}

        rows = conn.execute(
            """SELECT story_id,
                      COUNT(*) as iterations,
                      COALESCE(SUM(cost_usd), 0) as cost_usd,
                      COALESCE(SUM(duration), 0) as duration,
                      LIST(impl_status) as statuses
               FROM run_log
               WHERE project_id = ? AND story_id != '' AND mode = 'implement'
               GROUP BY story_id
               ORDER BY cost_usd DESC""",
            [project_id],
        )
        story_costs = {}
        for r in rows.fetchall():
            statuses = r[4] if r[4] else []
            story_costs[r[0]] = {
                "iterations": r[1],
                "cost_usd": r[2],
                "duration": r[3],
                "statuses": [s for s in statuses if s],
            }

        rows = conn.execute(
            "SELECT phase, COALESCE(SUM(cost_usd), 0) FROM run_log WHERE project_id = ? GROUP BY phase",
            [project_id],
        )
        phase_costs = {r[0]: r[1] for r in rows.fetchall()}

        row = conn.execute(
            "SELECT COALESCE(SUM(duration), 0) FROM run_log WHERE project_id = ?", [project_id]
        ).fetchone()
        total_duration = row[0] if row else 0.0

        active_story = current_phase.get("active_story_id", "") or None

        row = conn.execute(
            "SELECT phase, mode, story_id, impl_status FROM run_log WHERE project_id = ? ORDER BY logged_at DESC LIMIT 1",
            [project_id],
        ).fetchone()
        last_entry = None
        if row:
            last_entry = {"phase": row[0], "mode": row[1], "story_id": row[2], "impl_status": row[3]}

        # Cost projection
        implemented = status_counts.get("implemented", 0)
        pending = status_counts.get("pending", 0) + status_counts.get("rework", 0)
        avg_cost = sum(sc["cost_usd"] for sc in story_costs.values()) / max(implemented, 1) if story_costs else 0
        avg_duration = sum(sc["duration"] for sc in story_costs.values()) / max(implemented, 1) if story_costs else 0
    finally:
        conn.close()

    return {
        "phase_states": phase_states,
        "current_phase": current_phase,
        "stories": stories,
        "story_costs": story_costs,
        "phase_costs": phase_costs,
        "status_counts": status_counts,
        "total_duration": total_duration,
        "active_story": active_story,
        "last_entry": last_entry,
        "grand_total_cost": round(sum(phase_costs.values()), 4),
        "projection": {
            "implemented": implemented,
            "remaining": pending,
            "avg_cost_per_story": round(avg_cost, 4),
            "avg_duration_per_story": round(avg_duration, 1),
            "estimated_remaining_cost": round(avg_cost * pending, 2),
            "estimated_remaining_duration": round(avg_duration * pending, 1),
        },
    }


# -- formatting helpers --

def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    from datetime import timedelta
    td = timedelta(seconds=int(seconds))
    parts = []
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if td.days:
        hours += td.days * 24
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs:
        parts.append(f"{secs}s")
    return " ".join(parts) if parts else "0s"


def format_cost(cost: float) -> str:
    return f"${cost:.2f}"


def format_table(columns: list[str], rows: list[tuple]) -> str:
    """Format query results as an aligned text table."""
    if not rows:
        return "(no results)"
    str_rows = [[str(v) if v is not None else "" for v in row] for row in rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))
    header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    separator = "  ".join("-" * widths[i] for i in range(len(columns)))
    lines = [header, separator]
    for row in str_rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(columns))))
    return "\n".join(lines)


def format_csv(columns: list[str], rows: list[tuple]) -> str:
    """Format query results as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def format_json(columns: list[str], rows: list[tuple]) -> str:
    """Format query results as JSON."""
    data = [dict(zip(columns, row)) for row in rows]
    return json.dumps(data, indent=2, default=str)


# -- CLI report printers --

def print_report(data: dict) -> None:
    """Print the combined progress report to stdout via click."""
    import click

    ps = data["current_phase"]
    if not ps:
        click.echo("No phase state found. Has pralph been run in this project?")
        return

    click.echo("=" * 60)
    click.echo("  PRALPH PROGRESS REPORT")
    click.echo("=" * 60)

    click.echo()
    for phase_data in data["phase_states"]:
        phase = phase_data.get("phase", "?")
        iteration = phase_data.get("current_iteration", 0)
        completed = phase_data.get("completed", False)
        cost = phase_data.get("total_cost_usd", 0.0)
        status_str = "COMPLETED" if completed else "running"
        if completed:
            reason = phase_data.get("completion_reason", "")
            if reason:
                status_str += f" ({reason})"
        click.echo(f"  {phase:<12} iter={iteration:<4} cost={format_cost(cost):<10} {status_str}")

    last_summary = ps.get("last_summary", "")
    last_error = ps.get("last_error", "")
    if last_summary:
        click.echo(f"\n  Last result: {last_summary[:200]}")
    if last_error:
        click.echo(f"  Last error:  {last_error[:200]}")

    click.echo()
    click.echo("-" * 60)
    click.echo("  STORY SUMMARY")
    click.echo("-" * 60)
    counts = data["status_counts"]
    total_stories = sum(counts.values())
    click.echo(f"  Total stories: {total_stories}")
    click.echo()
    priority_order = ["implemented", "in_progress", "rework", "pending"]
    seen = set()
    for status in priority_order:
        if status in counts:
            click.echo(f"    {status:<20} {counts[status]:>3}")
            seen.add(status)
    for status, count in sorted(counts.items()):
        if status not in seen:
            click.echo(f"    {status:<20} {count:>3}")

    story_costs = data["story_costs"]
    if story_costs:
        click.echo()
        click.echo("-" * 60)
        click.echo("  COST PER STORY")
        click.echo("-" * 60)
        click.echo(f"  {'Story':<12} {'Description':<40} {'Status':<15} {'Cost':>8} {'Duration':>10} {'Iters':>5}")
        click.echo(f"  {chr(9472) * 12} {chr(9472) * 40} {chr(9472) * 15} {chr(9472) * 8} {chr(9472) * 10} {chr(9472) * 5}")

        for story_id, sc in story_costs.items():
            story_data = data["stories"].get(story_id, {})
            status = story_data.get("status", "unknown")
            title = story_data.get("title", "")
            if len(title) > 38:
                title = title[:35] + "..."
            final_status = sc["statuses"][-1] if sc["statuses"] else status
            click.echo(
                f"  {story_id:<12} {title:<40} {final_status:<15} {format_cost(sc['cost_usd']):>8} "
                f"{format_duration(sc['duration']):>10} {sc['iterations']:>5}"
            )

        impl_total = sum(sc["cost_usd"] for sc in story_costs.values())
        impl_duration = sum(sc["duration"] for sc in story_costs.values())
        click.echo(f"  {chr(9472) * 12} {chr(9472) * 40} {chr(9472) * 15} {chr(9472) * 8} {chr(9472) * 10} {chr(9472) * 5}")
        click.echo(
            f"  {'TOTAL':<12} {'':<40} {'':<15} {format_cost(impl_total):>8} "
            f"{format_duration(impl_duration):>10} {sum(sc['iterations'] for sc in story_costs.values()):>5}"
        )

    phase_costs = data["phase_costs"]
    if phase_costs:
        click.echo()
        click.echo("-" * 60)
        click.echo("  COST BY PHASE")
        click.echo("-" * 60)
        phase_order = ["plan", "stories", "webgen", "implement"]
        seen = set()
        for p in phase_order:
            if p in phase_costs:
                click.echo(f"    {p:<20} {format_cost(phase_costs[p]):>10}")
                seen.add(p)
        for p, cost in sorted(phase_costs.items()):
            if p not in seen:
                click.echo(f"    {p:<20} {format_cost(cost):>10}")
        grand_total = sum(phase_costs.values())
        click.echo(f"    {chr(9472) * 20} {chr(9472) * 10}")
        click.echo(f"    {'GRAND TOTAL':<20} {format_cost(grand_total):>10}")

    click.echo()
    click.echo("-" * 60)
    click.echo("  CURRENTLY ACTIVE")
    click.echo("-" * 60)
    active = data["active_story"]
    if active:
        story_data = data["stories"].get(active, {})
        title = story_data.get("title", "")
        category = story_data.get("category", "")
        sc = data["story_costs"].get(active, {})
        cost_so_far = sc.get("cost_usd", 0.0) if sc else 0.0
        iters = sc.get("iterations", 0) if sc else 0
        click.echo(f"  Story:    {active}")
        if title:
            click.echo(f"  Title:    {title}")
        if category:
            click.echo(f"  Category: {category}")
        if cost_so_far > 0:
            click.echo(f"  Cost:     {format_cost(cost_so_far)} ({iters} iterations)")
    elif data["current_phase"].get("completed"):
        click.echo("  All work completed.")
    else:
        last = data["last_entry"]
        if last:
            click.echo(f"  Phase: {last.get('phase', '?')}, Mode: {last.get('mode', '?')}")
        else:
            click.echo("  No activity recorded yet.")

    click.echo()
    click.echo("=" * 60)


def build_report_json(data: dict) -> str:
    """Build the progress report as JSON."""
    story_details = []
    for story_id, sc in data["story_costs"].items():
        story_data = data["stories"].get(story_id, {})
        story_details.append({
            "story_id": story_id,
            "title": story_data.get("title", ""),
            "status": story_data.get("status", "unknown"),
            "last_impl_status": sc["statuses"][-1] if sc["statuses"] else "",
            "cost_usd": round(sc["cost_usd"], 2),
            "duration_seconds": round(sc["duration"], 1),
            "iterations": sc["iterations"],
        })
    report = {
        "phase_states": data["phase_states"],
        "current_phase": data["current_phase"],
        "story_summary": data["status_counts"],
        "total_stories": sum(data["status_counts"].values()),
        "cost_by_phase": {k: round(v, 2) for k, v in data["phase_costs"].items()},
        "grand_total_cost": round(sum(data["phase_costs"].values()), 2),
        "total_duration_seconds": round(data["total_duration"], 1),
        "story_costs": story_details,
        "active_story": data["active_story"],
    }
    return json.dumps(report, indent=2, default=str)
