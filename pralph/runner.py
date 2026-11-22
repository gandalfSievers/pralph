from __future__ import annotations

import json
import os
import random
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False


@dataclass
class ClaudeResult:
    success: bool
    result: str = ""
    cost_usd: float = 0.0
    session_id: str = ""
    error: str = ""
    is_rate_limit: bool = False
    is_context_overflow: bool = False
    interrupted: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


_CONTEXT_OVERFLOW_MARKERS = (
    "prompt is too long",
    "context_length_exceeded",
    "context length exceeded",
    "input length",
    "input is too long",
    "input_tokens_exceed",
    "max_tokens_to_sample",
    "exceeds the maximum",
    "too many tokens",
    "context window",
)


def _is_context_overflow(error: str) -> bool:
    if not error:
        return False
    err = error.lower()
    return any(marker in err for marker in _CONTEXT_OVERFLOW_MARKERS)


# Tools per phase
PLAN_TOOLS = "Read,Glob,Grep,WebSearch,WebFetch,Edit,Write"
STORIES_TOOLS_EXTRACT = "Read,Glob,Grep"
STORIES_TOOLS_RESEARCH = "Read,Glob,Grep,WebSearch,WebFetch"
IMPLEMENT_TOOLS = "Read,Write,Edit,Bash,Glob,Grep"
REVIEW_TOOLS = "Read,Glob,Grep,Bash"
COMPOUND_TOOLS = "Read,Glob,Grep,Bash"
ADD_TOOLS = "Read,Glob,Grep"
IDEATE_TOOLS = "Read,Glob,Grep,WebSearch,WebFetch"
REFINE_TOOLS = "Read,Glob,Grep"
JUSTLOOP_TOOLS = "Read,Write,Edit,Bash,Glob,Grep"

# ── Elapsed timer ─────────────────────────────────────────────────────

_timer_lock = threading.Lock()
_timer_active = False


_timer_origin: list[float] = [0.0]  # mutable so timer thread sees resets


def _start_elapsed_timer(start_time: float) -> threading.Event:
    """Start a background thread that displays elapsed time on stderr."""
    global _timer_active
    _timer_active = False
    _timer_origin[0] = start_time
    stop = threading.Event()

    def _timer_fn():
        global _timer_active
        while not stop.is_set():
            with _timer_lock:
                elapsed = int(time.monotonic() - _timer_origin[0])
                m, s = divmod(elapsed, 60)
                sys.stderr.write(f"\r  \u23f3 running {m}:{s:02d}  \033[2m(ESC to interrupt)\033[0m")
                sys.stderr.flush()
                _timer_active = True
            stop.wait(1.0)

    t = threading.Thread(target=_timer_fn, daemon=True)
    t.start()
    return stop


def _clear_timer_line():
    """Clear the in-place timer line before printing an event."""
    global _timer_active
    if _timer_active:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
        _timer_active = False


def _reset_timer():
    """Reset the timer origin so it counts from now (call inside _timer_lock)."""
    _timer_origin[0] = time.monotonic()


def _stop_elapsed_timer(stop: threading.Event):
    """Stop the timer thread and clear the timer line."""
    stop.set()
    with _timer_lock:
        _clear_timer_line()


# ── Loop control: shrink max-iterations from the ESC menu ─────────────
#
# The ESC interrupt menu offers "set max iterations to current", which
# tells _run_loop to exit cleanly after the current iteration finishes.
# Implemented as a module-level flag so the interrupt handler (called
# from inside run_claude) can signal _run_loop in loop.py without
# threading state through every iteration_fn.

_max_iter_truncate = threading.Event()
_loop_context: dict = {}


def request_truncate_max_iter() -> None:
    _max_iter_truncate.set()


def cancel_truncate_max_iter() -> None:
    _max_iter_truncate.clear()


def consume_truncate_request() -> bool:
    """Return True (and clear) if a max-iter truncate request is pending."""
    if _max_iter_truncate.is_set():
        _max_iter_truncate.clear()
        return True
    return False


def is_truncate_requested() -> bool:
    return _max_iter_truncate.is_set()


def set_loop_context(phase: str, current: int, last: int) -> None:
    """Record loop position so the interrupt menu can show it.

    *last* is the last iteration number planned for this loop session
    (inclusive), i.e. start_iter + max_iterations - 1.
    """
    _loop_context["phase"] = phase
    _loop_context["current"] = current
    _loop_context["last"] = last


def clear_loop_context() -> None:
    _loop_context.clear()


def get_loop_context() -> dict:
    return dict(_loop_context)


@contextmanager
def _stdin_from_tty():
    """Temporarily point ``sys.stdin`` at /dev/tty when stdin isn't a TTY.

    When the user pipes a prompt at startup (``echo "..." | pralph justloop``),
    ``sys.stdin`` is the now-exhausted pipe and ``click.prompt`` can't read
    keystrokes. This re-opens /dev/tty for the duration of the block so any
    Python-level reads (click, input(), etc.) get the keyboard. No-op when
    stdin is already a TTY or /dev/tty is unavailable.

    Note: this only affects Python-level ``sys.stdin``. Subprocesses inherit
    file descriptor 0 directly and need ``stdin=...`` passed to ``Popen``.
    """
    if sys.stdin.isatty():
        yield
        return
    try:
        tty_fh = open("/dev/tty", "r")  # noqa: SIM115
    except OSError:
        yield
        return
    saved = sys.stdin
    sys.stdin = tty_fh
    try:
        yield
    finally:
        try:
            tty_fh.close()
        except OSError:
            pass
        sys.stdin = saved


def run_claude(
    prompt: str,
    *,
    model: str = "sonnet",
    allowed_tools: str = "",
    system_prompt: str = "",
    dangerously_skip_permissions: bool = False,
    max_budget_usd: float | None = None,
    timeout: int = 600,
    verbose: bool = False,
    project_dir: str | None = None,
    resume_session_id: str | None = None,
    session_id: str | None = None,
) -> ClaudeResult:
    """Invoke claude -p as a subprocess with streaming output.

    Monitors stdin for ESC key — on press, stops the subprocess (SIGSTOP) and
    presents an interrupt menu (continue / take over / skip / abort).
    """
    if resume_session_id:
        session_id = resume_session_id
        cmd = [
            "claude", "-p", "--resume", resume_session_id,
            "--verbose", "--output-format", "stream-json",
        ]
        if dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
    else:
        session_id = session_id or str(uuid.uuid4())
        cmd = [
            "claude", "-p", "--verbose", "--model", model,
            "--output-format", "stream-json", "--session-id", session_id,
        ]

        if allowed_tools:
            cmd.extend(["--allowedTools", allowed_tools])

        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        if dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")

        if max_budget_usd is not None:
            cmd.extend(["--max-budget-usd", str(max_budget_usd)])

    cwd = project_dir or None

    if verbose:
        _print_debug(cmd, prompt)

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
    except FileNotFoundError:
        return ClaudeResult(
            success=False,
            error="claude CLI not found — install it: https://docs.anthropic.com/en/docs/claude-code",
        )

    # Send prompt on stdin, then close
    assert proc.stdin is not None
    if prompt:
        proc.stdin.write(prompt)
    proc.stdin.close()

    # Read streaming NDJSON lines from stdout, printing progress as we go
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_fd = proc.stdout.fileno()

    # Start ESC key monitor
    monitor_state = _start_esc_monitor(proc)

    # Start elapsed time counter
    start_time = time.monotonic()
    timer_stop = _start_elapsed_timer(start_time)

    final_result: dict | None = None
    all_assistant_text: list[str] = []  # accumulate all assistant text blocks
    deadline = time.monotonic() + timeout
    error_result: ClaudeResult | None = None
    buf = ""

    try:
        while True:
            # Check if interrupted by ESC (process is SIGSTOP'd)
            if monitor_state and monitor_state[0].is_set():
                _stop_elapsed_timer(timer_stop)
                tty_file = monitor_state[3]
                _stop_esc_monitor(monitor_state)

                choice, verbose = _handle_interrupt(session_id, cwd, tty_file=tty_file, verbose=verbose)

                # Close /dev/tty if we opened it; a new one is opened if we continue
                if tty_file is not None:
                    try:
                        tty_file.close()
                    except OSError:
                        pass

                if choice == "continue":
                    try:
                        proc.send_signal(signal.SIGCONT)
                    except OSError:
                        pass
                    timer_stop = _start_elapsed_timer(time.monotonic())
                    monitor_state = _start_esc_monitor(proc)
                    continue
                elif choice == "takeover":
                    import click as _click
                    proc.kill()
                    proc.wait()
                    _click.echo(_click.style("\n  Entering interactive mode...\n", fg="cyan", bold=True))
                    resume_interactive(session_id, cwd, dangerously_skip_permissions)
                    post = _post_takeover_menu(session_id)
                    if post == "resume":
                        return run_claude(
                            "Continue where you left off.",
                            resume_session_id=session_id,
                            dangerously_skip_permissions=dangerously_skip_permissions,
                            timeout=timeout,
                            verbose=verbose,
                            project_dir=project_dir,
                        )
                    elif post == "implemented":
                        return ClaudeResult(
                            success=True,
                            result=json.dumps({
                                "status": "implemented",
                                "summary": "Completed via interactive takeover",
                            }),
                            session_id=session_id,
                            interrupted=True,
                        )
                    elif post == "continue":
                        return ClaudeResult(
                            success=False, error="interrupted",
                            session_id=session_id, interrupted=True,
                        )
                    else:  # abort
                        return ClaudeResult(
                            success=False, error="aborted",
                            session_id=session_id, interrupted=True,
                        )
                elif choice == "skip":
                    proc.kill()
                    proc.wait()
                    return ClaudeResult(
                        success=False, error="interrupted",
                        session_id=session_id, interrupted=True,
                    )
                else:  # abort
                    proc.kill()
                    proc.wait()
                    return ClaudeResult(
                        success=False, error="aborted",
                        session_id=session_id, interrupted=True,
                    )

            if time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                with _timer_lock:
                    _clear_timer_line()
                error_result = ClaudeResult(
                    success=False, error="timeout", session_id=session_id,
                )
                break

            # Wait for data with timeout so we can check the interrupt flag
            ready, _, _ = select.select([stdout_fd], [], [], 0.3)

            if not ready:
                if proc.poll() is not None:
                    break  # process ended
                continue

            chunk = os.read(stdout_fd, 8192)
            if not chunk:
                break  # EOF

            buf += chunk.decode("utf-8", errors="replace")

            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if verbose:
                        with _timer_lock:
                            _clear_timer_line()
                            print(f"  [stream] unparseable: {line[:120]}", file=sys.stderr)
                            _reset_timer()
                    continue

                etype = event.get("type", "")

                if etype == "result":
                    final_result = event
                    result_text = event.get("result", "")
                    if result_text:
                        with _timer_lock:
                            _clear_timer_line()
                            print(f"\n{result_text}", file=sys.stderr)
                            _reset_timer()
                    continue

                # Accumulate all assistant text for fallback parsing
                if etype == "assistant":
                    msg = event.get("message", {})
                    if isinstance(msg, dict):
                        for block in msg.get("content", []):
                            if isinstance(block, dict) and block.get("type") == "text":
                                t = block.get("text", "")
                                if t:
                                    all_assistant_text.append(t)

                with _timer_lock:
                    _clear_timer_line()
                    _print_event(event, verbose)
                    _reset_timer()

    except Exception as e:
        proc.kill()
        proc.wait()
        error_result = ClaudeResult(
            success=False, error=f"stream read error: {e}", session_id=session_id,
        )

    finally:
        _stop_elapsed_timer(timer_stop)
        _stop_esc_monitor(monitor_state)
        if monitor_state is not None and monitor_state[3] is not None:
            try:
                monitor_state[3].close()
            except OSError:
                pass

    if error_result:
        return error_result

    proc.wait()

    # Check stderr for errors
    stderr = proc.stderr.read().strip()

    if proc.returncode != 0 and final_result is None:
        is_rate = "rate" in stderr.lower() or "overloaded" in stderr.lower()
        err_text = stderr or f"exit code {proc.returncode}"
        return ClaudeResult(
            success=False,
            error=err_text,
            is_rate_limit=is_rate,
            is_context_overflow=_is_context_overflow(err_text),
            session_id=session_id,
        )

    if final_result is None:
        return ClaudeResult(
            success=False, error="no result event in stream", session_id=session_id,
        )

    result = _parse_result_event(final_result)
    if not result.session_id:
        result.session_id = session_id
    # If the result event had no text but we captured assistant text during
    # streaming, use the accumulated text so parsers can find structured output
    # that appeared in earlier assistant turns (e.g. before a tool call).
    if not result.result and all_assistant_text:
        result.result = "\n\n".join(all_assistant_text)
    return result


# ── ESC interrupt support ─────────────────────────────────────────────


def _start_esc_monitor(
    proc: subprocess.Popen,
) -> tuple[threading.Event, threading.Event, list, "io.BufferedReader | None"] | None:
    """Start monitoring stdin for ESC key.

    Returns (interrupted, stop, old_settings, tty_file) or None.
    tty_file is non-None when /dev/tty was opened because stdin is not a TTY.
    """
    if not _HAS_TERMIOS:
        return None

    tty_file = None
    try:
        stdin_fd = sys.stdin.fileno()
    except (ValueError, AttributeError):
        return None

    if os.isatty(stdin_fd):
        input_fd = stdin_fd
    else:
        try:
            tty_file = open("/dev/tty", "rb", buffering=0)  # noqa: SIM115
            input_fd = tty_file.fileno()
        except OSError:
            return None

    interrupted = threading.Event()
    stop = threading.Event()
    old_settings = termios.tcgetattr(input_fd)
    tty.setcbreak(input_fd)

    def monitor() -> None:
        try:
            while not stop.is_set():
                ready, _, _ = select.select([input_fd], [], [], 0.2)
                if stop.is_set():
                    return
                if ready:
                    ch = os.read(input_fd, 1)
                    if ch == b"\x1b":
                        interrupted.set()
                        try:
                            proc.send_signal(signal.SIGSTOP)
                        except OSError:
                            pass
                        return
        except (OSError, ValueError):
            pass

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return interrupted, stop, old_settings, tty_file


def _stop_esc_monitor(
    monitor_state: tuple[threading.Event, threading.Event, list, "io.BufferedReader | None"] | None,
) -> None:
    """Stop ESC monitor and restore terminal settings."""
    if monitor_state is None:
        return
    _, stop, old_settings, tty_file = monitor_state
    stop.set()
    try:
        if tty_file is not None:
            termios.tcsetattr(tty_file.fileno(), termios.TCSADRAIN, old_settings)
        else:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
    except (OSError, ValueError):
        pass


def _handle_interrupt(
    session_id: str,
    project_dir: str | None,
    tty_file: "io.BufferedReader | None" = None,
    verbose: bool = False,
) -> tuple[str, bool]:
    """Show interrupt menu and return (choice, verbose).

    Choice is one of: 'continue', 'takeover', 'skip', or 'abort'.
    verbose is the (possibly toggled) verbose state.
    """
    import click

    # Flush any stale input from the ESC detection
    try:
        flush_fd = tty_file.fileno() if tty_file is not None else sys.stdin.fileno()
        termios.tcflush(flush_fd, termios.TCIFLUSH)
    except (OSError, ValueError):
        pass

    verbose_label = "Off" if verbose else "On"

    ctx = get_loop_context()
    truncate_pending = is_truncate_requested()

    with _stdin_from_tty():
        click.echo()
        click.echo(click.style("  ⏸  Interrupted", fg="yellow", bold=True))
        if ctx:
            cur = ctx.get("current", 0)
            last = ctx.get("last", 0)
            remaining_after = max(0, last - cur)
            phase = ctx.get("phase", "")
            click.echo(click.style(
                f"  Phase '{phase}': iteration {cur} (last planned: {last}, "
                f"{remaining_after} more after this)", dim=True,
            ))
        if truncate_pending:
            click.echo(click.style(
                "  ⓘ Pending: this is the last iteration (max-iterations truncated)",
                fg="cyan",
            ))
        click.echo()
        click.echo("  [1] Continue  — resume where it left off")
        click.echo("  [2] Take over — open interactive session")
        click.echo("  [3] Skip      — continue to next iteration")
        if truncate_pending:
            click.echo("  [4] Cancel finish — keep running remaining iterations")
        else:
            click.echo("  [4] Finish    — set max-iterations to current, exit cleanly after this one")
        click.echo("  [5] Abort     — stop the loop now")
        click.echo()
        click.echo(f"  [6] Toggle verbose ({verbose_label})")
        click.echo()
        choice = click.prompt(
            "  Choice", type=click.Choice(["1", "2", "3", "4", "5", "6"]), default="1",
        )

    if choice == "6":
        verbose = not verbose
        state = "on" if verbose else "off"
        click.echo(click.style(f"  Verbose {state}", fg="cyan"))
        return "continue", verbose
    elif choice == "1":
        return "continue", verbose
    elif choice == "2":
        return "takeover", verbose
    elif choice == "3":
        return "skip", verbose
    elif choice == "4":
        if truncate_pending:
            cancel_truncate_max_iter()
            click.echo(click.style("  Loop will continue normally", fg="cyan"))
        else:
            request_truncate_max_iter()
            click.echo(click.style(
                "  Loop will exit cleanly after this iteration finishes",
                fg="cyan",
            ))
        return "continue", verbose
    else:
        return "abort", verbose


def _post_takeover_menu(session_id: str) -> str:
    """Ask user about story status after interactive takeover.

    Returns: 'implemented', 'resume', 'continue', or 'abort'.
    """
    import click

    with _stdin_from_tty():
        click.echo()
        click.echo(click.style("  Interactive session ended", fg="cyan", bold=True))
        click.echo()
        click.echo("  [1] Resume              — continue automated session")
        click.echo("  [2] Mark as implemented — story complete")
        click.echo("  [3] Continue loop       — story resets to pending")
        click.echo("  [4] Abort loop")
        click.echo()
        choice = click.prompt(
            "  Choice", type=click.Choice(["1", "2", "3", "4"]), default="1",
        )

    if choice == "1":
        return "resume"
    elif choice == "2":
        return "implemented"
    elif choice == "3":
        return "continue"
    else:
        return "abort"


def resume_interactive(
    session_id: str,
    project_dir: str | None = None,
    dangerously_skip_permissions: bool = False,
) -> int:
    """Resume a claude session interactively, inheriting the terminal.

    When pralph was invoked with a piped prompt (``echo "..." | pralph ...``)
    fd 0 is a closed pipe. Inherited as-is, the interactive ``claude`` would
    receive immediate EOF and the user couldn't type anything. Open /dev/tty
    explicitly in that case so the interactive session works.
    """
    cmd = ["claude", "--resume", session_id]
    if dangerously_skip_permissions:
        cmd.append("--dangerously-skip-permissions")

    tty_fd: int | None = None
    if not os.isatty(0):
        try:
            tty_fd = os.open("/dev/tty", os.O_RDONLY)
        except OSError:
            tty_fd = None
    try:
        return subprocess.call(cmd, cwd=project_dir, stdin=tty_fd)
    finally:
        if tty_fd is not None:
            try:
                os.close(tty_fd)
            except OSError:
                pass


def _parse_result_event(data: dict) -> ClaudeResult:
    """Parse the final 'result' event from the stream."""
    subtype = data.get("subtype", "")
    is_success = subtype in ("success", "error_max_turns")

    usage = data.get("usage", {})

    err_text = "" if is_success else data.get("error", subtype or "unknown")
    return ClaudeResult(
        success=is_success,
        result=data.get("result", ""),
        cost_usd=data.get("total_cost_usd", data.get("cost_usd", 0.0)),
        session_id=data.get("session_id", ""),
        error=err_text,
        is_rate_limit="rate" in data.get("error", "").lower(),
        is_context_overflow=_is_context_overflow(err_text),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
    )


def _print_event(event: dict, verbose: bool) -> None:
    """Print a streaming event as compact progress."""
    import click as _click

    etype = event.get("type", "")

    if etype == "assistant":
        # message is the full API message object with content blocks
        msg = event.get("message", {})
        if isinstance(msg, dict):
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        _click.echo(text, err=True)
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    tool = block.get("name", "?")
                    inp = block.get("input", {})
                    hint = _tool_hint(inp)
                    _click.echo(_click.style(f"  🔧 {tool}", fg='cyan') + _click.style(hint, dim=True), err=True)

    elif etype == "tool_use":
        tool = event.get("tool", event.get("name", "?"))
        hint = _tool_hint(event.get("input", {}))
        _click.echo(_click.style(f"  🔧 {tool}", fg='cyan') + _click.style(hint, dim=True), err=True)

    elif etype == "tool_result":
        content = str(event.get("content", ""))
        if verbose:
            _click.echo(_click.style(f"  📋 {content}", dim=True), err=True)
        elif content:
            # Show a brief snippet by default
            first_line = content.split("\n")[0][:200]
            _click.echo(_click.style(f"  📋 {first_line}", dim=True), err=True)

    elif etype == "user":
        # Tool result coming back from a tool call
        output = ""
        tool_result = event.get("tool_use_result")
        if isinstance(tool_result, dict):
            stdout = tool_result.get("stdout", "")
            stderr = tool_result.get("stderr", "")
            output = stderr if stderr else stdout
        if not output:
            # Fallback: extract content from message.content tool_result blocks
            msg = event.get("message", {})
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        c = block.get("content", "")
                        if isinstance(c, str) and c:
                            output = c
                            break
        if output:
            if verbose:
                _click.echo(_click.style(f"  📋 {output}", dim=True), err=True)
            else:
                lines = output.strip().split("\n")
                preview = lines[0]
                if len(lines) > 1:
                    preview += _click.style(f" (+{len(lines)-1} lines)", dim=True)
                _click.echo(_click.style(f"  📋 {preview}", dim=True), err=True)

    elif etype == "rate_limit_event":
        info = event.get("rate_limit_info", {})
        status = info.get("status", "unknown")
        if status != "allowed":
            limit_type = info.get("rateLimitType", "")
            resets_at = info.get("resetsAt", "")
            _click.echo(_click.style(f"  ⚠ rate limited ({limit_type}, resets {resets_at})", fg='yellow'), err=True)

    elif etype == "system":
        if verbose:
            subtype = event.get("subtype", "")
            model = event.get("model", "")
            cwd = event.get("cwd", "")
            version = event.get("claude_code_version", "")
            parts = [s for s in [subtype, model, cwd, version] if s]
            _click.echo(_click.style(f"  [system] {' | '.join(parts)}", dim=True), err=True)

    elif verbose:
        _click.echo(_click.style(f"  [{etype}] {str(event)[:120]}", dim=True), err=True)


def _tool_hint(tool_input: dict) -> str:
    """Extract a brief hint from tool input for display."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "pattern", "file_path", "query", "url", "old_string"):
        if key in tool_input:
            val = str(tool_input[key])[:80]
            return f" → {val}"
    return ""


def run_with_retry(
    prompt: str,
    *,
    max_retries: int = 4,
    verbose: bool = False,
    **kwargs,
) -> ClaudeResult:
    """Run claude with exponential backoff on rate limits."""
    delays = [30, 60, 120, 240]
    last_result = None

    for attempt in range(max_retries + 1):
        result = run_claude(prompt, verbose=verbose, **kwargs)
        last_result = result

        if result.success:
            return result

        if result.interrupted:
            return result

        if result.is_context_overflow:
            # Retrying with the same prompt won't help — bail immediately so
            # the caller can chunk down or surface the error.
            print(
                "  [retry] prompt too large for model context — not retrying",
                file=sys.stderr,
            )
            return result

        if result.error == "timeout" and attempt == 0:
            original_timeout = kwargs.get("timeout", 600)
            kwargs["timeout"] = original_timeout * 2
            kwargs.pop("session_id", None)  # avoid "already in use" on retry
            print(f"  [retry] timeout — retrying with {kwargs['timeout']}s", file=sys.stderr)
            continue

        if result.is_rate_limit and attempt < max_retries:
            delay = delays[min(attempt, len(delays) - 1)]
            jitter = random.uniform(0, delay * 0.2)
            wait = delay + jitter
            kwargs.pop("session_id", None)  # avoid "already in use" on retry
            print(f"  [retry] rate limited — waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue

        # Non-retryable error
        break

    return last_result  # type: ignore[return-value]


def _print_debug(cmd: list[str], prompt: str) -> None:
    safe_cmd = " ".join(cmd)
    print(f"  [claude] {safe_cmd}", file=sys.stderr)
    print(f"  [prompt] {prompt}", file=sys.stderr)
