from __future__ import annotations

import json
import threading
from pathlib import Path

import duckdb

from pralph import db
from pralph.db_state import DbStateMixin
from pralph.file_state import FileStateMixin
from pralph.migrate import migrate_project, needs_migration


class ProjectNotInitializedError(Exception):
    """Raised when a command is run in a directory without a project.json."""
    pass


class StateManager(FileStateMixin, DbStateMixin):
    def __init__(self, project_dir: str, *, project_name: str | None = None, readonly: bool = False) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.state_dir = self.project_dir / ".pralph"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._readonly = readonly
        self.__conn: duckdb.DuckDBPyConnection | None = None

        # Resolve project_id from project.json or create it
        self.project_id = self._resolve_project_id(project_name)

        # Initialize DuckDB — short-lived connection for setup only
        if readonly:
            pass  # readonly connections opened on demand
        else:
            with db.connection() as conn:
                db.register_project(conn, self.project_id, self.project_dir.name)
                if needs_migration(self.state_dir, self.project_id, conn):
                    migrate_project(self.state_dir, self.project_id, conn)

    @property
    def _conn(self) -> duckdb.DuckDBPyConnection:
        """Return the currently held DuckDB connection.

        All callers must be inside a _hold_conn() context. For read-only mode,
        a persistent snapshot connection is used.
        """
        if self.__conn is not None:
            return self.__conn
        if self._readonly:
            self.__conn = db.get_readonly_connection()
            return self.__conn
        raise RuntimeError("No held connection — wrap operation in _hold_conn()")

    def _hold_conn(self):
        """Context manager to hold a single connection open for batched operations.

        Usage:
            with self._hold_conn():
                self._conn.execute(...)  # reuses same connection
                self._conn.execute(...)
            # connection closed here
        """
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            if self.__conn is not None:
                yield  # already held (nested or readonly)
                return
            self.__conn = db.get_connection()
            try:
                yield
            finally:
                self.__conn.close()
                self.__conn = None

        return _cm()

    def refresh_readonly(self) -> None:
        """Re-snapshot the database so reads see the latest data."""
        if not self._readonly:
            return
        if self.__conn is not None:
            self.__conn.close()
        self.__conn = db.get_readonly_connection()

    def _transient_write(self, sql: str, params: list) -> None:
        """Execute a write via a short-lived connection to the real database.

        Retries briefly to handle transient lock contention with a running
        implement process (which releases the lock between iterations).
        """
        import time

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                with db.connection() as conn:
                    conn.execute(sql, params)
                return
            except duckdb.IOException as e:
                last_err = e
                time.sleep(0.5)
        raise last_err  # type: ignore[misc]

    @property
    def _project_config_path(self) -> Path:
        return self.state_dir / "project.json"

    def _resolve_project_id(self, project_name: str | None) -> str:
        """Resolve project_id: read from project.json, or create from project_name."""
        if self._project_config_path.exists():
            try:
                data = json.loads(self._project_config_path.read_text())
                stored_id = data.get("project_id", "")
                if stored_id:
                    return stored_id
            except (json.JSONDecodeError, OSError):
                pass

        if project_name:
            self._save_project_config(project_name)
            return project_name

        # Legacy project: has JSONL files but no project.json — auto-assign basename
        if self._has_legacy_data():
            legacy_name = self.project_dir.name
            self._save_project_config(legacy_name)
            return legacy_name

        # No project.json and no name provided — not initialized yet
        raise ProjectNotInitializedError(
            f"Project not initialized. Run 'pralph plan --name <project-name>' first.\n"
            f"  directory: {self.project_dir}"
        )

    def _has_legacy_data(self) -> bool:
        """Check if this project has old-style JSONL files (pre-DuckDB)."""
        return (
            (self.state_dir / "stories.jsonl").exists()
            or (self.state_dir / "phase-state.json").exists()
            or self.design_doc_path.exists()
        )

    def _save_project_config(self, project_id: str) -> None:
        self._project_config_path.write_text(
            json.dumps({"project_id": project_id}, indent=2) + "\n"
        )
