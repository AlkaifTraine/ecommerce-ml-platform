"""Independent audit that the training table cannot see the future.

This deliberately does NOT reuse build_features' SQL. It recomputes the
truncation logic from the raw events and checks the published training table
against it. A bug copied into both would defeat the point, so the audit is
written from the definition rather than from the implementation.

Checks
------
prefix_size        every row's session has exactly k events at or before its cutoff
no_prepurchase     no session purchased at or before its own cutoff
label_provenance   y=1 iff a purchase exists strictly after the cutoff
no_future_events   no cutoff_time is at or after the clock bound (when set)
no_forbidden_cols  identifiers and timestamps never reach the feature matrix
split_disjoint     train/valid/test/drift windows do not overlap in time

Run standalone:
    python -m src.features.leakage_audit --k 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import duckdb

from src.platform_core import get_logger, get_settings

log = get_logger(__name__)

# Anything here would either identify the row or encode the outcome's timing.
FORBIDDEN_FEATURE_COLS = {
    "session_key", "user_id", "session_start", "cutoff_time", "session_date",
    "y", "bought", "carted", "first_purchase_at", "revenue", "session_end",
    "n_events", "duration_sec",
}


@dataclass
class AuditResult:
    checks: dict[str, int] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(v == 0 for v in self.checks.values())

    def report(self) -> str:
        lines = []
        for name, violations in self.checks.items():
            status = "PASS" if violations == 0 else f"FAIL ({violations:,} violations)"
            lines.append(f"  {name:<20} {status}")
            if violations and name in self.details:
                lines.append(f"       -> {self.details[name]}")
        return "\n".join(lines)


def _con(settings) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    con.execute(f"SET temp_directory='{settings.duckdb_temp_dir.as_posix()}'")
    return con


def audit(settings, k: int, hard_mode: bool = False, until: str | None = None) -> AuditResult:
    suffix = f"k{k}" + ("_hard" if hard_mode else "")
    train_path = settings.features_dir / f"train_{suffix}.parquet"
    if not train_path.exists():
        raise SystemExit(f"missing {train_path}")

    con = _con(settings)
    events = f"read_parquet('{settings.events_dir.as_posix()}/**/*.parquet')"
    train = f"read_parquet('{train_path.as_posix()}')"
    res = AuditResult()

    # Recompute event ranks from first principles, matching the documented
    # ordering (event_time, then product_id as a deterministic tiebreak).
    #
    # Restricted to sessions that actually appear in the training table. Ranking
    # all 110M events is what killed an earlier run: the window function has to
    # sort the whole archive, and the audited sessions are a small fraction of
    # it. Narrowing here changes nothing about independence - the ranks are
    # still derived from raw events, not from the feature builder's output.
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE audited AS SELECT DISTINCT session_key FROM {train}"
    )
    # Must apply the same total ordering the definition specifies: exact
    # duplicate events collapsed, then ordered by (event_time, product_id,
    # event_type). Ordering by (event_time, product_id) alone leaves ties, and
    # row_number() over a tied key is arbitrary - which is the bug this audit
    # found in the first place.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ranked AS
        SELECT session_key, event_time, event_type,
               row_number() OVER (PARTITION BY session_key
                                  ORDER BY event_time, product_id, event_type) AS rn
        FROM (
            SELECT a.session_key, e.event_time, e.event_type, e.product_id
            FROM {events} e
            JOIN audited a ON a.session_key = e.user_session
            {f"WHERE e.event_time < TIMESTAMP '{until}'" if until else ""}
            GROUP BY 1, 2, 3, 4
        )
        """
    )

    # 1. The prefix must be exactly k events, and none of them may postdate the
    #    published cutoff. Counting purely by `event_time <= cutoff` does NOT
    #    work: one-second timestamp granularity means events tie, so a session
    #    can legitimately have more than k events at or before its cutoff. The
    #    check is therefore on rank, plus the separate guarantee that the
    #    cutoff really is the last moment the features saw.
    n = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT t.session_key, count(*) AS n_in_prefix
            FROM {train} t JOIN ranked r USING (session_key)
            WHERE r.rn <= {k}
            GROUP BY t.session_key
        ) WHERE n_in_prefix <> {k}
        """
    ).fetchone()[0]
    res.checks["prefix_size"] = int(n)
    res.details["prefix_size"] = f"sessions whose prefix is not exactly {k} events"

    # 1b. cutoff_time must equal the last event the features actually saw.
    n = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT t.session_key, t.cutoff_time, max(r.event_time) AS true_cutoff
            FROM {train} t JOIN ranked r USING (session_key)
            WHERE r.rn <= {k}
            GROUP BY t.session_key, t.cutoff_time
        ) WHERE cutoff_time <> true_cutoff
        """
    ).fetchone()[0]
    res.checks["cutoff_is_last_seen"] = int(n)
    res.details["cutoff_is_last_seen"] = "rows whose cutoff_time is not the last prefix event"

    # 2. no purchase inside the prefix
    n = con.execute(
        f"""
        SELECT count(DISTINCT t.session_key)
        FROM {train} t JOIN ranked r USING (session_key)
        WHERE r.rn <= {k} AND r.event_type = 'purchase'
        """
    ).fetchone()[0]
    res.checks["no_prepurchase"] = int(n)
    res.details["no_prepurchase"] = "sessions that already purchased before the cutoff"

    # 3. The label comes only from events strictly after the cutoff instant.
    #    Recomputed from the published cutoff_time using nothing but timestamps,
    #    so this stays independent of how the prefix was ranked.
    n = con.execute(
        f"""
        WITH truth AS (
            SELECT t.session_key,
                   max(CASE WHEN r.event_time > t.cutoff_time AND r.event_type='purchase'
                            THEN 1 ELSE 0 END) AS y_true
            FROM {train} t JOIN ranked r USING (session_key)
            GROUP BY t.session_key
        )
        SELECT count(*) FROM {train} t JOIN truth USING (session_key)
        WHERE t.y <> truth.y_true
        """
    ).fetchone()[0]
    res.checks["label_provenance"] = int(n)
    res.details["label_provenance"] = "rows whose y disagrees with recomputed post-cutoff truth"

    # 4. clock bound respected
    if until:
        n = con.execute(
            f"SELECT count(*) FROM {train} WHERE cutoff_time >= TIMESTAMP '{until}'"
        ).fetchone()[0]
        res.checks["no_future_events"] = int(n)
        res.details["no_future_events"] = f"rows with cutoff at or after clock bound {until}"

    # 5. the feature matrix must not contain identifiers or outcome timing
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {train}").fetchall()}
    from src.models.train import ID_COLS, LABEL

    model_feats = cols - set(ID_COLS) - {LABEL}
    bad = model_feats & FORBIDDEN_FEATURE_COLS
    res.checks["no_forbidden_cols"] = len(bad)
    res.details["no_forbidden_cols"] = f"forbidden columns reaching the model: {sorted(bad)}"

    con.close()
    return res


def audit_splits(splits: dict) -> AuditResult:
    """Confirm chronological split windows are ordered and non-overlapping."""
    res = AuditResult()
    order = ["train", "valid", "test", "drift"]
    present = [s for s in order if s in splits and splits[s].height > 0]
    violations = 0
    detail = []
    for a, b in zip(present, present[1:]):
        a_max = splits[a]["session_date"].max()
        b_min = splits[b]["session_date"].min()
        if a_max >= b_min:
            violations += 1
            detail.append(f"{a} ends {a_max} but {b} starts {b_min}")
    res.checks["split_disjoint"] = violations
    res.details["split_disjoint"] = "; ".join(detail)
    return res


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hard-mode", action="store_true")
    ap.add_argument("--until", default=None)
    args = ap.parse_args()

    res = audit(settings, k=args.k, hard_mode=args.hard_mode, until=args.until)
    log.info("LEAKAGE AUDIT (k=%d)", args.k)
    log.info("\n%s", res.report())
    if not res.passed:
        raise SystemExit("LEAKAGE AUDIT FAILED")
    log.info("all leakage checks passed")


if __name__ == "__main__":
    main()
