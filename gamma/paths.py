"""Unique output paths for result artifacts, so a rerun never silently
overwrites a previous run's numbers.

Applies going forward from Amendment 4: existing Phase 0/1 output files
(fixed names, already committed and already referenced by path in
reports/phase0_validation_report.md, reports/phase0_addendum_report.md,
reports/phase1_kickoff_report.md) are left as-is -- renaming them would
break those reports' links without adding any preservation benefit
they don't already have via git history. This convention is for new
result-writing code.
"""

import os
from datetime import datetime, timezone


def unique_path(dir_path: str, stem: str, ext: str) -> str:
    """{dir_path}/{stem}__{UTC timestamp}.{ext}, directory created if needed.

    Timestamp format YYYYmmddTHHMMSSZ is sortable and filesystem-safe.
    Not collision-proof against two calls in the same second -- callers
    doing that (e.g. a tight loop) should add their own disambiguator
    (seed, condition name, budget) into `stem`.
    """
    os.makedirs(dir_path, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(dir_path, f"{stem}__{ts}.{ext}")


def latest_matching(dir_path: str, stem: str, ext: str) -> str:
    """Most recent unique_path()-produced file for (dir_path, stem, ext).
    Convenience for plot scripts / manual inspection -- does not affect
    what gets written, only what gets read by default."""
    import glob

    pattern = os.path.join(dir_path, f"{stem}__*.{ext}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern}")
    return matches[-1]
