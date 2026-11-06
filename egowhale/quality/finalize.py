"""Merge deterministic curation and the L3 audit into one decision.

This is a pure function, not a DAG stage: `VlmAudit` calls it right after writing its audit, and the
module can also be run on an existing output directory to recompute `action/final.json` without
calling the API again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egowhale.paths import FINAL, QUALITY, VLM_AUDIT


def decide(quality: dict, audit: dict, fail_open: bool = True) -> dict:
    keep_l1_l2 = bool(quality.get("keep_l1_l2", quality.get("keep", False)))
    status = str(audit.get("status", "error"))
    consistent = audit.get("is_consistent")

    if not keep_l1_l2 and consistent is False:
        decision, reason = "drop_both", "l1_l2 and vlm rejected the episode"
    elif not keep_l1_l2:
        decision, reason = "drop_l1_l2", "l1_l2 rejected the episode"
    elif status == "disabled":
        decision, reason = "keep_audit_disabled", "l3 audit is disabled"
    elif status != "ok":
        if fail_open:
            decision, reason = "keep_vlm_unavailable", f"l3 audit unavailable: {status}"
        else:
            decision, reason = "drop_vlm_unavailable", f"l3 audit unavailable: {status}"
    elif consistent is False:
        decision, reason = "drop_vlm", "vlm found a major mismatch"
    elif consistent is None:
        decision, reason = "keep", "l1_l2 passed; the vlm gave no verdict"
    else:
        decision, reason = "keep", "l1_l2 and vlm agree"

    return {
        "keep": decision.startswith("keep"),
        "keep_l1_l2": keep_l1_l2,
        "vlm_status": status,
        "vlm_is_consistent": consistent,
        "decision": decision,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute action/final.json from existing quality and audit files.")
    parser.add_argument("out", type=Path, help="episode output directory")
    parser.add_argument("--fail-closed", action="store_true", help="drop when the audit is unavailable")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    quality = json.loads((out / QUALITY).read_text())
    audit = json.loads((out / VLM_AUDIT).read_text())
    final = decide(quality, audit, fail_open=not args.fail_closed)
    (out / FINAL).write_text(json.dumps(final, indent=2))
    print(f"wrote {out / FINAL}")


if __name__ == "__main__":
    main()
