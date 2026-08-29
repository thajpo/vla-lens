from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.validate_research_child import _print_human


def test_human_preflight_reports_non_contract_readiness_blockers(capsys):
    snapshot = {
        "authorized_to_start_child": False,
        "child_check": {
            "valid": False,
            "fingerprint": "sha256:" + "a" * 64,
            "files_verified": True,
        },
        "lock_check": {"valid": False},
        "program_check": {"issues": []},
        "campaign_ledger_check": {
            "valid": True,
            "state": {
                "status": {
                    "hardware_authorized": False,
                    "next_action": {
                        "action_id": "prepare_child",
                        "reason_code": "entry_conditions_satisfied",
                    },
                }
            },
            "issues": [],
        },
        "git_lock_check": {
            "valid": False,
            "errors": ["implementation_commit_not_found"],
        },
        "storage_check": {"valid": True},
        "output_freshness_check": {
            "claimed": False,
            "reason": "claim_output_required",
        },
    }

    _print_human(
        Namespace(verify_files=True, event_root=Path("events"), claim_output=False),
        snapshot,
    )

    output = capsys.readouterr().out
    assert "Ledger next action: prepare_child (entry_conditions_satisfied)" in output
    assert "Hardware authorized by ledger: False" in output
    assert "Git lock: INVALID" in output
    assert "- git_lock_check: implementation_commit_not_found" in output
    assert "Output claim: NOT CLAIMED (claim_output_required)" in output
