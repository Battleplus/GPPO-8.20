"""Build and verify the evidence-only GitHub publication manifest.

The publication intentionally excludes checkpoint binaries, training run
directories, logs, and controller state. Checkpoint identity is preserved by
cross-checking the immutable training seal, checkpoint index, Freeze manifests,
and Test ledger SHA-256 values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVIDENCE_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = EVIDENCE_DIR.parent
PRELIMINARY_DIR = CAMPAIGN_DIR / "preliminary"
WORKTREE = Path(__file__).resolve().parents[5]
CAMPAIGN_REL = CAMPAIGN_DIR.relative_to(WORKTREE).as_posix()
EVIDENCE_HEAD = "2afa8ec1cb481deb57645dbd30240d90d32d2233"
SOURCE_COMMIT = "32974ec85be71e192b12cae85d00eb877d5fe07d"
VARIANTS = ("PPO-MLP", "GPPO-Adaptive")
SEEDS = (1101, 2202, 3303)
SCENARIOS = (
    "Test-Single",
    "Test-Sequential",
    "Test-Overlap",
    "Test-Burst",
    "Test-Unseen",
)
PUBLICATION_FILES = {
    "publication_manifest.json",
    "publication_sha256_inventory.json",
    "publication_readonly_revalidation.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(WORKTREE), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def count_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def fail_if(condition: bool, message: str, errors: list[str]) -> None:
    if condition:
        errors.append(message)


def verify_evidence_graph() -> dict[str, Any]:
    errors: list[str] = []
    head = git("rev-parse", "HEAD")
    if head != EVIDENCE_HEAD:
        git("merge-base", "--is-ancestor", EVIDENCE_HEAD, head)

    tracked_changes = git("diff", "--name-only")
    staged_changes = git("diff", "--cached", "--name-only")
    fail_if(bool(tracked_changes or staged_changes), "tracked files modified before publication", errors)

    forbidden = [
        path
        for path in CAMPAIGN_DIR.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() == ".pt"
            or "runs" in path.relative_to(CAMPAIGN_DIR).parts
            or path.suffix.lower() == ".log"
        )
    ]
    fail_if(bool(forbidden), "publication contains checkpoint/run/log files", errors)
    fail_if((WORKTREE / "training_control").exists(), "publication contains training_control", errors)

    training_dir = CAMPAIGN_DIR / "training_evidence"
    training = load(training_dir / "training_evidence.json")
    training_inventory = load(training_dir / "sha256_inventory.json")
    training_revalidation = load(training_dir / "readonly_revalidation.json")
    fail_if(training.get("status") != "PASS", "training evidence is not PASS", errors)
    fail_if(training_revalidation.get("status") != "PASS", "training readonly revalidation is not PASS", errors)
    fail_if(training.get("summary", {}).get("completed_runs") != 6, "training runs are not 6/6", errors)
    fail_if(training.get("summary", {}).get("total_steps") != 300_000, "training total is not 300000", errors)
    fail_if(training.get("summary", {}).get("checkpoint_count") != 12, "training checkpoint count is not 12", errors)
    fail_if(training_inventory.get("artifact_count") != 48, "training inventory does not attest 48 artifacts", errors)
    fail_if(
        training.get("provenance", {}).get("runtime_evidence_head") != EVIDENCE_HEAD,
        "training evidence HEAD mismatch",
        errors,
    )
    fail_if(
        training.get("provenance", {}).get("attested_source_commit_sha") != SOURCE_COMMIT,
        "training source commit mismatch",
        errors,
    )

    checkpoint_index = load(PRELIMINARY_DIR / "checkpoint_index.json")
    fail_if(len(checkpoint_index) != 12, "checkpoint index is not 12 entries", errors)
    checkpoint_steps = Counter(int(row["decision_steps"]) for row in checkpoint_index)
    fail_if(checkpoint_steps != {25_000: 6, 50_000: 6}, "checkpoint index step grid mismatch", errors)

    fixed_candidates = {
        (row["variant"], int(row["seed"])): row
        for row in training.get("fixed_50000_evaluation_candidates", [])
    }
    expected_keys = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    fail_if(set(fixed_candidates) != expected_keys, "training seal 50k candidate matrix mismatch", errors)

    freeze_path = PRELIMINARY_DIR / "frozen_manifests.json"
    freeze = load(freeze_path)
    freeze_sha = sha256(freeze_path)
    freezes = {
        (row["variant"], int(row["training_seed"])): row
        for row in freeze.get("freezes", [])
    }
    fail_if(freeze.get("formal") is not True, "Freeze is not formal", errors)
    fail_if(freeze.get("checkpoint_selection") is not False, "Freeze used checkpoint selection", errors)
    fail_if(freeze.get("fixed_evaluation_checkpoint") != 50_000, "Freeze is not fixed at 50k", errors)
    fail_if(set(freezes) != expected_keys or len(freeze.get("freezes", [])) != 6, "Freeze matrix mismatch", errors)
    for key, row in freezes.items():
        fail_if(int(row["selected_step"]) != 50_000, f"non-50k Freeze: {key}", errors)
        candidate = fixed_candidates.get(key, {})
        fail_if(
            row.get("checkpoint_sha256") != candidate.get("sha256"),
            f"Freeze/training candidate SHA mismatch: {key}",
            errors,
        )

    manifest_path = PRELIMINARY_DIR / "tapes/preliminary_test_protocol/manifest.json"
    manifest = load(manifest_path)
    manifest_sha = sha256(manifest_path)
    entries = manifest.get("entries", [])
    tape_ids = {row["tape_id"] for row in entries}
    set_counts = Counter(row["set_name"] for row in entries)
    fail_if(len(entries) != 100 or len(tape_ids) != 100, "held-out bank is not 100 unique cases", errors)
    fail_if(set_counts != {scenario: 20 for scenario in SCENARIOS}, "held-out scenario balance mismatch", errors)
    fail_if(manifest.get("checkpoint_selection") is not False, "held-out manifest used checkpoint selection", errors)
    fail_if(manifest.get("reward_tuning") is not False, "held-out manifest used reward tuning", errors)

    lock = load(PRELIMINARY_DIR / "formal_test_bank_lock.json")
    ledger = load(PRELIMINARY_DIR / "test_ledger.json")
    fail_if(lock.get("completed") is not True, "formal Test lock is incomplete", errors)
    fail_if(ledger.get("completed") is not True, "Test ledger is incomplete", errors)
    fail_if(lock.get("test_manifest_sha256") != manifest_sha, "lock Test manifest SHA mismatch", errors)
    fail_if(lock.get("freeze_manifest_sha256") != freeze_sha, "lock Freeze SHA mismatch", errors)
    fail_if(ledger.get("test_manifest_sha256") != manifest_sha, "ledger Test manifest SHA mismatch", errors)
    fail_if(len(ledger.get("entries", {})) != 6, "ledger does not contain six entries", errors)

    result_files = sorted((PRELIMINARY_DIR / "test_results").glob("*.json"))
    fail_if(len(result_files) != 6, "publication does not contain six result files", errors)
    result_records: list[tuple[str, int, str]] = []
    result_inventory = []
    for path in result_files:
        payload = load(path)
        freeze_row = payload.get("freeze", {})
        key = (freeze_row.get("variant"), int(freeze_row.get("training_seed", 0)))
        results = payload.get("results", [])
        ids = {row["tape_id"] for row in results}
        fail_if(key not in expected_keys, f"unexpected result key: {key}", errors)
        fail_if(payload.get("tape_count") != 100 or len(results) != 100, f"result count mismatch: {key}", errors)
        fail_if(ids != tape_ids, f"result tape set mismatch: {key}", errors)
        fail_if(payload.get("test_manifest_sha256") != manifest_sha, f"result manifest mismatch: {key}", errors)
        ledger_matches = [
            row
            for row in ledger.get("entries", {}).values()
            if (row.get("variant"), int(row.get("training_seed", 0))) == key
        ]
        fail_if(len(ledger_matches) != 1, f"ledger result key count mismatch: {key}", errors)
        if len(ledger_matches) == 1:
            item = ledger_matches[0]
            fail_if(item.get("result_sha") != sha256(path), f"ledger result SHA mismatch: {key}", errors)
            fail_if(item.get("checkpoint_sha") != freezes[key]["checkpoint_sha256"], f"ledger checkpoint SHA mismatch: {key}", errors)
        for row in results:
            result_records.append((str(key[0]), int(key[1]), str(row["tape_id"])))
        result_inventory.append(
            {
                "variant": key[0],
                "training_seed": key[1],
                "path": path.relative_to(WORKTREE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "results": len(results),
            }
        )
    fail_if(len(result_records) != 600 or len(set(result_records)) != 600, "results are not 600 unique records", errors)

    state_files = sorted((PRELIMINARY_DIR / "test_state").glob("*.json"))
    fail_if(len(state_files) != 6, "publication does not contain six state journals", errors)
    for path in state_files:
        fail_if(load(path).get("state") != "consumed", f"state is not consumed: {path.name}", errors)

    analysis_counts = {
        "canonical_rows": count_csv(EVIDENCE_DIR / "evaluation_rows.csv"),
        "aggregate_records": count_csv(EVIDENCE_DIR / "aggregate_metrics.csv"),
        "paired_effect_records": count_csv(EVIDENCE_DIR / "paired_effects.csv"),
    }
    fail_if(analysis_counts != {"canonical_rows": 600, "aggregate_records": 480, "paired_effect_records": 240}, "analysis row counts mismatch", errors)
    local_audit = load(EVIDENCE_DIR / "result_integrity_audit.json")
    local_evidence = load(EVIDENCE_DIR / "evaluation_evidence.json")
    local_readonly = load(EVIDENCE_DIR / "readonly_revalidation.json")
    fail_if(local_audit.get("status") != "PASS", "local result audit is not PASS", errors)
    fail_if(local_evidence.get("status") != "PASS", "local evaluation seal is not PASS", errors)
    fail_if(local_readonly.get("status") != "PASS", "local readonly revalidation is not PASS", errors)

    if errors:
        raise SystemExit("PUBLICATION AUDIT FAIL\n" + "\n".join(f"- {error}" for error in errors))
    return {
        "runtime_head": head,
        "training": {
            "runs": 6,
            "total_steps": 300_000,
            "checkpoints": 12,
            "checkpoint_25000": 6,
            "checkpoint_50000": 6,
            "sealed_inventory_artifacts": 48,
        },
        "fixed_checkpoints": [
            {
                "variant": key[0],
                "training_seed": key[1],
                "step": 50_000,
                "sha256": freezes[key]["checkpoint_sha256"],
                "binary_published": False,
            }
            for key in sorted(expected_keys)
        ],
        "freeze_manifest_sha256": freeze_sha,
        "test_manifest_sha256": manifest_sha,
        "held_out_cases": 100,
        "held_out_sets": dict(set_counts),
        "model_case_results": 600,
        "result_files": result_inventory,
        "analysis_counts": analysis_counts,
    }


def inventory_inputs() -> list[Path]:
    return sorted(
        (
            path
            for path in CAMPAIGN_DIR.rglob("*")
            if path.is_file() and path.name not in PUBLICATION_FILES
        ),
        key=lambda path: path.relative_to(WORKTREE).as_posix(),
    )


def build() -> None:
    audit = verify_evidence_graph()
    paths = inventory_inputs()
    inventory = {
        "schema_version": 1,
        "status": "PASS",
        "algorithm": "SHA-256",
        "scope": "all evidence-only GitHub publication inputs excluding the three publication wrapper JSON files",
        "evidence_head": EVIDENCE_HEAD,
        "attested_source_commit_sha": SOURCE_COMMIT,
        "artifact_count": len(paths),
        "artifacts": [
            {
                "path": path.relative_to(WORKTREE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ],
        "explicit_exclusions": [
            "checkpoint binaries (*.pt)",
            "preliminary/runs/**",
            "training_control/**",
            "*.log",
            "publication_manifest.json",
            "publication_sha256_inventory.json",
            "publication_readonly_revalidation.json",
        ],
    }
    inventory_path = EVIDENCE_DIR / "publication_sha256_inventory.json"
    dump(inventory_path, inventory)
    inventory_sha = sha256(inventory_path)

    manifest = {
        "schema_version": 1,
        "publication_type": "minimum_validation_50k_training_and_heldout_evidence_only",
        "status": "PASS",
        "created_at": utc_now(),
        "repository": "Battleplus/GPPO-8.20",
        "target_branch": "plan/minimum-validation-50k",
        "evidence_head": EVIDENCE_HEAD,
        "attested_source_commit_sha": SOURCE_COMMIT,
        "publication_root": CAMPAIGN_REL,
        "audit": audit,
        "publication_inventory": {
            "path": inventory_path.relative_to(WORKTREE).as_posix(),
            "artifact_count": inventory["artifact_count"],
            "sha256": inventory_sha,
        },
        "scope_guards": {
            "source_files_changed": False,
            "checkpoint_binaries_published": False,
            "training_run_directories_published": False,
            "logs_published": False,
            "validation_performed": False,
            "checkpoint_selection_performed": False,
            "retraining_or_resume_performed": False,
            "protocol_or_algorithm_changed": False,
        },
        "report": f"{CAMPAIGN_REL}/evaluation_evidence/comparison_report.html",
    }
    manifest_path = EVIDENCE_DIR / "publication_manifest.json"
    dump(manifest_path, manifest)

    verify_inventory(inventory_path)
    readonly = {
        "schema_version": 1,
        "status": "PASS",
        "revalidation_type": "evidence_only_GitHub_publication_readonly_revalidation",
        "checked_at": utc_now(),
        "publication_manifest": {
            "path": manifest_path.relative_to(WORKTREE).as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "publication_inventory": {
            "path": inventory_path.relative_to(WORKTREE).as_posix(),
            "bytes": inventory_path.stat().st_size,
            "sha256": inventory_sha,
            "verified_artifacts": inventory["artifact_count"],
            "mismatches": 0,
        },
        "checks": {
            "training_seal": "PASS",
            "fixed_six_50k_checkpoint_sha_crosscheck": "PASS",
            "held_out_bank_100_unique_20x5": "PASS",
            "results_600_unique": "PASS",
            "analysis_600_480_240": "PASS",
            "forbidden_binary_run_log_control_files": "NONE",
            "checkpoint_selection": "NOT_PERFORMED",
            "validation": "NOT_PERFORMED",
            "source_or_protocol_changes": "NONE",
        },
    }
    dump(EVIDENCE_DIR / "publication_readonly_revalidation.json", readonly)


def verify_inventory(path: Path) -> None:
    payload = load(path)
    errors = []
    for row in payload.get("artifacts", []):
        artifact = WORKTREE / row["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row["bytes"])
            or sha256(artifact) != row["sha256"]
        ):
            errors.append(row["path"])
    if len(payload.get("artifacts", [])) != int(payload.get("artifact_count", -1)):
        errors.append("artifact_count")
    if errors:
        raise SystemExit("PUBLICATION INVENTORY FAIL\n" + "\n".join(f"- {item}" for item in errors))


def verify_only() -> None:
    verify_evidence_graph()
    inventory_path = EVIDENCE_DIR / "publication_sha256_inventory.json"
    manifest_path = EVIDENCE_DIR / "publication_manifest.json"
    readonly_path = EVIDENCE_DIR / "publication_readonly_revalidation.json"
    verify_inventory(inventory_path)
    manifest = load(manifest_path)
    readonly = load(readonly_path)
    if manifest.get("status") != "PASS" or readonly.get("status") != "PASS":
        raise SystemExit("publication wrapper status is not PASS")
    if manifest["publication_inventory"]["sha256"] != sha256(inventory_path):
        raise SystemExit("publication manifest inventory SHA mismatch")
    if readonly["publication_manifest"]["sha256"] != sha256(manifest_path):
        raise SystemExit("readonly manifest SHA mismatch")
    if readonly["publication_inventory"]["sha256"] != sha256(inventory_path):
        raise SystemExit("readonly inventory SHA mismatch")
    print(
        json.dumps(
            {
                "status": "PASS",
                "runtime_head": git("rev-parse", "HEAD"),
                "inventory_artifacts": load(inventory_path)["artifact_count"],
                "publication_manifest_sha256": sha256(manifest_path),
                "publication_inventory_sha256": sha256(inventory_path),
                "publication_readonly_revalidation_sha256": sha256(readonly_path),
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_only()
    else:
        build()
        verify_only()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
