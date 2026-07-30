#!/usr/bin/env python3
"""
Midge parser evaluation driver.

Runs the same natural-language command through:
  1. deterministic-only
  2. LLM-only
  3. hybrid (deterministic first, then LLM)

Safety:
- Does not open MIDI ports.
- Does not call midge.main().
- Replaces state-changing Midge functions with in-memory recorders.
- Restores Midge's in-memory state after every test.
- Does not save state.

Expected project layout:
    midge.py
    llm_parser.py
    rule_engine.py
    state_manager.py
    midge_state.json
    evaluate_midge.py
    midge_eval_cases.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import midge
    import llm_parser
except ImportError as exc:
    print(
        "Could not import the Midge project modules.\n"
        "Place evaluate_midge.py in the same directory as "
        "midge.py, llm_parser.py, rule_engine.py, and state_manager.py.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


METHODS = ("deterministic", "llm", "hybrid")

DETERMINISTIC_PARSERS: Tuple[str, ...] = (
    "parse_fast_capture_command",
    "parse_fast_nth_fixed_note_rule",
    "parse_fast_rule_update",
    "parse_source_mode_command",
    "parse_passthrough_command",
    "parse_natural_channel_command",
)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def median_or_blank(values: List[float]) -> Any:
    return round(statistics.median(values), 3) if values else ""


def mean_or_blank(values: List[float]) -> Any:
    return round(statistics.mean(values), 3) if values else ""


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def build_snapshot() -> Dict[str, Any]:
    if hasattr(midge, "build_state_snapshot"):
        return deepcopy(midge.build_state_snapshot())
    return deepcopy(midge.state)


def source_names() -> List[str]:
    return list(getattr(midge, "INPUT_PORTS", {}).keys())


class ActionRecorder:
    def __init__(self) -> None:
        self.actions: List[Dict[str, Any]] = []

    def record(self, action: Dict[str, Any]) -> None:
        self.actions.append(deepcopy(action))

    @property
    def last_action(self) -> Optional[Dict[str, Any]]:
        return self.actions[-1] if self.actions else None


def make_recorders(recorder: ActionRecorder) -> Dict[str, Callable[..., None]]:
    def route_alias(alias_name: str, midi_channel: int) -> None:
        recorder.record({
            "action": "route_alias",
            "alias": midge.normalize_text(alias_name),
            "channel": int(midi_channel),
        })

    def route_group(group_name: str, midi_channel: int) -> None:
        recorder.record({
            "action": "route_group",
            "group": midge.normalize_text(group_name),
            "channel": int(midi_channel),
        })

    def route_source(source_name: str, midi_channel: int) -> None:
        recorder.record({
            "action": "route_source",
            "source": midge.normalize_text(source_name),
            "channel": int(midi_channel),
        })

    def source_passthrough(source_name: str) -> None:
        recorder.record({
            "action": "source_passthrough",
            "source": midge.normalize_text(source_name),
        })

    def source_explicit_only(source_name: str) -> None:
        recorder.record({
            "action": "source_explicit_only",
            "source": midge.normalize_text(source_name),
        })

    def clear_alias(alias_name: str) -> None:
        recorder.record({
            "action": "clear_alias_route",
            "alias": midge.normalize_text(alias_name),
        })

    def clear_group(group_name: str) -> None:
        recorder.record({
            "action": "clear_group_route",
            "group": midge.normalize_text(group_name),
        })

    def capture(*args: Any, **kwargs: Any) -> None:
        trigger_alias = kwargs.get("trigger_alias", args[0] if args else None)
        capture_mode = kwargs.get("capture_mode", args[1] if len(args) > 1 else "note")
        capture_source = kwargs.get("capture_source", args[2] if len(args) > 2 else "kboard")
        midi_channel = kwargs.get("midi_channel", args[3] if len(args) > 3 else 1)
        recorder.record({
            "action": f"capture_{capture_mode}_trigger",
            "trigger_alias": midge.normalize_text(trigger_alias),
            "capture_source": midge.normalize_text(capture_source),
            "channel": int(midi_channel),
        })

    def clear_trigger(trigger_alias: str) -> None:
        recorder.record({
            "action": "clear_trigger_assignment",
            "trigger_alias": midge.normalize_text(trigger_alias),
        })

    def dynamic_trigger(*args: Any, **kwargs: Any) -> None:
        trigger_alias = kwargs.get("trigger_alias", args[0] if args else None)
        selector_group = kwargs.get("selector_group", args[1] if len(args) > 1 else None)
        note_source = kwargs.get("note_source", args[2] if len(args) > 2 else "lowest")
        transpose = kwargs.get("transpose", args[3] if len(args) > 3 else -24)
        midi_channel = kwargs.get("midi_channel", args[4] if len(args) > 4 else 1)
        recorder.record({
            "action": "assign_dynamic_chord_note_trigger",
            "trigger_alias": midge.normalize_text(trigger_alias),
            "selector_group": midge.normalize_text(selector_group),
            "note_source": midge.normalize_text(note_source).replace(" ", "_"),
            "transpose": int(transpose),
            "channel": int(midi_channel),
        })

    def clear_dynamic(trigger_alias: str) -> None:
        recorder.record({
            "action": "clear_dynamic_note_trigger",
            "trigger_alias": midge.normalize_text(trigger_alias),
        })

    def install_rule(rule: Dict[str, Any]) -> None:
        recorder.record({
            "action": "install_rule",
            "rule": deepcopy(rule),
        })

    def replace_rule(rule_id: str, rule: Dict[str, Any]) -> None:
        recorder.record({
            "action": "update_rule",
            "rule_id": rule_id,
            "rule": deepcopy(rule),
        })

    def patch_rule(rule_id: str, changes: Dict[str, Any]) -> None:
        recorder.record({
            "action": "update_rule",
            "rule_id": rule_id,
            "changes": deepcopy(changes),
        })

    def delete_rule(rule_id: str) -> None:
        recorder.record({
            "action": "delete_rule",
            "rule_id": rule_id,
        })

    def set_enabled(rule_id: str, enabled: bool) -> None:
        recorder.record({
            "action": "enable_rule" if enabled else "disable_rule",
            "rule_id": rule_id,
        })

    def mute(alias_name: str) -> None:
        recorder.record({
            "action": "mute_alias",
            "alias": midge.normalize_text(alias_name),
        })

    def unmute(alias_name: str) -> None:
        recorder.record({
            "action": "unmute_alias",
            "alias": midge.normalize_text(alias_name),
        })

    return {
        "route_alias": route_alias,
        "route_alias_group": route_group,
        "set_source_channel": route_source,
        "set_source_passthrough": source_passthrough,
        "set_source_explicit_only": source_explicit_only,
        "clear_alias_route": clear_alias,
        "clear_alias_group_route": clear_group,
        "arm_trigger_capture": capture,
        "clear_trigger_assignment": clear_trigger,
        "assign_dynamic_chord_note_trigger": dynamic_trigger,
        "clear_dynamic_note_trigger": clear_dynamic,
        "install_runtime_rule": install_rule,
        "replace_runtime_rule": replace_rule,
        "patch_runtime_rule": patch_rule,
        "delete_runtime_rule": delete_rule,
        "set_rule_enabled": set_enabled,
        "mute_alias": mute,
        "unmute_alias": unmute,
        "save_state": lambda *args, **kwargs: None,
        "send_channel_panic": lambda *args, **kwargs: None,
        "send_global_panic": lambda *args, **kwargs: None,
        "push_undo": lambda *args, **kwargs: None,
    }


@contextmanager
def isolated_midge_state(snapshot: Dict[str, Any], recorder: ActionRecorder):
    original_state = midge.state
    original_last_rule = getattr(midge, "last_modified_rule_id", None)
    original_undo = getattr(midge, "undo_stack", None)

    patches = make_recorders(recorder)
    originals: Dict[str, Any] = {}

    midge.state = deepcopy(snapshot)
    if hasattr(midge, "last_modified_rule_id"):
        midge.last_modified_rule_id = None
    if hasattr(midge, "undo_stack"):
        midge.undo_stack = []

    for name, replacement in patches.items():
        if hasattr(midge, name):
            originals[name] = getattr(midge, name)
            setattr(midge, name, replacement)

    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(midge, name, original)
        midge.state = original_state
        if hasattr(midge, "last_modified_rule_id"):
            midge.last_modified_rule_id = original_last_rule
        if hasattr(midge, "undo_stack") and original_undo is not None:
            midge.undo_stack = original_undo


def run_deterministic(command: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    recorder = ActionRecorder()
    recognized = False
    used_parser = ""

    started = time.perf_counter()
    output_buffer = StringIO()

    try:
        with isolated_midge_state(snapshot, recorder), redirect_stdout(output_buffer):
            for parser_name in DETERMINISTIC_PARSERS:
                parser = getattr(midge, parser_name, None)
                if parser is None:
                    continue
                if parser(command):
                    recognized = True
                    used_parser = parser_name
                    break

        error = ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000.0
    action = recorder.last_action

    if not recognized and not error:
        error = "Command not recognized by deterministic parsers."

    if recognized and action is None and not error:
        error = (
            f"{used_parser} recognized the command but produced no captured action. "
            "The parser may need one additional recorder hook."
        )

    return {
        "recognized": recognized,
        "valid_action": action is not None and not error,
        "action": action,
        "latency_ms": latency_ms,
        "used_llm": False,
        "parser": used_parser,
        "semantic_errors": [],
        "error": error,
        "console": output_buffer.getvalue().strip(),
    }


def run_llm(command: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    action: Optional[Dict[str, Any]] = None
    error = ""
    semantic_errors: List[str] = []

    try:
        action = llm_parser.parse_command(command, deepcopy(snapshot), source_names())
        semantic_errors = list(
            midge.validate_llm_action_against_command(command, action)
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000.0

    return {
        "recognized": action is not None,
        "valid_action": action is not None and not error and not semantic_errors,
        "action": action,
        "latency_ms": latency_ms,
        "used_llm": True,
        "parser": "llm_parser.parse_command",
        "semantic_errors": semantic_errors,
        "error": error,
        "console": "",
    }


def run_method(method: str, command: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if method == "deterministic":
        return run_deterministic(command, snapshot)

    if method == "llm":
        return run_llm(command, snapshot)

    if method == "hybrid":
        deterministic = run_deterministic(command, snapshot)
        if deterministic["recognized"]:
            deterministic["parser"] = f"hybrid:{deterministic['parser']}"
            return deterministic

        llm_result = run_llm(command, snapshot)
        llm_result["latency_ms"] += deterministic["latency_ms"]
        llm_result["parser"] = "hybrid:llm_fallback"
        return llm_result

    raise ValueError(f"Unknown method: {method}")


def nested_get(value: Any, path: str) -> Tuple[bool, Any]:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def flatten_expected(value: Any, prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict):
                flattened.update(flatten_expected(child, path))
            else:
                flattened[path] = child
    else:
        flattened[prefix] = value
    return flattened


def canonicalize_value(path: str, value: Any) -> Any:
    """Normalize harmless surface differences before semantic comparison."""
    if isinstance(value, str):
        value = value.strip().lower()

        if path.endswith((
            "alias",
            "trigger_alias",
            "group",
            "selector_group",
            "source",
            "capture_source",
        )):
            for article in ("the ", "a ", "an "):
                if value.startswith(article):
                    value = value[len(article):].strip()
                    break

    return value


def compare_action(expected: Dict[str, Any], actual: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    expected_fields = flatten_expected(expected)

    if not expected_fields:
        return {
            "semantic_correct": actual is not None,
            "field_score": 1.0 if actual is not None else 0.0,
            "matched_fields": 0,
            "expected_fields": 0,
            "mismatches": [],
        }

    if actual is None:
        return {
            "semantic_correct": False,
            "field_score": 0.0,
            "matched_fields": 0,
            "expected_fields": len(expected_fields),
            "mismatches": [
                {"field": path, "expected": value, "actual": "<missing action>"}
                for path, value in expected_fields.items()
            ],
        }

    matched = 0
    mismatches = []

    for path, expected_value in expected_fields.items():
        exists, actual_value = nested_get(actual, path)
        normalized_expected = canonicalize_value(path, expected_value)
        normalized_actual = canonicalize_value(path, actual_value)

        if exists and normalized_actual == normalized_expected:
            matched += 1
        else:
            mismatches.append({
                "field": path,
                "expected": normalized_expected,
                "actual": normalized_actual if exists else "<missing>",
            })

    score = matched / len(expected_fields)

    return {
        "semantic_correct": matched == len(expected_fields),
        "field_score": score,
        "matched_fields": matched,
        "expected_fields": len(expected_fields),
        "mismatches": mismatches,
    }


def unmet_requirements(case: Dict[str, Any], snapshot: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    aliases = snapshot.get("aliases", {})
    groups = snapshot.get("groups", {})
    sources = set(source_names())
    rules = snapshot.get("rules", {})

    for alias in case.get("required_aliases", []):
        if alias not in aliases:
            missing.append(f'alias "{alias}"')

    for group in case.get("required_groups", []):
        if group not in groups:
            missing.append(f'group "{group}"')

    for source in case.get("required_sources", []):
        if source not in sources:
            missing.append(f'source "{source}"')

    for rule in case.get("required_rules", []):
        if rule not in rules:
            missing.append(f'rule "{rule}"')

    return missing


def check_llm_server() -> Tuple[bool, str]:
    if requests is None:
        return False, "requests is not installed"

    base_url = llm_parser.LLM_URL.rsplit("/v1/", 1)[0]
    try:
        response = requests.get(f"{base_url}/health", timeout=3)
        if response.ok:
            return True, ""
    except requests.RequestException:
        pass

    try:
        response = requests.get(f"{base_url}/v1/models", timeout=3)
        if response.ok:
            return True, ""
        return False, f"server returned HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Test case file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit("The test case file must contain a JSON list.")

    required = {"id", "category", "command", "expected"}
    for index, case in enumerate(data, start=1):
        missing = required - set(case)
        if missing:
            raise SystemExit(
                f"Test case #{index} is missing fields: {sorted(missing)}"
            )

    return data


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []

    for method in METHODS:
        method_rows = [
            row for row in rows
            if row["method"] == method and not row["skipped"]
        ]
        latencies = [float(row["latency_ms"]) for row in method_rows]
        recognized = sum(bool(row["recognized"]) for row in method_rows)
        valid = sum(bool(row["valid_action"]) for row in method_rows)
        correct = sum(bool(row["semantic_correct"]) for row in method_rows)
        llm_calls = sum(bool(row["used_llm"]) for row in method_rows)
        field_scores = [float(row["field_score"]) for row in method_rows]

        summaries.append({
            "method": method,
            "commands": len(method_rows),
            "recognized_rate_pct": percent(recognized, len(method_rows)),
            "validation_rate_pct": percent(valid, len(method_rows)),
            "semantic_accuracy_pct": percent(correct, len(method_rows)),
            "mean_field_score_pct": round(100 * statistics.mean(field_scores), 2)
            if field_scores else 0.0,
            "mean_latency_ms": mean_or_blank(latencies),
            "median_latency_ms": median_or_blank(latencies),
            "llm_invocation_rate_pct": percent(llm_calls, len(method_rows)),
        })

    return summaries


def build_category_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    categories = sorted({row["category"] for row in rows})

    for category in categories:
        for method in METHODS:
            subset = [
                row for row in rows
                if row["category"] == category
                and row["method"] == method
                and not row["skipped"]
            ]
            if not subset:
                continue
            latencies = [float(row["latency_ms"]) for row in subset]
            correct = sum(bool(row["semantic_correct"]) for row in subset)
            result.append({
                "category": category,
                "method": method,
                "commands": len(subset),
                "semantic_accuracy_pct": percent(correct, len(subset)),
                "mean_latency_ms": mean_or_blank(latencies),
            })

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="midge_eval_cases.json",
        help="JSON test-case file (default: midge_eval_cases.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_outputs",
        help="Directory for CSV and JSON outputs",
    )
    parser.add_argument(
        "--methods",
        default="deterministic,llm,hybrid",
        help="Comma-separated methods: deterministic,llm,hybrid",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N test cases",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat every command per method (default: 1)",
    )
    parser.add_argument(
        "--no-llm-check",
        action="store_true",
        help="Skip the llama-server health check",
    )
    args = parser.parse_args()

    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    invalid = set(methods) - set(METHODS)
    if invalid:
        raise SystemExit(f"Unknown methods: {sorted(invalid)}")

    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    cases = load_cases(Path(args.cases))
    if args.limit is not None:
        cases = cases[: args.limit]

    if any(method in {"llm", "hybrid"} for method in methods) and not args.no_llm_check:
        healthy, detail = check_llm_server()
        if not healthy:
            raise SystemExit(
                "llama-server does not appear reachable.\n"
                f"Detail: {detail}\n\n"
                "Start it with your normal command, for example:\n"
                "llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M "
                "--host 127.0.0.1 --port 8080 -c 2048"
            )

    snapshot = build_snapshot()
    output_dir = Path(args.output_dir)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    print(
        f"Loaded {len(cases)} cases. "
        f"Methods: {', '.join(methods)}. Repeats: {args.repeat}."
    )
    print(
        f"State context: {len(snapshot.get('aliases', {}))} aliases, "
        f"{len(snapshot.get('groups', {}))} groups, "
        f"{len(snapshot.get('rules', {}))} rules."
    )

    for case_index, case in enumerate(cases, start=1):
        missing = unmet_requirements(case, snapshot)

        for method in methods:
            for repetition in range(1, args.repeat + 1):
                if missing:
                    row = {
                        "test_id": case["id"],
                        "category": case["category"],
                        "command": case["command"],
                        "method": method,
                        "repetition": repetition,
                        "skipped": True,
                        "skip_reason": "; ".join(missing),
                        "recognized": False,
                        "valid_action": False,
                        "semantic_correct": False,
                        "field_score": 0.0,
                        "latency_ms": 0.0,
                        "used_llm": method in {"llm", "hybrid"},
                        "parser": "",
                        "semantic_errors": "",
                        "error": "",
                        "expected": json_text(case["expected"]),
                        "actual": "",
                        "mismatches": "",
                        "console": "",
                    }
                    rows.append(row)
                    continue

                result = run_method(method, case["command"], snapshot)
                comparison = compare_action(case["expected"], result["action"])

                row = {
                    "test_id": case["id"],
                    "category": case["category"],
                    "command": case["command"],
                    "method": method,
                    "repetition": repetition,
                    "skipped": False,
                    "skip_reason": "",
                    "recognized": result["recognized"],
                    "valid_action": result["valid_action"],
                    "semantic_correct": (
                        result["valid_action"] and comparison["semantic_correct"]
                    ),
                    "field_score": round(comparison["field_score"], 4),
                    "latency_ms": round(result["latency_ms"], 3),
                    "used_llm": result["used_llm"],
                    "parser": result["parser"],
                    "semantic_errors": json_text(result["semantic_errors"])
                    if result["semantic_errors"] else "",
                    "error": result["error"],
                    "expected": json_text(case["expected"]),
                    "actual": json_text(result["action"]) if result["action"] else "",
                    "mismatches": json_text(comparison["mismatches"])
                    if comparison["mismatches"] else "",
                    "console": result["console"],
                }
                rows.append(row)

                status = "PASS" if row["semantic_correct"] else "FAIL"
                print(
                    f"[{case_index:02d}/{len(cases):02d}] "
                    f"{method:13s} {status:4s} "
                    f"{row['latency_ms']:9.1f} ms  {case['id']}"
                )

                if not row["semantic_correct"]:
                    failures.append({
                        "test_id": case["id"],
                        "category": case["category"],
                        "method": method,
                        "command": case["command"],
                        "expected": case["expected"],
                        "actual": result["action"],
                        "valid_action": result["valid_action"],
                        "semantic_errors": result["semantic_errors"],
                        "error": result["error"],
                        "mismatches": comparison["mismatches"],
                        "console": result["console"],
                    })

    result_fields = [
        "test_id", "category", "command", "method", "repetition",
        "skipped", "skip_reason", "recognized", "valid_action",
        "semantic_correct", "field_score", "latency_ms", "used_llm",
        "parser", "semantic_errors", "error", "expected", "actual",
        "mismatches", "console",
    ]
    write_csv(output_dir / "evaluation_results.csv", rows, result_fields)

    summary = build_summary(rows)
    summary_fields = [
        "method", "commands", "recognized_rate_pct", "validation_rate_pct",
        "semantic_accuracy_pct", "mean_field_score_pct",
        "mean_latency_ms", "median_latency_ms", "llm_invocation_rate_pct",
    ]
    write_csv(output_dir / "evaluation_summary.csv", summary, summary_fields)

    category_summary = build_category_summary(rows)
    category_fields = [
        "category", "method", "commands",
        "semantic_accuracy_pct", "mean_latency_ms",
    ]
    write_csv(
        output_dir / "evaluation_by_category.csv",
        category_summary,
        category_fields,
    )

    (output_dir / "evaluation_failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_metadata = {
        "cases_file": str(Path(args.cases).resolve()),
        "case_count": len(cases),
        "methods": list(methods),
        "repeat": args.repeat,
        "source_names": source_names(),
        "aliases": sorted(snapshot.get("aliases", {}).keys()),
        "groups": sorted(snapshot.get("groups", {}).keys()),
        "rules": sorted(snapshot.get("rules", {}).keys()),
        "llm_url": getattr(llm_parser, "LLM_URL", ""),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSummary")
    for item in summary:
        if item["method"] not in methods:
            continue
        print(
            f"  {item['method']:13s} "
            f"accuracy={item['semantic_accuracy_pct']:6.2f}%  "
            f"valid={item['validation_rate_pct']:6.2f}%  "
            f"median={str(item['median_latency_ms']):>9s} ms  "
            f"LLM={item['llm_invocation_rate_pct']:6.2f}%"
        )

    print(f"\nWrote results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

