import json
import os
import re
import tempfile
from copy import deepcopy


STATE_FILE = "midge_state.json"


DEFAULT_STATE = {
    "aliases": {},
    "groups": {},
    "alias_routes": {},
    "source_channels": {
        "kboard": None,
        "simmons": None,
    },

    # passthrough: unmatched MIDI is forwarded.
    # explicit_only: only configured aliases/rules/triggers/routes output.
    "source_modes": {
        "kboard": "passthrough",
        "simmons": "explicit_only",
    },

    # Existing capture-based note and chord assignments.
    "trigger_assignments": {},

    # Kept only so older state files can be migrated.
    "dynamic_note_triggers": {},

    # Runtime Level 2 rules.
    "rules": {},
    "rule_order": [],

    # Aliases blocked before routing or rule execution.
    "muted_aliases": [],

    # Entire performance configurations saved by name.
    "presets": {},

    # The most recently selected chord and its selector group.
    "active_harmonic_context": None,
}


PRESET_KEYS = (
    "aliases",
    "groups",
    "alias_routes",
    "source_channels",
    "source_modes",
    "trigger_assignments",
    "rules",
    "rule_order",
    "muted_aliases",
    "active_harmonic_context",
)


def normalize_name(value):
    if not isinstance(value, str):
        return ""

    value = value.lower().strip()
    value = value.replace("-", " ")
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def slugify(value):
    value = normalize_name(value)
    value = re.sub(r"\s+", "-", value)
    return value or "rule"


def legacy_dynamic_rule(
    trigger_alias,
    assignment,
):
    selector_group = normalize_name(
        assignment.get(
            "selector_group",
            "toms",
        )
    )

    note_source = normalize_name(
        assignment.get(
            "note_source",
            "lowest",
        )
    ).replace(" ", "_")

    transpose = assignment.get(
        "transpose",
        -24,
    )

    channel = assignment.get(
        "channel",
        1,
    )

    rule_id = slugify(
        f"{trigger_alias} follows {selector_group}"
    )

    return {
        "version": 1,
        "id": rule_id,
        "name": (
            f"{trigger_alias} follows "
            f"{selector_group}"
        ),
        "enabled": True,
        "priority": 100,
        "when": {
            "event": "note_on",
            "alias": normalize_name(
                trigger_alias
            ),
        },
        "condition": {},
        "derive": {
            "source": "active_chord",
            "selector_group": selector_group,
            "select": note_source,
            "transpose": int(transpose),
        },
        "output": {
            "type": "note",
            "channel": int(channel),
            "velocity": "input",
            "velocity_scale": 1.0,
            "velocity_min": 1,
            "velocity_max": 127,
            "suppress_original": bool(
                assignment.get(
                    "suppress_original",
                    True,
                )
            ),
        },
        "state_updates": [],
        "continue": False,
    }


def migrate_legacy_dynamic_triggers(state):
    legacy = state.get(
        "dynamic_note_triggers",
        {},
    )

    if not isinstance(legacy, dict):
        state["dynamic_note_triggers"] = {}
        return

    for trigger_alias, assignment in (
        legacy.items()
    ):
        if not isinstance(assignment, dict):
            continue

        rule = legacy_dynamic_rule(
            trigger_alias,
            assignment,
        )

        rule_id = rule["id"]

        if rule_id not in state["rules"]:
            state["rules"][rule_id] = rule

        if rule_id not in state["rule_order"]:
            state["rule_order"].append(
                rule_id
            )

    # Once represented as Level 2 rules, the legacy entries
    # must not also execute.
    state["dynamic_note_triggers"] = {}


def merge_defaults(loaded_state):
    state = deepcopy(DEFAULT_STATE)

    if not isinstance(loaded_state, dict):
        return state

    for key, value in loaded_state.items():
        state[key] = value

    dictionary_sections = (
        "aliases",
        "groups",
        "alias_routes",
        "source_channels",
        "source_modes",
        "trigger_assignments",
        "dynamic_note_triggers",
        "rules",
        "presets",
    )

    for key in dictionary_sections:
        if not isinstance(state.get(key), dict):
            state[key] = deepcopy(
                DEFAULT_STATE[key]
            )

    list_sections = (
        "rule_order",
        "muted_aliases",
    )

    for key in list_sections:
        if not isinstance(state.get(key), list):
            state[key] = deepcopy(
                DEFAULT_STATE[key]
            )

    state["muted_aliases"] = [
        normalize_name(alias)
        for alias in state["muted_aliases"]
        if normalize_name(alias)
    ]

    state["rule_order"] = [
        str(rule_id)
        for rule_id in state["rule_order"]
        if str(rule_id) in state["rules"]
    ]

    for rule_id in state["rules"]:
        if rule_id not in state["rule_order"]:
            state["rule_order"].append(
                rule_id
            )

    for source_name, channel in DEFAULT_STATE[
        "source_channels"
    ].items():
        state["source_channels"].setdefault(
            source_name,
            channel,
        )

    valid_modes = {"passthrough", "explicit_only"}
    for source_name, default_mode in DEFAULT_STATE[
        "source_modes"
    ].items():
        mode = state["source_modes"].get(
            source_name,
            default_mode,
        )
        if mode not in valid_modes:
            mode = default_mode
        state["source_modes"][source_name] = mode

    context = state.get(
        "active_harmonic_context"
    )

    if (
        context is not None
        and not isinstance(context, dict)
    ):
        state["active_harmonic_context"] = None

    migrate_legacy_dynamic_triggers(state)

    return state


def snapshot_for_preset(state):
    return {
        key: deepcopy(state.get(key))
        for key in PRESET_KEYS
    }


def apply_preset_snapshot(
    state,
    snapshot,
):
    if not isinstance(snapshot, dict):
        raise ValueError(
            "Preset data must be an object."
        )

    for key in PRESET_KEYS:
        if key in snapshot:
            state[key] = deepcopy(
                snapshot[key]
            )

    merged = merge_defaults(state)

    # Presets must remain available after loading one.
    merged["presets"] = deepcopy(
        state.get("presets", {})
    )

    return merged


def load_state():
    if not os.path.exists(STATE_FILE):
        state = deepcopy(DEFAULT_STATE)
        save_state(state)
        return state

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            loaded_state = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"Could not read {STATE_FILE}: {error}"
        )
        print(
            "Starting with a fresh Midge state."
        )
        return deepcopy(DEFAULT_STATE)

    return merge_defaults(loaded_state)


def save_state(state):
    """
    Write atomically so an interrupted save does not corrupt
    midge_state.json.
    """
    directory = os.path.dirname(
        os.path.abspath(STATE_FILE)
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            delete=False,
            prefix=".midge_state_",
            suffix=".json",
        ) as temporary_file:
            temporary_path = temporary_file.name

            json.dump(
                state,
                temporary_file,
                indent=2,
                sort_keys=True,
            )

            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(
                temporary_file.fileno()
            )

        os.replace(
            temporary_path,
            STATE_FILE,
        )

    except OSError as error:
        print(
            f"Could not save {STATE_FILE}: {error}"
        )

        if (
            temporary_path is not None
            and os.path.exists(
                temporary_path
            )
        ):
            try:
                os.remove(
                    temporary_path
                )
            except OSError:
                pass
