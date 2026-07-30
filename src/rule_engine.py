import random
import re
from copy import deepcopy
from dataclasses import dataclass, field

import mido


RULE_VERSION = 1

ALLOWED_EVENTS = {"note_on"}
ALLOWED_DERIVE_SOURCES = {
    "active_chord",
    "input_note",
    "fixed",
}
ALLOWED_SELECTORS = {
    "all",
    "lowest",
    "highest",
    "first",
    "second",
    "third",
    "root",
    "chord_third",
    "chord_fifth",
    "index",
    "cycle",
    "random",
}
ALLOWED_OUTPUT_TYPES = {
    "note",
    "chord",
    "cc",
    "mute",
    "passthrough",
}
ALLOWED_VELOCITY_MODES = {
    "input",
    "fixed",
}


class RuleValidationError(ValueError):
    pass


@dataclass
class RuleExecutionResult:
    matched_rule_ids: list[str] = field(default_factory=list)
    messages: list[mido.Message] = field(default_factory=list)
    suppress_original: bool = False
    muted: bool = False


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


def clamp_midi(value):
    return max(0, min(127, int(value)))


def deep_merge(base, patch):
    result = deepcopy(base)

    for key, value in patch.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = deepcopy(value)

    return result


def _as_int(value, field_name):
    if isinstance(value, bool):
        raise RuleValidationError(
            f'"{field_name}" must be an integer.'
        )

    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            pass

    if not isinstance(value, int):
        raise RuleValidationError(
            f'"{field_name}" must be an integer.'
        )

    return value


def _as_float(value, field_name):
    if isinstance(value, bool):
        raise RuleValidationError(
            f'"{field_name}" must be numeric.'
        )

    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass

    if not isinstance(value, (int, float)):
        raise RuleValidationError(
            f'"{field_name}" must be numeric.'
        )

    return float(value)


def _validate_channel(value):
    channel = _as_int(value, "channel")

    if not 1 <= channel <= 16:
        raise RuleValidationError(
            "MIDI channel must be between 1 and 16."
        )

    return channel


def _resolve_existing_name(
    value,
    valid_names,
    field_name,
):
    normalized = normalize_name(value)

    lookup = {
        normalize_name(name): name
        for name in valid_names
    }

    if normalized not in lookup:
        available = ", ".join(sorted(lookup))
        raise RuleValidationError(
            f'Unknown {field_name} "{normalized}". '
            f"Available values: {available or 'none'}."
        )

    return normalize_name(lookup[normalized])


def validate_rule(
    raw_rule,
    state,
    source_names,
    existing_rule_id=None,
):
    if not isinstance(raw_rule, dict):
        raise RuleValidationError(
            "A rule must be a JSON object."
        )

    aliases = state.get("aliases", {})
    groups = state.get("groups", {})

    rule = deepcopy(raw_rule)

    name = str(
        rule.get("name")
        or existing_rule_id
        or "runtime rule"
    ).strip()

    rule_id = str(
        rule.get("id")
        or existing_rule_id
        or slugify(name)
    ).strip()

    rule_id = slugify(rule_id)
    name = name or rule_id

    enabled = rule.get("enabled", True)
    if not isinstance(enabled, bool):
        raise RuleValidationError(
            '"enabled" must be true or false.'
        )

    priority = _as_int(
        rule.get("priority", 100),
        "priority",
    )

    when = rule.get("when")
    if not isinstance(when, dict):
        raise RuleValidationError(
            'A rule requires a "when" object.'
        )

    event = normalize_name(
        when.get("event", "note_on")
    ).replace(" ", "_")

    if event in {"note", "hit", "pad_hit"}:
        event = "note_on"

    if event not in ALLOWED_EVENTS:
        raise RuleValidationError(
            'Rules currently support only '
            '"when.event": "note_on".'
        )

    normalized_when = {
        "event": event,
    }

    if when.get("alias") is not None:
        normalized_when["alias"] = (
            _resolve_existing_name(
                when["alias"],
                aliases.keys(),
                "alias",
            )
        )

    if when.get("group") is not None:
        normalized_when["group"] = (
            _resolve_existing_name(
                when["group"],
                groups.keys(),
                "group",
            )
        )

    if when.get("source") is not None:
        normalized_when["source"] = (
            _resolve_existing_name(
                when["source"],
                source_names,
                "source",
            )
        )

    if when.get("note") is not None:
        note = _as_int(
            when["note"],
            "when.note",
        )

        if not 0 <= note <= 127:
            raise RuleValidationError(
                '"when.note" must be between 0 and 127.'
            )

        normalized_when["note"] = note

    if when.get("input_channel") is not None:
        normalized_when["input_channel"] = (
            _validate_channel(
                when["input_channel"]
            )
        )

    if len(normalized_when) == 1:
        raise RuleValidationError(
            'A rule must match at least one alias, '
            'group, source, note, or input channel.'
        )

    condition = rule.get("condition", {})
    if condition is None:
        condition = {}

    if not isinstance(condition, dict):
        raise RuleValidationError(
            '"condition" must be an object.'
        )

    normalized_condition = {}

    if condition.get("every_n") is not None:
        every_n = _as_int(
            condition["every_n"],
            "condition.every_n",
        )

        if every_n < 1:
            raise RuleValidationError(
                '"condition.every_n" must be at least 1.'
            )

        normalized_condition["every_n"] = every_n

    if condition.get("offset") is not None:
        normalized_condition["offset"] = _as_int(
            condition["offset"],
            "condition.offset",
        )

    if condition.get("min_velocity") is not None:
        minimum = _as_int(
            condition["min_velocity"],
            "condition.min_velocity",
        )

        if not 1 <= minimum <= 127:
            raise RuleValidationError(
                '"condition.min_velocity" must be 1-127.'
            )

        normalized_condition["min_velocity"] = minimum

    if condition.get("max_velocity") is not None:
        maximum = _as_int(
            condition["max_velocity"],
            "condition.max_velocity",
        )

        if not 1 <= maximum <= 127:
            raise RuleValidationError(
                '"condition.max_velocity" must be 1-127.'
            )

        normalized_condition["max_velocity"] = maximum

    if condition.get("active_context_group") is not None:
        normalized_condition[
            "active_context_group"
        ] = _resolve_existing_name(
            condition["active_context_group"],
            groups.keys(),
            "group",
        )

    if condition.get("state_key") is not None:
        state_key = normalize_name(
            condition["state_key"]
        ).replace(" ", "_")

        if not state_key:
            raise RuleValidationError(
                '"condition.state_key" cannot be empty.'
            )

        normalized_condition["state_key"] = state_key
        normalized_condition["equals"] = deepcopy(
            condition.get("equals")
        )

    output = rule.get("output")
    if not isinstance(output, dict):
        raise RuleValidationError(
            'A rule requires an "output" object.'
        )

    output_type = normalize_name(
        output.get("type", "note")
    ).replace(" ", "_")

    if output_type not in ALLOWED_OUTPUT_TYPES:
        raise RuleValidationError(
            f'Unsupported output type "{output_type}".'
        )

    normalized_output = {
        "type": output_type,
        "suppress_original": bool(
            output.get(
                "suppress_original",
                True,
            )
        ),
    }

    derive = rule.get("derive")
    normalized_derive = None

    if output_type in {"note", "chord"}:
        if not isinstance(derive, dict):
            raise RuleValidationError(
                f'Output type "{output_type}" '
                'requires a "derive" object.'
            )

        derive_source = normalize_name(
            derive.get("source", "input_note")
        ).replace(" ", "_")

        if derive_source not in ALLOWED_DERIVE_SOURCES:
            raise RuleValidationError(
                f'Unsupported derive source '
                f'"{derive_source}".'
            )

        selector = normalize_name(
            derive.get(
                "select",
                "all"
                if output_type == "chord"
                else "first",
            )
        ).replace(" ", "_")

        selector_aliases = {
            "minimum": "lowest",
            "maximum": "highest",
            "bottom": "lowest",
            "top": "highest",
            "root_note": "root",
            "chordthird": "chord_third",
            "chordfifth": "chord_fifth",
        }

        selector = selector_aliases.get(
            selector,
            selector,
        )

        if selector not in ALLOWED_SELECTORS:
            raise RuleValidationError(
                f'Unsupported selector "{selector}".'
            )

        normalized_derive = {
            "source": derive_source,
            "select": selector,
            "transpose": _as_int(
                derive.get("transpose", 0),
                "derive.transpose",
            ),
        }

        if derive_source == "active_chord":
            selector_group = derive.get(
                "selector_group"
            )

            if selector_group is None:
                raise RuleValidationError(
                    'An active-chord rule requires '
                    '"derive.selector_group".'
                )

            normalized_derive[
                "selector_group"
            ] = _resolve_existing_name(
                selector_group,
                groups.keys(),
                "group",
            )

        if derive_source == "fixed":
            notes = derive.get("notes")

            if not isinstance(notes, list) or not notes:
                raise RuleValidationError(
                    'A fixed-note rule requires a '
                    'non-empty "derive.notes" list.'
                )

            normalized_notes = []

            for raw_note in notes:
                note = _as_int(
                    raw_note,
                    "derive.notes",
                )

                if not 0 <= note <= 127:
                    raise RuleValidationError(
                        "Fixed MIDI notes must be 0-127."
                    )

                normalized_notes.append(note)

            normalized_derive["notes"] = normalized_notes

        if selector == "index":
            index = _as_int(
                derive.get("index", 0),
                "derive.index",
            )

            if index < 0:
                raise RuleValidationError(
                    '"derive.index" cannot be negative.'
                )

            normalized_derive["index"] = index

        normalized_output["channel"] = (
            _validate_channel(
                output.get("channel")
            )
        )

        velocity = output.get(
            "velocity",
            "input",
        )

        if isinstance(velocity, int):
            normalized_output["velocity"] = "fixed"
            normalized_output["velocity_value"] = (
                clamp_midi(velocity)
            )
        else:
            velocity_mode = normalize_name(
                velocity
            ).replace(" ", "_")

            if velocity_mode not in ALLOWED_VELOCITY_MODES:
                raise RuleValidationError(
                    f'Unsupported velocity mode '
                    f'"{velocity_mode}".'
                )

            normalized_output["velocity"] = velocity_mode

            if velocity_mode == "fixed":
                normalized_output[
                    "velocity_value"
                ] = clamp_midi(
                    _as_int(
                        output.get(
                            "velocity_value",
                            127,
                        ),
                        "output.velocity_value",
                    )
                )

        normalized_output["velocity_scale"] = (
            _as_float(
                output.get(
                    "velocity_scale",
                    1.0,
                ),
                "output.velocity_scale",
            )
        )

        normalized_output["velocity_min"] = (
            clamp_midi(
                _as_int(
                    output.get(
                        "velocity_min",
                        1,
                    ),
                    "output.velocity_min",
                )
            )
        )

        normalized_output["velocity_max"] = (
            clamp_midi(
                _as_int(
                    output.get(
                        "velocity_max",
                        127,
                    ),
                    "output.velocity_max",
                )
            )
        )

        if (
            normalized_output["velocity_min"]
            > normalized_output["velocity_max"]
        ):
            raise RuleValidationError(
                "velocity_min cannot exceed velocity_max."
            )

    elif output_type == "cc":
        normalized_output["channel"] = (
            _validate_channel(
                output.get("channel")
            )
        )

        control = _as_int(
            output.get("control"),
            "output.control",
        )

        if not 0 <= control <= 127:
            raise RuleValidationError(
                '"output.control" must be 0-127.'
            )

        normalized_output["control"] = control

        value = output.get("value", 127)

        if isinstance(value, str):
            normalized_value = normalize_name(
                value
            ).replace(" ", "_")

            if normalized_value not in {
                "input_velocity",
                "fixed",
            }:
                raise RuleValidationError(
                    'CC value must be 0-127, '
                    '"input_velocity", or "fixed".'
                )

            normalized_output["value"] = normalized_value

            if normalized_value == "fixed":
                normalized_output[
                    "value_number"
                ] = clamp_midi(
                    _as_int(
                        output.get(
                            "value_number",
                            127,
                        ),
                        "output.value_number",
                    )
                )
        else:
            normalized_output["value"] = "fixed"
            normalized_output["value_number"] = (
                clamp_midi(
                    _as_int(
                        value,
                        "output.value",
                    )
                )
            )

    elif output_type == "passthrough":
        if output.get("channel") is not None:
            normalized_output["channel"] = (
                _validate_channel(
                    output["channel"]
                )
            )

        normalized_output["suppress_original"] = True

    elif output_type == "mute":
        normalized_output["suppress_original"] = True

    state_updates = rule.get(
        "state_updates",
        [],
    )

    if state_updates is None:
        state_updates = []

    if not isinstance(state_updates, list):
        raise RuleValidationError(
            '"state_updates" must be a list.'
        )

    normalized_updates = []

    for update in state_updates:
        if not isinstance(update, dict):
            raise RuleValidationError(
                "Each state update must be an object."
            )

        operation = normalize_name(
            update.get("operation")
        ).replace(" ", "_")

        if operation not in {
            "set",
            "toggle",
            "increment",
        }:
            raise RuleValidationError(
                f'Unsupported state operation '
                f'"{operation}".'
            )

        key = normalize_name(
            update.get("key")
        ).replace(" ", "_")

        if not key:
            raise RuleValidationError(
                "A state update requires a key."
            )

        normalized_update = {
            "operation": operation,
            "key": key,
        }

        if operation == "set":
            normalized_update["value"] = deepcopy(
                update.get("value")
            )

        elif operation == "increment":
            normalized_update["amount"] = _as_int(
                update.get("amount", 1),
                "state_updates.amount",
            )

        normalized_updates.append(
            normalized_update
        )

    return {
        "version": RULE_VERSION,
        "id": rule_id,
        "name": name,
        "enabled": enabled,
        "priority": priority,
        "when": normalized_when,
        "condition": normalized_condition,
        "derive": normalized_derive,
        "output": normalized_output,
        "state_updates": normalized_updates,
        "continue": bool(
            rule.get("continue", False)
        ),
    }


def summarize_rule(rule):
    when = rule.get("when", {})
    output = rule.get("output", {})
    derive = rule.get("derive") or {}

    trigger_parts = []

    for key in (
        "alias",
        "group",
        "source",
        "note",
        "input_channel",
    ):
        if key in when:
            trigger_parts.append(
                f"{key}={when[key]}"
            )

    trigger_text = ", ".join(
        trigger_parts
    ) or "unknown trigger"

    if output.get("type") in {"note", "chord"}:
        derive_text = (
            f'{derive.get("source")} '
            f'{derive.get("select")} '
            f'{derive.get("transpose", 0):+d}'
        )

        if derive.get("selector_group"):
            derive_text += (
                f' from {derive["selector_group"]}'
            )

        output_text = (
            f'{output.get("type")} {derive_text} '
            f'-> ch {output.get("channel")}'
        )

    elif output.get("type") == "cc":
        output_text = (
            f'CC {output.get("control")} '
            f'-> ch {output.get("channel")}'
        )

    else:
        output_text = output.get("type", "unknown")

    status = "on" if rule.get("enabled", True) else "off"

    return (
        f'[{rule.get("id")}] {rule.get("name")} '
        f'({status}): {trigger_text} => {output_text}'
    )


class RuleEngine:
    def __init__(self):
        self.match_counts = {}
        self.cycle_positions = {}
        self.variables = {}

    def reset_runtime(self):
        self.match_counts.clear()
        self.cycle_positions.clear()
        self.variables.clear()

    def _rule_matches(
        self,
        rule,
        source_name,
        message,
        matching_aliases,
        state,
    ):
        when = rule.get("when", {})

        if message.type != "note_on":
            return False

        if message.velocity <= 0:
            return False

        if (
            "source" in when
            and when["source"] != source_name
        ):
            return False

        if (
            "note" in when
            and when["note"] != message.note
        ):
            return False

        if (
            "input_channel" in when
            and when["input_channel"] - 1
            != message.channel
        ):
            return False

        if (
            "alias" in when
            and when["alias"]
            not in matching_aliases
        ):
            return False

        if "group" in when:
            members = state.get(
                "groups",
                {},
            ).get(
                when["group"],
                [],
            )

            if not any(
                alias in members
                for alias in matching_aliases
            ):
                return False

        return True

    def _condition_passes(
        self,
        rule,
        message,
        state,
    ):
        rule_id = rule["id"]
        condition = rule.get(
            "condition",
            {},
        )

        count = self.match_counts.get(
            rule_id,
            0,
        ) + 1

        self.match_counts[rule_id] = count

        every_n = condition.get(
            "every_n",
            1,
        )

        offset = condition.get(
            "offset",
            0,
        )

        if every_n > 1:
            if (count - offset) % every_n != 0:
                return False

        minimum = condition.get(
            "min_velocity"
        )

        if (
            minimum is not None
            and message.velocity < minimum
        ):
            return False

        maximum = condition.get(
            "max_velocity"
        )

        if (
            maximum is not None
            and message.velocity > maximum
        ):
            return False

        context_group = condition.get(
            "active_context_group"
        )

        if context_group is not None:
            context = state.get(
                "active_harmonic_context"
            ) or {}

            if context_group not in context.get(
                "selector_groups",
                [],
            ):
                return False

        state_key = condition.get(
            "state_key"
        )

        if state_key is not None:
            if self.variables.get(state_key) != (
                condition.get("equals")
            ):
                return False

        return True

    def _source_notes(
        self,
        rule,
        message,
        state,
    ):
        derive = rule.get("derive") or {}
        source = derive.get(
            "source",
            "input_note",
        )

        if source == "input_note":
            return [message.note]

        if source == "fixed":
            return list(
                derive.get("notes", [])
            )

        if source == "active_chord":
            context = state.get(
                "active_harmonic_context"
            ) or {}

            group_name = derive.get(
                "selector_group"
            )

            if group_name not in context.get(
                "selector_groups",
                [],
            ):
                return []

            return list(
                context.get("notes", [])
            )

        return []

    def _select_notes(
        self,
        rule,
        notes,
    ):
        if not notes:
            return []

        derive = rule.get("derive") or {}
        selector = derive.get(
            "select",
            "first",
        )

        original = list(notes)
        ordered = sorted(
            dict.fromkeys(original)
        )

        if selector == "all":
            selected = ordered

        elif selector in {"lowest", "root"}:
            selected = [ordered[0]]

        elif selector == "highest":
            selected = [ordered[-1]]

        elif selector == "first":
            selected = [original[0]]

        elif selector == "second":
            selected = (
                [ordered[1]]
                if len(ordered) >= 2
                else []
            )

        elif selector == "third":
            selected = (
                [ordered[2]]
                if len(ordered) >= 3
                else []
            )

        elif selector == "chord_third":
            selected = (
                [ordered[1]]
                if len(ordered) >= 2
                else []
            )

        elif selector == "chord_fifth":
            selected = (
                [ordered[2]]
                if len(ordered) >= 3
                else []
            )

        elif selector == "index":
            index = derive.get("index", 0)

            selected = (
                [ordered[index]]
                if index < len(ordered)
                else []
            )

        elif selector == "cycle":
            rule_id = rule["id"]
            position = self.cycle_positions.get(
                rule_id,
                0,
            )

            selected = [
                ordered[
                    position % len(ordered)
                ]
            ]

            self.cycle_positions[
                rule_id
            ] = position + 1

        elif selector == "random":
            selected = [
                random.choice(ordered)
            ]

        else:
            selected = []

        transpose = derive.get(
            "transpose",
            0,
        )

        return [
            clamp_midi(note + transpose)
            for note in selected
        ]

    def _velocity(
        self,
        output,
        input_velocity,
    ):
        if output.get("velocity") == "fixed":
            base = output.get(
                "velocity_value",
                127,
            )
        else:
            base = input_velocity

        scaled = round(
            base
            * output.get(
                "velocity_scale",
                1.0,
            )
        )

        minimum = output.get(
            "velocity_min",
            1,
        )

        maximum = output.get(
            "velocity_max",
            127,
        )

        return max(
            minimum,
            min(maximum, clamp_midi(scaled)),
        )

    def _build_messages(
        self,
        rule,
        message,
        state,
    ):
        output = rule["output"]
        output_type = output["type"]

        if output_type == "mute":
            return [], True

        if output_type == "passthrough":
            if output.get("channel") is None:
                return [message.copy()], False

            return [
                message.copy(
                    channel=output["channel"] - 1
                )
            ], False

        if output_type == "cc":
            if output.get("value") == (
                "input_velocity"
            ):
                value = message.velocity
            else:
                value = output.get(
                    "value_number",
                    127,
                )

            return [
                mido.Message(
                    "control_change",
                    channel=(
                        output["channel"] - 1
                    ),
                    control=output["control"],
                    value=clamp_midi(value),
                )
            ], False

        source_notes = self._source_notes(
            rule,
            message,
            state,
        )

        selected_notes = self._select_notes(
            rule,
            source_notes,
        )

        if not selected_notes:
            return [], False

        if output_type == "note":
            selected_notes = selected_notes[:1]

        velocity = self._velocity(
            output,
            message.velocity,
        )

        messages = [
            mido.Message(
                "note_on",
                channel=output["channel"] - 1,
                note=note,
                velocity=velocity,
            )
            for note in selected_notes
        ]

        return messages, False

    def _apply_state_updates(self, rule):
        for update in rule.get(
            "state_updates",
            [],
        ):
            operation = update["operation"]
            key = update["key"]

            if operation == "set":
                self.variables[key] = deepcopy(
                    update.get("value")
                )

            elif operation == "toggle":
                self.variables[key] = not bool(
                    self.variables.get(key, False)
                )

            elif operation == "increment":
                self.variables[key] = (
                    self.variables.get(key, 0)
                    + update.get("amount", 1)
                )

    def execute(
        self,
        source_name,
        message,
        matching_aliases,
        state,
    ):
        result = RuleExecutionResult()

        if (
            message.type != "note_on"
            or message.velocity <= 0
        ):
            return result

        rules = state.get("rules", {})
        order = state.get(
            "rule_order",
            [],
        )

        ordered_ids = [
            rule_id
            for rule_id in order
            if rule_id in rules
        ]

        for rule_id in rules:
            if rule_id not in ordered_ids:
                ordered_ids.append(rule_id)

        ordered_rules = [
            rules[rule_id]
            for rule_id in ordered_ids
        ]

        ordered_rules.sort(
            key=lambda rule: (
                -int(
                    rule.get(
                        "priority",
                        100,
                    )
                ),
                ordered_ids.index(rule["id"]),
            )
        )

        for rule in ordered_rules:
            if not rule.get("enabled", True):
                continue

            if not self._rule_matches(
                rule,
                source_name,
                message,
                matching_aliases,
                state,
            ):
                continue

            if not self._condition_passes(
                rule,
                message,
                state,
            ):
                continue

            messages, muted = self._build_messages(
                rule,
                message,
                state,
            )

            result.matched_rule_ids.append(
                rule["id"]
            )

            result.messages.extend(messages)

            result.suppress_original = (
                result.suppress_original
                or rule["output"].get(
                    "suppress_original",
                    True,
                )
            )

            result.muted = (
                result.muted or muted
            )

            self._apply_state_updates(rule)

            if not rule.get("continue", False):
                break

        return result
