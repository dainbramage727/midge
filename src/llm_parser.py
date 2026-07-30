import json
import re

import requests

from rule_engine import (
    RuleValidationError,
    deep_merge,
    normalize_name,
    summarize_rule,
    validate_rule,
)


LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
LLM_TIMEOUT_SECONDS = 90


class LLMParserError(Exception):
    pass


class LLMConnectionError(LLMParserError):
    pass


SUPPORTED_ACTIONS = {
    "route_alias",
    "route_group",
    "route_source",
    "source_passthrough",
    "clear_alias_route",
    "clear_group_route",
    "capture_note_trigger",
    "capture_chord_trigger",
    "clear_trigger_assignment",
    "assign_dynamic_chord_note_trigger",
    "clear_dynamic_note_trigger",
    "install_rule",
    "update_rule",
    "delete_rule",
    "enable_rule",
    "disable_rule",
    "mute_alias",
    "unmute_alias",
}


def strip_code_fence(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def extract_json_object(text):
    text = strip_code_fence(text)

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    start_index = text.find("{")
    end_index = text.rfind("}")

    if start_index == -1 or end_index == -1:
        raise LLMParserError(
            "The local model did not return a JSON object."
        )

    candidate = text[
        start_index:end_index + 1
    ]

    try:
        parsed = json.loads(candidate)

    except json.JSONDecodeError as error:
        raise LLMParserError(
            "The local model returned invalid JSON: "
            f"{error}"
        ) from error

    if not isinstance(parsed, dict):
        raise LLMParserError(
            "The local model response must be a JSON object."
        )

    return parsed


def require_channel(action):
    channel = action.get("channel")

    if (
        isinstance(channel, str)
        and channel.isdigit()
    ):
        channel = int(channel)

    if not isinstance(channel, int):
        raise LLMParserError(
            "The action did not contain a valid "
            "MIDI channel."
        )

    if not 1 <= channel <= 16:
        raise LLMParserError(
            "MIDI channel must be between 1 and 16."
        )

    action["channel"] = channel


def require_integer(
    action,
    field_name,
    default=None,
):
    value = action.get(
        field_name,
        default,
    )

    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            pass

    if not isinstance(value, int):
        raise LLMParserError(
            f'"{field_name}" must be an integer.'
        )

    action[field_name] = value


def require_existing_name(
    action,
    field_name,
    valid_names,
):
    value = normalize_name(
        action.get(field_name)
    )

    if not value:
        raise LLMParserError(
            f'The action did not contain '
            f'"{field_name}".'
        )

    lookup = {
        normalize_name(name): name
        for name in valid_names
    }

    if value not in lookup:
        available = ", ".join(
            sorted(lookup)
        )

        raise LLMParserError(
            f'Unknown {field_name} "{value}". '
            f"Available values: "
            f"{available or 'none'}."
        )

    action[field_name] = (
        normalize_name(
            lookup[value]
        )
    )


def resolve_rule_id(
    requested,
    state,
):
    requested_normalized = (
        normalize_name(requested)
    )

    rules = state.get("rules", {})

    if requested in rules:
        return requested

    direct_lookup = {
        normalize_name(rule_id): rule_id
        for rule_id in rules
    }

    if requested_normalized in direct_lookup:
        return direct_lookup[
            requested_normalized
        ]

    name_lookup = {
        normalize_name(
            rule.get("name", "")
        ): rule_id
        for rule_id, rule in rules.items()
    }

    if requested_normalized in name_lookup:
        return name_lookup[
            requested_normalized
        ]

    # Let "kick" resolve to the only rule triggered by the
    # kick alias.
    matching_alias_rules = []

    for rule_id, rule in rules.items():
        alias = normalize_name(
            rule.get(
                "when",
                {},
            ).get(
                "alias",
                "",
            )
        )

        if (
            alias
            and requested_normalized
            in {
                alias,
                alias.replace(" drum", ""),
            }
        ):
            matching_alias_rules.append(
                rule_id
            )

    if len(matching_alias_rules) == 1:
        return matching_alias_rules[0]

    available = ", ".join(
        sorted(rules)
    )

    raise LLMParserError(
        f'Unknown or ambiguous rule "{requested}". '
        f"Available rule IDs: "
        f"{available or 'none'}."
    )


def validate_action(
    action,
    state,
    source_names,
):
    action_name = normalize_name(
        action.get("action")
    ).replace(" ", "_")

    if action_name not in SUPPORTED_ACTIONS:
        raise LLMParserError(
            f'Unsupported action "{action_name}".'
        )

    aliases = list(
        state.get("aliases", {}).keys()
    )

    groups = list(
        state.get("groups", {}).keys()
    )

    action["action"] = action_name

    if action_name == "route_alias":
        require_existing_name(
            action,
            "alias",
            aliases,
        )
        require_channel(action)

        return {
            "action": action_name,
            "alias": action["alias"],
            "channel": action["channel"],
        }

    if action_name == "route_group":
        require_existing_name(
            action,
            "group",
            groups,
        )
        require_channel(action)

        return {
            "action": action_name,
            "group": action["group"],
            "channel": action["channel"],
        }

    if action_name == "route_source":
        require_existing_name(
            action,
            "source",
            source_names,
        )
        require_channel(action)

        return {
            "action": action_name,
            "source": action["source"],
            "channel": action["channel"],
        }

    if action_name == "source_passthrough":
        require_existing_name(
            action,
            "source",
            source_names,
        )

        return {
            "action": action_name,
            "source": action["source"],
        }

    if action_name == "clear_alias_route":
        require_existing_name(
            action,
            "alias",
            aliases,
        )

        return {
            "action": action_name,
            "alias": action["alias"],
        }

    if action_name == "clear_group_route":
        require_existing_name(
            action,
            "group",
            groups,
        )

        return {
            "action": action_name,
            "group": action["group"],
        }

    if action_name in {
        "capture_note_trigger",
        "capture_chord_trigger",
    }:
        require_existing_name(
            action,
            "trigger_alias",
            aliases,
        )

        capture_source = normalize_name(
            action.get(
                "capture_source",
                "kboard",
            )
        )

        normalized_sources = {
            normalize_name(name)
            for name in source_names
        }

        if capture_source not in normalized_sources:
            raise LLMParserError(
                f'Unknown capture source '
                f'"{capture_source}".'
            )

        require_channel(action)

        return {
            "action": action_name,
            "trigger_alias": action[
                "trigger_alias"
            ],
            "capture_source": capture_source,
            "channel": action["channel"],
        }

    if action_name == "clear_trigger_assignment":
        require_existing_name(
            action,
            "trigger_alias",
            aliases,
        )

        return {
            "action": action_name,
            "trigger_alias": action[
                "trigger_alias"
            ],
        }

    if (
        action_name
        == "assign_dynamic_chord_note_trigger"
    ):
        require_existing_name(
            action,
            "trigger_alias",
            aliases,
        )

        require_existing_name(
            action,
            "selector_group",
            groups,
        )

        require_channel(action)

        note_source = normalize_name(
            action.get(
                "note_source",
                "lowest",
            )
        ).replace(" ", "_")

        if note_source not in {
            "lowest",
            "highest",
            "first",
            "second",
            "third",
            "root",
            "chord_third",
            "chord_fifth",
            "cycle",
            "random",
        }:
            raise LLMParserError(
                f'Unsupported note source '
                f'"{note_source}".'
            )

        require_integer(
            action,
            "transpose",
            default=-24,
        )

        return {
            "action": action_name,
            "trigger_alias": action[
                "trigger_alias"
            ],
            "selector_group": action[
                "selector_group"
            ],
            "note_source": note_source,
            "transpose": action["transpose"],
            "channel": action["channel"],
        }

    if action_name == "clear_dynamic_note_trigger":
        require_existing_name(
            action,
            "trigger_alias",
            aliases,
        )

        return {
            "action": action_name,
            "trigger_alias": action[
                "trigger_alias"
            ],
        }

    if action_name in {
        "mute_alias",
        "unmute_alias",
    }:
        require_existing_name(
            action,
            "alias",
            aliases,
        )

        return {
            "action": action_name,
            "alias": action["alias"],
        }

    if action_name == "install_rule":
        raw_rule = action.get("rule")

        try:
            rule = validate_rule(
                raw_rule,
                state,
                source_names,
            )

        except RuleValidationError as error:
            raise LLMParserError(
                f"Invalid runtime rule: {error}"
            ) from error

        return {
            "action": action_name,
            "rule": rule,
        }

    if action_name == "update_rule":
        requested_rule = (
            action.get("rule_id")
            or action.get("rule")
            or action.get("name")
            or action.get("alias")
        )

        if not requested_rule:
            raise LLMParserError(
                "An update requires a rule_id."
            )

        rule_id = resolve_rule_id(
            requested_rule,
            state,
        )

        changes = action.get("changes", {})

        if not isinstance(changes, dict):
            raise LLMParserError(
                '"changes" must be a JSON object.'
            )

        existing = state["rules"][rule_id]

        merged = deep_merge(
            existing,
            changes,
        )

        merged["id"] = rule_id

        try:
            rule = validate_rule(
                merged,
                state,
                source_names,
                existing_rule_id=rule_id,
            )

        except RuleValidationError as error:
            raise LLMParserError(
                f"Invalid rule update: {error}"
            ) from error

        return {
            "action": action_name,
            "rule_id": rule_id,
            "rule": rule,
        }

    if action_name in {
        "delete_rule",
        "enable_rule",
        "disable_rule",
    }:
        requested_rule = (
            action.get("rule_id")
            or action.get("rule")
            or action.get("name")
            or action.get("alias")
        )

        if not requested_rule:
            raise LLMParserError(
                f"{action_name} requires a rule_id."
            )

        rule_id = resolve_rule_id(
            requested_rule,
            state,
        )

        return {
            "action": action_name,
            "rule_id": rule_id,
        }

    raise LLMParserError(
        f'Could not validate action '
        f'"{action_name}".'
    )


def compact_rule_catalog(state):
    rules = state.get("rules", {})

    if not rules:
        return "[]"

    summaries = [
        summarize_rule(rule)
        for rule in rules.values()
    ]

    return json.dumps(
        summaries,
        ensure_ascii=False,
    )


def build_system_prompt(
    state,
    source_names,
):
    aliases = sorted(
        state.get("aliases", {}).keys()
    )

    groups = sorted(
        state.get("groups", {}).keys()
    )

    rules = compact_rule_catalog(state)

    return f"""
Translate the user's Midge request into exactly one JSON
object. Output JSON only. Use only the names listed below.
MIDI channels are 1-16.

For new musical behavior, install a runtime rule:
{{
 "action":"install_rule",
 "rule":{{
  "id":"short-id","name":"brief name","enabled":true,
  "priority":100,
  "when":{{"event":"note_on","alias":"<alias>"}},
  "condition":{{}},
  "derive":{{"source":"active_chord",
   "selector_group":"<group>","select":"lowest",
   "transpose":0}},
  "output":{{"type":"note","channel":1,
   "velocity":"input","suppress_original":true}},
  "state_updates":[],"continue":false
 }}
}}

when may use alias, group, source, note, or input_channel.
condition may use every_n, offset, min_velocity,
max_velocity, active_context_group, state_key, equals.
derive.source: active_chord, input_note, fixed.
derive.select: all, lowest, highest, first, second,
third, root, chord_third, chord_fifth, index, cycle,
random. Fixed uses "notes":[60,64,67].
output.type: note, chord, cc, mute, passthrough.
CC uses channel, control, and value="input_velocity".

Update only requested fields:
{{"action":"update_rule","rule_id":"<id>",
 "changes":{{"derive":{{"transpose":-12}}}}}}
Other rule actions:
{{"action":"delete_rule","rule_id":"<id>"}}
{{"action":"enable_rule","rule_id":"<id>"}}
{{"action":"disable_rule","rule_id":"<id>"}}

Simple actions:
{{"action":"route_alias","alias":"<alias>","channel":1}}
{{"action":"route_group","group":"<group>","channel":1}}
{{"action":"route_source","source":"<source>","channel":1}}
{{"action":"source_passthrough","source":"<source>"}}
{{"action":"clear_alias_route","alias":"<alias>"}}
{{"action":"clear_group_route","group":"<group>"}}
{{"action":"capture_note_trigger","trigger_alias":"<alias>",
 "capture_source":"kboard","channel":1}}
{{"action":"capture_chord_trigger","trigger_alias":"<alias>",
 "capture_source":"kboard","channel":1}}
{{"action":"clear_trigger_assignment","trigger_alias":"<alias>"}}
{{"action":"mute_alias","alias":"<alias>"}}
{{"action":"unmute_alias","alias":"<alias>"}}

Example: "Every fourth snare hit, play the highest note of
the current tom chord one octave up on channel 4" means an
install_rule with when.alias="snare", condition.every_n=4,
select="highest", transpose=12, channel=4.

Sources: {json.dumps(source_names)}
Aliases: {json.dumps(aliases)}
Groups: {json.dumps(groups)}
Installed rules: {rules}
""".strip()


def parse_command(
    command,
    state,
    source_names,
):
    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": build_system_prompt(
                    state,
                    source_names,
                ),
            },
            {
                "role": "user",
                "content": command,
            },
        ],
        "temperature": 0,
        "max_tokens": 320,
        "stream": False,
    }

    try:
        response = requests.post(
            LLM_URL,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise LLMConnectionError(
            "Could not reach llama-server at "
            f"{LLM_URL}: {error}"
        ) from error

    try:
        response_data = response.json()

        content = response_data[
            "choices"
        ][0]["message"]["content"]

    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        raise LLMParserError(
            "llama-server returned an "
            "unexpected response."
        ) from error

    action = extract_json_object(
        content
    )

    return validate_action(
        action,
        state,
        source_names,
    )
