import re
import threading
import time
from copy import deepcopy
from pathlib import Path

import mido

from llm_parser import (
    LLMConnectionError,
    LLMParserError,
    parse_command,
)
from rule_engine import (
    RuleEngine,
    RuleValidationError,
    deep_merge,
    normalize_name,
    slugify,
    summarize_rule,
    validate_rule,
)
from state_manager import (
    apply_preset_snapshot,
    load_state,
    save_state,
    snapshot_for_preset,
)


INPUT_PORTS = {
    "kboard": "K-Board",
    "simmons": "Simmons SD200",
}

OUTPUT_PORT = "ESI MIDIMATE eX Port 1"


SOURCE_ALIASES = {
    "keyboard": "kboard",
    "key board": "kboard",
    "k board": "kboard",
    "k-board": "kboard",
    "kboard": "kboard",
    "drums": "simmons",
    "drum kit": "simmons",
    "drumkit": "simmons",
    "simmons": "simmons",
    "simmons kit": "simmons",
}


stop_event = threading.Event()
state_lock = threading.RLock()
output_lock = threading.RLock()
rule_engine_lock = threading.RLock()

state = load_state()
rule_engine = RuleEngine()

last_event = None
current_output_port = None
last_modified_rule_id = None

pending_capture = None
capture_pressed_notes = set()
capture_collected_notes = []

# Physical input key -> generated notes and whether the
# original note_off should also pass through.
active_generated = {}

# In-memory undo avoids writing a second history file.
undo_stack = []
MAX_UNDO = 20


def normalize_text(text):
    text = text.lower().strip()

    # Preserve signed numbers such as -12 while still
    # treating hyphens in names such as K-Board as spaces.
    text = re.sub(
        r"(?<!\w)-(?=\d)",
        "__minus__",
        text,
    )

    text = text.replace("-", " ")

    text = re.sub(
        r"[^\w\s,+]",
        "",
        text,
    )

    text = text.replace(
        "__minus__",
        "-",
    )

    text = re.sub(r"\s+", " ", text)
    return text


def identify_source(text):
    normalized = normalize_text(text)

    for phrase, source_name in (
        SOURCE_ALIASES.items()
    ):
        normalized_phrase = normalize_text(
            phrase
        )

        if re.search(
            rf"\b{re.escape(normalized_phrase)}\b",
            normalized,
        ):
            return source_name

    return None


def is_note_on(message):
    return (
        message.type == "note_on"
        and message.velocity > 0
    )


def is_note_off(message):
    return (
        message.type == "note_off"
        or (
            message.type == "note_on"
            and message.velocity == 0
        )
    )


def incoming_trigger_key(
    source_name,
    message,
):
    return (
        source_name,
        getattr(
            message,
            "channel",
            None,
        ),
        getattr(
            message,
            "note",
            None,
        ),
    )


def clamp_midi_note(note):
    return max(
        0,
        min(127, int(note)),
    )


def safe_send(message):
    with output_lock:
        if current_output_port is None:
            return

        try:
            current_output_port.send(message)

        except Exception as error:
            print(
                "Could not send MIDI message: "
                f"{error}"
            )


def send_channel_panic(channel):
    if (
        channel is None
        or not 0 <= channel <= 15
    ):
        return

    safe_send(
        mido.Message(
            "control_change",
            channel=channel,
            control=123,
            value=0,
        )
    )

    safe_send(
        mido.Message(
            "control_change",
            channel=channel,
            control=120,
            value=0,
        )
    )


def send_global_panic():
    for channel in range(16):
        send_channel_panic(channel)

    with state_lock:
        active_generated.clear()

    print(
        "Sent all-notes-off on all MIDI channels."
    )


def push_undo(label):
    with state_lock:
        undo_stack.append(
            (
                label,
                deepcopy(state),
            )
        )

        if len(undo_stack) > MAX_UNDO:
            del undo_stack[0]


def undo_last_change():
    global state
    global last_modified_rule_id

    with state_lock:
        if not undo_stack:
            print("Nothing to undo.")
            return

        label, previous_state = (
            undo_stack.pop()
        )

        state = previous_state
        save_state(state)

    last_modified_rule_id = None
    with rule_engine_lock:
        rule_engine.reset_runtime()

    send_global_panic()

    print(f"Undid: {label}")


def save_current_state():
    with state_lock:
        save_state(state)


def message_matches_alias(
    source_name,
    message,
    alias_event,
):
    if message.type not in {
        "note_on",
        "note_off",
    }:
        return False

    if not hasattr(message, "channel"):
        return False

    return (
        alias_event.get("source")
        == source_name
        and alias_event.get("type")
        == "note"
        and alias_event.get("channel")
        == message.channel
        and alias_event.get("note")
        == message.note
    )


def find_matching_aliases(
    source_name,
    message,
):
    with state_lock:
        aliases = deepcopy(
            state.get("aliases", {})
        )

    matches = []

    for alias_name, alias_event in (
        aliases.items()
    ):
        if message_matches_alias(
            source_name,
            message,
            alias_event,
        ):
            matches.append(alias_name)

    return matches


def groups_for_alias(alias_name):
    with state_lock:
        return sorted(
            group_name
            for group_name, members in (
                state.get(
                    "groups",
                    {},
                ).items()
            )
            if alias_name in members
        )


def update_harmonic_context(
    selector_alias,
    assignment,
):
    """
    Runtime-only update. Disk writes here previously added
    latency on every tom hit.
    """
    notes = assignment.get("notes", [])

    if not notes:
        return

    selector_groups = groups_for_alias(
        selector_alias
    )

    if not selector_groups:
        return

    context = {
        "selector_alias": selector_alias,
        "selector_groups": (
            selector_groups
        ),
        "notes": list(notes),
        "captured_mode": assignment.get(
            "mode",
            "chord",
        ),
    }

    with state_lock:
        state[
            "active_harmonic_context"
        ] = context

    print(
        f'Active harmony selected by '
        f'"{selector_alias}": '
        f'{context["notes"]}'
    )


def release_previous_generated(key):
    with state_lock:
        previous = active_generated.pop(
            key,
            None,
        )

    if previous is None:
        return

    for generated in previous.get(
        "notes",
        [],
    ):
        safe_send(
            mido.Message(
                "note_off",
                channel=generated[
                    "channel"
                ],
                note=generated["note"],
                velocity=0,
            )
        )


def remember_generated(
    source_name,
    message,
    generated_messages,
    pass_original_note_off,
):
    key = incoming_trigger_key(
        source_name,
        message,
    )

    release_previous_generated(key)

    generated_notes = []

    for generated in generated_messages:
        if (
            generated.type == "note_on"
            and generated.velocity > 0
        ):
            generated_notes.append(
                {
                    "channel": (
                        generated.channel
                    ),
                    "note": generated.note,
                }
            )

    with state_lock:
        active_generated[key] = {
            "notes": generated_notes,
            "pass_original_note_off": bool(
                pass_original_note_off
            ),
        }


def generated_note_off_messages(
    source_name,
    message,
):
    key = incoming_trigger_key(
        source_name,
        message,
    )

    with state_lock:
        active = active_generated.pop(
            key,
            None,
        )

    if active is None:
        return None

    messages = [
        mido.Message(
            "note_off",
            channel=generated["channel"],
            note=generated["note"],
            velocity=0,
        )
        for generated in active.get(
            "notes",
            [],
        )
    ]

    if active.get(
        "pass_original_note_off",
        False,
    ):
        messages.append(
            message.copy(
                velocity=0
                if hasattr(
                    message,
                    "velocity",
                )
                else 0
            )
        )

    return messages


def build_static_trigger_messages(
    source_name,
    message,
    assignment,
):
    notes = [
        clamp_midi_note(note)
        for note in assignment.get(
            "notes",
            [],
        )
    ]

    if not notes:
        remember_generated(
            source_name,
            message,
            [],
            False,
        )
        return []

    output_channel = (
        assignment["channel"] - 1
    )

    velocity = max(
        1,
        min(127, message.velocity),
    )

    messages = [
        mido.Message(
            "note_on",
            channel=output_channel,
            note=note,
            velocity=velocity,
        )
        for note in notes
    ]

    remember_generated(
        source_name,
        message,
        messages,
        False,
    )

    return messages


def transform_message(
    source_name,
    message,
):
    matching_aliases = find_matching_aliases(
        source_name,
        message,
    )

    if is_note_off(message):
        generated_offs = (
            generated_note_off_messages(
                source_name,
                message,
            )
        )

        if generated_offs is not None:
            return generated_offs

        with state_lock:
            muted = set(
                state.get(
                    "muted_aliases",
                    [],
                )
            )

        if any(
            alias in muted
            for alias in matching_aliases
        ):
            return []

    if not is_note_on(message):
        with state_lock:
            alias_routes = dict(
                state.get(
                    "alias_routes",
                    {},
                )
            )

            source_channel = state.get(
                "source_channels",
                {},
            ).get(source_name)

            source_mode = state.get(
                "source_modes",
                {},
            ).get(source_name, "passthrough")

        for alias_name in matching_aliases:
            channel = alias_routes.get(
                alias_name
            )

            if channel is not None:
                return [
                    message.copy(
                        channel=channel
                    )
                ]

        if (
            hasattr(message, "channel")
            and source_channel is not None
        ):
            return [
                message.copy(
                    channel=source_channel
                )
            ]

        if source_mode == "explicit_only":
            return []

        return [message]

    with state_lock:
        muted = set(
            state.get(
                "muted_aliases",
                [],
            )
        )

        state_snapshot = deepcopy(state)

    if any(
        alias in muted
        for alias in matching_aliases
    ):
        remember_generated(
            source_name,
            message,
            [],
            False,
        )
        return []

    with rule_engine_lock:
        rule_result = rule_engine.execute(
            source_name=source_name,
            message=message,
            matching_aliases=matching_aliases,
            state=state_snapshot,
        )

    if rule_result.matched_rule_ids:
        generated_messages = list(
            rule_result.messages
        )

        outgoing = list(
            generated_messages
        )

        pass_original = not (
            rule_result.suppress_original
        )

        if pass_original:
            outgoing.append(message)

        # Track only messages created by the rule. The
        # original input note is released separately when
        # pass_original_note_off is true.
        remember_generated(
            source_name,
            message,
            generated_messages,
            pass_original,
        )

        return outgoing

    with state_lock:
        trigger_assignments = deepcopy(
            state.get(
                "trigger_assignments",
                {},
            )
        )

        alias_routes = dict(
            state.get(
                "alias_routes",
                {},
            )
        )

        source_channel = state.get(
            "source_channels",
            {},
        ).get(source_name)

        source_mode = state.get(
            "source_modes",
            {},
        ).get(source_name, "passthrough")

    for alias_name in matching_aliases:
        assignment = (
            trigger_assignments.get(
                alias_name
            )
        )

        if assignment is None:
            continue

        if assignment.get("mode") == "chord":
            update_harmonic_context(
                alias_name,
                assignment,
            )

        return build_static_trigger_messages(
            source_name,
            message,
            assignment,
        )

    for alias_name in matching_aliases:
        target_channel = alias_routes.get(
            alias_name
        )

        if target_channel is not None:
            return [
                message.copy(
                    channel=target_channel
                )
            ]

    if (
        hasattr(message, "channel")
        and source_channel is not None
    ):
        return [
            message.copy(
                channel=source_channel
            )
        ]

    if source_mode == "explicit_only":
        remember_generated(
            source_name,
            message,
            [],
            False,
        )
        return []

    return [message]


def finish_capture(notes):
    global pending_capture
    global capture_collected_notes
    global capture_pressed_notes

    if pending_capture is None:
        return

    unique_notes = []

    for note in notes:
        note = clamp_midi_note(note)

        if note not in unique_notes:
            unique_notes.append(note)

    if not unique_notes:
        print("No notes were captured.")

        pending_capture = None
        capture_collected_notes = []
        capture_pressed_notes = set()
        return

    trigger_alias = pending_capture[
        "trigger_alias"
    ]

    assignment = {
        "mode": pending_capture["mode"],
        "notes": unique_notes,
        "channel": pending_capture["channel"],
        "capture_source": pending_capture[
            "capture_source"
        ],
        "velocity_mode": "trigger",
        "suppress_original": True,
    }

    push_undo(
        f'capture for "{trigger_alias}"'
    )

    with state_lock:
        state[
            "trigger_assignments"
        ][trigger_alias] = assignment

        old_alias_channel = state[
            "alias_routes"
        ].pop(
            trigger_alias,
            None,
        )

        removed_rule_channels = (
            remove_rules_for_alias_locked(
                trigger_alias
            )
        )

        save_state(state)

    if old_alias_channel is not None:
        send_channel_panic(
            old_alias_channel
        )

    for channel in removed_rule_channels:
        send_channel_panic(channel)

    mode_label = (
        "note"
        if len(unique_notes) == 1
        else "chord"
    )

    print(
        f'Captured {mode_label} {unique_notes} '
        f'for "{trigger_alias}".'
    )

    print(
        f'Each "{trigger_alias}" hit will send '
        f'the captured {mode_label} on MIDI '
        f'channel {assignment["channel"]}.'
    )

    pending_capture = None
    capture_collected_notes = []
    capture_pressed_notes = set()


def process_pending_capture(
    source_name,
    message,
):
    global capture_pressed_notes
    global capture_collected_notes

    if pending_capture is None:
        return False

    if (
        source_name
        != pending_capture["capture_source"]
    ):
        return False

    if message.type not in {
        "note_on",
        "note_off",
    }:
        return True

    if pending_capture["mode"] == "note":
        if is_note_on(message):
            finish_capture(
                [message.note]
            )

        return True

    if pending_capture["mode"] == "chord":
        if is_note_on(message):
            capture_pressed_notes.add(
                message.note
            )

            if (
                message.note
                not in capture_collected_notes
            ):
                capture_collected_notes.append(
                    message.note
                )

            print(
                f"Captured note {message.note}; "
                f"release the chord to save it."
            )

            return True

        if is_note_off(message):
            capture_pressed_notes.discard(
                message.note
            )

            if (
                capture_collected_notes
                and not capture_pressed_notes
            ):
                finish_capture(
                    capture_collected_notes
                )

            return True

    return True


def midi_callback(
    source_name,
    output_port,
):
    def callback(message):
        global last_event

        if process_pending_capture(
            source_name,
            message,
        ):
            print(
                f"\n[{source_name}] "
                f"{message} [CAPTURED]"
            )

            print(
                "midge> ",
                end="",
                flush=True,
            )
            return

        if is_note_on(message):
            event = {
                "source": source_name,
                "type": "note",
                "channel": message.channel,
                "note": message.note,
                "velocity": (
                    message.velocity
                ),
            }

            with state_lock:
                last_event = event

        transformed_messages = (
            transform_message(
                source_name,
                message,
            )
        )

        print(f"\n[{source_name}] {message}")

        if not transformed_messages:
            print("       OUT suppressed")

        for transformed in (
            transformed_messages
        ):
            if transformed != message:
                print(
                    f"       OUT {transformed}"
                )

            try:
                output_port.send(
                    transformed
                )

            except Exception as error:
                print(
                    "Could not send MIDI message: "
                    f"{error}"
                )

        print(
            "midge> ",
            end="",
            flush=True,
        )

    return callback


def create_alias(alias_name):
    global last_event

    alias_name = normalize_text(
        alias_name
    )

    if not alias_name:
        print("Alias name cannot be empty.")
        return

    with state_lock:
        if last_event is None:
            print(
                "Play a note or drum pad first."
            )
            return

        event = last_event.copy()

    push_undo(
        f'create alias "{alias_name}"'
    )

    with state_lock:
        state["aliases"][alias_name] = {
            "source": event["source"],
            "type": event["type"],
            "channel": event["channel"],
            "note": event["note"],
        }

        save_state(state)

    print(
        f'Alias saved: "{alias_name}" = '
        f'{event["source"]} note '
        f'{event["note"]} on MIDI channel '
        f'{event["channel"] + 1}'
    )


def show_last_event():
    with state_lock:
        event = (
            last_event.copy()
            if last_event
            else None
        )

    if event is None:
        print(
            "No note has been played yet."
        )
        return

    print(
        f'Last event: {event["source"]}, '
        f'note {event["note"]}, '
        f'MIDI channel '
        f'{event["channel"] + 1}, '
        f'velocity {event["velocity"]}'
    )


def list_aliases():
    with state_lock:
        aliases = deepcopy(
            state["aliases"]
        )

        routes = dict(
            state["alias_routes"]
        )

        static_triggers = deepcopy(
            state["trigger_assignments"]
        )

        muted = set(
            state.get(
                "muted_aliases",
                [],
            )
        )

        rules = deepcopy(
            state.get("rules", {})
        )

    if not aliases:
        print(
            "No aliases have been created."
        )
        return

    print("\nAliases:")

    for alias_name, event in aliases.items():
        description = (
            f'  "{alias_name}" -> '
            f'{event["source"]}, '
            f'note {event["note"]}, '
            f'MIDI channel '
            f'{event["channel"] + 1}'
        )

        if alias_name in muted:
            description += " [MUTED]"

        matching_rules = [
            rule
            for rule in rules.values()
            if rule.get(
                "when",
                {},
            ).get(
                "alias"
            ) == alias_name
        ]

        if matching_rules:
            description += (
                f" [{len(matching_rules)} "
                f"runtime rule(s)]"
            )

        elif alias_name in static_triggers:
            assignment = static_triggers[
                alias_name
            ]

            description += (
                f' -> triggers '
                f'{assignment["notes"]} '
                f'on channel '
                f'{assignment["channel"]}'
            )

        elif alias_name in routes:
            description += (
                f' -> output channel '
                f'{routes[alias_name] + 1}'
            )

        print(description)


def delete_alias(alias_name):
    alias_name = normalize_text(
        alias_name
    )

    with state_lock:
        if alias_name not in state["aliases"]:
            print(
                f'Alias "{alias_name}" '
                f"does not exist."
            )
            return

    push_undo(
        f'delete alias "{alias_name}"'
    )

    with state_lock:
        old_route = state[
            "alias_routes"
        ].pop(
            alias_name,
            None,
        )

        old_static = state[
            "trigger_assignments"
        ].pop(
            alias_name,
            None,
        )

        removed_channels = (
            remove_rules_for_alias_locked(
                alias_name
            )
        )

        state["muted_aliases"] = [
            item
            for item in state.get(
                "muted_aliases",
                [],
            )
            if item != alias_name
        ]

        del state["aliases"][alias_name]

        for group_name, members in (
            state["groups"].items()
        ):
            state["groups"][group_name] = [
                member
                for member in members
                if member != alias_name
            ]

        context = state.get(
            "active_harmonic_context"
        )

        if (
            context
            and context.get(
                "selector_alias"
            ) == alias_name
        ):
            state[
                "active_harmonic_context"
            ] = None

        save_state(state)

    if old_route is not None:
        send_channel_panic(old_route)

    if old_static is not None:
        send_channel_panic(
            old_static["channel"] - 1
        )

    for channel in removed_channels:
        send_channel_panic(channel)

    print(
        f'Deleted alias "{alias_name}".'
    )


def create_alias_group(
    group_name,
    members_text,
):
    group_name = normalize_text(
        group_name
    )

    members = [
        normalize_text(member)
        for member in members_text.split(",")
        if normalize_text(member)
    ]

    if not group_name:
        print(
            "Group name cannot be empty."
        )
        return

    with state_lock:
        missing = [
            member
            for member in members
            if member not in state["aliases"]
        ]

    if missing:
        print(
            "Unknown aliases: "
            + ", ".join(missing)
        )
        return

    push_undo(
        f'create group "{group_name}"'
    )

    with state_lock:
        state["groups"][group_name] = (
            list(dict.fromkeys(members))
        )

        save_state(state)

    print(
        f'Group "{group_name}" = '
        f'{state["groups"][group_name]}'
    )


def show_groups():
    with state_lock:
        groups = deepcopy(
            state.get("groups", {})
        )

    if not groups:
        print(
            "No groups have been created."
        )
        return

    print("\nGroups:")

    for group_name, members in (
        groups.items()
    ):
        print(
            f'  "{group_name}": {members}'
        )


def route_alias(
    alias_name,
    midi_channel,
):
    alias_name = normalize_text(
        alias_name
    )

    if not 1 <= midi_channel <= 16:
        print(
            "MIDI channel must be "
            "between 1 and 16."
        )
        return

    with state_lock:
        if alias_name not in state["aliases"]:
            print(
                f'Alias "{alias_name}" '
                f"does not exist."
            )
            return

    push_undo(
        f'route "{alias_name}"'
    )

    new_channel = midi_channel - 1

    with state_lock:
        old_channel = state[
            "alias_routes"
        ].get(alias_name)

        old_static = state[
            "trigger_assignments"
        ].pop(
            alias_name,
            None,
        )

        removed_channels = (
            remove_rules_for_alias_locked(
                alias_name
            )
        )

        state["alias_routes"][
            alias_name
        ] = new_channel

        save_state(state)

    channels = {
        old_channel,
        (
            old_static["channel"] - 1
            if old_static
            else None
        ),
        *removed_channels,
    }

    for previous_channel in channels:
        if (
            previous_channel is not None
            and previous_channel != new_channel
        ):
            send_channel_panic(
                previous_channel
            )

    print(
        f'Routing "{alias_name}" exclusively '
        f"to MIDI channel {midi_channel}."
    )


def clear_alias_route(alias_name):
    alias_name = normalize_text(
        alias_name
    )

    with state_lock:
        if alias_name not in state[
            "alias_routes"
        ]:
            print(
                f'Alias "{alias_name}" '
                f"has no channel route."
            )
            return

    push_undo(
        f'clear route "{alias_name}"'
    )

    with state_lock:
        old_channel = state[
            "alias_routes"
        ].pop(
            alias_name
        )

        save_state(state)

    send_channel_panic(old_channel)

    print(
        f'Cleared routing for '
        f'"{alias_name}".'
    )


def route_alias_group(
    group_name,
    midi_channel,
):
    group_name = normalize_text(
        group_name
    )

    if not 1 <= midi_channel <= 16:
        print(
            "MIDI channel must be "
            "between 1 and 16."
        )
        return

    with state_lock:
        members = list(
            state["groups"].get(
                group_name,
                [],
            )
        )

    if not members:
        print(
            f'Group "{group_name}" '
            f"does not exist or is empty."
        )
        return

    push_undo(
        f'route group "{group_name}"'
    )

    previous_channels = set()
    new_channel = midi_channel - 1

    with state_lock:
        for member in members:
            old_route = state[
                "alias_routes"
            ].get(member)

            if old_route is not None:
                previous_channels.add(
                    old_route
                )

            old_static = state[
                "trigger_assignments"
            ].pop(
                member,
                None,
            )

            if old_static is not None:
                previous_channels.add(
                    old_static["channel"] - 1
                )

            previous_channels.update(
                remove_rules_for_alias_locked(
                    member
                )
            )

            state["alias_routes"][
                member
            ] = new_channel

        save_state(state)

    for channel in previous_channels:
        if channel != new_channel:
            send_channel_panic(channel)

    print(
        f'Routing group "{group_name}" '
        f"to MIDI channel {midi_channel}."
    )


def clear_alias_group_route(
    group_name,
):
    group_name = normalize_text(
        group_name
    )

    with state_lock:
        members = list(
            state["groups"].get(
                group_name,
                [],
            )
        )

    if not members:
        print(
            f'Group "{group_name}" '
            f"does not exist or is empty."
        )
        return

    push_undo(
        f'clear group route "{group_name}"'
    )

    old_channels = set()

    with state_lock:
        for member in members:
            channel = state[
                "alias_routes"
            ].pop(
                member,
                None,
            )

            if channel is not None:
                old_channels.add(channel)

        save_state(state)

    for channel in old_channels:
        send_channel_panic(channel)

    print(
        f'Cleared routes for group '
        f'"{group_name}".'
    )


def show_routes():
    with state_lock:
        source_channels = dict(
            state["source_channels"]
        )
        source_modes = dict(
            state.get("source_modes", {})
        )

        alias_routes = dict(
            state["alias_routes"]
        )

    print("\nSource routing:")

    for source_name, channel in (
        source_channels.items()
    ):
        mode = source_modes.get(
            source_name,
            "passthrough",
        )
        if mode == "explicit_only":
            print(
                f"  {source_name}: explicit-only "
                "(unassigned MIDI suppressed)"
            )
        elif channel is None:
            print(
                f"  {source_name}: passthrough"
            )
        else:
            print(
                f"  {source_name}: MIDI "
                f"channel {channel + 1}"
            )

    print("\nAlias routing:")

    if not alias_routes:
        print("  None")

    else:
        for alias_name, channel in (
            alias_routes.items()
        ):
            print(
                f'  "{alias_name}": MIDI '
                f"channel {channel + 1}"
            )


def clear_source_alias_routes_locked(
    source_name,
):
    cleared_channels = set()

    for alias_name, alias_event in (
        state.get(
            "aliases",
            {},
        ).items()
    ):
        if (
            alias_event.get("source")
            != source_name
        ):
            continue

        old_route = state[
            "alias_routes"
        ].pop(
            alias_name,
            None,
        )

        if old_route is not None:
            cleared_channels.add(
                old_route
            )

    return cleared_channels


def set_source_channel(
    source_name,
    midi_channel,
):
    if source_name not in INPUT_PORTS:
        print(
            f'Unknown source "{source_name}".'
        )
        return

    if not 1 <= midi_channel <= 16:
        print(
            "MIDI channel must be "
            "between 1 and 16."
        )
        return

    push_undo(
        f'route source "{source_name}"'
    )

    new_channel = midi_channel - 1

    with state_lock:
        old_source_channel = state[
            "source_channels"
        ].get(source_name)

        cleared_alias_channels = (
            clear_source_alias_routes_locked(
                source_name
            )
        )

        state[
            "source_channels"
        ][source_name] = new_channel
        state.setdefault(
            "source_modes",
            {},
        )[source_name] = "passthrough"

        save_state(state)

    channels_to_panic = set(
        cleared_alias_channels
    )

    if old_source_channel is not None:
        channels_to_panic.add(
            old_source_channel
        )

    for channel in channels_to_panic:
        if channel != new_channel:
            send_channel_panic(channel)

    print(
        f"{source_name} now outputs "
        f"exclusively on MIDI channel "
        f"{midi_channel}."
    )


def set_source_passthrough(source_name):
    if source_name not in INPUT_PORTS:
        print(
            f'Unknown source "{source_name}".'
        )
        return

    push_undo(
        f'passthrough source "{source_name}"'
    )

    with state_lock:
        old_channel = state[
            "source_channels"
        ].get(source_name)

        state[
            "source_channels"
        ][source_name] = None
        state.setdefault(
            "source_modes",
            {},
        )[source_name] = "passthrough"

        save_state(state)

    if old_channel is not None:
        send_channel_panic(old_channel)

    print(
        f"{source_name} now preserves "
        f"its original MIDI channel."
    )


def set_source_explicit_only(source_name):
    if source_name not in INPUT_PORTS:
        print(
            f'Unknown source "{source_name}".'
        )
        return

    push_undo(
        f'explicit-only source "{source_name}"'
    )

    with state_lock:
        old_channel = state.get(
            "source_channels",
            {},
        ).get(source_name)

        state.setdefault(
            "source_channels",
            {},
        )[source_name] = None
        state.setdefault(
            "source_modes",
            {},
        )[source_name] = "explicit_only"

        save_state(state)

    if old_channel is not None:
        send_channel_panic(old_channel)

    print(
        f"{source_name} is now explicit-only: "
        "unassigned MIDI is suppressed."
    )


def arm_trigger_capture(
    trigger_alias,
    capture_mode,
    capture_source,
    midi_channel,
):
    global pending_capture
    global capture_pressed_notes
    global capture_collected_notes

    trigger_alias = normalize_text(
        trigger_alias
    )

    capture_source = normalize_text(
        capture_source
    )

    if capture_mode not in {
        "note",
        "chord",
    }:
        print(
            f'Unknown capture mode '
            f'"{capture_mode}".'
        )
        return

    if not 1 <= midi_channel <= 16:
        print(
            "MIDI channel must be "
            "between 1 and 16."
        )
        return

    with state_lock:
        if trigger_alias not in state["aliases"]:
            print(
                f'Alias "{trigger_alias}" '
                f"does not exist."
            )
            return

    if capture_source not in INPUT_PORTS:
        print(
            f'Unknown capture source '
            f'"{capture_source}".'
        )
        return

    pending_capture = {
        "trigger_alias": trigger_alias,
        "mode": capture_mode,
        "capture_source": capture_source,
        "channel": midi_channel,
    }

    capture_pressed_notes = set()
    capture_collected_notes = []

    print(
        f'{capture_mode.title()} capture armed '
        f'for "{trigger_alias}".'
    )

    print(
        f"Play and release the "
        f"{capture_mode} on "
        f"{capture_source}; output channel "
        f"will be {midi_channel}."
    )


def cancel_capture():
    global pending_capture
    global capture_pressed_notes
    global capture_collected_notes

    if pending_capture is None:
        print(
            "No note or chord capture is armed."
        )
        return

    pending_capture = None
    capture_pressed_notes = set()
    capture_collected_notes = []

    print("Capture cancelled.")


def clear_trigger_assignment(
    trigger_alias,
):
    trigger_alias = normalize_text(
        trigger_alias
    )

    with state_lock:
        if trigger_alias not in state[
            "trigger_assignments"
        ]:
            print(
                f'Alias "{trigger_alias}" has '
                f"no static note or chord trigger."
            )
            return

    push_undo(
        f'clear static trigger '
        f'"{trigger_alias}"'
    )

    with state_lock:
        assignment = state[
            "trigger_assignments"
        ].pop(trigger_alias)

        save_state(state)

    send_channel_panic(
        assignment["channel"] - 1
    )

    print(
        f'Cleared static trigger for '
        f'"{trigger_alias}".'
    )


def rule_output_channel(rule):
    output = rule.get("output", {})

    channel = output.get("channel")

    if isinstance(channel, int):
        return channel - 1

    return None


def remove_rules_for_alias_locked(
    alias_name,
    active_chord_only=False,
):
    alias_name = normalize_text(
        alias_name
    )

    removed_channels = set()

    for rule_id, rule in list(
        state.get("rules", {}).items()
    ):
        when_alias = normalize_text(
            rule.get(
                "when",
                {},
            ).get(
                "alias",
                "",
            )
        )

        if when_alias != alias_name:
            continue

        if active_chord_only:
            derive_source = (
                rule.get(
                    "derive",
                    {},
                ) or {}
            ).get("source")

            if derive_source != "active_chord":
                continue

        channel = rule_output_channel(rule)

        if channel is not None:
            removed_channels.add(channel)

        del state["rules"][rule_id]

        state["rule_order"] = [
            item
            for item in state.get(
                "rule_order",
                [],
            )
            if item != rule_id
        ]

    return removed_channels


def unique_rule_id(base_id):
    with state_lock:
        rules = state.get("rules", {})

        if base_id not in rules:
            return base_id

        number = 2

        while f"{base_id}-{number}" in rules:
            number += 1

        return f"{base_id}-{number}"


def install_runtime_rule(
    raw_rule,
    replace=False,
):
    global last_modified_rule_id

    with state_lock:
        snapshot = deepcopy(state)

    try:
        validated = validate_rule(
            raw_rule,
            snapshot,
            list(INPUT_PORTS.keys()),
        )

    except RuleValidationError as error:
        print(
            f"Rule rejected: {error}"
        )
        return None

    requested_id = validated["id"]

    with state_lock:
        exists = (
            requested_id
            in state.get("rules", {})
        )

    if exists and not replace:
        validated["id"] = unique_rule_id(
            requested_id
        )

    push_undo(
        f'install rule "{validated["id"]}"'
    )

    with state_lock:
        rule_id = validated["id"]

        old_rule = state[
            "rules"
        ].get(rule_id)

        state["rules"][rule_id] = (
            validated
        )

        if rule_id not in state[
            "rule_order"
        ]:
            state["rule_order"].append(
                rule_id
            )

        save_state(state)

    old_channel = (
        rule_output_channel(old_rule)
        if old_rule
        else None
    )

    new_channel = rule_output_channel(
        validated
    )

    if (
        old_channel is not None
        and old_channel != new_channel
    ):
        send_channel_panic(old_channel)

    last_modified_rule_id = rule_id

    print("Installed runtime rule:")
    print(
        "  " + summarize_rule(validated)
    )

    return rule_id


def replace_runtime_rule(
    rule_id,
    validated_rule,
):
    global last_modified_rule_id

    with state_lock:
        if rule_id not in state["rules"]:
            print(
                f'Rule "{rule_id}" does not exist.'
            )
            return False

        old_rule = deepcopy(
            state["rules"][rule_id]
        )

    push_undo(
        f'update rule "{rule_id}"'
    )

    with state_lock:
        validated_rule["id"] = rule_id
        state["rules"][rule_id] = (
            validated_rule
        )

        save_state(state)

    old_channel = rule_output_channel(
        old_rule
    )

    new_channel = rule_output_channel(
        validated_rule
    )

    if (
        old_channel is not None
        and old_channel != new_channel
    ):
        send_channel_panic(old_channel)

    last_modified_rule_id = rule_id

    print("Updated runtime rule:")
    print(
        "  "
        + summarize_rule(validated_rule)
    )

    return True


def patch_runtime_rule(
    rule_id,
    changes,
):
    with state_lock:
        existing = deepcopy(
            state.get(
                "rules",
                {},
            ).get(rule_id)
        )

        snapshot = deepcopy(state)

    if existing is None:
        print(
            f'Rule "{rule_id}" does not exist.'
        )
        return False

    merged = deep_merge(
        existing,
        changes,
    )

    merged["id"] = rule_id

    try:
        validated = validate_rule(
            merged,
            snapshot,
            list(INPUT_PORTS.keys()),
            existing_rule_id=rule_id,
        )

    except RuleValidationError as error:
        print(
            f"Rule update rejected: {error}"
        )
        return False

    return replace_runtime_rule(
        rule_id,
        validated,
    )


def delete_runtime_rule(rule_id):
    global last_modified_rule_id

    with state_lock:
        rule = deepcopy(
            state.get(
                "rules",
                {},
            ).get(rule_id)
        )

    if rule is None:
        print(
            f'Rule "{rule_id}" does not exist.'
        )
        return

    push_undo(
        f'delete rule "{rule_id}"'
    )

    with state_lock:
        del state["rules"][rule_id]

        state["rule_order"] = [
            item
            for item in state[
                "rule_order"
            ]
            if item != rule_id
        ]

        save_state(state)

    channel = rule_output_channel(rule)

    if channel is not None:
        send_channel_panic(channel)

    if last_modified_rule_id == rule_id:
        last_modified_rule_id = None

    print(
        f'Deleted runtime rule '
        f'"{rule_id}".'
    )


def set_rule_enabled(
    rule_id,
    enabled,
):
    global last_modified_rule_id

    with state_lock:
        rule = state.get(
            "rules",
            {},
        ).get(rule_id)

        if rule is None:
            print(
                f'Rule "{rule_id}" '
                f"does not exist."
            )
            return

        if rule.get(
            "enabled",
            True,
        ) == enabled:
            print(
                f'Rule "{rule_id}" is already '
                f'{"enabled" if enabled else "disabled"}.'
            )
            return

    push_undo(
        f'{"enable" if enabled else "disable"} '
        f'rule "{rule_id}"'
    )

    with state_lock:
        state["rules"][rule_id][
            "enabled"
        ] = enabled

        save_state(state)

    if not enabled:
        channel = rule_output_channel(rule)

        if channel is not None:
            send_channel_panic(channel)

    last_modified_rule_id = rule_id

    print(
        f'Rule "{rule_id}" '
        f'{"enabled" if enabled else "disabled"}.'
    )


def find_rules_for_alias(alias_name):
    alias_name = normalize_text(
        alias_name
    )

    with state_lock:
        rules = deepcopy(
            state.get("rules", {})
        )

        order = list(
            state.get(
                "rule_order",
                [],
            )
        )

    matches = []

    for rule_id in order:
        rule = rules.get(rule_id)

        if rule is None:
            continue

        if normalize_text(
            rule.get(
                "when",
                {},
            ).get(
                "alias",
                "",
            )
        ) == alias_name:
            matches.append(rule_id)

    return matches


def best_rule_for_alias(alias_name):
    matches = find_rules_for_alias(
        alias_name
    )

    if not matches:
        return None

    if (
        last_modified_rule_id
        in matches
    ):
        return last_modified_rule_id

    return matches[-1]


def resolve_rule_reference(text):
    text = text.strip()

    with state_lock:
        rules = deepcopy(
            state.get("rules", {})
        )

    if text in rules:
        return text

    normalized = normalize_text(text)

    for rule_id, rule in rules.items():
        if normalize_text(rule_id) == normalized:
            return rule_id

        if normalize_text(
            rule.get("name", "")
        ) == normalized:
            return rule_id

    alias = find_alias_in_text(
        normalized
    )

    if alias:
        return best_rule_for_alias(alias)

    return None


def assign_dynamic_chord_note_trigger(
    trigger_alias,
    selector_group,
    note_source,
    transpose,
    midi_channel,
):
    trigger_alias = normalize_text(
        trigger_alias
    )

    selector_group = normalize_text(
        selector_group
    )

    rule_id = slugify(
        f"{trigger_alias} follows "
        f"{selector_group}"
    )

    existing = best_rule_for_alias(
        trigger_alias
    )

    if existing is not None:
        with state_lock:
            existing_rule = state[
                "rules"
            ][existing]

        if (
            (
                existing_rule.get(
                    "derive",
                    {},
                ) or {}
            ).get("source")
            == "active_chord"
        ):
            rule_id = existing

    rule = {
        "id": rule_id,
        "name": (
            f"{trigger_alias} follows "
            f"{selector_group}"
        ),
        "enabled": True,
        "priority": 100,
        "when": {
            "event": "note_on",
            "alias": trigger_alias,
        },
        "condition": {},
        "derive": {
            "source": "active_chord",
            "selector_group": (
                selector_group
            ),
            "select": note_source,
            "transpose": int(transpose),
        },
        "output": {
            "type": "note",
            "channel": int(
                midi_channel
            ),
            "velocity": "input",
            "suppress_original": True,
        },
        "state_updates": [],
        "continue": False,
    }

    if rule_id in state.get("rules", {}):
        with state_lock:
            snapshot = deepcopy(state)

        try:
            validated = validate_rule(
                rule,
                snapshot,
                list(INPUT_PORTS.keys()),
                existing_rule_id=rule_id,
            )

        except RuleValidationError as error:
            print(
                f"Rule rejected: {error}"
            )
            return

        replace_runtime_rule(
            rule_id,
            validated,
        )

    else:
        install_runtime_rule(rule)


def clear_dynamic_note_trigger(
    trigger_alias,
):
    trigger_alias = normalize_text(
        trigger_alias
    )

    matching = []

    with state_lock:
        for rule_id, rule in (
            state.get(
                "rules",
                {},
            ).items()
        ):
            if normalize_text(
                rule.get(
                    "when",
                    {},
                ).get(
                    "alias",
                    "",
                )
            ) != trigger_alias:
                continue

            if (
                (
                    rule.get(
                        "derive",
                        {},
                    ) or {}
                ).get("source")
                == "active_chord"
            ):
                matching.append(rule_id)

    if not matching:
        print(
            f'Alias "{trigger_alias}" '
            f"has no active-chord rule."
        )
        return

    for rule_id in matching:
        delete_runtime_rule(rule_id)


def clear_any_trigger(
    trigger_alias,
):
    trigger_alias = normalize_text(
        trigger_alias
    )

    with state_lock:
        static_exists = (
            trigger_alias
            in state["trigger_assignments"]
        )

        rule_ids = find_rules_for_alias(
            trigger_alias
        )

    if not static_exists and not rule_ids:
        print(
            f'Alias "{trigger_alias}" '
            f"has no trigger behavior."
        )
        return

    if static_exists:
        clear_trigger_assignment(
            trigger_alias
        )

    for rule_id in rule_ids:
        delete_runtime_rule(rule_id)


def show_runtime_rules():
    with state_lock:
        rules = deepcopy(
            state.get("rules", {})
        )

        order = list(
            state.get(
                "rule_order",
                [],
            )
        )

    print("\nRuntime rules:")

    if not rules:
        print("  None")
        return

    for rule_id in order:
        rule = rules.get(rule_id)

        if rule is not None:
            print(
                "  " + summarize_rule(rule)
            )


def show_trigger_assignments():
    with state_lock:
        static_assignments = deepcopy(
            state["trigger_assignments"]
        )

        context = deepcopy(
            state.get(
                "active_harmonic_context"
            )
        )

    print(
        "\nStatic capture assignments:"
    )

    if not static_assignments:
        print("  None")

    else:
        for alias_name, assignment in (
            static_assignments.items()
        ):
            print(
                f'  "{alias_name}" -> '
                f'{assignment["mode"]} '
                f'{assignment["notes"]} -> '
                f'MIDI channel '
                f'{assignment["channel"]}'
            )

    show_runtime_rules()

    print("\nActive harmonic context:")

    if context is None:
        print("  None")

    else:
        print(
            f'  Selected by '
            f'"{context["selector_alias"]}"'
        )

        print(
            f'  Groups: '
            f'{context["selector_groups"]}'
        )

        print(
            f'  Notes: {context["notes"]}'
        )


def clear_harmonic_context():
    with state_lock:
        state[
            "active_harmonic_context"
        ] = None

    print(
        "Cleared active harmonic context."
    )


def mute_alias(alias_name):
    alias_name = normalize_text(
        alias_name
    )

    with state_lock:
        if alias_name not in state["aliases"]:
            print(
                f'Alias "{alias_name}" '
                f"does not exist."
            )
            return

        if alias_name in state.get(
            "muted_aliases",
            [],
        ):
            print(
                f'Alias "{alias_name}" '
                f"is already muted."
            )
            return

    push_undo(
        f'mute "{alias_name}"'
    )

    with state_lock:
        state.setdefault(
            "muted_aliases",
            [],
        ).append(alias_name)

        save_state(state)

    send_global_panic()

    print(
        f'Muted "{alias_name}".'
    )


def unmute_alias(alias_name):
    alias_name = normalize_text(
        alias_name
    )

    with state_lock:
        if alias_name not in state.get(
            "muted_aliases",
            [],
        ):
            print(
                f'Alias "{alias_name}" '
                f"is not muted."
            )
            return

    push_undo(
        f'unmute "{alias_name}"'
    )

    with state_lock:
        state["muted_aliases"] = [
            item
            for item in state[
                "muted_aliases"
            ]
            if item != alias_name
        ]

        save_state(state)

    print(
        f'Unmuted "{alias_name}".'
    )


def save_preset(name):
    name = normalize_text(name)

    if not name:
        print(
            "Preset name cannot be empty."
        )
        return

    push_undo(
        f'save preset "{name}"'
    )

    with state_lock:
        state.setdefault(
            "presets",
            {},
        )[name] = snapshot_for_preset(
            state
        )

        save_state(state)

    print(
        f'Saved preset "{name}".'
    )


def load_preset(name):
    global state
    global last_modified_rule_id

    name = normalize_text(name)

    with state_lock:
        snapshot = deepcopy(
            state.get(
                "presets",
                {},
            ).get(name)
        )

    if snapshot is None:
        print(
            f'Preset "{name}" does not exist.'
        )
        return

    push_undo(
        f'load preset "{name}"'
    )

    with state_lock:
        state = apply_preset_snapshot(
            state,
            snapshot,
        )

        save_state(state)

    with rule_engine_lock:
        rule_engine.reset_runtime()

    last_modified_rule_id = None
    send_global_panic()

    print(
        f'Loaded preset "{name}".'
    )


def list_presets():
    with state_lock:
        names = sorted(
            state.get(
                "presets",
                {},
            )
        )

    if not names:
        print("No presets saved.")
        return

    print("\nPresets:")

    for name in names:
        print(f"  {name}")


def show_status():
    with state_lock:
        context = deepcopy(
            state.get(
                "active_harmonic_context"
            )
        )

        muted = list(
            state.get(
                "muted_aliases",
                [],
            )
        )

        rule_count = len(
            state.get("rules", {})
        )

        enabled_count = sum(
            1
            for rule in state.get(
                "rules",
                {},
            ).values()
            if rule.get("enabled", True)
        )

    print("\nMidge status:")
    print(
        f"  Runtime rules: "
        f"{enabled_count}/{rule_count} enabled"
    )

    print(
        f"  Muted aliases: "
        f"{muted or 'none'}"
    )

    if context:
        print(
            f'  Active harmony: '
            f'{context["selector_alias"]} '
            f'{context["notes"]}'
        )
    else:
        print(
            "  Active harmony: none"
        )

    with rule_engine_lock:
        runtime_variables = deepcopy(
            rule_engine.variables
        )

    if runtime_variables:
        print(
            f"  Runtime variables: "
            f"{runtime_variables}"
        )

    if last_modified_rule_id:
        print(
            f"  Last modified rule: "
            f"{last_modified_rule_id}"
        )


def find_alias_in_text(text):
    normalized = normalize_text(text)

    with state_lock:
        aliases = sorted(
            state.get(
                "aliases",
                {},
            ),
            key=len,
            reverse=True,
        )

    for alias in aliases:
        if re.search(
            rf"\b{re.escape(alias)}\b",
            normalized,
        ):
            return alias

        shortened = alias.replace(
            " drum",
            "",
        )

        if (
            shortened != alias
            and re.search(
                rf"\b{re.escape(shortened)}\b",
                normalized,
            )
        ):
            return alias

    return None


def find_group_in_text(text):
    normalized = normalize_text(text)

    with state_lock:
        groups = sorted(
            state.get(
                "groups",
                {},
            ),
            key=len,
            reverse=True,
        )

    for group in groups:
        if re.search(
            rf"\b{re.escape(group)}\b",
            normalized,
        ):
            return group

    return None


def parse_follow_shorthand(command):
    normalized = normalize_text(
        command
    )

    if not normalized.startswith(
        "follow "
    ):
        return False

    alias = find_alias_in_text(
        normalized
    )

    group = find_group_in_text(
        normalized
    )

    if alias is None or group is None:
        print(
            'Use: follow <alias> from '
            '<group> <selector> <transpose> '
            'ch<1-16>'
        )
        return True

    selector = "lowest"

    for candidate in (
        "chord third",
        "chord fifth",
        "lowest",
        "highest",
        "first",
        "second",
        "third",
        "root",
        "cycle",
        "random",
        "all",
    ):
        if candidate in normalized:
            selector = candidate.replace(
                " ",
                "_",
            )
            break

    channel_match = re.search(
        r"\b(?:ch|channel)\s*(\d{1,2})\b",
        normalized,
    )

    channel = (
        int(channel_match.group(1))
        if channel_match
        else 1
    )

    # Remove the channel fragment before looking for the
    # signed transposition number.
    transpose_text = normalized

    if channel_match:
        transpose_text = (
            normalized[:channel_match.start()]
            + normalized[channel_match.end():]
        )

    transpose_match = re.search(
        r"(?<!\w)([+-]?\d{1,3})(?!\w)",
        transpose_text,
    )

    transpose = (
        int(transpose_match.group(1))
        if transpose_match
        else 0
    )

    assign_dynamic_chord_note_trigger(
        alias,
        group,
        selector,
        transpose,
        channel,
    )

    return True


def parse_fast_capture_command(command):
    """
    Handle common note/chord capture requests without the
    LLM. This keeps capture setup immediate even when the
    local model is stopped or slow.
    """
    normalized = normalize_text(command)

    explicit_shorthand = (
        normalized.startswith("capture chord ")
        or normalized.startswith("capture note ")
    )

    next_chord_request = (
        "next chord" in normalized
        and any(
            word in normalized.split()
            for word in {
                "play",
                "trigger",
                "capture",
                "remember",
                "use",
            }
        )
    )

    next_note_request = (
        "next note" in normalized
        and any(
            word in normalized.split()
            for word in {
                "play",
                "trigger",
                "capture",
                "remember",
                "use",
            }
        )
    )

    if not (
        explicit_shorthand
        or next_chord_request
        or next_note_request
    ):
        return False

    if (
        "chord" in normalized
        and "note" not in normalized
    ):
        capture_mode = "chord"
    elif "next chord" in normalized:
        capture_mode = "chord"
    else:
        capture_mode = "note"

    trigger_alias = find_alias_in_text(
        normalized
    )

    if trigger_alias is None:
        print(
            "Could not determine which existing "
            "alias should receive the capture."
        )
        return True

    channel_match = re.search(
        r"\b(?:midi\s+)?(?:channel|ch)\s*"
        r"(\d{1,2})\b",
        normalized,
    )

    if channel_match:
        midi_channel = int(
            channel_match.group(1)
        )
    else:
        with state_lock:
            existing = state.get(
                "trigger_assignments",
                {},
            ).get(trigger_alias)

        midi_channel = (
            int(existing.get("channel", 1))
            if existing
            else 1
        )

    if not 1 <= midi_channel <= 16:
        print(
            "MIDI channel must be between "
            "1 and 16."
        )
        return True

    capture_source = identify_source(
        normalized
    )

    # A capture command normally means the keyboard. Do
    # not infer the trigger's own source (for example the
    # Simmons kit) from its alias.
    if capture_source is None:
        capture_source = "kboard"

    arm_trigger_capture(
        trigger_alias=trigger_alias,
        capture_mode=capture_mode,
        capture_source=capture_source,
        midi_channel=midi_channel,
    )

    return True


ORDINAL_VALUES = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def parse_ordinal_value(token):
    token = normalize_text(token)
    if token in ORDINAL_VALUES:
        return ORDINAL_VALUES[token]

    match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)?", token)
    if match:
        return int(match.group(1))

    return None


def parse_fast_nth_fixed_note_rule(command):
    """Parse common every-Nth-hit fixed-note rules locally."""
    normalized = normalize_text(command)

    patterns = [
        # every fourth snare hit, play note 60 on channel 3
        re.compile(
            r"\bevery\s+(\w+)\s+(.+?)\s+hit\b.*?"
            r"(?:play|send|trigger|output)\s+(?:midi\s+)?note\s+"
            r"(\d{1,3})\b(?:.*?(?:on\s+)?(?:midi\s+)?(?:channel|ch)\s*(\d{1,2})\b)?"
        ),
        # make the snare play note 60 every fourth hit [on channel 3]
        re.compile(
            r"\b(?:make\s+)?(?:the\s+)?(.+?)\s+"
            r"(?:play|send|trigger|output)\s+(?:midi\s+)?note\s+"
            r"(\d{1,3})\s+every\s+(\w+)\s+hit\b"
            r"(?:.*?(?:on\s+)?(?:midi\s+)?(?:channel|ch)\s*(\d{1,2})\b)?"
        ),
    ]

    match = patterns[0].search(normalized)
    alternate = False
    if match is None:
        match = patterns[1].search(normalized)
        alternate = True
    if match is None:
        return False

    if alternate:
        alias_text, note_text, ordinal_text, channel_text = match.groups()
    else:
        ordinal_text, alias_text, note_text, channel_text = match.groups()

    every_n = parse_ordinal_value(ordinal_text)
    alias_text = normalize_text(alias_text)
    note = int(note_text)
    # Channel 3 is the current musical output default used by captured chords.
    channel = int(channel_text) if channel_text else 3

    if every_n is None or every_n < 1:
        print("Could not determine the hit interval.")
        return True

    alias = find_alias_in_text(alias_text)
    if alias is None:
        print(f'Unknown alias "{alias_text}". Create or name the pad first.')
        return True

    if not 0 <= note <= 127:
        print("MIDI note must be between 0 and 127.")
        return True

    if not 1 <= channel <= 16:
        print("MIDI channel must be between 1 and 16.")
        return True

    rule = {
        "version": 1,
        "id": slugify(f"every {every_n} {alias} note {note}"),
        "name": f"Every {every_n}th {alias}: note {note}",
        "enabled": True,
        "priority": 100,
        "when": {"event": "note_on", "alias": alias},
        "condition": {"every_n": every_n},
        "derive": {"source": "fixed", "notes": [note], "select": "first", "transpose": 0},
        "output": {
            "type": "note", "channel": channel, "velocity": "input",
            "velocity_scale": 1.0, "velocity_min": 1, "velocity_max": 127,
            "suppress_original": True,
        },
        "state_updates": [],
        "continue": False,
    }

    print("Local rule preview:")
    print(f"  Every {every_n}th {alias} hit -> fixed note {note} on channel {channel}; other hits remain silent.")
    install_runtime_rule(rule)
    return True


def validate_llm_action_against_command(command, action):
    """Reject obvious semantic contradictions before state changes."""
    normalized = normalize_text(command)
    if action.get("action") not in {"install_rule", "update_rule"}:
        return []

    rule = action.get("rule", {})
    errors = []

    nth = re.search(r"\bevery\s+(\w+)\b", normalized)
    if nth:
        requested_n = parse_ordinal_value(nth.group(1))
        returned_n = rule.get("condition", {}).get("every_n")
        if requested_n is not None and returned_n != requested_n:
            errors.append(
                f"requested every {requested_n} hits, but returned "
                f"every_n={returned_n!r}"
            )

    fixed_note = re.search(r"\b(?:midi\s+)?note\s+(\d{1,3})\b", normalized)
    if fixed_note:
        requested_note = int(fixed_note.group(1))
        derive = rule.get("derive", {})
        if (
            derive.get("source") != "fixed"
            or derive.get("notes") != [requested_note]
        ):
            errors.append(
                f"requested fixed note {requested_note}, but the returned "
                "rule derives something else"
            )

    channel_match = re.search(
        r"\b(?:midi\s+)?(?:channel|ch)\s*(\d{1,2})\b",
        normalized,
    )
    if channel_match:
        requested_channel = int(channel_match.group(1))
        returned_channel = rule.get("output", {}).get("channel")
        if returned_channel != requested_channel:
            errors.append(
                f"requested channel {requested_channel}, but returned "
                f"channel {returned_channel!r}"
            )

    return errors


def parse_fast_rule_update(command):
    global last_modified_rule_id

    normalized = normalize_text(
        command
    )

    if parse_follow_shorthand(
        normalized
    ):
        return True

    if normalized.startswith("mute "):
        mute_alias(
            normalized[len("mute "):]
        )
        return True

    if normalized.startswith(
        "unmute "
    ):
        unmute_alias(
            normalized[
                len("unmute "):
            ]
        )
        return True

    for prefix, enabled in (
        ("enable rule ", True),
        ("disable rule ", False),
    ):
        if normalized.startswith(prefix):
            reference = command[
                len(prefix):
            ].strip()

            rule_id = resolve_rule_reference(
                reference
            )

            if rule_id is None:
                print(
                    f'Could not find rule '
                    f'"{reference}".'
                )
            else:
                set_rule_enabled(
                    rule_id,
                    enabled,
                )

            return True

    if normalized.startswith(
        "delete rule "
    ):
        reference = command[
            len("delete rule "):
        ].strip()

        rule_id = resolve_rule_reference(
            reference
        )

        if rule_id is None:
            print(
                f'Could not find rule '
                f'"{reference}".'
            )
        else:
            delete_runtime_rule(rule_id)

        return True

    # Contextual correction after an LLM or shorthand edit.
    correction_channel = re.search(
        r"\b(?:i meant )?"
        r"(?:midi )?channel\s+"
        r"(\d{1,2})\b",
        normalized,
    )

    alias = find_alias_in_text(
        normalized
    )

    if (
        correction_channel
        and (
            "meant" in normalized
            or "rule" in normalized
            or "trigger" in normalized
            or (
                alias is not None
                and normalized.startswith(
                    alias
                )
            )
        )
    ):
        rule_id = (
            best_rule_for_alias(alias)
            if alias
            else last_modified_rule_id
        )

        if rule_id is None:
            return False

        channel = int(
            correction_channel.group(1)
        )

        patch_runtime_rule(
            rule_id,
            {
                "output": {
                    "channel": channel,
                }
            },
        )
        return True

    # "kick -12"
    if alias is not None:
        compact_pattern = re.fullmatch(
            rf"(?:set )?"
            rf"{re.escape(alias)}\s+"
            r"([+-]?\d{1,3})",
            normalized,
        )

        semitone_match = re.search(
            r"([+-]?\d{1,3})\s+"
            r"semitones?\b",
            normalized,
        )

        transpose_match = re.search(
            r"\btranspose(?: to)?\s+"
            r"([+-]?\d{1,3})\b",
            normalized,
        )

        chosen = (
            compact_pattern
            or semitone_match
            or transpose_match
        )

        if chosen is not None:
            rule_id = best_rule_for_alias(
                alias
            )

            if rule_id is None:
                return False

            transpose = int(
                chosen.group(1)
            )

            patch_runtime_rule(
                rule_id,
                {
                    "derive": {
                        "transpose": (
                            transpose
                        ),
                    }
                },
            )
            return True

        ch_match = re.fullmatch(
            rf"(?:set )?"
            rf"{re.escape(alias)}\s+"
            r"(?:ch|channel)\s*"
            r"(\d{1,2})",
            normalized,
        )

        if ch_match:
            rule_id = best_rule_for_alias(
                alias
            )

            if rule_id is None:
                return False

            patch_runtime_rule(
                rule_id,
                {
                    "output": {
                        "channel": int(
                            ch_match.group(1)
                        ),
                    }
                },
            )
            return True

    return False


def extract_channel(command):
    normalized = normalize_text(
        command
    )

    match = re.search(
        r"(?:midi\s+)?channel\s+"
        r"(\d{1,2})\b",
        normalized,
    )

    if match is None:
        match = re.search(
            r"\b(\d{1,2})\s*$",
            normalized,
        )

    if match is None:
        return None

    return int(match.group(1))


def extract_route_target(command):
    normalized = normalize_text(
        command
    )

    patterns = [
        (
            r"^(?:route|send|output|move|assign|"
            r"put|switch)\s+(?:the\s+)?(.+?)\s+"
            r"(?:to|onto|on)\s+"
            r"(?:midi\s+)?channel\s+\d{1,2}$"
        ),
        (
            r"^(?:route|send|output|move|assign|"
            r"put|switch)\s+(?:the\s+)?(.+?)\s+"
            r"(?:to|onto|on)\s+\d{1,2}$"
        ),
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            normalized,
        )

        if match:
            return normalize_text(
                match.group(1)
            )

    return None


def parse_natural_channel_command(
    command,
):
    normalized = normalize_text(
        command
    )

    # Prevent intervals from being mistaken for channels.
    semantic_words = {
        "trigger",
        "rule",
        "note",
        "chord",
        "capture",
        "remember",
        "tune",
        "lowest",
        "highest",
        "octave",
        "semitone",
        "semitones",
        "transpose",
        "recent",
        "latest",
        "context",
        "follow",
        "cycle",
        "random",
    }

    if any(
        word in normalized.split()
        for word in semantic_words
    ):
        return False

    midi_channel = extract_channel(
        normalized
    )

    if midi_channel is None:
        return False

    if not 1 <= midi_channel <= 16:
        print(
            "MIDI channel must be "
            "between 1 and 16."
        )
        return True

    source_name = identify_source(
        normalized
    )

    if source_name is not None:
        set_source_channel(
            source_name,
            midi_channel,
        )
        return True

    target_name = extract_route_target(
        normalized
    )

    if target_name is None:
        return False

    with state_lock:
        alias_exists = (
            target_name in state["aliases"]
        )

        group_exists = (
            target_name in state["groups"]
        )

    if alias_exists:
        route_alias(
            target_name,
            midi_channel,
        )
        return True

    if group_exists:
        route_alias_group(
            target_name,
            midi_channel,
        )
        return True

    return False


def parse_source_mode_command(command):
    normalized = normalize_text(command)
    explicit_phrases = {
        "explicit only",
        "explicit-only",
        "assigned only",
        "configured only",
        "suppress unassigned",
        "block unassigned",
        "passthrough off",
        "pass through off",
    }

    if not any(phrase in normalized for phrase in explicit_phrases):
        return False

    source_name = identify_source(normalized)
    if source_name is None:
        return False

    set_source_explicit_only(source_name)
    return True


def parse_passthrough_command(command):
    normalized = normalize_text(
        command
    )

    passthrough_phrases = {
        "passthrough",
        "pass through",
        "original channel",
        "native channel",
        "clear channel",
        "remove channel",
        "stop routing",
    }

    if not any(
        phrase in normalized
        for phrase in passthrough_phrases
    ):
        return False

    source_name = identify_source(
        normalized
    )

    if source_name is None:
        return False

    set_source_passthrough(source_name)
    return True


def execute_llm_action(action):
    action_name = action["action"]

    if action_name == "route_alias":
        route_alias(
            action["alias"],
            action["channel"],
        )

    elif action_name == "route_group":
        route_alias_group(
            action["group"],
            action["channel"],
        )

    elif action_name == "route_source":
        set_source_channel(
            action["source"],
            action["channel"],
        )

    elif action_name == "source_passthrough":
        set_source_passthrough(
            action["source"]
        )

    elif action_name == "clear_alias_route":
        clear_alias_route(
            action["alias"]
        )

    elif action_name == "clear_group_route":
        clear_alias_group_route(
            action["group"]
        )

    elif action_name == "capture_note_trigger":
        arm_trigger_capture(
            trigger_alias=action[
                "trigger_alias"
            ],
            capture_mode="note",
            capture_source=action[
                "capture_source"
            ],
            midi_channel=action[
                "channel"
            ],
        )

    elif action_name == "capture_chord_trigger":
        arm_trigger_capture(
            trigger_alias=action[
                "trigger_alias"
            ],
            capture_mode="chord",
            capture_source=action[
                "capture_source"
            ],
            midi_channel=action[
                "channel"
            ],
        )

    elif action_name == "clear_trigger_assignment":
        clear_trigger_assignment(
            action["trigger_alias"]
        )

    elif (
        action_name
        == "assign_dynamic_chord_note_trigger"
    ):
        assign_dynamic_chord_note_trigger(
            trigger_alias=action[
                "trigger_alias"
            ],
            selector_group=action[
                "selector_group"
            ],
            note_source=action[
                "note_source"
            ],
            transpose=action[
                "transpose"
            ],
            midi_channel=action[
                "channel"
            ],
        )

    elif action_name == "clear_dynamic_note_trigger":
        clear_dynamic_note_trigger(
            action["trigger_alias"]
        )

    elif action_name == "install_rule":
        install_runtime_rule(
            action["rule"]
        )

    elif action_name == "update_rule":
        replace_runtime_rule(
            action["rule_id"],
            action["rule"],
        )

    elif action_name == "delete_rule":
        delete_runtime_rule(
            action["rule_id"]
        )

    elif action_name == "enable_rule":
        set_rule_enabled(
            action["rule_id"],
            True,
        )

    elif action_name == "disable_rule":
        set_rule_enabled(
            action["rule_id"],
            False,
        )

    elif action_name == "mute_alias":
        mute_alias(
            action["alias"]
        )

    elif action_name == "unmute_alias":
        unmute_alias(
            action["alias"]
        )

    else:
        print(
            f'LLM returned unsupported '
            f'action "{action_name}".'
        )


def print_help():
    print(
        """
Core commands:

  call that <name>
  name that <name>
  aliases
  group <name> as <alias>, <alias>
  groups

  send <alias or group> to channel <1-16>
  keyboard channel <1-16>
  drums channel <1-16>
  keyboard passthrough
  drums passthrough
  drums explicit only
  drums passthrough off
  clear route <alias or group>
  routes

Capture:

  Fast local capture (no LLM required):
    Make crash play the next chord I play on
    channel 3.
    Capture chord crash ch3
    Capture note snare ch4

  triggers
  clear static trigger <alias>
  cancel capture

Runtime Level 2 rules:

  rules
  status
  disable rule <id>
  enable rule <id>
  delete rule <id>
  clear dynamic trigger <alias>
  clear trigger <alias>
  undo

Fast shorthand:

  every fourth snare hit, play note 60 on channel 3
  follow kick drum from toms lowest -12 ch16
  kick drum -24
  kick drum ch16
  mute snare
  unmute snare
  I meant channel 16

Novel natural-language examples:

  Every fourth snare hit, play the highest note
  of the current tom chord one octave up on
  channel 4.

  Make the kick cycle through the latest tom
  chord on channel 16.

  Make the ride send CC 1 using its velocity on
  channel 2 without suppressing the ride note.

Presets:

  save preset <name>
  load preset <name>
  presets

Other:

  clear harmonic context
  panic
  last
  delete <alias>
  help
  quit
"""
    )


def build_state_snapshot():
    with state_lock:
        return deepcopy(state)


def handle_command(command):
    command = command.strip()

    if not command:
        return

    normalized = normalize_text(
        command
    )

    if normalized.startswith("call that "):
        create_alias(
            normalized[
                len("call that "):
            ]
        )

    elif normalized.startswith("name that "):
        create_alias(
            normalized[
                len("name that "):
            ]
        )

    elif normalized == "last":
        show_last_event()

    elif normalized in {
        "aliases",
        "show aliases",
    }:
        list_aliases()

    elif (
        normalized.startswith("group ")
        and " as " in normalized
    ):
        definition = normalized[
            len("group "):
        ]

        group_name, members_text = (
            definition.split(
                " as ",
                1,
            )
        )

        create_alias_group(
            group_name,
            members_text,
        )

    elif normalized in {
        "groups",
        "show groups",
        "list groups",
    }:
        show_groups()

    elif normalized.startswith(
        "clear group route "
    ):
        clear_alias_group_route(
            normalized[
                len("clear group route "):
            ]
        )

    elif normalized.startswith(
        "clear route "
    ):
        target_name = normalize_text(
            normalized[
                len("clear route "):
            ]
        )

        with state_lock:
            group_exists = (
                target_name
                in state["groups"]
            )

        if group_exists:
            clear_alias_group_route(
                target_name
            )
        else:
            clear_alias_route(
                target_name
            )

    elif normalized in {
        "routes",
        "show routes",
        "show routing",
    }:
        show_routes()

    elif normalized in {
        "triggers",
        "show triggers",
        "trigger assignments",
        "show trigger assignments",
    }:
        show_trigger_assignments()

    elif normalized in {
        "rules",
        "show rules",
        "list rules",
    }:
        show_runtime_rules()

    elif normalized in {
        "status",
        "show status",
    }:
        show_status()

    elif normalized == "undo":
        undo_last_change()

    elif normalized.startswith(
        "clear static trigger "
    ):
        clear_trigger_assignment(
            normalized[
                len("clear static trigger "):
            ]
        )

    elif normalized.startswith(
        "clear dynamic trigger "
    ):
        clear_dynamic_note_trigger(
            normalized[
                len("clear dynamic trigger "):
            ]
        )

    elif normalized.startswith(
        "clear trigger "
    ):
        clear_any_trigger(
            normalized[
                len("clear trigger "):
            ]
        )

    elif normalized in {
        "clear harmonic context",
        "clear active chord",
        "forget active chord",
    }:
        clear_harmonic_context()

    elif normalized in {
        "cancel capture",
        "cancel note capture",
        "cancel chord capture",
    }:
        cancel_capture()

    elif normalized in {
        "panic",
        "all notes off",
        "stop all notes",
    }:
        send_global_panic()

    elif normalized.startswith(
        "save preset "
    ):
        save_preset(
            normalized[
                len("save preset "):
            ]
        )

    elif normalized.startswith(
        "load preset "
    ):
        load_preset(
            normalized[
                len("load preset "):
            ]
        )

    elif normalized in {
        "presets",
        "show presets",
        "list presets",
    }:
        list_presets()

    elif normalized.startswith(
        "delete alias "
    ):
        delete_alias(
            normalized[
                len("delete alias "):
            ]
        )

    elif (
        normalized.startswith("delete ")
        and not normalized.startswith(
            "delete rule "
        )
    ):
        delete_alias(
            normalized[
                len("delete "):
            ]
        )

    elif normalized == "help":
        print_help()

    elif normalized in {
        "quit",
        "exit",
        "stop midge",
    }:
        stop_event.set()

    elif parse_fast_capture_command(
        command
    ):
        pass

    elif parse_fast_nth_fixed_note_rule(
        command
    ):
        pass

    elif parse_fast_rule_update(
        command
    ):
        pass

    elif parse_source_mode_command(
        normalized
    ):
        pass

    elif parse_passthrough_command(
        normalized
    ):
        pass

    elif parse_natural_channel_command(
        normalized
    ):
        pass

    else:
        print("Interpreting command...")

        try:
            action = parse_command(
                command,
                build_state_snapshot(),
                list(INPUT_PORTS.keys()),
            )

            print(
                f"LLM action: {action}"
            )

            semantic_errors = (
                validate_llm_action_against_command(
                    command,
                    action,
                )
            )
            if semantic_errors:
                print("Rejected LLM action:")
                for error in semantic_errors:
                    print(f"  - {error}")
            else:
                execute_llm_action(action)

        except LLMConnectionError as error:
            print(
                f"Local LLM unavailable: {error}"
            )

            print(
                "Make sure llama-server is "
                "running on port 8080."
            )

        except LLMParserError as error:
            print(
                "Could not interpret command: "
                f"{error}"
            )


def main():
    global current_output_port

    input_ports = []
    output_port = None

    try:
        output_port = mido.open_output(
            OUTPUT_PORT
        )

        current_output_port = output_port

        for source_name, port_name in (
            INPUT_PORTS.items()
        ):
            port = mido.open_input(
                port_name,
                callback=midi_callback(
                    source_name,
                    output_port,
                ),
            )

            input_ports.append(port)

            print(
                f"Opened input: {source_name} "
                f"-> {port_name}"
            )

        print(
            f"Opened output: {OUTPUT_PORT}"
        )
        print("\nMidge Level 2 is running.")

        print(
            'Type "status", "rules", '
            '"help", or "quit".\n'
        )

        while not stop_event.is_set():
            try:
                command = input("midge> ")
                handle_command(command)

            except EOFError:
                stop_event.set()

            except KeyboardInterrupt:
                print(
                    "\nStopping Midge..."
                )
                stop_event.set()

    finally:
        send_global_panic()
        time.sleep(0.05)

        for port in input_ports:
            try:
                port.close()
            except Exception:
                pass

        if output_port is not None:
            try:
                output_port.close()
            except Exception:
                pass

        current_output_port = None
        print("Midge stopped.")


if __name__ == "__main__":
    main()
