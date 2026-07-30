from typing import Any, Callable


def normalize_group_name(
    group_name: str,
    normalize_text: Callable[[str], str],
) -> str:
    group_name = normalize_text(group_name)

    # Let commands use either "tom" or "toms".
    if group_name == "tom":
        return "toms"

    return group_name


def create_group(
    state: dict[str, Any],
    group_name: str,
    members: list[str],
    normalize_text: Callable[[str], str],
) -> tuple[bool, str]:
    group_name = normalize_group_name(
        group_name,
        normalize_text,
    )

    normalized_members = [
        normalize_text(member)
        for member in members
        if normalize_text(member)
    ]

    if not group_name:
        return False, "Group name cannot be empty."

    if not normalized_members:
        return False, "A group must contain at least one alias."

    unknown_members = [
        member
        for member in normalized_members
        if member not in state["aliases"]
        and member not in state["groups"]
    ]

    if unknown_members:
        names = ", ".join(unknown_members)

        return (
            False,
            f"These aliases or groups do not exist: {names}",
        )

    state["groups"][group_name] = normalized_members

    return (
        True,
        f'Created group "{group_name}" with: '
        + ", ".join(normalized_members),
    )


def resolve_group(
    state: dict[str, Any],
    group_name: str,
    normalize_text: Callable[[str], str],
    visited: set[str] | None = None,
) -> list[str]:
    group_name = normalize_group_name(
        group_name,
        normalize_text,
    )

    if visited is None:
        visited = set()

    if group_name in visited:
        return []

    visited.add(group_name)

    members = state["groups"].get(group_name, [])
    resolved_aliases = []

    for member in members:
        if member in state["aliases"]:
            resolved_aliases.append(member)

        elif member in state["groups"]:
            nested_aliases = resolve_group(
                state,
                member,
                normalize_text,
                visited,
            )

            resolved_aliases.extend(nested_aliases)

    # Remove duplicates while keeping the original order.
    return list(dict.fromkeys(resolved_aliases))


def route_group(
    state: dict[str, Any],
    group_name: str,
    midi_channel: int,
    normalize_text: Callable[[str], str],
) -> tuple[bool, str]:
    group_name = normalize_group_name(
        group_name,
        normalize_text,
    )

    if not 1 <= midi_channel <= 16:
        return False, "MIDI channel must be between 1 and 16."

    if group_name not in state["groups"]:
        return False, f'Group "{group_name}" does not exist.'

    aliases = resolve_group(
        state,
        group_name,
        normalize_text,
    )

    if not aliases:
        return False, f'Group "{group_name}" contains no aliases.'

    for alias_name in aliases:
        state["alias_routes"][alias_name] = midi_channel - 1

    return (
        True,
        f'Routing group "{group_name}" to MIDI channel '
        f"{midi_channel}: {', '.join(aliases)}",
    )


def clear_group_route(
    state: dict[str, Any],
    group_name: str,
    normalize_text: Callable[[str], str],
) -> tuple[bool, str]:
    group_name = normalize_group_name(
        group_name,
        normalize_text,
    )

    if group_name not in state["groups"]:
        return False, f'Group "{group_name}" does not exist.'

    aliases = resolve_group(
        state,
        group_name,
        normalize_text,
    )

    removed = []

    for alias_name in aliases:
        if alias_name in state["alias_routes"]:
            del state["alias_routes"][alias_name]
            removed.append(alias_name)

    if not removed:
        return (
            False,
            f'Group "{group_name}" has no active routes.',
        )

    return (
        True,
        f'Cleared routing for group "{group_name}": '
        + ", ".join(removed),
    )


def list_groups(
    state: dict[str, Any],
) -> list[str]:
    lines = []

    for group_name, members in state["groups"].items():
        lines.append(
            f'  "{group_name}" -> {", ".join(members)}'
        )

    return lines