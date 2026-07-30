import mido

from rule_engine import RuleEngine, validate_rule


def build_state():
    return {
        "aliases": {
            "kick drum": {
                "source": "simmons",
                "type": "note",
                "channel": 9,
                "note": 36,
            },
            "snare": {
                "source": "simmons",
                "type": "note",
                "channel": 9,
                "note": 38,
            },
        },
        "groups": {
            "toms": [
                "tom 1",
                "tom 2",
                "tom 3",
            ]
        },
        "rules": {},
        "rule_order": [],
        "active_harmonic_context": {
            "selector_alias": "tom 1",
            "selector_groups": ["toms"],
            "notes": [48, 52, 55, 59],
            "captured_mode": "chord",
        },
    }


def main():
    state = build_state()

    kick_rule = validate_rule(
        {
            "id": "kick-follows-toms",
            "name": "Kick follows toms",
            "when": {
                "event": "note_on",
                "alias": "kick drum",
            },
            "derive": {
                "source": "active_chord",
                "selector_group": "toms",
                "select": "lowest",
                "transpose": -12,
            },
            "output": {
                "type": "note",
                "channel": 16,
                "velocity": "input",
                "suppress_original": True,
            },
        },
        state,
        ["kboard", "simmons"],
    )

    state["rules"][kick_rule["id"]] = kick_rule
    state["rule_order"].append(kick_rule["id"])

    engine = RuleEngine()

    message = mido.Message(
        "note_on",
        channel=9,
        note=36,
        velocity=100,
    )

    result = engine.execute(
        source_name="simmons",
        message=message,
        matching_aliases=["kick drum"],
        state=state,
    )

    assert result.matched_rule_ids == [
        "kick-follows-toms"
    ]

    assert len(result.messages) == 1
    assert result.messages[0].note == 36
    assert result.messages[0].channel == 15
    assert result.messages[0].velocity == 100

    print("Midge Level 2 smoke test passed.")


if __name__ == "__main__":
    main()
