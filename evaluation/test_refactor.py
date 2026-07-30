from copy import deepcopy
import mido

import midge
from state_manager import DEFAULT_STATE, merge_defaults


def synthetic_state():
    state = merge_defaults(deepcopy(DEFAULT_STATE))
    state["aliases"] = {
        "snare": {
            "source": "simmons",
            "type": "note",
            "channel": 9,
            "note": 38,
        },
        "hi hat": {
            "source": "simmons",
            "type": "note",
            "channel": 9,
            "note": 46,
        },
    }
    return state


midge.state = synthetic_state()
midge.rule_engine.reset_runtime()
midge.active_generated.clear()

# Unassigned Simmons note and pedal CC must be suppressed.
hi_hat = mido.Message("note_on", channel=9, note=46, velocity=70)
assert midge.transform_message("simmons", hi_hat) == []
pedal = mido.Message("control_change", channel=9, control=4, value=20)
assert midge.transform_message("simmons", pedal) == []

# Keyboard stays passthrough by default.
key = mido.Message("note_on", channel=0, note=60, velocity=80)
assert midge.transform_message("kboard", key) == [key]

# Parse the formerly failing nth-hit command locally.
command = "every fourth snare hit, play note 60 on channel 3"
assert midge.parse_fast_nth_fixed_note_rule(command)
rule = next(iter(midge.state["rules"].values()))
assert rule["condition"]["every_n"] == 4
assert rule["derive"]["source"] == "fixed"
assert rule["derive"]["notes"] == [60]
assert rule["output"]["channel"] == 3

# First three hits silent; fourth produces note 60 on zero-based channel 2.
for _ in range(3):
    assert midge.transform_message(
        "simmons",
        mido.Message("note_on", channel=9, note=38, velocity=90),
    ) == []

out = midge.transform_message(
    "simmons",
    mido.Message("note_on", channel=9, note=38, velocity=90),
)
assert len(out) == 1
assert out[0].type == "note_on"
assert out[0].note == 60
assert out[0].channel == 2

# Semantic validator rejects the exact bad shape returned earlier.
bad_action = {
    "action": "install_rule",
    "rule": {
        "condition": {},
        "derive": {
            "source": "active_chord",
            "select": "highest",
            "transpose": 12,
        },
        "output": {"channel": 3},
    },
}
errors = midge.validate_llm_action_against_command(command, bad_action)
assert len(errors) >= 2

print("Midge parser/routing refactor tests passed.")
