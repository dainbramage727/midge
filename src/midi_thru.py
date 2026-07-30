import mido

INPUT_NAME = "K-Board"
OUTPUT_NAME = "ESI MIDIMATE eX Port 1"
TARGET_CHANNEL = 15  # MIDI channel 3

with mido.open_input(INPUT_NAME) as inp, mido.open_output(OUTPUT_NAME) as out:
    print("K-Board → Streichfett on MIDI channel 3")
    print("Press Ctrl+C to stop.\n")

    for msg in inp:
        print("IN: ", msg)

        if hasattr(msg, "channel"):
            msg = msg.copy(channel=TARGET_CHANNEL)

        print("OUT:", msg)
        out.send(msg)