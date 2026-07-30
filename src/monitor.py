import mido

INPUT_NAME = "Simmons SD200"
OUTPUT_NAME = "ESI MIDIMATE eX Port 1"

with mido.open_input(INPUT_NAME) as inp, mido.open_output(OUTPUT_NAME) as out:
    print("Simmons MIDI thru running...")
    print("Press Ctrl+C to stop.\n")

    try:
        for msg in inp:
            print(msg)
            out.send(msg)
    except KeyboardInterrupt:
        print("\nStopped.")