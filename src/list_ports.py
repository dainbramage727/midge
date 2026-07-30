# list_ports.py

import mido

print("INPUTS")
for p in mido.get_input_names():
    print(" ", p)

print()

print("OUTPUTS")
for p in mido.get_output_names():
    print(" ", p)