"""Bridge a Muse 2 headset onto the lab streaming layer (LSL).

Finds the headset over Bluetooth and publishes its EEG as an LSL stream
named "Muse". Leave this running, then read the data in another terminal
with src/print_eeg.py or src/plot_live.py.

Usage:
    python src/stream_eeg.py                 # connect to first Muse found
    python src/stream_eeg.py --address <MAC or UUID>
"""

import argparse

from muselsl import list_muses, stream


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Muse 2 EEG as an LSL stream")
    parser.add_argument("--address", default=None, help="Device address (default: first Muse found)")
    args = parser.parse_args()

    address = args.address
    if address is None:
        print("Searching for Muse devices (make sure it is on and not paired to another app)...")
        muses = list_muses()
        if not muses:
            raise SystemExit("No Muse found. Is the headset on and in range?")
        address = muses[0]["address"]
        print(f"Found {muses[0]['name']} at {address}")

    # Blocks until the headset disconnects or the process is killed.
    stream(address)


if __name__ == "__main__":
    main()
