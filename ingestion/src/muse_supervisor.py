"""Keep the Muse-to-LSL bridge alive for a whole collection session.

Scans for the headset and connects in a loop: whenever the headset sleeps,
drops, or changes heads, this reconnects automatically as soon as it is
advertising again. Leave it running for the entire session so participants
never wait on a manual reconnect.

Usage:
    python ingestion/src/muse_supervisor.py                    # first Muse found
    python ingestion/src/muse_supervisor.py --address <UUID>   # specific headset
"""

import argparse
import time

from muselsl import list_muses, stream


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-reconnecting Muse-to-LSL bridge")
    parser.add_argument("--address", default=None, help="Device address (default: first Muse found)")
    args = parser.parse_args()

    address = args.address
    while True:
        try:
            if address is None:
                print("[supervisor] scanning for Muse...", flush=True)
                muses = list_muses()
                if not muses:
                    print("[supervisor] none found (asleep? connected elsewhere?), retrying in 5 s", flush=True)
                    time.sleep(5)
                    continue
                address = muses[0]["address"]
                print(f"[supervisor] found {muses[0]['name']} at {address}", flush=True)

            print(f"[supervisor] connecting to {address}", flush=True)
            stream(address)  # blocks until the headset disconnects
            print("[supervisor] headset disconnected, reconnecting...", flush=True)
        except KeyboardInterrupt:
            print("[supervisor] stopped", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - keep the session alive no matter what
            print(f"[supervisor] error: {exc}; retrying in 5 s", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
