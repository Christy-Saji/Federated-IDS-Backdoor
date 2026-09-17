"""python -m flids.dashboard [--port 8765] [--no-browser]"""

from __future__ import annotations

import argparse

from .engine import PROCESSED
from .server import serve


def main():
    p = argparse.ArgumentParser(description="Local dashboard for the federated "
                                            "IDS backdoor project.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--processed", default=PROCESSED,
                   help="cached real arrays (default: data/processed). Without "
                        "them the demo falls back to the synthetic generator "
                        "and says so on the page.")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    serve(args.host, args.port, args.processed, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
