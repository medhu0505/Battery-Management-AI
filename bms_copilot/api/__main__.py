import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the BMS Copilot dashboard and API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("bms_copilot.api.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
