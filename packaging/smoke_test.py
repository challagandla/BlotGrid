"""Smoke-test a built BlotGrid bundle without a display.

Starts the frozen executable in server mode, waits for Streamlit's health
endpoint, checks the page and bundled component assets are served, then stops
it. Used by CI on every OS.

    python packaging/smoke_test.py dist/BlotGrid/BlotGrid[.exe]
    python packaging/smoke_test.py dist/BlotGrid.app/Contents/MacOS/BlotGrid
"""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url: str, timeout: float = 3.0) -> tuple[int, bytes]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.status, response.read()


def main(executable: str) -> int:
    check = subprocess.run([executable, "--self-test"], capture_output=True,
                           text=True, timeout=300)
    print(check.stdout.strip()[-4000:])
    if check.returncode != 0:
        print(check.stderr[-4000:])
        print("FAILED: bundle self-test")
        return 1

    port = free_port()
    home = tempfile.mkdtemp(prefix="blotgrid-smoke-")
    env = dict(os.environ, BLOTGRID_HOME=home,
               BLOTGRID_LAUNCH_TOKEN=secrets.token_urlsafe(16))
    log_path = Path(home) / "server.log"
    started = time.monotonic()
    with log_path.open("wb") as log:
        proc = subprocess.Popen([executable, "--serve", "--port", str(port)],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")
            try:
                if get(f"http://127.0.0.1:{port}/_stcore/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not become healthy in 180 s")
        ready = time.monotonic() - started
        status, body = get(f"http://127.0.0.1:{port}/")
        assert status == 200 and b"<html" in body.lower(), "index page missing"
        for sub in ("models", "runs", "data"):
            assert (Path(home) / sub).is_dir(), f"user dir {sub} not created"
        print(f"OK: healthy after {ready:.1f}s on 127.0.0.1:{port}")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}\n--- server log ---")
        print(log_path.read_text(errors="replace")[-6000:])
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
