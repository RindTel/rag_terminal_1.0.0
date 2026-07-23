"""
desktop.py
──────────
Native desktop launcher for the packaged QwenRAG app — the entry point that
PyInstaller freezes.

It runs the existing Streamlit app headless on a private localhost port, then
opens a native OS window (pywebview) pointing at it. No terminal, no browser tab,
no separate server to start — double-click and a window appears.

Why a self-reexec instead of a thread: Streamlit registers OS signal handlers,
which only work on a process's main thread. Running its bootstrap in a worker
THREAD raises "signal only works in main thread" under a frozen build. So the
launcher runs the server in a child PROCESS (this same executable, re-entered
with QWENRAG_STREAMLIT_WORKER=1) and keeps the GUI on the parent's main thread.

NOTE (packaging risk — see the roadmap's timebox): Streamlit under PyInstaller is
the fiddly part (static assets, metadata, entry point). Validate this on a real
frozen build in CI; if it resists within a few iterations, stop and report.
"""

import os
import sys
import socket
import subprocess
import time


APP_SCRIPT = "app.py"
WINDOW_TITLE = "QwenRAG"


def _base_dir() -> str:
    """Dir containing app.py — the PyInstaller bundle root when frozen."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    """Ask the OS for an unused localhost port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(port: int, timeout: float = 90.0) -> bool:
    """Block until the server accepts connections (models can take a while to load)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _run_streamlit_worker(port: int) -> None:
    """This process IS the Streamlit server. Blocks running the app."""
    base = _base_dir()
    # chdir into the bundle so Streamlit finds the bundled .streamlit/config.toml
    # (theme). config.py resolves data/model paths absolutely, so this is safe.
    os.chdir(base)

    from streamlit.web import bootstrap

    # flag_options are the CLI-equivalent, highest-precedence config channel.
    # load_config_options() is what actually APPLIES them (run() alone only watches
    # for changes) — it reparses config.toml and overlays these, resolving the
    # developmentMode/server.port conflict on the final merged state. Skipping it
    # was why the server ignored our port and bound the default 8501.
    flag_options = {
        "global.developmentMode": False,
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "browser.gatherUsageStats": False,
    }
    bootstrap.load_config_options(flag_options)
    # Streamlit 1.32+ signature: run(main_script_path, is_hello, args, flag_options)
    bootstrap.run(os.path.join(base, APP_SCRIPT), False, [], flag_options)


def _spawn_worker(port: int) -> subprocess.Popen:
    """Re-launch this executable as the Streamlit server child process."""
    env = dict(os.environ, QWENRAG_STREAMLIT_WORKER="1", QWENRAG_PORT=str(port))
    # Frozen: sys.executable is the app exe. Dev: python + this script.
    argv = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__)]
    return subprocess.Popen(argv, env=env)


def main() -> None:
    # Child role: just be the server.
    if os.environ.get("QWENRAG_STREAMLIT_WORKER") == "1":
        _run_streamlit_worker(int(os.environ["QWENRAG_PORT"]))
        return

    # Parent role: start the server child, then own the GUI window.
    port = _free_port()
    worker = _spawn_worker(port)
    try:
        if not _wait_until_up(port):
            sys.stderr.write("QwenRAG: Streamlit server failed to start.\n")
            worker.terminate()
            sys.exit(1)

        import webview
        webview.create_window(
            WINDOW_TITLE, f"http://127.0.0.1:{port}", width=1200, height=820
        )
        webview.start()   # blocks until the window is closed
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except Exception:
            worker.kill()


if __name__ == "__main__":
    main()
