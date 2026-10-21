"""Launch a BinderHub session and run bash commands in it without a browser.

launch_binder_and_get_jupyter_url() calls the BinderHub build API and returns the
session URL and token. BinderTerminal then uses the Jupyter Server terminal API,
which is the same channel the JupyterLab "Terminal" tab uses:

    POST   {url}api/terminals                  -> {"name": "1"}
    WS     {url}terminals/websocket/{name}     <- ["stdin", "..."] / -> ["stdout", "..."]
    DELETE {url}api/terminals/{name}

The terminal is a real bash process inside the session container, so shell
state (cwd, exported variables, sourced setup files) persists across `run()`
calls, and the terminal also shows up in JupyterLab's "Running Terminals" list.
"""

import json
import re
import time
import uuid
from urllib.parse import urlparse, urlunparse

import requests
import websocket  # pip install websocket-client

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[=>]")


def launch_binder_and_get_jupyter_url(binder_url, connect_timeout=10, verbose=True):
    """
    Launch a BinderHub build using requests with streaming enabled.

    Returns:
        (True, {"url": ..., "token": ...}) on success
        (False, error_message) on failure

    Notes:
    - This function blocks until BinderHub reports either success or failure.
    - No exceptions are raised; all errors are converted to return values.
    """

    # Step 1: initiate the HTTP request (may block until headers are received)
    try:
        resp = requests.get(
            binder_url,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=(connect_timeout, None),  # timeout only for connection, not for streaming
        )
    except Exception as e:
        return False, f"Failed to connect to BinderHub: {e}"

    # Step 2: validate HTTP response
    if resp.status_code != 200:
        return False, f"BinderHub returned HTTP {resp.status_code}"

    # Step 3: read Server-Sent Events (SSE) line by line
    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            # BinderHub sends events in SSE format: "data: {...}"
            if not raw_line or not raw_line.startswith("data:"):
                continue

            try:
                event = json.loads(raw_line[5:].strip())
            except json.JSONDecodeError:
                continue

            phase = event.get("phase")

            if verbose:
                print("EVENT:", event, flush=True)

            # Build failed
            if phase == "failed":
                return False, event.get("message") or "Binder build failed"

            # Build succeeded and Jupyter server is ready
            if phase == "ready":
                url = event.get("url")
                if not url:
                    return False, "Binder reported ready phase without a URL"
                return True, {"url": url, "token": event.get("token")}

        # Stream ended without receiving a final "ready" or "failed" event
        return False, "Binder event stream ended without a final result"

    except Exception as e:
        return False, f"Error while reading Binder event stream: {e}"


def shutdown_session(url, token, timeout=10):
    """Shut down the session's Jupyter server, the same as File > Shut Down in JupyterLab.

    The pod is then removed by JupyterHub/BinderHub. Returns True if the server accepted the request.
    """
    url = url if url.endswith("/") else url + "/"
    try:
        r = requests.post(url + "api/shutdown", headers={"Authorization": f"token {token}"}, timeout=timeout)
    except requests.RequestException:
        return False
    return r.ok


class BinderTerminal:
    def __init__(self, url, token, name=None, connect_timeout=30):
        """
        url:   session URL from the Binder "ready" event, e.g. https://binder.intel4coro.de/user/xxx/
        token: session token from the same event
        name:  attach to an existing terminal instead of creating a new one
        """
        self.url = url if url.endswith("/") else url + "/"
        self.token = token
        self.headers = {"Authorization": f"token {token}"}

        if name is None:
            r = requests.post(self.url + "api/terminals", headers=self.headers, timeout=connect_timeout)
            r.raise_for_status()
            name = r.json()["name"]
        self.name = name

        p = urlparse(self.url)
        ws_url = urlunparse(p._replace(
            scheme="wss" if p.scheme == "https" else "ws",
            path=p.path + f"terminals/websocket/{name}",
            query=f"token={token}",
        ))
        self.ws = websocket.create_connection(ws_url, header=self.headers, timeout=connect_timeout)

        # Turn off echo and the prompt so the output stream only contains command output.
        self._send("stty -echo; export PS1='' PS2='' PROMPT_COMMAND=''\n")
        self._read_until_marker(self._send_marker(), timeout=connect_timeout)

    def _send(self, text):
        self.ws.send(json.dumps(["stdin", text]))

    def _send_marker(self):
        # The marker is written split in the command ("__E""ND_") so that, even if the
        # input is echoed, only the real `echo` output matches the pattern.
        uid = uuid.uuid4().hex
        self._send(f'__rc=$?; echo "__E""ND_{uid}_${{__rc}}__"\n')
        return re.compile(rf"__END_{uid}_(\d+)__\r?\n?")

    def _read_until_marker(self, pattern, timeout=None, on_output=None):
        buf = ""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"command did not finish within {timeout}s; output so far:\n{buf}")
                self.ws.settimeout(remaining)
            else:
                self.ws.settimeout(None)
            try:
                msg = json.loads(self.ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if msg[0] == "stdout":
                chunk = msg[1]
                buf += chunk
                m = pattern.search(buf)
                if on_output:
                    # Don't stream the marker line itself.
                    on_output(_clean(chunk if not m else chunk[: max(0, len(chunk) - (len(buf) - m.start()))]))
                if m:
                    return _clean(buf[: m.start()]), int(m.group(1))
            elif msg[0] == "disconnect":
                raise ConnectionError(f"terminal {self.name} was closed; output so far:\n{buf}")

    def run(self, cmd, timeout=None, stream=False):
        """Run `cmd` in the terminal's bash and wait for it to finish.

        Returns (output, exit_code). With stream=True the output is also printed live.
        On timeout the command is interrupted with Ctrl-C and TimeoutError is raised.
        Don't run commands that wait for keyboard input (use -y flags etc.).
        """
        self._send(cmd.rstrip("\n") + "\n")
        on_output = (lambda s: print(s, end="", flush=True)) if stream else None
        try:
            out, rc = self._read_until_marker(self._send_marker(), timeout=timeout, on_output=on_output)
        except TimeoutError:
            # Interrupt the command (Ctrl-C) and resync, so the terminal is usable for the next run().
            self._send("\x03")
            self._read_until_marker(self._send_marker(), timeout=10)
            raise
        return out.strip("\n"), rc

    def close(self, kill=True):
        """Close the websocket; with kill=True also terminate the terminal's bash process."""
        try:
            self.ws.close()
        finally:
            if kill:
                requests.delete(self.url + f"api/terminals/{self.name}", headers=self.headers, timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _clean(s):
    return _ANSI_RE.sub("", s).replace("\r\n", "\n").replace("\r", "")
