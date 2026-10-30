# The shared-world server behind the session's jupyter-server-proxy, at
# <session URL>/robot-lobby/ (the path the Share panel's Join a server connects to:
# "path" in deploy.json). It starts on the first request there, and again
# after it has died. It is also where the session opens (default_url)
# instead of JupyterLab, which stays at <session URL>/lab. postBuild copies
# this file to ~/.jupyter/.
import os
import sys

c = get_config()  # noqa: F821
repo = os.environ.get('REPO_DIR', os.path.expanduser('~'))
c.ServerProxy.servers = {
  'robot-lobby': {
    'command': [sys.executable, os.path.join(repo, 'server', 'server.py'), '{port}'],
    'timeout': 120,
    'launcher_entry': {'title': 'Robot Lobby'},
  },
}
c.ServerApp.default_url = '/robot-lobby/'
