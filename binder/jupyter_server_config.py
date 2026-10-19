# The shared-world server behind the session's jupyter-server-proxy, at
# <session URL>/mujoco/ (the path the page's Server panel connects to:
# "path" in deploy.json). It starts on the first request there, and again
# after it has died. postBuild copies this file to ~/.jupyter/.
import os
import sys

c = get_config()  # noqa: F821
repo = os.environ.get('REPO_DIR', os.path.expanduser('~'))
c.ServerProxy.servers = {
  'mujoco': {
    'command': [sys.executable, os.path.join(repo, 'server', 'server.py'), '{port}'],
    'timeout': 120,
    'launcher_entry': {'title': 'MuJoCo world'},
  },
}
