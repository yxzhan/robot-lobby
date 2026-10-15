"""Serves public/ like `python3 -m http.server`, but tells every cache not to
keep anything (Cache-Control: no-store). Without it, a caching proxy in front
of the page (e.g. Cloudflare) keeps serving old .js files after an edit, which
DevTools' "Disable cache" cannot bypass: it only skips the browser's cache.

    python3 scripts/serve.py [port]   # default 4000
"""

import functools
import http.server
import pathlib
import sys


class NoStoreHandler(http.server.SimpleHTTPRequestHandler):
  def end_headers(self):
    self.send_header('Cache-Control', 'no-store')
    super().end_headers()


def main():
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
  public = pathlib.Path(__file__).resolve().parent.parent / 'public'
  handler = functools.partial(NoStoreHandler, directory=str(public))
  with http.server.ThreadingHTTPServer(('', port), handler) as server:
    print(f'serving {public} on http://localhost:{port}/ (no-store)')
    server.serve_forever()


if __name__ == '__main__':
  main()
