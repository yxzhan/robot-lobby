# GitHub Pages and Binder

The page and the server can also live apart: the page as a static site on
GitHub Pages with a part of the library, running locally, and the server on
Binder with the whole library, joined from the page's **Server** panel.
`public/js/endpoint.js` decides where the server is: `?server=<url>`, else
the last one joined while it still answers, else the page's own origin
(server.py serving the page, see [the shared world](server.md)), else none. With a server, every
data file (`json/assets.json`, thumbnails, `mjcf_assets/`, `pretrained_policies/`)
and `/render` come from it. The server answers any origin (CORS); on Binder
the session's token, which the page puts on every request, WebSockets
included, is what guards it.

**The static site.** Its models are an archive of part of the library,
a GitHub release asset that `pages.json`'s `assets` names
(`mjcf_assets/<category>/<source>/<model>/` at its top, e.g.
`tar czf mjcf_assets.tar.gz ./mjcf_assets` of a trimmed copy; screenshots
included make the thumbnails). `scripts/fetch_assets.mjs` downloads it and
unpacks it as `public/mjcf_assets`, which `scripts/vendor.mjs` indexes like
the library (menagerie directories without their README or scenes are
indexed by their model files) and `scripts/check_assets.py` compiles model
by model. `pages.json` also lists the policy directories the site carries.
The site serves the models at `mjcf_assets/`, so asset paths are the same
as with the library. `build_pages.mjs` publishes whatever
`public/mjcf_assets` is: locally that is usually the link to the whole
library, so build from a checkout without it, as CI does.

```sh
node scripts/fetch_assets.mjs                # public/mjcf_assets, unless there
node scripts/vendor.mjs                      # runtimes and indexes
python3 scripts/thumbnails.py
python3 scripts/check_assets.py              # every model compiles
node scripts/build_pages.mjs                 # into dist/
node scripts/build_pages.mjs --push origin   # or force-push it as branch gh-pages
```

`.github/workflows/pages.yml` does this on every push to `main` (and by
hand, Actions → Pages → Run workflow): it fetches and checks the models, builds `dist/`
and deploys it. Set the repository's Settings → Pages → Source to "GitHub
Actions" for it, or to the `gh-pages` branch for `--push` (one commit each
time, so model files never pile up in its history). The build fails on a
file over 100 MB and warns over 1 GB, GitHub's limits.
`json/deploy.json` on the site names the Binder repository the panel starts:
`pages.json`'s `binder`, by default this checkout's GitHub remote and branch
on mybinder.org (`--binder gh/<owner>/<repo>/<ref>`, `--hub <url>` to
change it).

**The Binder repository** is this one (`binder/`), with the policies of
`pretrained_policies/` in git. The model library is not in the image: it is
a volume mounted into the user pod, `/mnt/dev-tools/mjcf_assets`
(`MJCF_ASSETS` at build time to change it). `environment.yml` has Python,
MuJoCo, onnxruntime and Node; `postBuild` vendors the page's runtimes
(`npm ci`), links the volume as `public/mjcf_assets` and installs
`jupyter_server_config.py`, which puts server.py behind the session's
jupyter-server-proxy at `<session>/robot-lobby/`, started on the first request,
and makes it the page a session opens on (`default_url`; JupyterLab stays
at `<session>/lab`). A launch link can also say so itself, whatever the
hub's own default: `https://<binderhub>/v2/gh/<owner>/robot-lobby/<branch>?urlpath=robot-lobby/`.
`start` runs as the pod starts, when the volume is there: it indexes the
library and the policies (`scripts/vendor.mjs`, which indexes through
`public/mjcf_assets` when there is no `mjcf_assets/` next to
`package.json`), and makes the thumbnails (`mjcf_assets/thumbs`, kept on
the volume) only when there are none yet; the log is `/tmp/index.log`.
`binder_build.ipynb` launches a session from a script, copies the local
library into it (`sync_assets.py`: only what is new or changed, through the
Jupyter Contents API and the session's terminal; the thumbnails go with
it), then indexes it again there.

**Joining.** The Server panel is hidden for now (its button in
`index.html` has `hidden`). *Start a server on Binder* there calls
BinderHub's build API, shows the build and launch, and reloads the page on
the new session's `<url>robot-lobby/?token=…`. *Join a server* takes such a URL
directly (or any server.py's). *Copy link* gives a link to the page on that
server, token included, so others join the same world; *Go local* leaves
it. A Binder session ends after a while without anyone connected (on
mybinder.org about 10 minutes); the page then starts locally again.

**Share panel.** Gets the page onto another device (a phone, a headset)
without typing its URL (`public/js/share.js`, after bambot's `shareLink.ts`):
the pairing service at join.aicor.dev (`?pair=<url>` for another) keeps a
short code for the link for 30 minutes; the other device opens the service
and types the code, or scans the QR code of `join.aicor.dev/<code>`. The
link is this page with the server it is on, token included (the token the
page was opened with is remembered for the tab, as Jupyter's proxy may drop
it from the address bar), so the other device joins the same world; in
local mode it gets a world of its own. A code is asked for when the panel
is shown; *New code* asks again. It warns when the page is on localhost or
was opened behind the proxy without a token.
