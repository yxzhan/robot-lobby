#!/bin/bash
# Gets the model library, mjcf_assets/ next to package.json (its own repository),
# or updates it, then indexes it for the Assets panel (scripts/vendor.mjs).
#
#   scripts/get_assets.sh          # lite: the models the GitHub Pages site carries (~280 MB)
#   scripts/get_assets.sh main     # everything (~4.5 GB, mujoco_menagerie a submodule)
#   scripts/get_assets.sh --full   # with the whole history (default: the latest commit only)
#
# Run again to update, or to switch between lite and main (local changes stay:
# git refuses a switch that would overwrite them).

set -eu
URL="https://gitlab.informatik.uni-bremen.de/yanxiang/mjcf_assets.git"
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
DEST="${ROOT}/mjcf_assets"

BRANCH=lite
DEPTH=(--depth 1)
for arg in "$@"; do
    case "${arg}" in
        lite|main) BRANCH="${arg}" ;;
        --full) DEPTH=() ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: ${arg} (lite, main, --full)"; exit 1 ;;
    esac
done
SUB=()
[ "${BRANCH}" = main ] && SUB=(--recurse-submodules ${DEPTH[@]+--shallow-submodules})

if [ ! -e "${DEST}" ]; then
    echo "cloning mjcf_assets (${BRANCH}) into ${DEST}"
    git clone ${DEPTH[@]+"${DEPTH[@]}"} ${SUB[@]+"${SUB[@]}"} --branch "${BRANCH}" "${URL}" "${DEST}"
elif [ -d "${DEST}/.git" ] || [ -f "${DEST}/.git" ]; then
    echo "updating ${DEST} (${BRANCH})"
    git -C "${DEST}" fetch ${DEPTH[@]+"${DEPTH[@]}"} origin "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
    if git -C "${DEST}" rev-parse --verify -q "refs/heads/${BRANCH}" > /dev/null; then
        git -C "${DEST}" checkout "${BRANCH}"
    else
        git -C "${DEST}" checkout -b "${BRANCH}" --track "origin/${BRANCH}"
    fi
    git -C "${DEST}" merge --ff-only "origin/${BRANCH}"
    if [ "${BRANCH}" = main ]; then
        git -C "${DEST}" submodule update --init ${DEPTH[@]+"${DEPTH[@]}"} --recursive
    fi
else
    echo "${DEST} exists but is no git checkout: move it away first"; exit 1
fi

# Index it (and link public/mjcf_assets) when the lobby's runtimes are there;
# otherwise npm install does both.
cd "${ROOT}"
if command -v node > /dev/null && [ -d node_modules ]; then
    node scripts/vendor.mjs
    echo "thumbnails for new models: python3 scripts/thumbnails.py"
else
    echo "now: npm install (indexes the library)"
fi
