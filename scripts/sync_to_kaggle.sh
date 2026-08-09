#!/usr/bin/env bash
# Push src/rsnaknee to Kaggle as a private dataset so notebooks can import it.
#
# The inference and cache-building notebooks import `rsnaknee` from this dataset rather than
# carrying pasted copies of the code. One sync per change, in exchange for the Kaggle side and
# the repo never silently diverging — a notebook running last week's preprocessing against this
# week's weights loads cleanly, runs, and returns predictions computed from the wrong pixels.
#
#   scripts/sync_to_kaggle.sh [ "version notes" ]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${REPO_ROOT}/.kaggle-staging"
NOTES="${1:-sync from $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo working-tree)}"

# Kaggle needs the dataset slug to match the account that owns it.
USERNAME="$(python3 -c 'import json,pathlib,sys
for p in (pathlib.Path.home()/".kaggle/credentials.json", pathlib.Path.home()/".kaggle/kaggle.json"):
    if p.exists():
        d=json.loads(p.read_text())
        print(d.get("username") or d.get("UserName") or ""); sys.exit()
print("")')"

if [ -z "$USERNAME" ]; then
  echo "Could not read your Kaggle username from ~/.kaggle/. Run: kaggle auth login" >&2
  exit 1
fi

rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -R "${REPO_ROOT}/src/rsnaknee" "$STAGING/"
cp "${REPO_ROOT}/pyproject.toml" "$STAGING/"
find "$STAGING" -name '__pycache__' -type d -prune -exec rm -rf {} +

cat > "${STAGING}/dataset-metadata.json" <<EOF
{
  "title": "rsnaknee-src",
  "id": "${USERNAME}/rsnaknee-src",
  "licenses": [{"name": "Apache 2.0"}]
}
EOF

if kaggle datasets status "${USERNAME}/rsnaknee-src" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGING" -m "$NOTES" --dir-mode zip
else
  echo "Dataset does not exist yet — creating it."
  kaggle datasets create -p "$STAGING" --dir-mode zip
fi

rm -rf "$STAGING"
echo "Synced to https://www.kaggle.com/datasets/${USERNAME}/rsnaknee-src"
