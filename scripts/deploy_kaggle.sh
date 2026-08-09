#!/usr/bin/env bash
# Sync the package, wait for Kaggle to finish processing it, then push the notebook.
#
#   scripts/deploy_kaggle.sh notebooks/01_build_cache.ipynb
#
# These three steps must happen in this order and they have burned us twice. Pushing a notebook
# without syncing the package runs the previous version of the code against the new notebook,
# which fails as an ImportError if you are lucky and as wrong numbers if you are not. Pushing
# before the dataset finishes processing runs against a mount that does not exist yet.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK="${1:-notebooks/01_build_cache.ipynb}"
KERNEL_ID="${KERNEL_ID:-rsna-knee-build-pixel-cache}"
KERNEL_TITLE="${KERNEL_TITLE:-RSNA Knee - build pixel cache}"

USERNAME="$(python3 -c 'import json,pathlib,sys
for p in (pathlib.Path.home()/".kaggle/credentials.json", pathlib.Path.home()/".kaggle/kaggle.json"):
    if p.exists():
        d=json.loads(p.read_text()); print(d.get("username") or d.get("UserName") or ""); sys.exit()
print("")')"
[ -n "$USERNAME" ] || { echo "No Kaggle username found. Run: kaggle auth login" >&2; exit 1; }

echo "==> 1/3 syncing package"
"${REPO_ROOT}/scripts/sync_to_kaggle.sh" "deploy $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "==> 2/3 waiting for dataset to finish processing"
for _ in $(seq 1 60); do
  STATUS="$(kaggle datasets status "${USERNAME}/rsnaknee-src" 2>&1 || true)"
  case "$STATUS" in
    *ready*) echo "    dataset ready"; break ;;
    *error*) echo "    dataset failed to process: $STATUS" >&2; exit 1 ;;
    *) printf '.'; sleep 10 ;;
  esac
done

echo "==> 3/3 pushing notebook"
STAGING="$(mktemp -d)"
cp "${REPO_ROOT}/${NOTEBOOK}" "${STAGING}/"
cat > "${STAGING}/kernel-metadata.json" <<EOF
{
  "id": "${USERNAME}/${KERNEL_ID}",
  "title": "${KERNEL_TITLE}",
  "code_file": "$(basename "$NOTEBOOK")",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["${USERNAME}/rsnaknee-src"],
  "competition_sources": ["rsna-knee-abnormality-detection"],
  "kernel_sources": []
}
EOF
kaggle kernels push -p "$STAGING"
rm -rf "$STAGING"

echo
echo "watch it:  scripts/watch_kernel.sh ${USERNAME}/${KERNEL_ID}"
