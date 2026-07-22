#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/home/song/InternNav/data}"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
ARCHIVE="${DATA_ROOT}/downloads/mp3d_pe.tar.gz"
SCENE_ROOT="${DATA_ROOT}/scene_data/mp3d_pe"
URL="${MIRROR}/datasets/InternRobotics/Scene-N1/resolve/main/mp3d_pe.tar.gz"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required because Scene-N1 is gated" >&2
  exit 3
fi

mkdir -p "$(dirname "${ARCHIVE}")" "$(dirname "${SCENE_ROOT}")"
curl --fail --location --continue-at - \
  --config <(printf 'header = "Authorization: Bearer %s"\n' "${HF_TOKEN}") \
  --output "${ARCHIVE}" \
  "${URL}"

sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
mkdir -p "${SCENE_ROOT}"
tar -xzf "${ARCHIVE}" -C "$(dirname "${SCENE_ROOT}")"

# Scene-N1's archive is currently rooted at scans/ rather than mp3d_pe/.
# Normalize it without copying the large asset tree so downstream paths stay stable.
EXTRACTED_SCANS="$(dirname "${SCENE_ROOT}")/scans"
if [[ -d "${EXTRACTED_SCANS}" ]]; then
  if find "${SCENE_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to replace non-empty ${SCENE_ROOT}" >&2
    exit 5
  fi
  rmdir "${SCENE_ROOT}"
  mv "${EXTRACTED_SCANS}" "${SCENE_ROOT}"
fi

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "mp3d_pe extraction did not produce a scene tree" >&2
  exit 4
fi

find "${SCENE_ROOT}" -type f \( -name '*.usd' -o -name '*.usda' -o -name '*.usdc' \) \
  -printf '%P\n' | LC_ALL=C sort > "${SCENE_ROOT}/scene_index.txt"
find "${SCENE_ROOT}" -mindepth 4 -maxdepth 4 -type f \
  -name 'isaacsim_*.usd' ! -name '*_non_metric.usd' \
  -printf '%P\n' | LC_ALL=C sort > "${SCENE_ROOT}/scene_root_candidates.txt"
du -sb "${SCENE_ROOT}" > "${SCENE_ROOT}/directory_size_bytes.txt"
printf 'archive_sha256=%s\nscene_files=%s\nscene_root_candidates=%s\n' \
  "$(cut -d' ' -f1 "${ARCHIVE}.sha256")" \
  "$(wc -l < "${SCENE_ROOT}/scene_index.txt")" \
  "$(wc -l < "${SCENE_ROOT}/scene_root_candidates.txt")" \
  > "${SCENE_ROOT}/installation_manifest.txt"

echo "Installed mp3d_pe at ${SCENE_ROOT}"
