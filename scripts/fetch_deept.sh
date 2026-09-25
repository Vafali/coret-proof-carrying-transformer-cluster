#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO/research_hab/public_benchmarks/DeepT"
REV=16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf
if [[ ! -d "$TARGET/.git" ]]; then
  mkdir -p "$(dirname "$TARGET")"
  git clone --filter=blob:none --no-checkout https://github.com/eth-sri/DeepT.git "$TARGET"
fi
git -C "$TARGET" fetch --filter=blob:none origin "$REV"
test "$(git -C "$TARGET" rev-parse "$REV")" = "$REV"
git -C "$TARGET" show "$REV:Robustness-Verification-for-Transformers/LICENSE.md" >/dev/null
echo "DEEPT_PIN_PASS $REV"
