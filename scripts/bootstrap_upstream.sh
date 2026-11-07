#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ ! -e upstream && ! -e openrlhf ]] || { echo 'Use a clean checkout without upstream/openrlhf'; exit 1; }
git clone --filter=blob:none https://github.com/Qwen-Applications/STAR.git upstream
git -C upstream checkout ee3202122bb4af1829709a1a0694c2a8f02310a5
git clone --filter=blob:none https://github.com/OpenRLHF/OpenRLHF.git openrlhf
git -C openrlhf checkout c1fc63a9f7e1837577a76b0c688809b3c0bdc644
git -C openrlhf apply "$ROOT/upstream/0001-add-ckd.patch"
git -C openrlhf apply "$ROOT/notes/local-path-compatibility.patch"
echo 'Pinned STAR / OpenRLHF ready. Install the upstream GPU requirements separately.'
