#!/bin/bash
# Download an English Wikipedia pages-articles dump (all split parts) and verify md5s.
# Usage: bash sources/get_wikipedia.sh [DUMP_DATE]   (default 20260901)
# Safe to re-run: wget -c resumes partial files and skips finished ones.
set -euo pipefail

D=${1:-20260901}
BASE=https://dumps.wikimedia.org/enwiki/$D
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/raw/enwiki-$D"
mkdir -p "$OUT"

# File list + md5s from the dump's own status file (a dated dir, not latest/, so parts never mix dumps)
curl -sf "$BASE/dumpstatus.json" \
  | python3 -c "import json,sys; [print(v['md5']+'  '+k) for k,v in json.load(sys.stdin)['jobs']['articlesdump']['files'].items()]" \
  > "$OUT/md5sums.txt"
echo "Downloading $(wc -l < "$OUT/md5sums.txt") files to $OUT"

# Wikimedia asks for at most 2 concurrent connections
cut -d' ' -f3 "$OUT/md5sums.txt" | xargs -P2 -I{} wget -q -c -P "$OUT" "$BASE/{}"

(cd "$OUT" && md5sum -c md5sums.txt)
