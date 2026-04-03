#!/usr/bin/env bash
# Counts false negatives and false positives in a results JSON file.
# Example: ./count_yesno_errors.sh path/to/results.json

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 path/to/results.json" >&2
  exit 1
fi

results_file=$1

if [[ ! -f "$results_file" ]]; then
  echo "file not found: $results_file" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found in PATH" >&2
  exit 1
fi

counts=$(
  jq -r '
    .results[]
    | select(test("✗"))
    | if test(": YES\\b") then "FP"
      elif test(": NO\\b") then "FN"
      else empty
      end
  ' "$results_file" | awk '
    BEGIN { fp = 0; fn = 0 }
    $1 == "FP" { fp++ }
    $1 == "FN" { fn++ }
    END { printf "FN=%d,FP=%d\n", fn, fp }
  '
)

echo "$counts"
