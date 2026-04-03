#!/usr/bin/env bash
# Prints the top N prompt IDs from a summary.jsonl file, sorted by score descending.
# Example: ./get_top_scoring_prompt_ids.sh results/yesno_nothink/summary.jsonl 10

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <summary.jsonl> [count]" >&2
  exit 1
fi

summary_file=$1
count=${2:-10}

if [[ ! -f "$summary_file" ]]; then
  echo "file not found: $summary_file" >&2
  exit 1
fi

if ! [[ "$count" =~ ^[0-9]+$ ]] || [[ "$count" -lt 1 ]]; then
  echo "count must be a positive integer: $count" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found in PATH" >&2
  exit 1
fi

jq -r '[.id, .score] | @tsv' "$summary_file" \
  | sort -k2,2nr \
  | head -n "$count" \
  | cut -f1
