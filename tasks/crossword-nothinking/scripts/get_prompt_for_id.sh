#!/usr/bin/env bash
# Prints the full JSONL record for a given prompt ID from a summary.jsonl file.
# Example: ./get_prompt_for_id.sh p123 results/yesno_nothink/summary.jsonl

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <prompt_id> <summary.jsonl>" >&2
  exit 1
fi

PROMPT_ID="$1"
SUMMARY_JSONL="$2"

if [[ ! -f "$SUMMARY_JSONL" ]]; then
  echo "file not found: $SUMMARY_JSONL" >&2
  exit 1
fi

jq -c --arg id "$PROMPT_ID" 'select(.id == $id)' "$SUMMARY_JSONL"
