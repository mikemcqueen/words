#!/usr/bin/env bash
# Appends a new prompt entry to prompts/crossword.json.
# Example: ./append-prompt.sh p123 "Describe the crossword solving strategy."

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <prompt_id> <prompt_text>" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found in PATH" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
CROSSWORD_JSON="$REPO_ROOT/prompts/crossword.json"

PROMPT_ID=$1
shift
PROMPT_TEXT=$*

if [[ ! -f "$CROSSWORD_JSON" ]]; then
  echo "File not found: $CROSSWORD_JSON" >&2
  exit 1
fi

tmp_file=$(mktemp)
trap 'rm -f "$tmp_file"' EXIT

jq --indent 4 \
  --arg id "$PROMPT_ID" \
  --arg text "$PROMPT_TEXT" \
  '. + [{"id": $id, "text": $text}]' \
  "$CROSSWORD_JSON" > "$tmp_file"

mv "$tmp_file" "$CROSSWORD_JSON"
