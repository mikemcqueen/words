#!/usr/bin/env bash
# Outputs the next available prompt ID from the crossword prompt list.
# Example: ./get_next_prompt_id.sh

CROSSWORD_JSON="${1:-$HOME/code/words/prompts/crossword.json}"

max_id=$(jq -r '.[].id' "$CROSSWORD_JSON" \
    | grep -oP '\d+' \
    | sort -n \
    | tail -1)

echo "p$((max_id + 1))"
