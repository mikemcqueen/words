---
name: summarize-prompt
description: summarize the results of a prompt
---

Load the prompt file at $ARGUMENTS

Analyze the prompt within this file.  I want you to capture what original idea or ideas were used
to construct this prompt. It could be either an incremental improvement, or a more significant
change from other prompts.

Create a one-sentence summary describing the idea(s) used to construct this prompt.

Calculate the number of False Positives (FP) and False Negatives (FN) from the results by calling
scripts/count_errors.sh with the results filename.

Append a one-line JSONL result to summary.jsonl in the same directory as the supplied prompt file. It
shoud include fields for score, FP, FN, your one-sentence "summary", and the prompt text itself.

