---
name: validate-prompt
description: validate that a generated prompt is adequately generalized
---

Consider the prompt: $ARGUMENTS

Review the word pairs that you loaded from the supplied pairs file. The prompt needs to be general,
and not specific to any of those word pairs, or even to single words from those word pairs.

One of the mistakes you make a lot, in trying to refine a prompt, is to identify specific word pairs
that are failing, and then tailor the prompt with too-specific hints related to a specific word pair.

An example of that would be actually using one or both words of a word pair, to tailor the prompt. That
is not general enough.

Another example is when you use too-specific of a category, which really only applies to one (or maybe
two) word pairs, in an attempt to get that "one last failing word pair" to pass. That is not general
enough.

If you determine the prompt is general enough, then this skill check passes, otherwise it fails, and
you should provide reasoning for why it failed, so that you can do better in your next prompt generation
attempt.
