---
name: validate-prompt
description: validate that a generated prompt meets all requirements
---

Consider the prompt: $ARGUMENTS

Review the word pairs that you loaded from the supplied pairs file.

For this skill check to pass, it must meet all of the following requirements:

* The prompt needs to be general, and not specific to any of those word pairs, or even to single words
  from those word pairs.

* The prompt must not contain hints specifically related to any ot those words or word pairs.

* The prompt must not contain too-specific of a category, which really only applies to one (or maybe
  two) word pairs, in an attempt to get that "one last failing word pair" to pass. That is not general
  enough.

* The prompt should not specify anything about word order, such as "in any order". The prompt evaluator
  will test the prompt with the word pair in both orders, and will succeed if either word order succeeds,
  and only fail if both word orders fail.

* The {PAIR} substitution should be at the end of the prompt, preceded by some text that indicates it
  represents the words the prompt is referring to, such as "Clue: ".

If the prompt does not meet any one of these requirements, then this skill check fails.

Otherwise, this skill check passes.

