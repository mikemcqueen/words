Keep in mind that you are modifying the *system* prompt, which is meant to address how the model thinks and
generates responses more generally. The system prompt shouldn't contain anything specific to crossword puzzle
clues or word-pairs, or about limiting the response to YES/NO - that is all contained with the regular prompt.

This should be a general solution, widely applicable to many different possible word substitutions, so you
cannot cheat by mentioning specific words or token sequences that you see in the results.

One of the mistakes you make a lot, in trying to refine a prompt, is to identify specific word pairs
that are failing, and then tailor the prompt with too-specific hints related to a specific word pair.

An example of that would be actually using one or both words of a word pair, to tailor the prompt. That
is not general enough.

Another example is when you use too-specific of a category, which really only applies to one (or maybe
two) word pairs, in an attempt to get that "one last failing word pair" to pass. That is not general
enough.

If you determine the prompt is general enough, then this validation passes, otherwise it fails, and
you should provide reasoning for why it failed, so that you can do better in your next prompt generation
attempt.
