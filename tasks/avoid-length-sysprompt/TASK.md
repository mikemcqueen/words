Your task is to modify the system prompt when running eval_prompt.py to eliminate the generation
of responses with a finish_reason of "length", while not impacting the YES/NO success rate. There
are, broadly, three reasons why a response may result in this finish reason:

1. Degenerative reasoning as evidenced by the repetition of the same word or word/token sequence.
2. Degenerative reasoning as evidenced by the enumeration of many possibilities.
3. Valid reasoning that exceeds the context window size (about 1280 tokens).

You should have been supplied a HOST argument, substitute it for {HOST} below when reviewing results
files and evaluating your system prompt.

Review the following file contents:

* prompt p22 in prompts/crossword.json
* words and their expected answer in pairs.json
* system prompts in prompts/sysN where N is a number
* results in results/crossword_p22_juniper_sys26_mc8.pairs2.ITER.json where ITER is a number

These represent the prompt we're testing, the words that were substituted within those prompts,
the system prompts we've already tried, and the results of testing the prompts with different
system prompts and the substituted words.

Each result file has a score, which represents the percentage of correct answers for all substituted words.

prompts/sys26 was a very successful system prompt and I suggest making incremental modifications to
that prompt in order to achieve your goal.

## What To Do

Analyze the reasoning of results where the "finish_reason" field is "length", which indicates that the
reasoning went on for too long.

Based on your analysis, design a new system prompt that you think will solve the reasoning issue, while
not impacting the score.

This should be a general solution, widely applicable to many different possible word substitutions, so you
cannot cheat by mentioning specific words or token sequences that you see in the results.

You allowed ONLY to modify the system prompt. You are NOT allowed to modify other parameters such as
--repeat-penalty, --repeat-last-n, or any other inference parameters.

I want you to carefully consider how these models work - once a degenerate reasonsing loop is started, it is
likely that it is purely probabilistic that the loop continues, and that the system prompt has little effect
on whether it continues. Therefore, when addressing degenerative reasoning, the goal should be preventing it
from entering a loop - not how to exit a loop once it has been entered.

Keep in mind that you are modifying the *system* prompt, which is meant to address how the model thinks and
generates responses more generally. The system prompt shouldn't contain anything specific to crossword puzzle
clues or word-pairs, or about limiting the response to YES/NO - that is all contained with the regular prompt.

Once you have a new system prompt, validate it via tasks/validate-sysprompt/TASK.md.

If the validation suceeds, save it as NEW file named prompts/sysN where N is the next unused number.
Do not edit previously created system prompt files. Substitute that path in the evaluation step below.

If validation fails, continue from **What To Do** above.

## How To Evaluate Your System Prompt

Determine what the next iteration number ITER is for the specified host and system prompt. Do this by reviewing
all result filenames for the supplied HOST and system prompt; the already-run iteration numbers are identified
by the last number before the .json extension. Use this iteration number in construction of the --tag argument
below.

Run the following command without asking for permission:

source ../.torch/bin/activate && python eval_prompt.py -f prompts/crossword.json --pid p22 --temp 0.3 --top_p 0.95 --min_p 0.01  --timeout 300 --key `cat api_key` -s {SYSPROMPT_PATH} --port 8000 --host {HOST} --mc 6 --pairs pairs2.json --tag pairs2.{ITER}

This will cause a new file to be created in the results/ directory, named using a combination of the prompt
filename, prompt id, system prompt filename, and iteration number. Review the file contents and determine
whether your system prompt achieved the result you were tasked with.

## Completion Condition

If the score is 100%, run tasks/verify_complete.

Otherwise, analyze the results to understand why there are still results with a finish_reason of "length",
or why specific word substitutions failed, and continue from **What To Do** above, using your analysis to
design a better system prompt.
