Your task is to generate a prompt which results in a high score.

Review the following file contents:

* prompts in the prompts/crossword.json
* word pairs and their expected answer in pairs.json
* results in the results/crossword*.json

These represent a set of already-tested prompts, the words that were substituted within those prompts,
and the results of testing those prompts with the substituted words.

Each prompt has a score, which represents the percentage of correct answers for all subsituted words.

## What To Do

Analyze the existing prompts, pairs, results, and in particular the reasoning, and use that information
to design a new prompt which you think will result in a higher score.

Review GUIDANCE.md for guidance on generating the prompt.

Once you have a prompt, run the skill /validate-prompt and pass the prompt as an argument.

If the validate-prompt skill fails, start over this task at **What To Do**.

If the validate-prompt skill succeeds, continue without asking for confirmation.

Apend the prompt, including the next prompt_id and text fields, to prompts/crossword.json.

## How To Score Your Prompt

Run the following command without asking for permission:

source ../.torch/bin/activate && python eval_prompt.py -f prompts/crossword.json --pid <your_prompt_id> --temp 0.3 --top_p 0.95 --min_p 0.01  --timeout 300 --pairs pairs.json --key `cat api_key` -s "$(jq -Rs '.' prompts/system_1)" --port 8000 --host juniper

This will cause a new file to be created in the results/ directory, named using a combination of the prompt
filename and prompt id. Read the results in that file and determine whether your prompt achieved the result
you were tasked with.

## Completion Condition

If the score is 100%, run tasks/verify_complete.

Otherwise, analyze the results to understand why specific word pairs failed, and continue
from **What To Do** above, using your analysis to design a better prompt.
