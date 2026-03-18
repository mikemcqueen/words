Your task is to modify an existing prompt to result in a score of 100.

Review the following file contents:

* prompt files in prompts/eden_wood_118*.json
* words and their expected answer in pairs.json
* results in the results/eden_wood_118*.json

These represent a set of already-tested prompts, the words that were substituted within those prompts,
and the results of testing those prompts with the substituted words.

Each result file has a score, which represents the percentage of correct answers for all subsituted words.

## What To Do

Analyze existing prompts, pairs, and results, and specifically the reasoning behind failed results,
and use that information to modify the prompt in prompts/eden_wood_118.json in a way that you think will
result in a score of 100.

Review tasks/eden_wood_118/GUIDANCE.md for additional guidance on modifying the prompt.

Once you have a prompt, run the skill /validate-prompt and pass the prompt as an argument.

If the validate-prompt skill fails, start over this task at **What To Do**.

If the validate-prompt skill succeeds, continue without asking for confirmation.

Write the prompt, including prompt_id and text fields, to a new prompts/eden_wood_118.N.json file, where N
represents the next unused iteration.

## How To Score Your Prompt

Run the following command without asking for permission:

source ../.torch/bin/activate && python eval_prompt.py -f <your_prompt_filename> --pid <your_prompt_id> --temp 0.3 --top_p 0.95 --min_p 0.01  --timeout 300 --pairs pairs.json --key `cat api_key` -s "$(jq -Rs '.' prompts/system_1)" --mc 1 --host juniper --port 8000

This will cause a new file to be created in the results/ directory, named using a combination of your prompt
filename and prompt id. Read the results in that file and determine whether your prompt achieved the score
you were tasked with.

## Completion Condition

If the score is 100%, run the task in tasks/verify_complete.

Otherwise, analyze the results to understand why specific word pairs failed, and continue
from **What To Do** above, using your analysis to design a better prompt.
