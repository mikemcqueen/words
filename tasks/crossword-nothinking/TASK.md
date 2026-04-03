Your task is to generate a prompt which results in a high score.

## What To Do

Review the scripts in the scripts/ subfolder.

Determine the top scoring prompts by running scripts/get_top_scoring_prompt_ids.sh, passing it the
supplied summary.jsonl.

Analyze the prompts and results for those prompt ids by running scripts/get_prompt_for_id.sh, passing
it the prompt id and supplied summary.jsonl.

Generate a new prompt which you think will result in a higher score. You should look at False
Positive (FA) and False Negatives (FN) of prior prompts when designing the new prompt. You typically
want to focus on reducing one of these values.

Generate and score one new prompt at a time.  Each new prompt you generate should be based on the
results of the previously generated and executed prompts. You are not allowed to generate multiple
prompts upfront, and/or attempt to run them in parallel.

Review GUIDANCE.md for guidance on generating the prompt.

It is perfectly OK to think outside the box and come up with completely new approaches when designing
a new prompt. Don't limit yourself to small tweaks of existing prompts.

Once you have generated a new prompt, run the skill /validate-prompt and pass the prompt
as an argument.

If the validate-prompt skill fails, start over this task at **What To Do**.

Append the prompt, including the next prompt_id and text fields, to prompts/crossword.json.

## How To Score Your Prompt

Run the following command without asking for permission. Do not run parallel instances of this command.

source ../.torch/bin/activate && python eval_prompt.py -f prompts/crossword.json --pid <your_prompt_id> --temp 0.7 --top_p 0.95 --min_p 0.01  --timeout 300 --pairs pairs/yesno.200.json --key `cat api_key` -s prompts/sys26 --port 8000 --host juniper --mc 16 -q --compact --thinking off 

Move the result file to the summary.jsonl folder.

Run the skill /summarize-prompt passing the new result file path as an argument.

## Completion Condition

If the score is 100%, run tasks/verify_complete.
