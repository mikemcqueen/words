Your task is to generate a prompt which generates a high score.

Review the following file contents:

* word pairs and their expected answer in pairs.json
* prompts in the prompts/*.json
* results in the results/*.json

These represent a set of already-tested prompts, the word pairs that were substituted within those prompts,
and the results of testing those prompts with the substituted word-pairs.

Each prmopt has a score, which represents the percentage of correct answers for all subsituted pair words.

Here's what you should do:

Analyze the existing prompts, pairs, and results, and use that information to design a new prompt which you
think will result in a high score.

Write that prompt, including prompt_id and text fields, to a new test_prompts_N.json file in the prompts/
directory, where N is one greater than the highest existing test_prompts filename.

Run:

source ../.torch/bin/activate && python eval_prompt.py -f <your_prompt_filename> --pid <your_prompt_id>

This will cause a new file to be created in the results/ directory, named using a combination of your prompt
filename and prompt id. Read that file and determine whether your prompt achieved the result you were tasked
with.
