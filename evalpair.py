# evalpair.py
#
# Prompts a model with word pairs and outputs the first line of the response.

import argparse
import sys

from info import info
from model import load_model, get_model_name, is_gemma_model, is_instruct, is_instruct_model, specialize_prompt

def generate_response(model, tokenizer, prompt: str, skip_special = True, max_new_tokens: int = 1000 ) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    """
    response = tokenizer.decode(outputs[0], skip_special_tokens=skip_special)
    return response

def get_first_line(text: str) -> str:
    """Extract the first line from the text."""
    lines = text.split('\n')
    return lines[0] if lines else text

def parse_response(model, text: str) -> list[str]:
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        if line == "model":
            return lines[idx+1:]

    return None

def make_single_question_prompt(model, pair: str) -> str:
    PROMPT = "Given a phrase, answer the following question. " \
        "Respond with a single YES or NO." \
        "Is the phrase unusual?\n" \
        "Phrase: "

    PROMPT = "You are given a pair of English words in order." \
        " Determine whether the pair could function as a sensible, literal English noun phrase" \
        " without metaphor, wordplay, or special domain knowledge. " \
        " Answer YES or NO only." \
        "\nWord pair: "

    #FOLLOWUP = "What?"
    #"Does reading it make you think of something?\n" \
    FOLLOWUP = "Why?"

    return specialize_prompt(model, PROMPT+pair), FOLLOWUP

def make_prompt(model, pair: str, include_questions) -> str:
    prefix = "Answer the following questions about the given phrase. " 

    yes_no = "Provide your answer to each question as either YES or NO.\n\n"
    # "Respond with all answers on a single line, separated by commas.\n\n" \

    yes_no_2 = "Respond with a single line of comma-separated YES or NO values, " \
        "which represent the answers to each question, in order.\n\n" \

    q_a = "Your response should **ONLY** include each question, " \
        "followed by your answer to the question: either YES or NO.\n\n" \

    questions = "Question: Is there any obvious connection between the words?\n" \
        "Question: Is the combination of words unusual?\n" \
        "Question: Is the phrase nonsensical?\n" \
        
    """
        "Could someone say it and be understood?\n" \
        "Could a person mean something by it?\n" \
        "Would you know what someone was talking about if they said it?\n" \
        "Could it be what something is called?\n" \
        "Could it be the name of a person, place, or thing?\n" \
        "Could you label something with it?\n" \
        "Does it feel off?\n" \
        "Is there something wrong with it?\n" \
        "Does it make you wince?\n" \
    """

    # gemma-3-bad
    #"Is it a trick question?\n" \


    # gemma-2-bad
    #"Does it bring anything to mind?\n" \
    #"Does it conjure anything?\n" \

    #"Respond by repeating each question, followed by YES or NO.\n\n" \
    #"Respond as a single line of text with comma-separated YES or NO values, " \
    #"which represent the answers to each question, in order.\n\n" \
    #"Is this word combination meaningless or absurd?\n" 
    #"Could these words appear adjacent in a grammatically correct English sentence?\n" \
    #"Is there a coherent meaning when these words are combined?\n" \
    #"Is there any context where a speaker might plausibly use this word combination?\n" \
    #"Is this word combination nonsensical?\n" \

    phrase = "Phrase: "

    style = q_a if include_questions else yes_no
    prompt = prefix + style + questions + phrase + pair

    return specialize_prompt(model, prompt)

def custom_context(model, tokenizer, ctx: str, pair: str):
    if pair:
        ctx = ctx.replace("%p", pair)
    prompt = specialize_prompt(model, ctx)
    #print(f"prompt: {prompt}")
    response = generate_response(model, tokenizer, prompt, False)
    #print(f"{ctx}: {response.strip()}")
    return prompt, response

def single_question(model, tokenizer, pair: str):
    pair = pair.strip()
    if not pair:
        return None

    prompt, followup = make_single_question_prompt(model, pair)
    #print(f"prompt: {prompt}")
    response = generate_response(model, tokenizer, prompt, False)
    print(f"{pair}: {response}")
    lines = response.split('\n')
    yes = lines[-2].strip() # NOTE: Gemma specific
    if lines:
        yessir = yes.startswith("YES")
        print(f"{pair}: {yes}: {yessir}")
        yessir = True # always ask why
        return yessir, response, followup

    return False, None

def process_pair(model, tokenizer, pair: str, include_questions):
    pair = pair.strip()
    if not pair:
        return None

    prompt = make_prompt(model, pair, include_questions)
    #print(f"prompt: {prompt}")
    response = generate_response(model, tokenizer, prompt)
    print(f"{pair}: {response}")
    """
    lines = parse_response(model, response)
    if lines:
        print(f"{pair}: {lines[0]}")
    """

def follow_up(model, tokenizer, response: str, prompt: str):
    p = response
    p += "<start_of_turn>user\n"
    p += prompt
    p += "<end_of_turn>\n"
    p += "<start_of_turn>model\n"
    
    response = generate_response(model, tokenizer, p)
    print(f"-----------\n{response}")

def parse_args():
    DEFAULT_MODEL = "g2it"

    parser = argparse.ArgumentParser(description="Evaluate word pairs with a language model")
    parser.add_argument('ctx', nargs='?', help='Optional context')
    parser.add_argument('-s', '--single', action="store_true", help='ask single question with followup')
    parser.add_argument('-q', '--questions', action="store_true", help='include questions in response')
    parser.add_argument('-p', '--pair', type=str, help='Single pair to evaluate (e.g., "foo,bar")')
    parser.add_argument('-f', '--file', type=str, help='File containing pairs (one per line)')
    parser.add_argument('-m', '--model', metavar='g2it', type=str, default=DEFAULT_MODEL,
                        help=f"Select model (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    if not is_instruct(args.model):
        print(f"{get_model_name(args.model)} is not an instruct model")
        exit()

    if not args.pair and not args.file and not args.ctx:
        parser.error("Either --ctx, -p/--pair, or -f/--file is required")

    if args.pair and args.file:
        parser.error("Cannot specify both -p/--pair and -f/--file")

    return args

def flip(pair: str) -> str:
    flipped = pair.split(',')
    assert len(flipped) == 2, f"{pair} len: {len(flipped)}"
    flipped = [flipped[1], flipped[0]]
    return ",".join(flipped)

def single_pair_generator(pair_string):
    """Generator that yields a single pair string, then ends."""
    yield pair_string

def file_pair_generator(filename):
    """Generator that yields all pairs read from a file, one per line."""
    with open(filename, 'r') as f:
        for line in f:
            pair = line.strip()
            if pair:
                yield pair

def is_yes_response(response: str) -> bool:
    lines = response.split('\n')
    answer = lines[-2].strip() # NOTE: Gemma specific
    return answer.lower().startswith("yes")

def main():
    args = parse_args()

    device, model, tokenizer = load_model(args.model)
    if is_gemma_model(model) and is_instruct_model(model):
        print("gemma instruct: True")

    if args.ctx:
        # Use generators for common handling of --pair and --file
        pairs = None
        if args.pair:
            pairs = single_pair_generator(args.pair)
        elif args.file:
            pairs = file_pair_generator(args.file)
        if not pairs:
            assert False, "TBD"

        for orig_pair in pairs:
            pair = orig_pair.replace(',', ' ')
            prompt, response = custom_context(model, tokenizer, args.ctx, pair)
            yes = is_yes_response(response)
            as_txt = ""
            if not yes:
                pair = flip(orig_pair).replace(',', ' ')
                prompt, response = custom_context(model, tokenizer, args.ctx, pair)
                yes = is_yes_response(response)
                if yes:
                    as_txt = f"as {pair}"

            if not args.file:
                print(response)
            print(f"{orig_pair} {'YES' if yes else 'NO'} {as_txt}")
    elif args.pair:
        # Single pair mode (no ctx)
        yes = False
        if args.single:
            yes, response, followup = single_question(model, tokenizer, args.pair)
        else:
            process_pair(model, tokenizer, args.pair, args.questions)

        if yes:
            follow_up(model, tokenizer, response, followup)
    else: # args.file without ctx
        # File mode
        try:
            for pair in file_pair_generator(args.file):
                process_pair(model, tokenizer, pair, args.questions)
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
