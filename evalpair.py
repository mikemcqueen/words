# evalpair.py
#
# Prompts a model with word pairs and outputs the first line of the response.

import argparse
import asyncio
import sys

import httpx

from client import MAX_CONCURRENT, SERVER_URL, run_concurrent, get_inference_params, send_yesno_request, send_openai_request
from info import info
from model import load_model, get_model_name, is_gemma_model, is_instruct, is_instruct_model, specialize_prompt, generate_text

def generate_response(model, tokenizer, prompt: str, skip_special = True, max_new_tokens: int = 1000 ) -> str:
    return generate_text(model, tokenizer, prompt, max_new_tokens)

def generate_response_fast(args, prompt: str) -> str:
    """Send prompt to server for generation using shared request functions (sync wrapper)."""
    async def _do():
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            if args.client == "yesno":
                actual, _, _ = await send_yesno_request(client, args, prompt)
            else:
                actual, _, _ = await send_openai_request(client, args, prompt)
            return actual
    try:
        return asyncio.run(_do())
    except httpx.ConnectError:
        print(f"Error: Cannot connect to server at {args.host}:{args.port}", file=sys.stderr)
        print("Start the server with: ./server", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error calling server: {e}", file=sys.stderr)
        sys.exit(1)


async def process_pair_async(client: httpx.AsyncClient, args, ctx: str, orig_pair: str) -> tuple:
    """Process a single pair with retry on flipped pair if NO."""
    pair = orig_pair.replace(',', ' ')
    prompt = ctx.replace("{PAIR}", pair)
    if args.client == "yesno":
        response, _, _ = await send_yesno_request(client, args, prompt)
    else:
        response, _, _ = await send_openai_request(client, args, prompt)
    yes = response.upper().startswith("YES")
    as_txt = ""

    if not yes:
        pair = flip(orig_pair).replace(',', ' ')
        prompt = ctx.replace("{PAIR}", pair)
        if args.client == "yesno":
            response, _, _ = await send_yesno_request(client, args, prompt)
        else:
            response, _, _ = await send_openai_request(client, args, prompt)
        yes = response.upper().startswith("YES")
        if yes:
            as_txt = f"as {pair}"

    return orig_pair, yes, as_txt


async def process_pairs_fast(ctx: str, pairs, args):
    """Process all pairs with MAX_CONCURRENT in flight at once."""
    async def process(client, orig_pair):
        return await process_pair_async(client, args, ctx, orig_pair)

    async for result in run_concurrent(pairs, process, MAX_CONCURRENT, args.timeout):
        print(f"{result[0]} {'YES' if result[1] else 'NO'} {result[2]}")


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

def make_single_question_prompt(model, tokenizer, pair: str) -> str:
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

    prompt = PROMPT + pair
    if model is not None:
        prompt = specialize_prompt(model, tokenizer, prompt)
    return prompt, FOLLOWUP

def make_prompt(model, tokenizer, pair: str, include_questions) -> str:
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

    if model is None:
        return prompt
    return specialize_prompt(model, tokenizer, prompt)

def custom_context(model, tokenizer, args, ctx: str, pair: str):
    if pair:
        ctx = ctx.replace("{PAIR}", pair)
    if model is None:
        prompt = ctx  # Server handles specialization
        response = generate_response_fast(args, prompt)
    else:
        prompt = specialize_prompt(model, tokenizer, ctx)
        response = generate_response(model, tokenizer, prompt, False)
    #print(f"prompt: {prompt}")
    #print(f"{ctx}: {response.strip()}")
    return prompt, response

def single_question(model, tokenizer, args, pair: str):
    pair = pair.strip()
    if not pair:
        return None

    prompt, followup = make_single_question_prompt(model, tokenizer, pair)
    #print(f"prompt: {prompt}")
    if model is None:
        response = generate_response_fast(args, prompt)
    else:
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

def process_pair(model, tokenizer, args, pair: str, include_questions):
    pair = pair.strip()
    if not pair:
        return None

    prompt = make_prompt(model, tokenizer, pair, include_questions)
    #print(f"prompt: {prompt}")
    if model is None:
        response = generate_response_fast(args, prompt)
    else:
        response = generate_response(model, tokenizer, prompt)
    print(f"{pair}: {response}")
    """
    lines = parse_response(model, response)
    if lines:
        print(f"{pair}: {lines[0]}")
    """

def follow_up(model, tokenizer, args, response: str, prompt: str):
    p = response
    p += "<start_of_turn>user\n"
    p += prompt
    p += "<end_of_turn>\n"
    p += "<start_of_turn>model\n"

    if model is None:
        response = generate_response_fast(args, p)
    else:
        response = generate_response(model, tokenizer, p)
    print(f"-----------\n{response}")

def parse_args():
    DEFAULT_MODEL = "g2it"

    parser = argparse.ArgumentParser(description="Evaluate word pairs with a language model")
    parser.add_argument('--single', action="store_true", help='ask single question with followup')
    parser.add_argument('-q', '--questions', action="store_true", help='include questions in response')

    # Pair input: either --pair or --pair-file (mutually exclusive)
    pair_group = parser.add_mutually_exclusive_group()
    pair_group.add_argument('--pair', type=str, help='Single pair to evaluate (e.g., "foo,bar")')
    pair_group.add_argument('--pair-file', type=str, help='File containing pairs (one per line)')

    # Prompt input: either --prompt or (--prompt-file + --pid) (mutually exclusive)
    parser.add_argument('--prompt', type=str, help='Prompt context (use {PAIR} as placeholder)')
    parser.add_argument('--prompt-file', type=str, help='JSON file containing prompts')
    parser.add_argument('--pid', '--prompt-id', type=str, dest='prompt_id',
                        help='Prompt ID to use from prompt file')

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('-m', '--model', metavar='g2it', type=str, default=DEFAULT_MODEL,
                        help=f"Select model (default: {DEFAULT_MODEL})")
    model_group.add_argument('--fast', action="store_true",
                        help=f"Use server at {SERVER_URL} instead of loading model")

    # Server/client options (used with --fast)
    parser.add_argument('-s', '--system-prompt', type=str,
                        help='System prompt (optional, used with --fast)')
    parser.add_argument('-c', '--client', choices=["yesno", "openai"], default="openai",
                        help="Client type: 'yesno' or 'openai' (default)")
    parser.add_argument('--host', default="http://localhost",
                        help="Server host (default: http://localhost)")
    parser.add_argument('--port', type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument('--key', type=str, help="API key")
    parser.add_argument('--temp', type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument('--top_p', type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    parser.add_argument('--min_p', type=float, default=0.01,
                        help="Min-p sampling (default: 0.01)")
    parser.add_argument('--repeat-penalty', '--rp', type=float, default=1.0,
                        help="Repeat penalty (default: 1.0)")
    parser.add_argument('--repeat-last-n', '--rln', type=int, default=32,
                        help="Repeat last n tokens (default: 32)")
    parser.add_argument('--timeout', type=float, default=300.0,
                        help="Request timeout in seconds (default: 300)")

    args = parser.parse_args()

    if not args.fast and not is_instruct(args.model):
        print(f"{get_model_name(args.model)} is not an instruct model")
        exit()

    # Validate --prompt-file and --pid must be used together
    if args.prompt_file and not args.prompt_id:
        parser.error("--prompt-file requires --pid/--prompt-id")
    if args.prompt_id and not args.prompt_file:
        parser.error("--pid/--prompt-id requires --prompt-file")

    # Validate --prompt and --prompt-file/--pid are mutually exclusive
    if args.prompt and args.prompt_file:
        parser.error("Cannot specify both --prompt and --prompt-file/--pid")

    # Require at least one of pair or prompt input
    if not args.pair and not args.pair_file and not args.prompt and not args.prompt_file:
        parser.error("Either --pair, --pair-file, --prompt, or --prompt-file/--pid is required")

    # If prompt is provided, pairs are required
    if (args.prompt or args.prompt_file) and not (args.pair or args.pair_file):
        parser.error("--prompt/--prompt-file requires --pair or --pair-file")

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

def is_yes_response(model, response: str) -> bool:
    if not model:
        answer = response 
    elif is_gemma_model(model):
        lines = response.split('\n')
        answer = lines[-2].strip() # NOTE: Gemma specific
    else:
        assert False, "is_yes_response not implemented for this model"
    return answer.upper().startswith("YES")

def load_prompt_from_file(prompt_file: str, prompt_id: str) -> str:
    """Load a prompt from a JSON prompt file by ID."""
    import json
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    for p in prompts:
        if p.get('id') == prompt_id:
            return p.get('text')
    raise ValueError(f"Prompt ID '{prompt_id}' not found in {prompt_file}")


def main():
    args = parse_args()

    if args.fast:
        model, tokenizer = None, None
    else:
        _, model, tokenizer = load_model(args.model)
        if is_gemma_model(model) and is_instruct_model(model):
            print("gemma instruct: True")

    # Resolve prompt from --prompt or --prompt-file/--pid
    ctx = args.prompt
    if args.prompt_file:
        ctx = load_prompt_from_file(args.prompt_file, args.prompt_id)

    if ctx:
        # Use generators for common handling of --pair and --pair-file
        pairs = None
        if args.pair:
            pairs = single_pair_generator(args.pair)
        elif args.pair_file:
            pairs = file_pair_generator(args.pair_file)
        if not pairs:
            assert False, "TBD"

        # Fast async path for file processing
        if args.fast and args.pair_file:
            asyncio.run(process_pairs_fast(ctx, pairs, args))
            return

        for orig_pair in pairs:
            pair = orig_pair.replace(',', ' ')
            prompt, response = custom_context(model, tokenizer, args, ctx, pair)
            yes = is_yes_response(model, response)
            as_txt = ""
            if not yes:
                pair = flip(orig_pair).replace(',', ' ')
                prompt, response = custom_context(model, tokenizer, args, ctx, pair)
                yes = is_yes_response(model, response)
                if yes:
                    as_txt = f"as {pair}"

            if not args.pair_file:
                print(response)
            print(f"{orig_pair} {'YES' if yes else 'NO'} {as_txt}")
    elif args.pair:
        # Single pair mode (no ctx)
        yes = False
        if args.single:
            yes, response, followup = single_question(model, tokenizer, args, args.pair)
        else:
            process_pair(model, tokenizer, args, args.pair, args.questions)

        if yes:
            follow_up(model, tokenizer, args, response, followup)
    else: # args.pair_file without ctx
        # File mode
        try:
            for pair in file_pair_generator(args.pair_file):
                process_pair(model, tokenizer, args, pair, args.questions)
        except FileNotFoundError:
            print(f"Error: File not found: {args.pair_file}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
