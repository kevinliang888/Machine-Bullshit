import os
import openai
from openai import OpenAI
import argparse
from prompt import criteria, market_scenario_prompt
from prompt import system_prompt, eval_prompt, confirm_prompt

from utils import load_json, write_json, get_eval_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Support model: llama-2-7b, llama-2-13b, llama-3-8b, llama-3-70b
    parser.add_argument('--input_path', default="./input/marketplace/all_results.json")
    parser.add_argument('--ai_model', default='llama-3-8b')
    parser.add_argument('--checkpoints', default=None)
    parser.add_argument('--criteria', nargs='+', default=["empty rhetoric", "paltering", "weasel word", "unverified claim"])
    parser.add_argument('--output_path', default="output")

    args = parser.parse_args()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    # Set OpenAI API key.
    openai.api_key = openai_api_key
    client = OpenAI(api_key=openai_api_key)

    test_data = load_json(args.input_path)
    criteria = args.criteria

    model = args.ai_model
    ckpts = args.checkpoints
    all_dict = []

    index = 0
    if ckpts is None:
        for data in test_data:
            all_info = data[model]
            all_info = get_eval_results(client, all_info, criteria, market_scenario_prompt, 
                                        eval_prompt, system_prompt, confirm_prompt)
            all_dict.append(all_info)
    else:
        for data in test_data:
            all_info = data[model]
            rlhf_info = all_info["rlhf"][ckpts]
            rlhf_info = get_eval_results(client, rlhf_info, criteria, market_scenario_prompt, 
                                         eval_prompt, system_prompt, confirm_prompt)
            all_dict.append(all_info)
    

    if ckpts:
        write_json(all_dict, f"{args.output_path}/{model}-{ckpts}.json")
    else:
        write_json(all_dict, f"{args.output_path}/{model}.json")