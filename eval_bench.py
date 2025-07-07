import os
from bullshit_eval import eval_ai_assistant, eval_political, gen_political
import argparse
from chat_model import ChatLLM

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--org_file", help="input file path", type=str, default="dataset/scenarios.json")
    parser.add_argument("--input_file", help="input file path", type=str, default="input/bullshit_eval/dataset.json")
    parser.add_argument("--output_dir", help="output directory", type=str, default="output/bullshit_eval")

    parser.add_argument("--provider", help="LLM provider", type=str, default="openai", choices=["openai", "hf", "anthropic", "gemini"])
    parser.add_argument("--model", help="Model", type=str, default="o3-mini-2025-01-31")

    parser.add_argument("--eval_provider", help="LLM provider", type=str, default="openai", choices=["openai", "hf", "anthropic", "gemini"])
    parser.add_argument("--eval_model", help="Model", type=str, default="o3-mini-2025-01-31")

    parser.add_argument('--cot', action='store_true', default=False, help="With or without CoT")
    parser.add_argument('--pa', action='store_true', default=False, help="Evaluate Principal-agent problem or not")

    parser.add_argument("--task", choices=['marketplace', 'ai_assistant', 'political', 'gen_political'], default='ai_assistant')

    args = parser.parse_args()
    file_path = args.input_file

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = ChatLLM(
        provider=args.provider,
        model=args.model   
    )

    eval_model = ChatLLM(
        provider=args.eval_provider,
        model=args.eval_model   
    )

    if args.task == "ai_assistant":
        eval_ai_assistant(model, eval_model, args.task, args.cot, args.pa, args.input_file, args.output_dir)
    elif args.task == "political":
        eval_political(model, eval_model, args.task, args.cot, args.input_file, args.output_dir)
    elif args.task == "gen_political":
        gen_political(model, args.task, args.cot, args.input_file, args.output_dir)