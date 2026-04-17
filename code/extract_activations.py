# this file is to ease getting and extracting the activations
# and then save the activations to the respective directory
# this is the first step when starting from scratch with the
# experimenting
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# to use this use this command as a template
# python3 extract_activations.py \
#   --model MODEL_NAME \
#   --suffix MODEL_SIZE \
#   --target-dir TARGET_DIRECTORY \
#   --n PROMPTS_NUM    

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import gc
from pathlib import Path

N = 100 #make sure to change this if we increase prompt #
DEFAULT_MAX_LENGTH = 512

def load_model(model_name):
    '''
    given the model name it loads the model and its tokenizer
    we use huggingface, after loading this becomes ready for 
    inference and activation extraction
    returns the model and its tokenizer
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

def load_prompts(n=N):
    '''
    n is number of prompts we wish to test with
    we load n prompts each dataset, AdvBench,
    alpaca, and helpsteer
    returns 3 lists (in a tuple) containing the prompts from each dataset
    '''
    # harmful prompts
    advbench = load_dataset("walledai/AdvBench")
    harmful_prompts = advbench['train']['prompt']

    # harmless prompts
    alpaca = load_dataset("tatsu-lab/alpaca")
    harmless_prompts = [entry['instruction'] for entry in alpaca['train']]

    # utility prompts
    helpsteer = load_dataset("nvidia/HelpSteer")
    utility_prompts = [entry['prompt'] for entry in helpsteer['train']]

    harmful_prompts = harmful_prompts[:n]
    harmless_prompts = harmless_prompts[:n]
    utility_prompts = utility_prompts[:n]

    return harmful_prompts, harmless_prompts, utility_prompts

def extract_all_and_save(harmful_prompts, harmless_prompts, utility_prompts,
                         model, tokenizer, suffix, output_dir):
    '''
    runs all three prompt through the model, 
    saves activations respective to each to disk as .npy
    returns the three activation arrays
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_activations(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return [h[:, -1, :].float().cpu().numpy()[0] for h in outputs.hidden_states]

    harmful  = np.array([get_activations(p) for p in tqdm(harmful_prompts,  desc="harmful")])
    np.save(output_dir / f"harmful_{suffix}_activations.npy", harmful)

    harmless = np.array([get_activations(p) for p in tqdm(harmless_prompts, desc="harmless")])
    np.save(output_dir / f"harmless_{suffix}_activations.npy", harmless)

    utility  = np.array([get_activations(p) for p in tqdm(utility_prompts,  desc="utility")])
    np.save(output_dir / f"utility_{suffix}_activations.npy", utility)

    return harmful, harmless, utility

def parse_args():
    p = argparse.ArgumentParser(description="Extract and save model activations.")
    p.add_argument("--model", required=True,
                   help="HF model ID, e.g. google/gemma-3-4b-it")
    p.add_argument("--suffix", required=True,
                   help="Filename suffix, e.g. 4b")
    p.add_argument("--target-dir", required=True,
                   help="Directory for saved .npy files, e.g. activations/gemma3acts")
    p.add_argument("--n", type=int, default=N,
                   help=f"Prompts per category (default: {N})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    harmful_prompts, harmless_prompts, utility_prompts = load_prompts(n=args.n)
    model, tokenizer = load_model(args.model)

    extract_all_and_save(
        harmful_prompts, harmless_prompts, utility_prompts,
        model, tokenizer, args.suffix, args.target_dir,
    )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()