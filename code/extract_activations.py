# this file is to ease getting and extracting the activations
# and then save the activations to the respective directory
# this is the first step when starting from scratch with the
# experimenting
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# to use this use this command as a template
# refer to TEMPLATES for which template name to use
# python3 extract_activations.py \
#   --model MODEL_NAME \
#   --suffix MODEL_SIZE \
#   --target-dir TARGET_DIRECTORY \
#   --n PROMPTS_NUM
#   --template TEMPLATE_NAME 

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from datasets import load_dataset
import argparse
import gc
from pathlib import Path

N = 500

## The following part is for chat templates
def _no_template(prompt: str, tokenizer) -> str:
    '''
    raw tokens
    '''
    print("Note: you may have mispelled something or this model activations extraction is not supported currently\n")
    return prompt
 
 
def _gemma_3_family(prompt: str, tokenizer) -> str:
    '''
    gemma3-it models chat template (gemma-3-1b-itand gemma-3-4b-it)
    use the tokenizer's built in chat template 
    gemma-3-12b-it and gemma-3-27b-it are multimodal 
    and ARE NOT CURRENTLY SUPPORTED
    '''
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )

def _gemma_4_family(prompt: str, tokenizer) -> str:
    '''
    template for gemma-4-E2B-it and gemma-4-E4B-it
    '''
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def _qwen_3_5_family(prompt: str, tokenizer) -> str:
    '''
    template for Qwen3.5-9B and Qwen3.5-4B
    Qwen3.5 is multimodal and uses AutoProcessor.
    '''
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def _llama_3_family(prompt: str, tokenizer) -> str:
    '''
    template for Llama-3.2 Instruct models
    '''
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
 
# template_name -> formatter function
TEMPLATES = {
    "none": _no_template,
    "gemma-3": _gemma_3_family,
    "gemma-4": _gemma_4_family,
    "qwen-3-5": _qwen_3_5_family,
    "llama-3": _llama_3_family,
}

def get_template(name: str):
    '''
    looks up a formatter by name
    '''
    if name not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES.keys()))
        raise ValueError(
            f"Unknown template '{name}'. Available templates: {available}"
        )
    return TEMPLATES[name]

# the following part is for loading and getting activations etc

def load_model(model_name, use_processor=False):
    '''
    given the model name it loads the model and its tokenizer
    we use huggingface, after loading this becomes ready for
    inference and activation extraction
    returns the model and its tokenizer or processor
    pass use_processor=True to use AutoProcessor for Gemma4
    '''
    if use_processor:
        tok_or_proc = AutoProcessor.from_pretrained(model_name)
        # processors wrap an underlying tokenizer; patch pad token there
        inner = getattr(tok_or_proc, "tokenizer", None)
        if inner is not None and inner.pad_token is None:
            inner.pad_token = inner.eos_token
    else:
        tok_or_proc = AutoTokenizer.from_pretrained(model_name)
        if tok_or_proc.pad_token is None:
            tok_or_proc.pad_token = tok_or_proc.eos_token
 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tok_or_proc

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

    # pick only utility prompts that are rated helpfulness=4
    helpsteer = load_dataset("nvidia/HelpSteer")
    utility_prompts = [
        entry['prompt']
        for entry in helpsteer['train']
        if entry['helpfulness'] == 4
    ]
    utility_prompts = list(dict.fromkeys(utility_prompts))

    harmful_prompts = harmful_prompts[:n]
    harmless_prompts = harmless_prompts[:n]
    utility_prompts = utility_prompts[:n]

    return harmful_prompts, harmless_prompts, utility_prompts

def extract_all_and_save(harmful_prompts, harmless_prompts, utility_prompts,
                         model, tok_or_proc, suffix, output_dir, template_fn, is_processor=False):
    '''
    runs all three prompt through the model, 
    saves activations respective to each to disk as .npy
    returns the three activation arrays
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_activations(prompt):
        formatted=template_fn(prompt, tok_or_proc)
        if is_processor:
            inputs = tok_or_proc(
                text=formatted,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
        else:
            inputs = tok_or_proc(
                formatted,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
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
    p.add_argument("--template", default="none",
                   choices=sorted(TEMPLATES.keys()),
                   help="Chat template to apply to prompts (default: none = raw)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    template_fn = get_template(args.template)
    use_processor = args.template in {"gemma-4", "qwen-3-5"}

 
    harmful_prompts, harmless_prompts, utility_prompts = load_prompts(n=args.n)
    model, tok_or_proc = load_model(args.model, use_processor=use_processor)

    extract_all_and_save(
    harmful_prompts, harmless_prompts, utility_prompts,
    model, tok_or_proc, args.suffix, args.target_dir,
    template_fn=template_fn,
    is_processor=use_processor,
    )

    del model, tok_or_proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()