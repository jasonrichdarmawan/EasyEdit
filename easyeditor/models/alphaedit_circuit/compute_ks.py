from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_z import get_module_input_output_at_words
from .AlphaEdit_Circuit_hparams import AlphaEditCircuitHyperParams
from ...util import nethook


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: AlphaEditCircuitHyperParams,
    layer: int,
    context_templates: List[str],
):
    all_prompts = []
    prompt_request_idx = []
    if not hparams.edit_with_chat_template:
        for i in range(len(requests)):
            request = requests[i]
            for context_types in context_templates:
                for context in context_types:
                    prompt = context.replace("{prompt}", request["prompt"])
                    all_prompts.append(prompt)
                    prompt_request_idx.append(i)
                    all_prompts.append(request["prompt"])
                    prompt_request_idx.append(i)
    else:
        for i in range(len(requests)):
            # request = requests[i]
            # for context_types in context_templates:
            #     for context in context_types:
            #         prompt = context.replace("{prompt}", request["prompt"])
            prompt = requests[i]["prompt"]
            chat = [
                {"role": "system", "content": "Only respond with the answer. Do not include any explanations."},
                # {"role": "user", "content": "Suppose Jack wears a red shirt, Jill wears a green shirt, and Terry Fox wears a blue shirt. Therefore, the person wearing the blue shirt is a citizen of"},
                # {"role": "assistant", "content": "Canada"},
                {"role": "user", "content": prompt},
            ]
            prompt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            all_prompts.append(prompt)
            prompt_request_idx.append(i)
    
    tok.padding_side = "left"
    input_tok = tok(
        all_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=not hparams.edit_with_chat_template,
    ).to(model.device)
    
    idxs = []
    if hparams.fact_token == "subject_first":
        for i in range(len(all_prompts)):
            request = requests[prompt_request_idx[i]]
            start_char = all_prompts[i].find(request["subject"])
            start_tok = input_tok.char_to_token(i, start_char)
            idxs.append(start_tok)
    elif hparams.fact_token == "subject_last":
        for i in range(len(all_prompts)):
            request = requests[prompt_request_idx[i]]
            start_char = all_prompts[i].find(request["subject"])
            end_char = start_char + len(request["subject"]) - 1
            start_tok = input_tok.char_to_token(i, start_char)
            end_tok = input_tok.char_to_token(i, end_char)
            idxs.append(end_tok)
    
    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.rewrite_module_tmp.format(layer),
            retain_input=True,
            stop=True,
        ) as tr:
            model(**input_tok)
            
    layer_ks = tr.input[list(range(tr.input.shape[0])), idxs]

    # return the hidden representation per request by averaging across all prompts for that request
    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    # context_type_lens = [0, 1]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()
    ans = []
    for i in range(0, layer_ks.size(0), context_len):
        tmp = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            tmp.append(layer_ks[i + start : i + end].mean(0))
        ans.append(torch.stack(tmp, 0).mean(0))
    
    return torch.stack(ans, dim=0)
