import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z
from .AlphaEdit_Circuit_hparams import AlphaEditCircuitHyperParams
from .position_utils import get_lookup_positions_in_target
from .eap_graph import (
    EAPGraph, get_kl_div_metric, 
    find_top_mlp_hubs_by_token_sample,
)
import json

from tqdm.auto import tqdm

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P_loaded = False
cache_c_new = False

def apply_AlphaEdit_Circuit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditCircuitHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
  #-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded, cache_c, cache_c_new
    
    weights_copy = {}
    if copy:
        model = deepcopy(model)
    
    # Calculate the null-space projection matrix P
    # Please ensure that you have downloaded "null_space_project.pt" to the easyedit folder beforehand, or get the P by following calculation
    model_name = model_name = model.config._name_or_path.rsplit("/")[-1]
    size_suffix = "" if hparams.mom2_n_samples is None else f"_{hparams.mom2_n_samples}"
    if hparams.mom2_batch_tokens is not None:
        size_suffix = f"_t{hparams.mom2_batch_tokens}" + size_suffix
    P_filepath = Path(hparams.stats_dir) / model_name / f"{hparams.mom2_dataset}_stats" / f"null_space_project_{hparams.mom2_dtype}_{size_suffix}.pt"
    if not os.path.exists(P_filepath):
        print(f"The null-space projection matrix P does not exist and now calculate.")
        P = [None for _ in range(hparams.num_hidden_layers)]
        for i, layer in tqdm(enumerate(range(hparams.num_hidden_layers)), desc="Computing projection matrix"):
            P[i] = get_project(model, tok, layer, hparams).to("cpu")
        print("Saving null-space projection matrix P to avoid redundant future computations...")
        torch.save(P, P_filepath)
        P_loaded = True
    elif P_loaded == False:
        P = torch.load(P_filepath)
        P = [P[i].contiguous() for i in range(len(P))]
        P_loaded = True

    # Maintain the global variable cache_c to avoid redundant computations.
    # If this is the first calculation (i.e., cache_c_new == false), then initialize cache_c first
    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(0)}.weight")
        if any(1 for item in ["llama", "gpt-j-6b", "qwen3-4b", "tiny-aya-global"] if item in hparams.model_name.lower()):
            cache_c_shape = (W_out.shape[1], W_out.shape[1])
        elif "gpt2-xl" in hparams.model_name.lower():
            cache_c_shape = (W_out.shape[0], W_out.shape[0])
        else:
            raise NotImplementedError(f"Model {hparams.model_name} not recognized. Please specify cache_c_shape for this model in the code.")
        cache_c = [torch.zeros(cache_c_shape) for _ in range(hparams.num_hidden_layers)]
        del W_out
        cache_c_new = True
    
    deltas = execute_AlphaEdit_Circuit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_m.to(w.device)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    if len(deltas) == 0:
        print("No updates were made for these requests.")
    else:
        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit_Circuit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditCircuitHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject: {request['subject']} do not exist in prompt: {request['prompt']}")
        else:
            requests[i]['prompt'] = request['prompt'].replace('{}', request['subject'])

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in range(hparams.num_hidden_layers)
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model=model, tokenizer=tok)
    print(f"Context templates used for computing z and k/v pairs: {context_templates}")

    for request in requests:
        print(
            f"Executing AlphaEdit_Circuit algo\n"
            f"Request: [{request['prompt']}] [{request['target_true']}] -> [{request['target_new']}]"
        )
        
        eap = EAPGraph(model)
        eap_prompts = []
        if not hparams.edit_with_chat_template:
            eap_prompts.append(request["prompt"])
        else:
            chat = [
                {"role": "system", "content": "Only respond with the answer. Do not include any explanations."},
                # {"role": "user", "content": "Suppose Jack wears a red shirt, Jill wears a green shirt, and Terry Fox wears a blue shirt. Therefore, the person wearing the blue shirt is a citizen of"},
                # {"role": "assistant", "content": "Canada"},
                {"role": "user", "content": request["prompt"]},
            ]
            prompt = tok.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            eap_prompts.append(prompt)
        
        tok.padding_side = "right"
        eap_tok = tok(
            eap_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not hparams.edit_with_chat_template,
        ).to(model.device)
        print(f"EAP Input IDs: {eap_tok['input_ids']}")
        
        subject_spans = []
        for i in range(len(eap_prompts)):
            start_char = eap_prompts[i].find(request["subject"])
            end_char = start_char + len(request["subject"]) - 1
            start_tok = eap_tok.char_to_token(i, start_char)
            end_tok = eap_tok.char_to_token(i, end_char)
            subject_spans.append((start_tok, end_tok))
        print(f"Subject spans: {subject_spans}")
        
        corrupted_input_ids = None
        corrupted_attention_mask = None
        if not hparams.use_subject_noise_baseline:
            corrupted_input_ids = eap_tok["input_ids"].clone()
            corrupted_attention_mask = eap_tok["attention_mask"].clone()
            for subject_span in subject_spans:
                start, end = subject_span
                corrupted_input_ids[:, start:end+1] +=1
        
        metric_fn = get_kl_div_metric()
        scores = eap.attribute(
            input_ids=eap_tok["input_ids"], 
            attention_mask=eap_tok["attention_mask"],
            corrupted_input_ids=corrupted_input_ids,
            corrupted_attention_mask=corrupted_attention_mask,
            metric_fn=metric_fn,
            subject_spans=subject_spans,
            integrated_gradients=5,
            return_per_head_attribution=False,
            return_per_token_scores=True,
        )
        top_mlp_hubs_by_token_sample = find_top_mlp_hubs_by_token_sample(scores, topk_hubs=8, topk_sources=8)
        
        hubs_skipped = []
        hubs = []
        for hub in deepcopy(top_mlp_hubs_by_token_sample[0]):
            if not hub["destination"]["raw"].endswith("hook_mlp_in"):
                hubs_skipped.append(hub)
                continue
            
            shallow_layers = list(range(0, int(hparams.num_hidden_layers * 1/8)))
            deep_layers = list(range(int(hparams.num_hidden_layers * 7/8), hparams.num_hidden_layers))
            layers_not_to_edit = (
                shallow_layers
                + deep_layers
            )
            if hub["destination"]["layer"] in layers_not_to_edit:
                hubs_skipped.append(hub)
                continue
            
            hubs.append(hub)
        
        sources_skipped = [[] for _ in range(len(hubs))]
        for hub_idx, hub in enumerate(hubs):
            sources = []
            for source in hub["sources"]:
                if not source["raw"].endswith("hook_mlp_out"):
                    sources_skipped[hub_idx].append(source)
                    continue
                
                shallow_layers = list(range(0, int(hparams.num_hidden_layers * 1/8)))
                deep_layers = list(range(int(hparams.num_hidden_layers * 7/8), hparams.num_hidden_layers))
                layers_not_to_edit = shallow_layers + deep_layers
                if source["layer"] in layers_not_to_edit:
                    sources_skipped[hub_idx].append(source)
                    continue
                
                sources.append(source)
            if len(sources[3:]) > 0:
                sources_skipped[hub_idx].extend(sources[3:])
            sources = sorted(sources[:3], key=lambda x: x["layer"])
            
            if len(sources) == 0:
                hubs.remove(hub)
                hubs_skipped.append(hub)
                continue
            
            hub["sources"] = sources
        
        if len(hubs[3:]) > 0:
            hubs_skipped.extend(hubs[3:])
        hubs = sorted(hubs[:3], key=lambda x: x["destination"]["layer"])
        
        print(f"Hubs:\n{json.dumps(hubs, indent=4)}")
        if len(hubs_skipped) > 0:
            print(f"Skipped hubs:\n{json.dumps(hubs_skipped, indent=4)}")
        if len(hubs) == 0:
            print("No significant hubs found for this request. Skipping...")
            print(f"Top MLP hubs by token/sample-level scores:\n{json.dumps(top_mlp_hubs_by_token_sample, indent=4)}")
            continue

        total_source_updates = sum(len(hub_item["sources"]) for hub_item in hubs)
        updates_done = 0
            
        # Insert
        for hub_idx, hub in enumerate(hubs):
            print(f"Hub:\n{json.dumps(hub, indent=4)}")
            if len(sources_skipped[hub_idx]) > 0:
                print(f"Skipped sources for this hub:\n{json.dumps(sources_skipped[hub_idx], indent=4)}")
            
            for source in hub["sources"]:
                z_list = []
                cur_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    hub["destination"]["layer"],
                    context_templates,
                    source_enc=eap_tok,
                    source_lookup_idx=hub["position"],
                    rendered_source_prompt=eap_prompts[0],
                    raw_source_prompt=request["prompt"],
                )
                
                z_list.append(cur_z)
                zs = torch.stack(z_list, dim=1) # shape [d_model, num_requests]
                
                # Get current model activations
                layer_ks = compute_ks(
                    model,
                    tok,
                    [request],
                    hparams,
                    source["layer"],
                    context_templates,
                    source_enc=eap_tok,
                    source_lookup_idx=hub["position"],
                    rendered_source_prompt=eap_prompts[0],
                    raw_source_prompt=request["prompt"],
                ).T # shape [mlp_hidden_size, num_requests]
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {source['layer']}")

                # Compute residual error
                all_prompts = []
                if not hparams.edit_with_chat_template:
                    all_prompts.append(request["prompt"])
                else:
                    chat = [
                        {"role": "system", "content": "Only respond with the answer. Do not include any explanations."},
                        # {"role": "user", "content": "Suppose Jack wears a red shirt, Jill wears a green shirt, and Terry Fox wears a blue shirt. Therefore, the person wearing the blue shirt is a citizen of"},
                        # {"role": "assistant", "content": "Canada"},
                        {"role": "user", "content": request["prompt"]},
                    ]
                    prompt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                    all_prompts.append(prompt)
                
                cur_zs_tok = tok(
                    all_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=not hparams.edit_with_chat_template,
                ).to(model.device)
                
                idxs = get_lookup_positions_in_target(
                    source_enc=eap_tok,
                    source_lookup_idx=hub["position"],
                    rendered_source_prompt=eap_prompts[0],
                    raw_source_prompt=request["prompt"],
                    target_enc=cur_zs_tok,
                    rendered_target_prompts=all_prompts,
                )
                # idxs = []
                # if hparams.fact_token == "subject_first":
                #     start_char = all_prompts[0].find(request["subject"])
                #     start_tok = cur_zs_tok.char_to_token(0, start_char)
                #     idxs.append(start_tok)
                # elif hparams.fact_token == "subject_last":
                #     start_char = all_prompts[0].find(request["subject"])
                #     start_tok = cur_zs_tok.char_to_token(0, start_char)
                #     end_char = start_char + len(request["subject"]) - 1
                #     end_tok = cur_zs_tok.char_to_token(0, end_char)
                #     idxs.append(end_tok)
                
                with torch.no_grad():
                    with nethook.Trace(
                        module=model,
                        layer=hparams.layer_module_tmp.format(hub["destination"]["layer"]),
                        retain_output=True,
                        stop=True,
                    ) as tr:
                        model(**cur_zs_tok)
                
                cur_zs = tr.output[list(range(tr.output.shape[0])), idxs].T # shape [d_model, num_requests]
                targets = zs - cur_zs
                print("z error", torch.linalg.norm(targets, dim=0).mean())

                repeat_factor = (layer_ks.size(1) // targets.size(1))
                targets = targets.repeat_interleave(repeat_factor, dim=1)
                resid = targets / (total_source_updates - updates_done)  # Distribute residual across remaining hub/source edits
                weight_name = f"{hparams.rewrite_module_tmp.format(source['layer'])}.weight"
                layer_device = weights[weight_name].device
                proj = P[source["layer"]].to(device=layer_device, dtype=torch.float)
                layer_ks = layer_ks.to(device=layer_device, dtype=torch.float)
                resid = resid.to(device=layer_device, dtype=torch.float)
                c = cache_c[source["layer"]].to(device=layer_device, dtype=torch.float)
                k1k1 = layer_ks @ layer_ks.T
                lhs = (
                    proj @ (c + k1k1)
                    + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device=layer_device)
                )
                rhs = (
                    proj @ layer_ks @ resid.T
                )
                upd_matrix = torch.linalg.solve(lhs, rhs)

                # Adjust update matrix shape
                upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

                print("orig norm", torch.linalg.norm(weights[weight_name]))
                print("upd norm", torch.linalg.norm(upd_matrix))

                # Update model weights and record desired changes in `delta` variable
                with torch.no_grad():
                    weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
                    if deltas.get(weight_name) is None:
                        deltas[weight_name] = upd_matrix.detach().cpu()
                    else:
                        deltas[weight_name] += upd_matrix.detach().cpu()
                
                cache_c[source["layer"]] += (k1k1).to(cache_c[source["layer"]].device)
                updates_done += 1
                
                # Clear GPU memory
                #del U,S,cov
                # for x in [layer_ks, cur_zs, targets]:
                #     x.cpu()
                #     del x
                # torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    mom2_batch_tokens: int = None,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            batch_tokens=mom2_batch_tokens,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float()

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to(f"cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tokenizer):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{prompt}"]]
        prompt_tok = tokenizer(
            ["The", "Therefore", "Because", "I", "You"],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        for length, n_gen in [(10, 5)]:
            gen_token = model.generate(
                **prompt_tok,
                max_new_tokens=length,
                num_beams=n_gen // 5,
                num_return_sequences=n_gen // 5,
                pad_token_id=tokenizer.eos_token_id,
            )
            templates = tokenizer.batch_decode(gen_token, skip_special_tokens=True)
            templates = [f"{template}. {{prompt}}" for template in templates]
            CONTEXT_TEMPLATES_CACHE.append(templates)

    return CONTEXT_TEMPLATES_CACHE

def get_project(model, tok, layer, hparams):
    """
    Caveat: Compute with Eigendecomposition instead of Singular Value Decomposition for efficiency because covariance is positive semi-definite.
    """
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        hparams.mom2_batch_tokens,
        force_recompute=force_recompute,
        hparams=hparams
    )
    print(f"Computing projection matrix for layer {layer}")
    # if cov shape is (mlp_hidden_size, mlp_hidden_size), then
    #   U shape is (mlp_hidden_size, mlp_hidden_size)
    #   S shape is (mlp_hidden_size,)
    #   U[:, small_singular_indices] shape is (mlp_hidden_size, num_small_singular_values)
    #   P shape is (mlp_hidden_size, mlp_hidden_size)
    vals, vecs = torch.linalg.eigh(cov)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (vals < threshold).nonzero(as_tuple=True)[0]
    print(f"{len(small_singular_indices)} small singular values found below threshold {threshold} out of {len(vals)} total singular values.")
    return vecs[:, small_singular_indices] @ vecs[:, small_singular_indices].T
