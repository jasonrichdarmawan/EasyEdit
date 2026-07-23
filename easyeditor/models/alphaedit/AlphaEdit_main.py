import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams

from tqdm.auto import tqdm

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P_loaded = False
cache_c_new = False

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
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
    if not os.path.exists(hparams.P_loc):
        print(f"The null-space projection matrix P does not exist and now calculate.")
        P = {}
        for layer in tqdm(hparams.layers, desc="Computing projection matrix"):
            P[layer] = get_project(model, tok, layer, hparams)
        print("Saving null-space projection matrix P to avoid redundant future computations...")
        torch.save(P, hparams.P_loc)
        P_loaded = True
    elif P_loaded == False:
        P = torch.load(hparams.P_loc)
        P = {k: v.contiguous() for k, v in P.items()}
        P_loaded = True

    # Maintain the global variable cache_c to avoid redundant computations.
    # If this is the first calculation (i.e., cache_c_new == false), then initialize cache_c first
    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if any(1 for item in ["llama", "gpt-j-6b", "qwen3-4b", "tiny-aya-global"] if item in hparams.model_name.lower()):
            cache_c_shape = (W_out.shape[1], W_out.shape[1])
        elif "gpt2-xl" in hparams.model_name.lower():
            cache_c_shape = (W_out.shape[0], W_out.shape[0])
        else:
            raise NotImplementedError(f"Model {hparams.model_name} not recognized. Please specify cache_c_shape for this model in the code.")
        cache_c = [torch.zeros(cache_c_shape, device=W_out.device) for _ in hparams.layers]
        del W_out
        cache_c_new = True

    return_statistics = kwargs.pop("return_statistics", False)
    execution_result = execute_AlphaEdit(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        return_statistics=return_statistics,
    )
    if return_statistics:
        deltas, statistics = execution_result
    else:
        deltas = execution_result

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            upd_matrix = upd_m.to(f"cuda:{hparams.device}")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if return_statistics:
        return model, weights_copy, statistics
    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    return_statistics: bool = False,
) -> Dict[str, Tuple[torch.Tensor]] | Tuple[Dict[str, Tuple[torch.Tensor]], List[Dict[str, Any]]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    statistics = []

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
        requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    temp_statistics = {}

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            z_result = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                return_statistics=return_statistics,
            )
            if return_statistics:
                cur_z, target_statistics = z_result
                target_statistics.update(
                    {
                        "case_id": request.get("case_id"),
                        "edit_id": request.get("edit_id"),
                        "target_layer": z_layer,
                    }
                )
                temp_statistics[request.get("edit_id")] = target_statistics
            else:
                cur_z = z_result

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        z_error = torch.linalg.norm(targets, dim=0).mean()
        print("z error", z_error)

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        layer_device = weights[f"{hparams.rewrite_module_tmp.format(layer)}.weight"].device
        proj = P[layer].to(device=layer_device, dtype=torch.float)
        layer_ks = layer_ks.to(device=layer_device, dtype=torch.float)
        resid = resid.to(device=layer_device, dtype=torch.float)
        c = cache_c[i].to(device=layer_device, dtype=torch.float)
        lhs = (
            proj @ (layer_ks @ layer_ks.T + c)
            + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device=layer_device)
        )
        rhs = (
            proj @ layer_ks @ resid.T
        )
        upd_matrix = torch.linalg.solve(lhs, rhs)

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        original_weight_norm = torch.linalg.norm(weights[weight_name])
        update_norm = torch.linalg.norm(upd_matrix)
        print("orig norm", original_weight_norm)
        print("upd norm", update_norm)

        if return_statistics:
            for request_index, request in enumerate(requests):
                statistics.append(
                    {
                        "edited_layer": layer,
                        "z_error": float(torch.linalg.norm(targets[:, request_index * repeat_factor]).detach().cpu()),
                        "weight_norm_before_update": float(original_weight_norm.detach().cpu()),
                        "weight_update_norm": float(update_norm.detach().cpu()),
                        **temp_statistics[request.get("edit_id")],
                    }
                )

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                upd_matrix.detach().cpu()
            )
        
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i] += layer_ks @ layer_ks.T

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    print(f"Deltas successfully computed for {list(weights.keys())}")

    if return_statistics:
        return deltas, statistics
    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_batch_tokens: int,
    mom2_dtype: str,
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
            batch_tokens=mom2_batch_tokens,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
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


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_batch_tokens,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        hparams=hparams
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T