import os
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
import logging
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
from ...util.device import normalize_device
from ...util.nethook import TraceDict, set_requires_grad
from ...util.runningstats import (
    CombinedStat,
    Mean,
    NormMean,
    SecondMoment,
    load_cached_state,
    make_loader,
    save_cached_state,
)

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

from .fake_dataset import FakeTokenDataset

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

LOGGER = logging.getLogger(__name__)

def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=[
        "gpt2-xl", "EleutherAI/gpt-j-6B",
        "Qwen/Qwen3-8B", "ibm-granite/granite-4.2-8b",
    ])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikitext2", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default="runs/stats")
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    device = normalize_device(None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().to(device)
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name: str | list[str],
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
    fake_samples=0,
    fake_seq_len=2**13,
):
    """
    Function to load or compute cached stats.
    """
    if isinstance(layer_name, str):
        layer_name = [layer_name]

    # Load_From_File
    # from datasets import Dataset
    # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
    # raw_ds = {'train': raw_ds}
    dataset_map = {
        "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
        "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        "wikipedia": ("wikimedia/wikipedia", "20231101.en"),
    }
    dataset_name, dataset_config = dataset_map[ds_name]

    if hasattr(model.config, 'n_positions'):
        maxlen = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        maxlen = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        maxlen = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        maxlen = model.config.seq_length
    else:
        raise NotImplementedError
            
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            maxlen = model.config.sliding_window or 4096
        else:
            maxlen = 4096
    if hasattr(model.config, 'model_type') and any(pattern in model.config.model_type for pattern in ["qwen", "granite"]):
        maxlen = min(maxlen, 4096)

    if batch_tokens is not None and batch_tokens < maxlen:
        maxlen = batch_tokens

    def get_ds():
        raw_ds = load_dataset(dataset_name, dataset_config)

        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen' in model.config.model_type:
            npos = min(npos, 4096)

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    sample_size = sample_size if fake_samples == 0 else fake_samples
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    size_suffix = f"_t{maxlen}" + size_suffix
    if batch_tokens < npos:
        size_suffix = f"_bt{batch_tokens}" + size_suffix
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)

    args = {"sample_size": sample_size}
    stats = {}
    for module_name in layer_name:
        file_extension = f"{model_name}/{ds_name}/{dataset_config}/{module_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
        file_name = stats_dir / file_extension

        stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
        logging.info(f"Trying to load cached stats from {file_name}...")
        cached_state = load_cached_state(file_name, args)
        if cached_state is not None and not force_recompute:
            logging.info(f"Loaded cached stats from {file_name}.")
            stat.load_state_dict(cached_state)

        stats[module_name] = (file_name, stat, cached_state)

    # backward compatibility
    if all(cached_state for _, _, cached_state in stats.values()):
        return stats[layer_name[0]][1]

    logging.info(f"Computing Cov locally....")

    needs_computation = force_recompute or any(cached_state is None for _, _, cached_state in stats.values())
    if needs_computation:
        ds = FakeTokenDataset(fake_samples, fake_seq_len) if fake_samples > 0 else get_ds()

    if progress is None:
        progress = lambda x: x

    loader = []
    if needs_computation:
        loader = make_loader(
            ds,
            sample_size=sample_size,
            batch_size=batch_size,
            collate_fn=length_collation(batch_tokens),
            pin_memory=True,
            random_sample=1,
            num_workers=2,
        )
    
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count, desc="Computing Cov", unit="batch_group"):
            for batch in tqdm(batch_group, unit="batch"):
                batch = dict_to_(batch, normalize_device(getattr(hparams, "device", None)))
                with TraceDict(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch, use_cache=False)

                for module_name, (_, stat, cached_state) in stats.items():
                    if cached_state is not None and not force_recompute:
                        continue
                    feats = flatten_masked_batch(tr[module_name].input, batch["attention_mask"])
                    # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                    feats = feats.to(dtype=dtype)
                    stat.add(feats)

    for module_name, (file_name, stat, cached_state) in stats.items():
        if cached_state is not None and not force_recompute:
            continue
        save_cached_state(file_name, stat, args)
    
    # backward compatibility
    return stats[layer_name[0]][1]


if __name__ == "__main__":
    main()
