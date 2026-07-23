from dataclasses import dataclass
from typing import List, Literal

from ...util.hparams import HyperParams
import yaml


@dataclass
class AlphaEditCircuitHyperParams(HyperParams):
    # Method
    num_hidden_layers: int
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    nullspace_threshold: float
    L2: float
    alg_name: str
    model_name: str
    edit_with_chat_template: bool
    stats_dir: str
    
    # Destination node
    hook_q_input: str
    hook_k_input: str
    hook_v_input: str
    hook_mlp_in: str
    hook_resid_post: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False
    mom2_batch_tokens: int = None
    
    use_subject_noise_baseline: bool = False
    # When true, compute the target vector once for each request and reuse it
    # for all selected hub/source updates. The default recomputes after each
    # update, matching the existing AlphaEdit-Circuit behavior.
    no_target_recompute: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'AlphaEdit_Circuit') \
            or print(
                f'AlphaEditCircuitHyperParams can not load from {hparams_name_or_path}, '
                f'alg_name is {config["alg_name"]}'
            )
        return cls(**config)