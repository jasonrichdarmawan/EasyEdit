from dataclasses import dataclass
import json
from typing import List, Literal

from ...util.hparams import HyperParams


@dataclass
class ENCOREHyperParams(HyperParams):
    # Method
    alg_name: Literal["ENCORE"]
    model_name: str
    stats_dir: str
    layers: List[int]
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
    mom2_batch_tokens: int

    #Objective
    calculate_objective_value: bool
    add_prev_edits: bool
    update_norm_lambda: float
    emmet_lambda: float

    #new hparams
    prob_cutoff: float
    norm_control: float

    #dynamic multipliplier
    dynamic_multiplier: int
    random_vector_preservation: bool

    #gd params
    use_gd: bool
    grad_desc_lr: float
    grad_desc_steps: int
    gd_lambda: float
    spectral_norm: float
    condition_num: float

    calculate_norms: bool
