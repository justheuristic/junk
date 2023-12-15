from dataclasses import dataclass

__all__ = ["QuantizationConfig"]


@dataclass
class QuantizationConfig:
    model_path: str = None
    dataset: str = "pajama"
    seed: int = 0
    new_eval: bool = False
    nsamples: int = 1024
    groupsize: int = None
    permutation_order: str = "act_order"
    max_epochs: int = 10000
    relative_mse_tolerance: float = 0.01
    steps_per_epoch: int = 100
    model_seqlen: int = 4096
    nbits_per_codebook: int = 16
    scale_nbits: int = 0
    init_max_iter: int = 100
    num_codebooks: int = 1
    out_group_size: int = 1
    in_group_size: int = 8
    codebook_value_nbits: int = 16
    codebook_value_num_groups: int = 1
    true_sequential: bool = False
    beam_size: int = 1
    print_frequency: int = 10
    devices: str = None
    sparsity_regularizer: float = 0.
    use_faiss: bool = False
    dtype:str  = "auto"
    max_points_per_centroid: int = None
    lr: float = 1e-4
    offload_activations: bool = False
    load: bool = False
    save: bool = False
    skip_out_loss: bool = False
    wandb: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
