from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from fairseq.dataclass import FairseqDataclass
from fairseq.models import register_model, BaseFairseqModel
from omegaconf import II
from lean_transformer.models.gpt import LeanGPTModel, LeanGPTConfig


DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class LeanTransformerLanguageModelConfig(FairseqDataclass):
    hf_model_config: str = field(default="./model_config.json", metadata=dict(help="path to a huggingface config json"))
    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")


@register_model("lean_lm", dataclass=LeanTransformerLanguageModelConfig)
class LeanGPTLanguageModel(BaseFairseqModel):
    def __init__(self, decoder: LeanGPTModel):
        super().__init__()
        self.decoder = decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        config = LeanGPTConfig.from_json_file(args.hf_model_config)
        assert config.pad_token_id == task.source_dictionary.pad(), task.source_dictionary.pad()
        assert config.eos_token_id == task.source_dictionary.eos(), task.source_dictionary.eos()
        assert config.bos_token_id == task.source_dictionary.bos(), task.source_dictionary.bos()
        model = LeanGPTModel(config)
        model.resize_token_embeddings(len(task.source_dictionary))
        return cls(model)

    def forward(self, src_tokens, *, src_lengths: torch.Tensor):
        batch_size, max_len = src_tokens.shape
        attention_mask = torch.arange(max_len, device=src_tokens.device) <= src_lengths.unsqueeze(1)
        return self.decoder(input_ids=src_tokens, attention_mask=attention_mask)

    def get_normalized_probs(self, net_output, log_probs: bool, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""


        print('keys:', net_output.keys())
        logits = net_output.logits.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @property
    def supported_targets(self):
        return {"future"}
