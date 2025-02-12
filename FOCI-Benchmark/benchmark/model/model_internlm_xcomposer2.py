#https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/tree/main

import math
import re

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


def build_vision_tower():
    vision_tower = 'openai/clip-vit-large-patch14-336'
    return CLIPVisionTower(vision_tower)


def build_vision_projector():
    projector_type = 'mlp2x_gelu'
    mm_hidden_size = 1024
    hidden_size = 4096

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {'mm_projector_type': 'identity'}


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower):
        super().__init__()

        self.is_loaded = False
        self.is_resize_pos = False

        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.select_feature = 'patch'
        self.load_model()
        self.resize_pos()

    def load_model(self):
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def resize_pos(self):
        pos_embed_checkpoint = self.vision_tower.vision_model.embeddings.position_embedding.weight
        pos_embed_checkpoint = pos_embed_checkpoint.unsqueeze(0)
        orig_size = 24
        new_size = 35

        if pos_embed_checkpoint.shape[1] == new_size**2 + 1:
            self.is_resize_pos = True
        else:
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = 1
            new_num = new_size**2 + num_extra_tokens
            #print('Position interpolate from %dx%d to %dx%d' %
            #      (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).permute(
                                                0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens.float(),
                size=(new_size, new_size),
                mode='bicubic',
                align_corners=False).half()
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

            new_pos_embed = new_pos_embed.squeeze(0)

            self.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(
                new_num, 1024)
            self.vision_tower.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(
                new_pos_embed.to(pos_embed_checkpoint.dtype))
            self.vision_tower.vision_model.embeddings.position_ids = torch.arange(
                new_num).expand((1, -1))

            self.is_resize_pos = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(
                f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device,
                             dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(
                    image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(
                images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(
            1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size)**2


class PLoRA(nn.Linear):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 lora_len=0,
                 **kwargs) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_len = lora_len
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.lora_scaling = self.lora_alpha / self.lora_r

        self.Plora_A = nn.Linear(
            in_features, self.lora_r, bias=False, device=device, dtype=dtype)
        self.Plora_B = nn.Linear(
            self.lora_r, out_features, bias=False, device=device, dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, im_mask=None):
        res = super().forward(x)
        if im_mask is not None:
            if torch.sum(im_mask) > 0:
                part_x = x[im_mask]
                res[im_mask] += self.Plora_B(
                    self.Plora_A(
                        self.lora_dropout(part_x))) * self.lora_scaling
            else:
                part_x = x[:, :1]
                res[:, :1] += self.Plora_B(
                    self.Plora_A(self.lora_dropout(part_x))) * 0
        return res

# coding=utf-8
# Copyright (c) InternLM. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" InternLM model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

INTERNLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class InternLM2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternLMModel`]. It is used to instantiate
    an InternLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the InternLM-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the InternLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`InternLMModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:

    ```python
    >>> from transformers import InternLMModel, InternLMConfig

    >>> # Initializing a InternLM internlm-7b style configuration
    >>> configuration = InternLMConfig()

    >>> # Initializing a model from the internlm-7b style configuration
    >>> model = InternLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "internlm"
    _auto_class = "AutoConfig"

    def __init__(  # pylint: disable=W0102
        self,
        vocab_size=103168,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        bias=True,
        rope_theta=10000,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.bias = bias

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor < 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float >= 1, got {rope_scaling_factor}")

# # Copyright (c) InternLM. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch InternLM2 model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging)

try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = 'InternLM2Config'


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size,
                      dtype: torch.dtype,
                      device: torch.device,
                      past_key_values_length: int = 0):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len),
                      torch.tensor(torch.finfo(dtype).min, device=device),
                      device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([
            torch.zeros(
                tgt_len, past_key_values_length, dtype=dtype, device=device),
            mask
        ],
                         dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                         tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor,
                 dtype: torch.dtype,
                 tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len,
    src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                  src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool),
        torch.finfo(dtype).min)


class InternLM2RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """InternLM2RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class InternLM2RotaryEmbedding(nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            **(torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            'cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            'sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class InternLM2LinearScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            'cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            'sin_cached', emb.sin().to(dtype), persistent=False)


class InternLM2DynamicNTKScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len /
                                 self.max_position_embeddings) -
                                (self.scaling_factor - 1))**(
                                    self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base
                **(torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            'cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            'sin_cached', emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos.unsqueeze(0).unsqueeze(0).expand(len(position_ids), -1, -1, -1)
    sin = sin.unsqueeze(0).unsqueeze(0).expand(len(position_ids), -1, -1, -1)
    if q.size(2) == 1:
        q_embed = (q * cos[:, :, -1:, :]) + (
            rotate_half(q) * sin[:, :, -1:, :])
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    if k.size(2) == 1:
        k_embed = (k * cos[:, :, -1:, :]) + (
            rotate_half(k) * sin[:, :, -1:, :])
    else:
        k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class InternLM2MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.w1 = PLoRA(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)
        self.w3 = PLoRA(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)
        self.w2 = PLoRA(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, im_mask):
        down_proj = self.w2(
            self.act_fn(self.w1(x, im_mask)) * self.w3(x, im_mask), im_mask)

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


class InternLM2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).')

        self.wqkv = PLoRA(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)

        self.wo = PLoRA(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.bias,
            lora_r=256,
            lora_alpha=256,
            lora_len=576)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'dynamic':
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor)
            else:
                raise ValueError(
                    "Currently we only support rotary embedding's type being 'dynamic'."
                )
        return self.rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        im_mask: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`')

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states, im_mask)

        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., :self.num_key_value_groups, :]
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.size()}')

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.size()}')

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output, im_mask)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class InternLM2FlashAttention2(InternLM2Attention):
    """InternLM2 flash attention module.

    This module inherits from `InternLM2Attention` as the weights of the module
    stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal
    with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        im_mask: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`')

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop('padding_mask')

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states, im_mask)

        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=self.num_heads + 2 * self.num_key_value_heads,
            d=self.head_dim,
            q=q_len,
        )

        query_states = qkv_states[..., :self.num_key_value_groups, :]
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (InternLM2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f'The input hidden states seems to be silently casted in float32, this might be related to'
                f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back '
                f'the input in {target_dtype}.')

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate)

        attn_output = attn_output.reshape(bsz, q_len,
                                          self.hidden_size).contiguous()
        attn_output = self.wo(attn_output, im_mask)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class InternLM2DecoderLayer(nn.Module):

    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = (
            InternLM2Attention(config=config)
            if not getattr(config, '_flash_attn_2_enabled', False) else
            InternLM2FlashAttention2(config=config))
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        im_mask: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`')

        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            im_mask=im_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, im_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


InternLM2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`InternLM2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    'The bare InternLM2 Model outputting raw hidden-states without any specific head on top.',
    InternLM2_START_DOCSTRING,
)
class InternLM2PreTrainedModel(PreTrainedModel):
    config_class = InternLM2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternLM2DecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


InternLM2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    'The bare InternLM2 Model outputting raw hidden-states without any specific head on top.',
    InternLM2_START_DOCSTRING,
)
class InternLM2Model(InternLM2PreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`InternLM2DecoderLayer`]

    Args:
        config: InternLM2Config
    """

    _auto_class = 'AutoModel'

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.tok_embeddings = nn.Embedding(config.vocab_size,
                                           config.hidden_size,
                                           self.padding_idx)
        self.layers = nn.ModuleList([
            InternLM2DecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = InternLM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else
                expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, BaseModelOutputWithPast]:

        im_mask = kwargs.get('im_mask', None)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)
            im_mask = torch.zeros(inputs_embeds.shape[:2]).to(
                inputs_embeds.device).bool()
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds,
            past_key_values_length)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = past_key_values[
                idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None,
                                      im_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    im_mask=im_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# # Copyright (c) InternLM. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch InternLMXComposer2 model."""
import copy
import queue
import threading
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (add_start_docstrings_to_model_forward,
                                replace_return_docstrings)

try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

_CONFIG_FOR_DOC = 'InternLMXcomposer2Config'


class InternLMXComposer2ForCausalLM(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.tokenizer = None

        self.max_length = config.max_length
        print(f'Set max length to {self.max_length}')
        # Initialize weights and apply final processing
        self.post_init()

        self.vit = build_vision_tower()
        self.vision_proj = build_vision_projector()

        self.vis_processor = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternLM2Model):
            module.gradient_checkpointing = value
        if value:
            self.vit.vision_tower.vision_model.encoder.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def encode_text(self, text, add_special_tokens=False):
        token = self.tokenizer(
            text, return_tensors='pt',
            add_special_tokens=add_special_tokens).input_ids.to(self.device)
        embs = self.model.tok_embeddings(token)
        return embs

    def encode_img(self, image):
        if image is None:
            return None
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = self.vis_processor(image).unsqueeze(0).to(self.device)
        else:
            assert isinstance(image, torch.Tensor)

        img_embeds, atts_img, img_target = self.img2emb(image)
        return img_embeds

    def img2emb(self, image):
        img_embeds = self.vision_proj(self.vit(image.to(self.device)))
        atts_img = torch.ones(
            img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        img_target = torch.ones(
            img_embeds.size()[:2], dtype=torch.long).to(
            img_embeds.device) * -100

        return img_embeds, atts_img, img_target

    def prompt_wrap(self, img_embeds, prompt):
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.tokenizer(
            p_before, return_tensors='pt',
            add_special_tokens=True).to(img_embeds.device)

        p_before_embeds = self.model.tok_embeddings(
            p_before_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds], dim=1)

        wrapped_atts_img = torch.ones(
            wrapped_img_embeds.size()[:-1],
            dtype=torch.long).to(img_embeds.device)

        wrapped_target = torch.ones(
            batch_size, wrapped_img_embeds.shape[1], dtype=torch.long).to(
            img_embeds.device) * -100

        return wrapped_img_embeds, wrapped_atts_img, wrapped_target

    def text2emb(self, text, add_special=False):
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special).to(self.device)

        targets = self.mask_human_targets(to_regress_tokens.input_ids)
        targets = targets.to(self.device)
        return to_regress_tokens, targets

    def interleav_wrap_chat(self, tokenizer, query, image, history, meta_instruction):
        prompt = ''
        if meta_instruction:
            prompt += f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""

        im_len = image.shape[1]
        image_nums = len(image)
        parts = prompt.split('<ImageHere>')
        wrap_embeds, wrap_im_mask = [], []
        temp_len = 0

        if len(parts) != image_nums + 1:
            raise ValueError('Invalid <ImageHere> prompt format.')

        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = tokenizer(part, return_tensors='pt').to(self.device)
                part_embeds = self.model.tok_embeddings(
                    part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]))
                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                wrap_embeds.append(image[idx].unsqueeze(0))
                wrap_im_mask.append(torch.ones(1, image[idx].shape[0]))
                temp_len += im_len

            if temp_len > self.max_length:
                break

        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)
        wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
        wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device).bool()
        inputs = {
            'inputs_embeds': wrap_embeds
        }
        return inputs, wrap_im_mask

    def interleav_wrap(self, img_list, text_list):
        wrap_embeds_list, wrap_atts_list = [], []
        wrap_target_list, wrap_im_mask_list = [], []

        for image, text in zip(img_list, text_list):
            img_embeds, atts_img, img_target = self.img2emb(image)
            text = text[0]
            parts = text.split('<ImageHere>')
            wrap_tokens, wrap_embeds, wrap_atts, wrap_im_mask = [], [], [], []
            temp_len = 0
            image_nums, im_len = img_embeds.shape[:2]
            need_bos = True
            for idx, part in enumerate(parts):
                if len(part) > 0:
                    part_tokens = self.tokenizer(
                        part,
                        return_tensors='pt',
                        padding='longest',
                        add_special_tokens=need_bos).to(self.device)
                    if need_bos:
                        need_bos = False
                    wrap_tokens.append(part_tokens.input_ids)
                    part_embeds = self.model.tok_embeddings(
                        part_tokens.input_ids)
                    wrap_embeds.append(part_embeds)
                    wrap_atts.append(part_tokens.attention_mask)
                    wrap_im_mask.append(
                        torch.zeros(part_embeds.shape[:2]).to(self.device))

                    temp_len += part_embeds.shape[1]
                if idx < image_nums:
                    wrap_tokens.append(img_target[idx].unsqueeze(0))
                    wrap_embeds.append(img_embeds[idx].unsqueeze(0))
                    wrap_atts.append(atts_img[idx].unsqueeze(0))
                    wrap_im_mask.append(
                        torch.ones_like(atts_img[idx].unsqueeze(0)))

                    temp_len += im_len
                if temp_len > self.max_length:
                    break

            wrap_tokens = torch.cat(wrap_tokens, dim=1)
            wrap_embeds = torch.cat(wrap_embeds, dim=1)
            wrap_atts = torch.cat(wrap_atts, dim=1)
            wrap_im_mask = torch.cat(wrap_im_mask, dim=1)

            wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)

            wrap_embeds = wrap_embeds[:, :self.max_length].to(self.device)
            wrap_atts = wrap_atts[:, :self.max_length].to(self.device)
            wrap_target = wrap_target[:, :self.max_length].to(self.device)
            wrap_im_mask = wrap_im_mask[:, :self.max_length].to(self.device)

            wrap_embeds_list.append(wrap_embeds)
            wrap_atts_list.append(wrap_atts)
            wrap_target_list.append(wrap_target)
            wrap_im_mask_list.append(wrap_im_mask)

        wrap_embeds = torch.cat(wrap_embeds_list)
        wrap_atts = torch.cat(wrap_atts_list)
        wrap_target = torch.cat(wrap_target_list)
        wrap_im_mask = torch.cat(wrap_im_mask_list)
        return wrap_embeds, wrap_atts, wrap_target, wrap_im_mask

    def mask_human_targets(self, input_ids, pure=False):
        target_batch = []
        for bs in range(input_ids.shape[0]):
            ids = input_ids[bs]
            targets = copy.deepcopy(ids)
            end_count = 0
            last_eoa = 0
            for i, temp_id in enumerate(ids):
                if temp_id == 92542:
                    if end_count % 2 == 0:
                        targets[last_eoa:i + 6] = -100
                    else:
                        last_eoa = i + 1
                    end_count += 1
                # # eos and following pad
                elif temp_id == 2:
                    # loss on eos, but not on pad
                    targets[i + 1:] = -100
                    break
            # trunction, end at last question
            if temp_id != 2 and end_count % 2 == 0:
                # mask all after the last answer
                targets[last_eoa + 1:] = -100
            target_batch.append(targets.unsqueeze(0))
        target_batch = torch.cat(target_batch, dim=0)
        return target_batch

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """

        samples = kwargs.get('samples', None)
        if samples:
            if samples['data_type'][0] == 'text':
                has_img = False
            elif samples['data_type'][0] == 'multi':
                has_img = True
            else:
                raise NotImplementedError

            # encode text
            text = samples['text_input']
            # encode image
            if has_img:
                image = samples['image']
                to_regress_embeds, attention_mask, targets, im_mask = self.interleav_wrap(
                    image, text)
            else:
                to_regress_tokens, targets = self.text2emb(
                    text, add_special=True)
                to_regress_embeds = self.model.tok_embeddings(
                    to_regress_tokens.input_ids)
                attention_mask = to_regress_tokens.attention_mask
                im_mask = torch.zeros(to_regress_embeds.shape[:2]).cuda()

            inputs_embeds = to_regress_embeds[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            targets = targets[:, :self.max_length]
            im_mask = im_mask[:, :self.max_length].bool()
            labels = targets
        else:
            im_mask = kwargs.get('im_mask', None)
            if im_mask is None and inputs_embeds is not None:
                im_mask = torch.zeros(inputs_embeds.shape[:2]).to(
                    inputs_embeds.device)
                im_mask = im_mask.bool()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=im_mask,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      im_mask=None,
                                      **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        im_mask = im_mask

        model_inputs.update({
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
            'im_mask': im_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past),)
        return reordered_past

    def build_inputs(self,
                     tokenizer,
                     query: str,
                     history: List[Tuple[str, str]] = [],
                     meta_instruction=''):
        prompt = ''
        if meta_instruction:
            prompt += f"""<s>[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        else:
            prompt += '<s>'
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        return tokenizer([prompt], return_tensors='pt')

    @torch.no_grad()
    def chat(
            self,
            tokenizer,
            query: str,
            image: torch.Tensor = None,
            history: List[Tuple[str, str]] = [],
            streamer: Optional[BaseStreamer] = None,
            max_new_tokens: int = 1024,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_p: float = 0.8,
            repetition_penalty: float = 1.005,
            meta_instruction:
            str = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
                  '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
                  '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
                  '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.',
            **kwargs,
    ):
        if image is None:
            inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
            im_mask = torch.zeros(inputs['input_ids'].shape[:2]).cuda().bool()
        else:
            image = self.encode_img(image)
            inputs, im_mask = self.interleav_wrap_chat(tokenizer, query, image, history, meta_instruction)
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
        ]
        outputs = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            im_mask=im_mask,
            **kwargs,
        )
        if image is None:
            outputs = outputs[0].cpu().tolist()[len(inputs['input_ids'][0]):]
        else:
            outputs = outputs[0].cpu().tolist()
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split('[UNUSED_TOKEN_145]')[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(
            self,
            tokenizer,
            query: str,
            history: List[Tuple[str, str]] = [],
            max_new_tokens: int = 1024,
            do_sample: bool = True,
            temperature: float = 0.8,
            top_p: float = 0.8,
            **kwargs,
    ):
        """Return a generator in format: (response, history) Eg.

        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')]) ('你好，有什么可以帮助您的吗？', [('你好',
        '你好，有什么可以帮助您的吗？')])
        """
        if BaseStreamer is None:
            raise ModuleNotFoundError(
                'The version of `transformers` is too low. Please make sure '
                'that you have installed `transformers>=4.28.0`.')

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):

            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ''
                self.received_inputs = False
                self.queue.put(
                    (self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError('ChatStreamer only supports batch size 1')
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                token = self.tokenizer.decode([value[-1]],
                                              skip_special_tokens=True)
                if token.strip() != '[UNUSED_TOKEN_145]':
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()
# Copyright (c) InternLM. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for IntermLM."""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': './tokenizer.model'}

PRETRAINED_VOCAB_FILES_MAP = {}


class InternLMXComposer2Tokenizer(PreTrainedTokenizer):
    """Construct a InternLM tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ['input_ids', 'attention_mask']
    _auto_class = 'AutoTokenizer'

    def __init__(
        self,
        vocab_file,
        unk_token='<unk>',
        bos_token='<s>',
        eos_token='</s>',
        pad_token='</s>',
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        decode_with_prefix_space=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.decode_with_prefix_space = decode_with_prefix_space
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        self._no_prefix_space_tokens = None
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        """ Initialization"""

    @property
    def no_prefix_space_tokens(self):
        if self._no_prefix_space_tokens is None:
            vocab = self.convert_ids_to_tokens(list(range(self.vocab_size)))
            self._no_prefix_space_tokens = {
                i
                for i, tok in enumerate(vocab) if not tok.startswith('▁')
            }
        return self._no_prefix_space_tokens

    @property
    def vocab_size(self):
        """Returns vocab size."""
        return self.sp_model.get_piece_size()

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp_model.eos_id()

    def get_vocab(self):
        """Returns vocab as a dict."""
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def _maybe_add_prefix_space(self, tokens, decoded):
        if tokens and tokens[0] not in self.no_prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ''
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += ' '
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        out_string = self.clean_up_tokenization(out_string)
        out_string = self._maybe_add_prefix_space(
            tokens=tokens, decoded=out_string)
        return out_string[1:]

    def save_vocabulary(self,
                        save_directory,
                        filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(
                f'Vocabulary path ({save_directory}) should be a directory')
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + '-' if filename_prefix else '') +
            VOCAB_FILES_NAMES['vocab_file'])

        if os.path.abspath(self.vocab_file) != os.path.abspath(
                out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, 'wb') as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file, )

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is not None:
            output = output + token_ids_1

        if self.add_eos_token:
            output = output + [self.eos_token_id]

        return output

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        """Retrieve sequence ids from a token list that has no special tokens
        added. This method is called when adding special tokens using the
        tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + (
            [0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """Create a mask from the two sequences passed to be used in a
        sequence-pair classification task. T5 does not make use of token type
        ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]
