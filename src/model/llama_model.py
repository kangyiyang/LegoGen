# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn
from typing import List

# working_path = os.getcwd()
# import sys
# sys.path.append(f'{working_path}/src/model/dinov2')

# from dinov2.models.vision_transformer import DinoVisionTransformer


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 512
    position_dim: int = 128
    out_pad_dim: int = 1

    rank: int = 2
    c_n_heads: int = 32
    image_dim: int = 384
    patch_h: int = 32
    patch_w: int = 24
    add_cross: int = 4


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cisq = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cisk = reshape_for_broadcast(freqs_cis, xk_)
    xq_out = torch.view_as_real(xq_ * freqs_cisq).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cisk).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Lora_layer(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        r: int, 
        lora_o: bool = False,
        lora_alpha: int = 1, 
        lora_dropout: float = 0,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        self.scaling = self.lora_alpha / self.r

        # self.lora_A = nn.Parameter(torch.zeros(input_features, self.r))
        # self.lora_B = nn.Parameter(torch.zeros(self.r, output_features))
        if lora_o:
            self.lora_A = RowParallelLinear(
                            input_features,
                            self.r,
                            bias=False,
                            input_is_parallel=True,
                            )
            self.lora_B = RowParallelLinear(
                            self.r,
                            output_features,
                            bias=False,
                            )
        else:
            self.lora_A = RowParallelLinear(
                            input_features,
                            self.r,
                            bias=False,
                            )
            self.lora_B = ColumnParallelLinear(
                            self.r,
                            output_features,
                            bias=False,
                            gather_output=False,
                            )
        
        # self.reset_parameters()
    
    def reset_parameters(self):
        # if hasattr(self, 'lora_A'):
        #     # initialize A the same way as the default for nn.Linear and B to zero
        #     nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        #     nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    
    def forward(self, x):
        # output = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        output = self.lora_dropout(self.lora_B(self.lora_A(x)))
        # output = self.lora_dropout(self.lora_B(self.lora_A(x))) * self.scaling
        
        return output


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs, pretrain: bool):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.pretrain=pretrain

        if not pretrain:
            self.lora_q = Lora_layer(
                args.dim,
                args.n_heads * self.head_dim,
                r=args.rank,
                lora_alpha=1,
                # lora_dropout=0.2,
            )
            # self.lora_k = Lora_layer(
            #     args.dim,
            #     self.n_kv_heads * self.head_dim,
            #     r=args.rank
            # )
            self.lora_v = Lora_layer(
                args.dim,
                self.n_kv_heads * self.head_dim,
                r=args.rank,
                lora_alpha=1,
                # lora_dropout=0.2,
            )
            # self.lora_o = Lora_layer(
            #     args.n_heads * self.head_dim,
            #     args.dim,
            #     r=args.rank,
            #     lora_o=True
            # )

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        autoregressive:bool=False,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if not self.pretrain:
            xq+=self.lora_q(x)
            # xk+=self.lora_k(x)
            xv+=self.lora_v(x)


        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        if autoregressive:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys=xk
            values=xv

        # repeat k/v heads if n_kv_heads < n_heads   
        xk = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq=xq.transpose(1,2)
        k=xk.transpose(1,2)
        v=xv.transpose(1,2)

        scores = torch.matmul(xq, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 那照这么说，这里的mask是atten mask，而在整个代码中没有使用padding mask
        # 当然这对于batch_size=1来讲没有问题，不过后续或许可以调整
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, v)  # (bs, n_local_heads, seqlen, head_dim)
        output_view = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        output=self.wo(output_view)
        # if not self.pretrain:
        #     output+=self.lora_o(output_view)

        return output

class Image_Cross_Atten(nn.Module):
    def __init__(
            self,
            layer_id: int,
            params: ModelArgs,
            n_kv_heads: Optional[int] =None,
            ):
        super().__init__()
        self.params=params
        dim=params.dim
        n_heads=params.c_n_heads
        image_dim=params.image_dim
        
        self.layer_id = layer_id

        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = ColumnParallelLinear(
            dim,
            n_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wk = ColumnParallelLinear(
            image_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wv = ColumnParallelLinear(
            image_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wo = RowParallelLinear(
            n_heads * self.head_dim,
            dim,
            bias=False,
            input_is_parallel=True,
        )

        self.norm=RMSNorm(params.dim,params.norm_eps)
        self.image_norm=RMSNorm(params.image_dim,params.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        images: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        x_norm=self.norm(x)
        images=self.image_norm(images)

        xq, xk, xv = self.wq(x_norm), self.wk(images), self.wv(images)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, self.params.patch_h*self.params.patch_w, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, self.params.patch_h*self.params.patch_w, self.n_local_kv_heads, self.head_dim)

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # # repeat k/v heads if n_kv_heads < n_heads   
        # xk = repeat_kv(xk, self.n_rep)  # (bs, image_seqlen, n_local_heads, head_dim)
        # xv = repeat_kv(xv, self.n_rep)  # (bs, image_seqlen, n_local_heads, head_dim)

        xq=xq.transpose(1,2)
        k=xk.transpose(1,2)
        v=xv.transpose(1,2)

        scores = torch.matmul(xq, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # if mask is not None:
        #     scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, v)  # (bs, n_local_heads, seqlen, head_dim)
        output_view = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        output=self.wo(output_view)

        return output+x_norm
        # return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, pretrain: bool):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, pretrain)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.pretrain=pretrain
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        autoregressive: bool = False,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, autoregressive)
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out

# class ConditionBlock(nn.Module):
#     def __init__(self, layer_id: int, args: ModelArgs):
#         super().__init__()
#         self.n_heads = args.n_heads
#         self.dim = args.dim
#         self.head_dim = args.dim // args.n_heads
#         self.attention = Image_Cross_Atten(args)
#         self.feed_forward = FeedForward(
#             dim=args.dim,
#             hidden_dim=4 * args.dim,
#             multiple_of=args.multiple_of,
#             ffn_dim_multiplier=args.ffn_dim_multiplier,
#         )
#         self.layer_id = layer_id
#         self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
#         self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

#     def forward(
#         self,
#         x: torch.Tensor,
#         images: torch.Tensor,
#         freqs_cis: torch.Tensor,
#     ):
        
#         h = x + self.attention.forward(self.attention_norm(x), images, freqs_cis)
#         # h = x + self.attention.forward(self.attention_norm(x), self.attention_norm(images), start_pos, freqs_cis, mask, autoregressive, pretrain)
#         out = h + self.feed_forward.forward(self.ffn_norm(h))

#         return out

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(MLP, self).__init__()
        self.fc1 = ColumnParallelLinear(input_dim, hidden_dim, bias=False)
        self.fc2 = ColumnParallelLinear(hidden_dim, output_dim, bias=False)
        
    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


# class CrossMLP(nn.Module):
#     def __init__(self, layer_id, input_dim, output_dim, hidden_dim=512):
#         super(MLP, self).__init__()
#         self.layer_id=layer_id
#         self.fc1 = ColumnParallelLinear(input_dim, hidden_dim, bias=False)
#         self.fc2 = ColumnParallelLinear(hidden_dim, output_dim, bias=False)
        
#     def forward(self, x):
#         return self.fc2(F.silu(self.fc1(x)))


class PositionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(PositionMLP, self).__init__()
        self.fc1 = ColumnParallelLinear(input_dim, 3*hidden_dim, bias=False)
        self.fc2 = ColumnParallelLinear(3*hidden_dim, hidden_dim, bias=False)
        self.fc3 = ColumnParallelLinear(hidden_dim, output_dim, bias=False)
        
    def forward(self, x):
        return self.fc3(F.silu(self.fc2(F.silu(self.fc1(x)))))

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, image_model = None):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        pretrain=False if image_model is not None else True

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.out_pad_dim=params.out_pad_dim

        if image_model is not None:
            self.image_model=image_model
            self.add_cross=params.add_cross
            self.image_layers = torch.nn.ModuleList()
            for layer_id in range(params.n_layers):
                self.image_layers.append(Image_Cross_Atten(
                                            layer_id=layer_id,
                                            params=params
                                            ))

        self.token_embeddings = ParallelEmbedding(params.vocab_size, params.dim-params.position_dim)
        self.position_embeddings = ColumnParallelLinear(12, params.position_dim, bias=False)
        self.cat_embeddings = MLP(params.dim, params.dim, 1024)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, pretrain))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.image_norm = RMSNorm(params.image_dim, eps=params.norm_eps)

        self.output_fusion = MLP(params.dim, params.dim)
        self.output_id = MLP(params.dim, params.vocab_size)

        # self.output_trans = MLP(params.dim, 3+self.out_pad_dim)
        # self.output_rot = MLP(params.dim, params.out_rot_dim)

        # self.output_position = PositionMLP(params.dim, 9+self.out_pad_dim)
        self.output_position = MLP(params.dim, 9+self.out_pad_dim)
        # self.output_position = MLP(params.dim, 12+self.out_pad_dim)

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        # self.image_freqs_cis = precompute_freqs_cis(
        #     self.params.image_dim // self.params.n_heads, self.params.max_seq_len * 2
        # )


    def forward(
            self,
            tokens: torch.Tensor, 
            tokens_position: torch.Tensor,
            images = None,
            start_pos: int = 0,
            autoregressive: bool = False,
            ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        
        h = self.token_embeddings(tokens)
        h_p=self.position_embeddings(tokens_position)
        # h:batch_size*pad_seq_len*token_embedding_dim
        # h_p:batch_size*pad_seq_len*position_embedding_dim
        h=torch.cat((h,h_p),dim=2)
        h=self.cat_embeddings(h)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        if images is not None and len(images) > 0:
            image_embeddings=self.image_model.forward_features(images)['x_norm_patchtokens']

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

            # image_mask=
        
        for i in range(len(self.layers)):
            h = self.layers[i](h, start_pos, freqs_cis, mask, autoregressive)
            if images is not None and len(images) > 0 and i%self.add_cross==0:
                h=self.image_layers[i].forward(h,image_embeddings,freqs_cis)
            # if images is not None and len(images) > 0:
            #     h=self.image_layers[i].forward(h,image_embeddings,freqs_cis)
            
            
        h = self.norm(h)
        # h=self.output_fusion(h)
        
        output_id=self.output_id(h).float()
        # output_trans=self.output_trans(h).float()
        # output_rot=self.output_rot(h).float()

        # if self.out_pad_dim>0:
        #     output_trans=output_trans[:,:,:-self.out_pad_dim]
        #     # output_rot=output_rot[:,:,:-self.out_pad_dim]
        

        output_position=self.output_position(h).float()
        if self.out_pad_dim>0:
            output_position=output_position[:,:,:-self.out_pad_dim]

        return output_id, output_position
        # return output_id, output_trans, output_rot
