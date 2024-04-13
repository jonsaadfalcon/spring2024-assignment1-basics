#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch
import math
import numpy as np

import torch
import torch.nn.functional as F


def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):

    from torch import mean
    
    stable_logits = inputs - torch.max(inputs, dim=1, keepdim=True)[0]
    sum_logits = torch.sum(torch.exp(stable_logits), dim=1)
    sum_of_log_exp = torch.log(sum_logits)

    logits_of_true_class = stable_logits.gather(dim=1, index=targets.unsqueeze(1))
    logits_of_true_class = logits_of_true_class.squeeze()
    
    loss_per_example = sum_of_log_exp - logits_of_true_class
    cross_entropy_mean = mean(loss_per_example)
    
    return cross_entropy_mean


def rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
    weight_key="weight"
) -> torch.FloatTensor:
    from torch.nn import Parameter
    
    rms_norm = torch.sqrt(eps + torch.mean(in_features ** 2, dim=-1, keepdim=True))

    features_normalized = in_features / rms_norm
    final_output = Parameter(weights[weight_key]) * features_normalized
    
    return final_output



def positionwise_feedforward(
    d_model: int,
    d_ff: int,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
    weight_1="w1.weight",
    weight_2="w2.weight",
) -> torch.FloatTensor:
    
    import numpy as np

    w1_weights = weights[weight_1]
    w2_weights = weights[weight_2]

    #breakpoint()

    first_linear_transformation_output = in_features@w1_weights.t()
    first_linear_transformation_output = run_gelu(first_linear_transformation_output)
    output = first_linear_transformation_output@w2_weights.t()
    
    return output


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
    weight_keys: dict[str, str],
) -> torch.FloatTensor:
    
    if weight_keys is None:
        breakpoint()
        weight_keys = {
            "rms_norm_1": "ln1.weight",
            "rms_norm_2": "ln2.weight",
            "positionwise_feedforward_1": "ffn.w1.weight",
            "positionwise_feedforward_2": "ffn.w2.weight",
        }

    eps = 1e-5

    rms_norm_output = rmsnorm(d_model=d_model, eps=eps, in_features=in_features, weights=weights, weight_key=weight_keys["rms_norm_1"])
    attention_output = multihead_self_attention(d_model, num_heads, attn_pdrop, weights, rms_norm_output, weight_keys=weight_keys)
    causal_attention_output = F.dropout(attention_output, p=residual_pdrop, inplace=False)
    attention_output = in_features + causal_attention_output

    rms_norm_output = rmsnorm(d_model=d_model, eps=eps, in_features=attention_output, weights=weights, weight_key=weight_keys["rms_norm_2"])
    position_feedforward_output = positionwise_feedforward(d_model, d_ff, weights=weights, in_features=rms_norm_output,
                                                           weight_1=weight_keys["positionwise_feedforward_1"], 
                                                           weight_2=weight_keys["positionwise_feedforward_2"])
    position_feedforward_output = F.dropout(position_feedforward_output, p=residual_pdrop, inplace=False)
    final_output = attention_output + position_feedforward_output

    return final_output





def multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
    weight_keys: dict[str, str] = None,
) -> torch.FloatTensor:
    
    
    import torch.nn.functional as F

    #breakpoint()

    d_key =  d_model // num_heads
    batch_size, seq_length, _ = in_features.size()

    try:
        Q_weights = torch.cat([weights[f"q_heads.{row}.weight"] for row in range(num_heads)], dim=0)
        K_weights = torch.cat([weights[f"k_heads.{row}.weight"] for row in range(num_heads)], dim=0)
        V_weights = torch.cat([weights[f"v_heads.{row}.weight"] for row in range(num_heads)], dim=0)
    except:
        try:
            Q_weights = weights[f"attn.q_proj.weight"]
            K_weights = weights[f"attn.k_proj.weight"] 
            V_weights = weights[f"attn.v_proj.weight"]
        except:
            Q_weights = weights[weight_keys["q_proj"]]
            K_weights = weights[weight_keys["k_proj"]] 
            V_weights = weights[weight_keys["v_proj"]]

    query_output = torch.matmul(in_features, Q_weights.transpose(0, 1)).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)
    key_output = torch.matmul(in_features, K_weights.transpose(0, 1)).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)
    value_output = torch.matmul(in_features, V_weights.transpose(0, 1)).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)

    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1) > 0
    attention_output = SDPA(query_output, key_output, value_output, 
                            mask, pdrop=attn_pdrop)
    attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
    try:
        final_attention_output = torch.matmul(attention_output, weights["output_proj.weight"].transpose(0, 1))
    except:
        try:
            final_attention_output = torch.matmul(attention_output, weights["attn.output_proj.weight"].transpose(0, 1))
        except:
            final_attention_output = torch.matmul(attention_output, weights[weight_keys["output_proj"]].transpose(0, 1))

    return final_attention_output


########################################################################

def SDPA(Q, K, V, mask, pdrop):
    import torch.nn.functional as F
    query_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
    query_scores.masked_fill_(mask, -1e9) if mask is not None else query_scores
    softmax_scores = run_softmax(query_scores, dim=-1)
    dropout_scores = F.dropout(input=softmax_scores, p=pdrop) if pdrop is not None else softmax_scores
    final_attention_output = torch.matmul(dropout_scores, V)
    return final_attention_output


def run_positionwise_feedforward(
    d_model: int,
    d_ff: int,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a position-wise feedforward network, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are `w1.weight` and `w2.weight`.
            `w1` is the first linear transformation, and `w2` is the second
            linear transformation (eq. 2 of Vaswani et al., 2017).
            `w1.weight` is of shape (d_ff, d_model).
            `w2.weight` is of shape (d_model, d_ff).
    )
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your position-wise feedforward network
        with the provided `weights` on the provided `in_features`.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # my_ffn.load_state_dict(weights)
    # You can also manually assign the weights
    # my_ffn.w1.weight.data = weights["w1.weight"]
    # my_ffn.w2.weight.data = weights["w2.weight"]

    import numpy as np

    return positionwise_feedforward(d_model, d_ff, weights, in_features)


    raise NotImplementedError


def run_scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    return SDPA(Q, K, V, mask, pdrop)

    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `q_heads.{N}.weight`, `q_heads.{N}.weight`:
                Weights for the query projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `k_heads.{N}.weight`, `k_heads.{N}.weight`:
                Weights for the key projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `v_heads.{N}.weight`, `v_heads.{N}.weight`:
                Weights for the value projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_value, d_model).
            - `output_proj.weight`:
                Weight of the output projection
                (W^{O} in the original Transformer paper)
                Shape of (d_model, d_value * num_heads).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    import torch.nn.functional as F

    #breakpoint()

    return multihead_self_attention(d_model, num_heads, attn_pdrop, weights, in_features)
        

    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    Args:
        d_model: int
            The dimensionality of the Transformer block input.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the output of each sub-layer, before it
            is added to the sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
            Shape is (batch_size, sequence_length, d_model).

    Returns:
        FloatTensor of shape (batch_size, sequence_length, d_model) with the output of
        running the Transformer block on the input features.
    """

    import torch.nn.functional as F

    weight_keys = None
    return transformer_block(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, weights, in_features, weight_keys=weight_keys)

    


    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the sum of the token and position embeddings
            as well as the output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `position_embeddings.weight`
                Positional embedding matrix. Shape is (context_length, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for layernorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices: torch.LongTensor
            Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the predicted unnormalized
        next-word distribution for each token.
    """

    import torch.nn.functional as F


    
    #breakpoint()
    token_embeddings = weights['token_embeddings.weight'][in_indices]
    position_ids = torch.arange(in_indices.shape[1]).repeat(in_indices.shape[0], 1)
    position_embeddings = weights['position_embeddings.weight'][position_ids]
    input_embeddings = token_embeddings + position_embeddings
    input_embeddings = F.dropout(input_embeddings, p=residual_pdrop, inplace=False)

    current_hidden_state = input_embeddings
    for layer_number in range(num_layers):
        #breakpoint()
        weight_keys = {
            "rms_norm_1": f"layers.{layer_number}.ln1.weight",
            "rms_norm_2": f"layers.{layer_number}.ln2.weight",
            "positionwise_feedforward_1": f"layers.{layer_number}.ffn.w1.weight",
            "positionwise_feedforward_2": f"layers.{layer_number}.ffn.w2.weight",
            "q_proj": f"layers.{layer_number}.attn.q_proj.weight",
            "k_proj": f"layers.{layer_number}.attn.k_proj.weight",
            "v_proj": f"layers.{layer_number}.attn.v_proj.weight",
            "output_proj": f"layers.{layer_number}.attn.output_proj.weight",
        }
        
        current_hidden_state = transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, attn_pdrop=attn_pdrop, 
                                                 residual_pdrop=residual_pdrop, weights=weights, in_features=current_hidden_state, 
                                                 weight_keys=weight_keys)

    
    #breakpoint()

    from torch.nn import Linear, Parameter
    linear_transformation = Linear(d_model, vocab_size)
    linear_transformation.weight = Parameter(weights['lm_head.weight'])

    rms_norm_output = rmsnorm(d_model=d_model, eps=1e-5, weights=weights, in_features=current_hidden_state, weight_key="ln_final.weight")
    #linear_output = torch.matmul(rms_norm_output, weights['lm_head.weight'].t())
    linear_output = linear_transformation(rms_norm_output)
    softmax_output = run_softmax(linear_output, dim=-1)
    
    return softmax_output

    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor
) -> torch.FloatTensor:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model: int
            The dimensionality of the layernorm input.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `weight`
                Weights of the RMSNorm affine transform.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Input features to run RMSNorm on. Tensor of (*, d_model), where *
            can be an arbitrary number of dimensions with arbitrary values.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of running
        layernorm of the `in_features`.
    """
    from torch.nn import Parameter
    
    return rmsnorm(d_model, eps, weights, in_features)

    breakpoint()

    #raise NotImplementedError


def run_gelu(in_features: torch.FloatTensor) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of applying GELU
    to each element.

    Args:
        in_features: torch.FloatTensor
            Input features to run GELU on. Shape is arbitrary.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of applying
        GELU to each element.
    """
    import math 
    from torch import tanh
    import numpy as np
    #breakpoint()
    
    return in_features * (0.5) * (1 + torch.erf(in_features / np.sqrt(2)))
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features: torch.FloatTensor
            Input features to softmax. Shape is arbitrary.
        dim: int
            Dimension of the `in_features` to apply softmax to.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    import numpy as np
    from torch import exp, max, sum

    output = in_features - max(in_features, dim=dim, 
                               keepdim=True)[0]
    divided_output = exp(output) / sum(exp(output), 
                                       dim=dim, keepdim=True)
    return divided_output




    raise NotImplementedError


def run_cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """

    return cross_entropy(inputs, targets)


    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """
    raise NotImplementedError


def get_adamw_cls() -> Type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters: int
            T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model: torch.nn.Module
            Serialize the state of this model.
        optimizer: torch.optim.Optimizer,
            Serialize the state of this optimizer.
        iteration: int
            Serialize this value, which represents the number of training iterations
            we've completed.
        out: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialized checkpoint.
        model: torch.nn.Module
            Restore the state of this model.
        optimizer: torch.optim.Optimizer,
            Restore the state of this optimizer.
    Returns:
        int, the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
):
    """Given the path to a JSON vocab, a file with BPE merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab: dict[int, bytes]
            The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens: Optional[list[str]]
            A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    #raise NotImplementedError
    from tests.tokenizer import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    #raise NotImplementedError

    from collections import defaultdict, Counter
    from tqdm import tqdm
    import regex as re
    from tests.tokenizer import add_new_tokens, most_common_bp, update_frequencies
    
    ###################################################

    vocabulary = {}
    for i in range(0, 256):
        vocabulary.update({i: bytes([i])})

    ###################################################
    
    starting_vocab_len = len(vocabulary)
    for special_token_count, token in enumerate(special_tokens) :
        if token.encode("utf-8") not in vocabulary.values():
            vocabulary.update({special_token_count + starting_vocab_len: token.encode("utf-8")})

    ###################################################

    token_freq = Counter()
    pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    print("Loading text!")
    with open(input_path, 'rb') as f:
        
        for line in tqdm(f.readlines()):
            
            tokens_converted = []
            for token in re.findall(pretokenization_pattern, line.decode("utf-8")):
                if token.encode() not in vocabulary.values():
                    tokens_converted.append(token.encode())

            ####################
                    
            token_freq.update(tokens_converted)

            ####################

            token_mapping = {}
            for token in token_freq.keys():
                token_mapping[token] = token

    ###################################################

    pairs_of_bytes = {}
    byte_pair_freqs = Counter()

    print("Gathering frequencies!")
    for token, freq in tqdm(token_freq.items()):
        
        pairs_of_bytes_in_token = []
        for curr_token in range(len(token) - 1):
            current_pair = (token[curr_token], token[curr_token + 1])
            pairs_of_bytes_in_token.append(current_pair)

        for token_pair in pairs_of_bytes_in_token:
            pairs_of_bytes = add_new_tokens(token, token_pair, pairs_of_bytes)
            byte_pair_freqs[token_pair] = byte_pair_freqs[token_pair] + freq

    ###################################################

    merges_overall = []
    
    print("Performing merges!")
    for merge_count in tqdm(range(len(vocabulary), vocab_size)):

        most_common_pair = most_common_bp(vocabulary, byte_pair_freqs)

        vocabulary[len(vocabulary)] = (vocabulary[most_common_pair[0]] + vocabulary[most_common_pair[1]])
        merges_overall.append((vocabulary[most_common_pair[0]], vocabulary[most_common_pair[1]]))

        byte_pair_freqs[most_common_pair] = 0
        for word_token in pairs_of_bytes[most_common_pair]:

            final_merged_token = list(token_mapping[word_token],)
            count = 0
            while count < len(final_merged_token) - 1: 

                token_1, token_2 = final_merged_token[count], final_merged_token[count + 1]

                if not (token_1, token_2) == most_common_pair:
                    count += 1
                else:
                    final_merged_token = final_merged_token[:count] + [len(vocabulary) - 1] + final_merged_token[count + 2:]

            token_mapping[word_token] = final_merged_token

            new_id_for_merged_token = len(vocabulary) - 1
            byte_pair_freqs, pairs_of_bytes = update_frequencies(new_id_for_merged_token, final_merged_token, token_freq, byte_pair_freqs, 
                                                                 pairs_of_bytes, word_token, most_common_pair[0], most_common_pair[1])

    return vocabulary, merges_overall


