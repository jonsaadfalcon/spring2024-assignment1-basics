
import os
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch
import math
import numpy as np

import torch
import torch.nn.functional as F

########################################################

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

########################################################

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

########################################################

def gelu(in_features: torch.FloatTensor) -> torch.FloatTensor:
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

########################################################

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
    first_linear_transformation_output = gelu(first_linear_transformation_output)
    output = first_linear_transformation_output@w2_weights.t()
    
    return output

########################################################

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
        #breakpoint()
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

########################################################

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

def SDPA(Q, K, V, mask, pdrop):
    import torch.nn.functional as F
    query_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
    query_scores.masked_fill_(mask, -1e9) if mask is not None else query_scores
    softmax_scores = softmax(query_scores, dim=-1)
    dropout_scores = F.dropout(input=softmax_scores, p=pdrop) if pdrop is not None else softmax_scores
    final_attention_output = torch.matmul(dropout_scores, V)
    return final_attention_output

def softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
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

########################################################