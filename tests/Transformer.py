
import os
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch
import math
import numpy as np

import torch
import torch.nn.functional as F

from numpy import random, zeros, int32
from torch import tensor, long

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

import torch.nn as nn
class Transformer_Block(nn.Module):

    def __init__(self, d_model:int, num_heads:int, d_ff:int, attn_pdrop:float, residual_pdrop:float, 
                 weights:dict[str, torch.FloatTensor], weight_keys: dict[str, str]):
        
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.weights = weights
        self.weights_keys = weight_keys
        
        super(transformer_block, self).__init__()

    ################################################

    def forward(self, in_features: torch.FloatTensor):
    
        eps = 1e-5

        rms_norm_output = rmsnorm(d_model=self.d_model, eps=eps, in_features=in_features, weights=self.weights, weight_key=self.weight_keys["rms_norm_1"])
        attention_output = multihead_self_attention(self.d_model, self.num_heads, self.attn_pdrop, self.weights, rms_norm_output, weight_keys=self.weight_keys)
        causal_attention_output = F.dropout(attention_output, p=self.residual_pdrop, inplace=False)
        attention_output = in_features + causal_attention_output

        rms_norm_output = rmsnorm(d_model=self.d_model, eps=eps, in_features=attention_output, weights=self.weights, weight_key=self.weight_keys["rms_norm_2"])
        position_feedforward_output = positionwise_feedforward(self.d_model, self.d_ff, weights=self.weights, in_features=rms_norm_output,
                                                            weight_1=self.weight_keys["positionwise_feedforward_1"], 
                                                            weight_2=self.weight_keys["positionwise_feedforward_2"])
        position_feedforward_output = F.dropout(position_feedforward_output, p=self.residual_pdrop, inplace=False)
        final_output = attention_output + position_feedforward_output

        return final_output
    
########################################################

class Transformer_LM(nn.Module):

    def __init__(self,
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
        weight_keys: dict[str, str]
    ):

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.weights = weights
        self.in_indices = in_indices
        self.weights_keys = weight_keys

        super(Transformer_LM, self).__init__()

    ################################################

    def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str, # | os.PathLike | BinaryIO | IO[bytes],
    ):
        #raise NotImplementedError
        checkpoint = {
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'iteration': iteration,
        }
        torch.save(checkpoint, out)

    ################################################
    
    def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        #raise NotImplementedError
        
        loaded_checkpoint = torch.load(src)
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        model.load_state_dict(loaded_checkpoint['model_state_dict'])

        return loaded_checkpoint['iteration']


    ########################################################

    def load_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:

        #raise NotImplementedError
    
        inputs = zeros((batch_size, context_length)) #, dtype=int32
        target_labels = zeros((batch_size, context_length)) #, dtype=int32

        ################################################
        
        starting_index = random.randint(0, len(dataset) - context_length, batch_size)
        for row, start_index in enumerate(starting_index):
            
            inputs[row] = dataset[start_index : start_index + context_length]
            target_labels[row] = dataset[start_index + 1 : start_index + context_length + 1]

        ################################################
        
        return tensor(inputs, dtype=long, device=device), tensor(target_labels, dtype=long, device=device)

    ########################################################

    def forward(self,
                in_indices: torch.LongTensor,):
        
        token_embeddings = self.weights['token_embeddings.weight'][in_indices]
        position_ids = torch.arange(in_indices.shape[1]).repeat(in_indices.shape[0], 1)
        position_embeddings = self.weights['position_embeddings.weight'][position_ids]
        input_embeddings = token_embeddings + position_embeddings
        input_embeddings = F.dropout(input_embeddings, p=self.residual_pdrop, inplace=False)

        current_hidden_state = input_embeddings
        for layer_number in range(self.num_layers):
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
            
            current_hidden_state = transformer_block(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, attn_pdrop=self.attn_pdrop, 
                                                    residual_pdrop=self.residual_pdrop, weights=self.weights, in_features=current_hidden_state, 
                                                    weight_keys=weight_keys)

        
        #breakpoint()

        from torch.nn import Linear, Parameter
        linear_transformation = Linear(self.d_model, self.vocab_size)
        linear_transformation.weight = Parameter(self.weights['lm_head.weight'])

        rms_norm_output = rmsnorm(d_model=self.d_model, eps=1e-5, weights=self.weights, in_features=current_hidden_state, weight_key="ln_final.weight")
        #linear_output = torch.matmul(rms_norm_output, weights['lm_head.weight'].t())
        linear_output = torch.nn.functional.linear(rms_norm_output, self.weights['lm_head.weight'])
        #softmax_output = run_softmax(linear_output, dim=-1)
        
        
        return linear_output
        