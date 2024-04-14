
import os
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter, Embedding

from numpy import random, zeros, int32
from torch import tensor, long, mean

from transformers.modeling_outputs import CausalLMOutput

########################################################

def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):

    #breakpoint()

    if len(inputs.shape) == 3:
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)

    assert inputs.size(0) == targets.size(0)
    
    stable_logits = inputs - torch.max(inputs, dim=1, keepdim=True)[0]
    sum_logits = torch.sum(torch.exp(stable_logits), dim=1)
    sum_of_log_exp = torch.log(sum_logits)

    logits_of_true_class = torch.gather(stable_logits, dim=1, index=targets.unsqueeze(1)).squeeze(1)
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
















########################################################

import torch.nn as nn

class positionwise_feedforward_params(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        weights: dict[str, torch.FloatTensor],
        weight_1="w1.weight",
        weight_2="w2.weight",
        device: str="cuda:0",
    ) -> torch.FloatTensor:
        
        super(positionwise_feedforward_params, self).__init__()
        
        self.d_model = d_model
        self.dff = d_ff
        self.device = torch.device(device)

        self.w1_weights = Parameter(torch.randn(d_ff, d_model)).to(self.device)
        self.w2_weights = Parameter(torch.randn(d_model, d_ff)).to(self.device)

        if weights is not None:
            self.w1_weights.weights = weights[weight_1]
            self.w2_weights.weights = weights[weight_2]
            

    ########################################################

    def perform_positionwise_feedforward(self, in_features: torch.FloatTensor):

        first_linear_transformation_output = in_features @ self.w1_weights.t()
        first_linear_transformation_output = gelu(first_linear_transformation_output)
        output = first_linear_transformation_output @ self.w2_weights.t()
        
        return output
    
########################################################

class multihead_self_attention_params(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_pdrop: float,
        weights: dict[str, torch.FloatTensor] | None,
        weight_keys: dict[str, str] | None,
        device: str="cuda:0",
    ) -> torch.FloatTensor:
        
        super(multihead_self_attention_params, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.weights = weights
        self.weight_keys = weight_keys
        self.device = torch.device(device)

        self.Q_weights = Parameter(torch.randn(self.d_model, self.d_model)).to(self.device)
        self.K_weights = Parameter(torch.randn(self.d_model, self.d_model)).to(self.device)
        self.V_weights = Parameter(torch.randn(self.d_model, self.d_model)).to(self.device)
        self.output_proj = Parameter(torch.randn(self.d_model, self.d_model)).to(self.device)

        if weights is not None:
            self.Q_weights.weights = weights[weight_keys["q_proj"]]
            self.K_weights.weights = weights[weight_keys["k_proj"]] 
            self.V_weights.weights = weights[weight_keys["v_proj"]]
            self.output_proj.weights = weights[weight_keys["output_proj"]]

    ########################################

    def perform_multihead_self_attention(self, 
                                         d_model: int,
                                         num_heads: int,
                                         attn_pdrop: float,
                                         in_features: torch.FloatTensor):
        
        ########################################
    
        d_key =  d_model // num_heads
        batch_size, seq_length, _ = in_features.size()

        ########################################

        query_output = torch.matmul(in_features, self.Q_weights.transpose(0, 1)).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)
        key_output = torch.matmul(in_features, self.K_weights.transpose(0, 1)).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)
        value_output = torch.matmul(in_features, self.V_weights.transpose(0, 1)).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)
        #query_output = self.Q_weights(in_features).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)
        #key_output = self.K_weights(in_features).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)
        #value_output = self.V_weights(in_features).view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)

        ########################################

        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1) > 0
        attention_output = SDPA(query_output, key_output, value_output, 
                                mask.to(self.device), pdrop=attn_pdrop)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        final_attention_output = torch.matmul(attention_output, self.output_proj.transpose(0, 1))
        #final_attention_output = self.output_proj(attention_output)

        ########################################

        return final_attention_output
    
########################################################

class rmsnorm_params(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        eps: float,
        weights: dict[str, torch.FloatTensor] | None,
        weight_key="weight",
        device: str="cuda:0",
    ) -> torch.FloatTensor:
        
        super(rmsnorm_params, self).__init__()
        
        self.eps = eps
        self.weights = weights
        self.d_model = d_model
        self.device = torch.device(device)

        if self.weights is not None:
            self.rmsnorm = Parameter(weights[weight_key]).to(self.device)
        else:
            self.rmsnorm = Parameter(torch.randn(self.d_model)).to(self.device)

    def perform_rmsnorm(self, in_features: torch.FloatTensor):
    
        current_state = torch.sqrt(self.eps + torch.mean(in_features ** 2, dim=-1, keepdim=True))

        features_normalized = in_features / current_state
        final_output = self.rmsnorm * features_normalized
        
        return final_output
    
########################################################

import torch.nn as nn
class transformer_block_params(nn.Module):

    def __init__(self, d_model:int, num_heads:int, d_ff:int, attn_pdrop:float, residual_pdrop:float, 
                 weights:dict[str, torch.FloatTensor], weight_keys: dict[str, str], eps: float=1e-5, device:str="cuda:0"):
        
        super(transformer_block_params, self).__init__()
        
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.weights = weights
        self.weights_keys = weight_keys
        self.eps = eps

        self.device = torch.device(device)

        #breakpoint()

        self.first_rms_norm = rmsnorm_params(d_model=self.d_model, eps=self.eps, weights=self.weights, weight_key=weight_keys["rms_norm_1"], device=device)
        self.multihead_self_attention = multihead_self_attention_params(d_model=self.d_model, num_heads=self.num_heads, attn_pdrop=self.attn_pdrop, weights=self.weights, weight_keys=weight_keys, device=device)
        self.second_rms_norm = rmsnorm_params(d_model=self.d_model, eps=self.eps, weights=self.weights, weight_key=weight_keys["rms_norm_2"], device=device)
        self.positionwise_feedforward = positionwise_feedforward_params(d_model=self.d_model, d_ff=self.d_ff, weights=self.weights, weight_1=weight_keys["positionwise_feedforward_1"], weight_2=weight_keys["positionwise_feedforward_2"], device=device)

    ################################################

    def forward(self, in_features: torch.FloatTensor):

        rms_norm_output = self.first_rms_norm.perform_rmsnorm(in_features=in_features)
        attention_output = self.multihead_self_attention.perform_multihead_self_attention(d_model=self.d_model, num_heads=self.num_heads, attn_pdrop=self.attn_pdrop, in_features=rms_norm_output)
        causal_attention_output = F.dropout(attention_output, p=self.residual_pdrop, inplace=False)
        attention_output = in_features + causal_attention_output

        ###############################################################

        rms_norm_output = self.second_rms_norm.perform_rmsnorm(in_features=attention_output)
        position_feedforward_output = self.positionwise_feedforward.perform_positionwise_feedforward(in_features=rms_norm_output)
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
        weights: dict[str, torch.FloatTensor] | None,
        eps: float=1e-5,
        #in_indices: torch.LongTensor,
        #weight_keys: dict[str, str] | None,
        device: str = "cuda:0"
    ):
        
        super(Transformer_LM, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.weights = weights
        self.eps = eps
        #self.in_indices = in_indices
        #self.weights_keys = weight_keys
        self.device = torch.device(device)

        ########################

        if self.weights is not None:
            self.token_embeddings = Embedding(self.vocab_size, self.d_model).to(self.device)
            self.token_embeddings.weights = weights['token_embeddings.weight']
            self.position_embeddings = Embedding(self.context_length, self.d_model).to(self.device)
            self.position_embeddings.weights = weights['token_embeddings.weight']
        else:
            self.token_embeddings = Embedding(self.vocab_size, self.d_model).to(self.device)
            self.token_embeddings.weights = Parameter(torch.randn(self.vocab_size, self.d_model))
            self.position_embeddings = Embedding(self.context_length, self.d_model).to(self.device)
            self.position_embeddings.weights = Parameter(torch.randn(self.context_length, self.d_model))

        ########################

        self.transformer_blocks = []
        for layer_number in range(self.num_layers):
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
            self.transformer_blocks.append(transformer_block_params(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, attn_pdrop=self.attn_pdrop, 
                                                                    residual_pdrop=self.residual_pdrop, weights=self.weights, weight_keys=weight_keys, eps=self.eps, device=device))

        ########################

        self.final_rms_norm = rmsnorm_params(d_model=self.d_model, eps=self.eps, weights=self.weights, weight_key="ln_final.weight", device=device)
        
        if self.weights is not None:
            self.linear_transformation = Linear(self.d_model, self.vocab_size, bias=False).to(self.device)
            self.linear_transformation.weight = Parameter(self.weights['lm_head.weight'])
        else:
            self.linear_transformation = Linear(self.d_model, self.vocab_size, bias=False).to(self.device)


    ################################################

    def forward(self,
                in_indices: torch.LongTensor,
                labels: torch.LongTensor | None = None,):
        
        ########################################################

        #print("in_indices: ", in_indices.shape)
        in_indices = in_indices.to(self.token_embeddings.weight.device)
        token_embeddings = self.token_embeddings(in_indices)
            
        position_ids = torch.arange(in_indices.shape[1]).repeat(in_indices.shape[0], 1).to(self.position_embeddings.weight.device)
        position_embeddings = self.position_embeddings(position_ids)

        input_embeddings = token_embeddings + position_embeddings
        input_embeddings = F.dropout(input_embeddings, p=self.residual_pdrop, inplace=False)

        ########################################################

        current_hidden_state = input_embeddings
        for layer_number in range(self.num_layers):
            current_hidden_state = self.transformer_blocks[layer_number](in_features=current_hidden_state)

        ########################################################

        rms_norm_output = self.final_rms_norm.perform_rmsnorm(in_features=current_hidden_state)

        linear_output = self.linear_transformation(rms_norm_output)

        #breakpoint()

        if labels is not None:
            loss = cross_entropy(linear_output, labels)
            return CausalLMOutput(
                loss=loss,
                logits=linear_output,
                hidden_states=None,
                attentions=None
            )
        else:
            return linear_output
    
    ########################################################
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
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
        self,
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

    def decode_from_logits(self, logits, tokenizer, end_of_text_token_id):

        breakpoint()
        
        batch_size, seq_length, vocab_size = logits.size()
        assert batch_size == 1, "Batch size should be 1 for this decoding function."

        # Reshape logits to [seq_length, vocab_size] for easier processing
        logits = logits.view(seq_length, vocab_size)

        # Decode each position in the sequence
        decoded_tokens = []
        for i in range(seq_length):
            token_probabilities = torch.softmax(logits[i], dim=0)
            next_token_id = torch.argmax(token_probabilities).item()  # Greedy decoding
            decoded_tokens.append(next_token_id)
            
            if next_token_id == end_of_text_token_id:
                break

        # Convert token ids to text
        decoded_text = tokenizer.decode(decoded_tokens)
        return decoded_text
        