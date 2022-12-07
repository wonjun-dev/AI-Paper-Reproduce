
# Define Transformer
# I follow style of official Pytorch Transformer source code and adjust it simply
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules import LayerNorm
from typing import Union, Callable, Optional, Any, Tuple


class Transformer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, \
        dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, \
        layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False, device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()
   
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, **factory_kwargs)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_params()

        self.d_model = d_model
        self.n_head = n_head
        self.batch_first = batch_first
    
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None, \
                memory_mask: Optional[Tensor] = None) -> Tensor:

        memory = self.encoder(src, mask=src_mask)
        output  = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output
    
    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_feedforward: int = 2048, dropout: float = 0.1, \
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,  
                device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        
        # Define params
        self.self_attn = MultiheadAttention(d_model, n_head, dropout, batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.droput = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
    
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._self_attn_block(self.norm1(src), src_mask)
            x = x + self._feedforward_block(self.norm2(x))
        else:
            x = self.norm1(x + self._self_attn_block(x, src_mask))
            x = self.norm2(x + self._feedforward_block(x))
        return x

    def _self_attn_block(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, mask=mask)
        return self.dropout1(x)
    
    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.droput(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_feedforward: int = 2048, droput: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, \
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()

        # Define params
        self.self_attn = MultiheadAttention(d_model, n_head, droput, batch_first, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, n_head, droput, batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.droput = nn.Dropout(droput)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(droput)
        self.dropout2 = nn.Dropout(droput)
        self.dropout3 = nn.Dropout(droput)

        self.activation = activation
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._self_attn_block(self.norm1(x), tgt_mask)
            x = x + self._multihead_attn_block(self.norm2(x), memory, memory_mask)
            x = x + self._feedforward_block(self.norm3(x))
        else:
            x = self.norm1(x + self._self_attn_block(x, tgt_mask))
            x = self.norm2(x + self._multihead_attn_block(x, memory, memory_mask))
            x = self.norm3(x + self._feedforward_block(x))
        return x
    
    def _self_attn_block(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.self_attn(x, x, x, mask=mask)
        return self.dropout1(x)

    def _multihead_attn_block(self, x: Tensor, mem: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.multihead_attn(x, mem, mem, mask=mask)
        return self.dropout2(x)
    
    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.droput(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = int(embed_dim // num_heads)
        assert self.embed_dim == self.head_dim * num_heads

        self.wq = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.wk = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.wv = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key ,value = self._split_heads(query), self._split_heads(key), self._split_heads(value)
        attn_out = self._attention(query, key, value, mask)
        attn_out = self.out_proj(attn_out)
        return attn_out
    
    def _split_heads(self, proj: Tensor) -> Tensor:
        if self.batch_first:    # (N, L, E)
            bs = proj.size(0)
            proj = proj.view(bs, -1, self.num_heads, self.head_dim)    # (N, L, H, E_Hi)
            proj = proj.transpose(1, 2) # (N, H, L, E_Hi)
        else:   # (L, N, E)
            bs = proj.size(1)
            proj = proj.view(-1, bs, self.num_heads, self.head_dim)    # (L, N, H, E_Hi)
        return proj
    
    def _attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.batch_first:
            score = torch.matmul(query, key.transpose(-1, -2))  # (N, H, QL, KL), score = Q * K^T
            score = score / math.sqrt(query.size(-1)) # score = Q * K^T divided by sqrt(E_Hi)
            
            if mask is not None:
                score = score.masked_fill(mask == 0, -1e10)

            softmax = F.softmax(score, dim=-1)
            attn_out = torch.matmul(self.dropout(softmax), value)   # (N, H, QL, E_Hi)
            attn_out = attn_out.transpose(1, 2) # (N, QL, H, E_Hi)
            attn_out = attn_out.contiguous().view(attn_out.size(0), -1, self.embed_dim)  # (N, QL, E)
            return attn_out
        else:
            raise NotImplementedError()


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
