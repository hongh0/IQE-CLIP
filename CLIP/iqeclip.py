from collections import OrderedDict
from typing import Tuple, Union, Optional, List
import math
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, Tensor
from .tokenizer import tokenize
import warnings
from .iqm import IQM,IQMConfig
from torch.overrides import (
    has_torch_function, has_torch_function_variadic,
    handle_torch_function)



def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor: 
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(linear, (input, weight, bias), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)

def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
    ):
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)

    assert w_q.shape == (Eq,Eq)
    assert w_k.shape == (Eq, Ek)
    assert w_v.shape == (Eq,Ev)
    assert b_q is None or b_q.shape == (Eq,)
    assert b_k is None or b_k.shape == (Eq,)
    assert b_v is None or b_v.shape == (Eq,)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)  

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v: 
        if q is k:
            return linear(q, w, b).chunk(3, dim=-1)  # L,E E,3E   = L,3E 
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None 
            else:
                b_q, b_kv = b.split([E, E * 2])
            
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)  
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None 
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q,w_q,b_q), linear(k,w_k,b_k), linear(v, w_v, b_v)



class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor) :
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x:torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class NonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features:int, out_features:int, bias:bool = True, device = None, dtype = None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)


class MultiheadAttention(nn.Module):

    __constants__ = ['batch_first']  
    bias_k : Optional[torch.Tensor] 
    bias_v: Optional[torch.Tensor]  

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn = False,
                 kdim = None, vdim = None, batch_first = False, device = None, dtype = None):
        #super().__init__()
        factory_kwards = {'device':device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim 
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim 
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim 

        self.num_heads = num_heads 

        self.dropout = dropout 
        self.batch_first = batch_first 
        self.head_dim = embed_dim // num_heads 

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim // num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwards))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwards))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwards))

            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwards))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwards))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias = bias, **factory_kwards)
        if add_bias_kv :
            self.bias_k = Parameter(torch.empty((1,1,embed_dim), **factory_kwards))
            self.bias_v = Parameter(torch.empty((1,1,embed_dim), **factory_kwards))
        else:
            self.bias_k = self.bias_v = None 

        self.add_zero_attn = add_zero_attn 

        self._reset_parameters()
    
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)
    
    def initialize_model_params(self,model):  
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
    
    def Scaled_dot_product_attention(self, q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,):
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.

        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            dropout_p: dropout probability. If greater than 0.0, dropout is applied.

        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.

            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns) 
        attn = torch.bmm(q, k.transpose(-2, -1))  
        if attn_mask is not None:
            attn += attn_mask
        attn = torch.nn.functional.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

    def Multi_head_attention_forward(self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None
        ):
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
        if has_torch_function(tens_ops):
            return handle_torch_function(
                self.multi_head_attention_forward,
                tens_ops,
                query,
                key,
                value,
                embed_dim_to_check,
                num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                add_zero_attn,
                dropout_p,
                out_proj_weight,
                out_proj_bias,
                training=training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                static_k=static_k,
                static_v=static_v,
            )
        
        tgt_len, bsz, embed_dim = query.shape

        src_len,_,_ = key.shape 
        assert embed_dim == embed_dim_to_check 

        if isinstance(embed_dim, torch.Tensor):
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        
        assert head_dim * num_heads  == embed_dim 

        if use_separate_proj_weight:
            assert key.shape[:2] == value.shape[:2]
        else:
            assert key.shape == value.shape 

        if not use_separate_proj_weight:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:  
            assert q_proj_weight is not None 
            assert k_proj_weight is not None
            assert v_proj_weight is not None 
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        

        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
        
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool 
            
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
        

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])  
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = nn.functional.pad(attn_mask, (0, 1)) 
            if key_padding_mask is not None:
                key_padding_mask = nn.functional.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None
        


        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)   
        if static_k is None:
            k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, \
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == head_dim, \
                f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, \
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, \
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)  
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = nn.functional.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = nn.functional.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)


        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        attn_output, attn_output_weights = self.Scaled_dot_product_attention(q, k, v, attn_mask, dropout_p) 
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)   
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None




    def forward (self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None,
                 need_weights:bool= True, attn_mask: Optional[Tensor] = None):
        

        if self.batch_first:
            query, key, value = [x.transpose(1,0) for x in (query, key, value)]
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.Multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, 
                self.in_proj_weight , self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias, 
                training = self.training,
                key_padding_mask = key_padding_mask, need_weights = need_weights, 
                attn_mask = attn_mask, use_separate_proj_weight = True, 
                q_proj_weight = self.q_proj_weight, k_proj_weight = self.k_proj_weight,
                v_proj_weight = self.v_proj_weight)
        
        else:
            attn_output, attn_output_weights = self.Multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, 
                self.in_proj_weight , self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias, 
                training = self.training,
                key_padding_mask = key_padding_mask, need_weights = need_weights, 
                attn_mask = attn_mask)
            
        if self.batch_first:
            return attn_output.transpose(1,0), attn_output_weights 
        else:
            return attn_output, attn_output_weights
        

class VVAttention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        k = v
        q = k

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model:int, n_head: int, attn_mask:torch.Tensor = None):
        super().__init__()
        
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),("gelu",nn.GELU()), ("c_proj", nn.Linear(d_model * 4, d_model ))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def attention(self, x:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype = x.dtype, device = x.device) if self.attn_mask is not None else None 
        #return self.attn(x,x,x, need_weights = False, attn_mask = self.attn_mask)[0]
        if isinstance(self.attn, VVAttention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x,x,x, need_weights = True, attn_mask = self.attn_mask)[0]  

    def forward(self, x:torch.Tensor, ffn = False):
        if isinstance(self.attn, VVAttention):
            if isinstance(x, list):
                if not ffn:
                    x, x_ori = x
                    x_res = self.attention(self.ln_1(x_ori))
                    x_res, x_ori_res = x_res
                    x_ori += x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res 
                    return [x, x_ori]
                else:
                    x, x_ori_1 = x
                    x_res = self.attention(self.ln_1(x_ori_1))
                    x_res, x_ori_res = x_res
                    x_ori = x_ori_1 +  x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res 
                    x = x_res + x_ori_1
                    x = x + self.mlp(self.ln_2(x))
                    return [x, x_ori]
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x
        
   






class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None) :
        super().__init__()
        self.width = width 
        self.layers = layers 
        #self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        #add
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, attn_mask
                )
            for idx in range(layers)
        ])


    def ori_CLIP_with_patch_forward(self, x, out_layers):
        idx = 0
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[1])
                else:
                    out_tokens.append(x)

        return x[0], out_tokens

    def vvattn_forward(self, x, out_layers, ffn):
        idx = 0
        out_tokens = []
        ori_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x, ffn = ffn)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[0])
                    ori_tokens.append(x[1])
                else:
                    out_tokens.append(x)
                    ori_tokens.append(x)

        return x, out_tokens, ori_tokens
    def forward(self, x: torch.Tensor, out_layers: list = [3, 6, 9], ori=True, VVattn_layer = None, ffn = False
                ):
        if ori:
            idx = 0
            out_tokens = []
            for r in self.resblocks:
                idx += 1
                if idx == 24:
                    x = r(x)
                else:
                    x = r(x)
                if idx in out_layers:
                    out_tokens.append(x)
            return x, out_tokens
        if VVattn_layer is None:
            x, out_tokens  = self.ori_CLIP_with_patch_forward(x, out_layers)
            return x, out_tokens 
        else:
            x, out_tokens, ori_tokens = self.vvattn_forward(x, out_layers, ffn)
            return x, out_tokens, ori_tokens
       


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int , patch_size: int , width: int, layers: int , heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution 
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= width, kernel_size = patch_size, stride= patch_size, bias=False)

        scale = width ** -0.5  #   
        self.class_embedding =  nn.Parameter(scale * torch.randn(width))

        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) **2 + 1, width))
        self.ln_pre = LayerNorm(width)   

        self.transformer = Transformer( width, layers, heads) 
        self.width = width
        self.heads = heads

        self.ln_post = LayerNorm(width)
        
        #add
        self.patch_size = patch_size
        self.grid_size = input_resolution // patch_size

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
    @torch.no_grad()
    def VVattn_replace(self, VVattn_layer):
        if VVattn_layer is not None:
            for i in range(1, VVattn_layer):
                self.attn = VVAttention(self.width, self.width, self.heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn    

    def forward(self, x:torch.Tensor, out_layers: list, return_x=False, ori=True, VVattn_layer = None, ffn = False):
        img = x
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0,2,1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],dim = 1) 

        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1,0,2)

        if VVattn_layer:
            [x, x_ori], patch_tokens, ori_tokens = self.transformer(x, out_layers, ori=ori, VVattn_layer = VVattn_layer, ffn = ffn)
        else:
            x_ori, patch_tokens = self.transformer(x, out_layers, ori=ori)
        

        x_ori = x_ori.permute(1,0,2)
        
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]  # LND -> NLD

        x_ori = self.ln_post(x_ori[:, 0, :])   

        if self.proj is not None:
            x_ = x_ori @ self.proj
        if return_x:
            if VVattn_layer:
                return x_, patch_tokens, [ori_tokens[t].permute(1, 0, 2) for t in range(len(ori_tokens))], x_ori
            else:
                return x_, patch_tokens, x_ori
        else:   
            return x_, patch_tokens
    
class LinearLayer(nn.Module): # linear layers used for mapping patch-level features.
    def __init__(self, dim_in, dim_out, k ):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
    def forward(self, tokens):
        tokens_list = []
        for i in range(len(tokens)):
            tokens_list.append(self.fc[i](tokens[i][:, 1:, :]))
        return tokens_list




class Linear1(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, k = 5):
        super(Linear1, self).__init__()
        self.fc = nn.Conv1d(1, k, 3, stride=1, padding="same")
        self.num_layer = k
    def forward(self, tokens):
        
        result = self.fc(tokens)
        return result

class Instanse_Prompting(nn.Module):
    def __init__(self, model_config, cla_len):
        super().__init__()
        assert model_config['text_cfg']['width'] == model_config['embed_dim']
        self.prompt_query = nn.Parameter(torch.randn(1, cla_len, model_config['text_cfg']['width']))
        self.cla_len = cla_len
        self.prompt_linear1 = Linear1(model_config['text_cfg']['width'], model_config['text_cfg']['width'] + 256, model_config['text_cfg']['width'], k = cla_len)
        self.prompt_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        nn.init.trunc_normal_(self.prompt_query)
        self._initialize_weights()


    def _initialize_weights(self):
        """Initialize the weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std= 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    


    def before_extract_feat(self, x, img_feature, use_global = True):
        B, C = img_feature.shape
        global_feat = img_feature
        global_feat_new = self.prompt_linear1(global_feat.reshape(B, 1, C))
        prompt_query = self.prompt_query + torch.zeros((B, self.prompt_query.shape[-2], self.prompt_query.shape[-1]), dtype=self.prompt_query.dtype, device=self.prompt_query.device)
        if use_global:
            class_feature =  prompt_query  +  global_feat_new 
        else:
            class_feature = prompt_query 
        return class_feature 


class Prompt_Ensemble():
    def __init__(self, cla_len, tokenizer):
        super().__init__()
      
        self.special_token = '<|class|>'   # special_token denotes the learnable product category

        self.prompt_templates_abnormal = ["a photo of a abnormal <|class|>."]
        self.prompt_templates_normal = ["a photo of a normal <|class|>."]
        self.tokenizer =  tokenizer


    
    def forward_ensemble(self, model, vison_feature, device, prompt_id = 0):

        prompted_sentence_normal = self.tokenizer(self.prompt_templates_normal).to(device)
        prompted_sentence_abnormal = self.tokenizer(self.prompt_templates_abnormal).to(device)
        normal_embeddings = model.encode_text(prompted_sentence_normal, vison_feature) 
        abnormal_embeddings = model.encode_text(prompted_sentence_abnormal, vison_feature)
        text_prompts = torch.cat([normal_embeddings, abnormal_embeddings], dim =1)

        return text_prompts
               
class CusCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 deep_prompt_len: int,
                 total_d_layer_len: int
                 ):
        super().__init__()

        self.context_length = context_length


        vision_heads = vision_width // 64  
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.prompt_text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        ## Add the prompt parameters # exclude_key=prompt:
        self.num_layers = transformer_layers    # 5 
        #self.total_d_layer = transformer_layers - 1
        self.total_d_layer = total_d_layer_len
        if self.total_d_layer != 0:
            assert self.total_d_layer == transformer_layers - 1
        self.num_tokens = deep_prompt_len
        self.prompt_dim = transformer_width

        self._init_prompt(self.num_tokens, self.prompt_dim, self.total_d_layer)


    def _init_prompt(self, num_tokens, prompt_dim, total_d_layer):
        val = 1
        if total_d_layer >= 0:
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = nn.Dropout(0.1)

        else: # total_d_layer < 0
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(abs(total_d_layer), num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = nn.Dropout(0.1)



    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            nn.init.normal_(self.prompt_text_projection, std=self.transformer.width ** -0.5)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, out_layers, return_x=False, ori=True, VVattn_layer = None, ffn = False):  
        #return self.visual(image.type(self.dtype), out_layers)
        return self.visual(image, out_layers, return_x, ori, VVattn_layer, ffn)

    def encode_text_(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, tokens = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def encode_text(self, text, visual_feature, out_layers = [2,5,8,12]):  

        pos_x, pos_y = torch.where(text == 49407)

        x = self.token_embedding(text).type(self.dtype)  
        N, L, D = x.shape

        text_feature_list = []
        for i in range(visual_feature.shape[0]):
            x_new = torch.zeros_like(x).to(x.device)
            for j in range(x.shape[0]):
                x_new[j, :, :] = torch.cat([x[j, 0:pos_y[j], :], visual_feature[i,:,:], x[j, (pos_y[j]+1):(self.context_length - visual_feature.shape[1] + 1)]], dim = 0).unsqueeze(0)

            x_new = x_new + self.positional_embedding.type(self.dtype)

            if self.total_d_layer > 0:
                # concat prompt
                x_new = torch.cat((
                    x_new[:, :1, :],
                        self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(N, -1, -1)), 
                        x_new[:, 1:-self.num_tokens, :]
                    ), dim=1)

            x_new = x_new.permute(1,0,2)

            features = []
            attns = []
            if self.total_d_layer == 0: #shallow
                # x_new, attns, tokens = self.transformer(x_new)
                x_new, tokens = self.transformer(x_new)
            elif self.total_d_layer > 0: # deep
               x_new, features, attns = self.forward_deep_prompt(x_new, features,attns,out_layers)   
            else:
                AttributeError('Input correct total_d_layer')

            x_new = x_new.permute(1, 0, 2)  # LND -> NLD
            x_new = self.ln_final(x_new).type(self.dtype)

            x_new = x_new[torch.arange(x_new.shape[0]), torch.where(text == 49407)[1] + visual_feature.shape[1] - 1] @ self.text_projection  
            x_new = x_new / x_new.norm(dim=-1, keepdim=True)
            x_new = x_new.mean(dim = 0, keepdim = True)
            x_new = x_new / x_new.norm(dim=-1, keepdim=True)
            text_feature_list.append(x_new)
            
        result = torch.stack(text_feature_list, dim = 0)
        return result

    def forward_deep_prompt(self, embedding_output, features,attns, out_layers,out_last=False):   
        N,B = embedding_output.shape[0], embedding_output.shape[1]

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0) 
  
                hidden_states = self.transformer.resblocks[i](hidden_states)  
            else:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0)
                hidden_states = self.transformer.resblocks[i](hidden_states)  
            if len(out_layers) > 1:

                if (i+1) in out_layers:
                    xp = hidden_states.permute(1, 0, 2)
                    xp = torch.cat([xp[:,:1,:], xp[:, (1+self.num_tokens):, :]], dim = 1)
                    features.append(xp.contiguous())
            
            if i == (self.num_layers-2): 
                before_last_feats = self.prompt_norm(hidden_states)

        hidden_states_new = hidden_states.permute(1, 0, 2)
        hidden_states_new = torch.cat([hidden_states_new[:,:1,:], hidden_states_new[:, (1+self.num_tokens):, :]], dim = 1)
        hidden_states = hidden_states_new.permute(1, 0, 2)
        encoded = self.prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features , attns
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)   

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


class IQE_CLIP(nn.Module):
    def __init__(self, freeze_clip, features_list, model_configs, prompt_len, iqm_config, query_vison=False):
        super(IQE_CLIP, self).__init__()

        self.clip = freeze_clip
        self.trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                  len(features_list))
        self.New_Lan_Embed = Instanse_Prompting(model_configs, cla_len = prompt_len)
        self.prompt_pre = Prompt_Ensemble(prompt_len, tokenize)
        iqm_config = IQMConfig.from_pretrained(iqm_config)
        self.iqm = IQM(iqm_config)
        if query_vison==True:
            self.query_tokens = nn.Parameter(torch.randn(1, 2, model_configs['text_cfg']['width']))
        else:
            self.query_tokens = nn.Parameter(torch.randn(1, 2, model_configs["vision_cfg"]['width']))
        
        self.query_linear = Linear1(model_configs['text_cfg']['width'], model_configs['text_cfg']['width'] + 256, model_configs['text_cfg']['width'], k = 2, in_channel=1)
        nn.init.trunc_normal_(self.query_tokens)
        self.New_Lan_Embed.train()
        self.trainable_layer.train()
        self.iqm.train()
    
    def get_query(self, img_feature, use_global = True):
        B, C = img_feature.shape
        global_feat = img_feature
        global_feat_new = self.query_linear(global_feat.reshape(B, 1, C))
        query_tokens = self.query_tokens + torch.zeros((B, self.query_tokens.shape[-2], self.query_tokens.shape[-1]), dtype=self.query_tokens.dtype, device=self.query_tokens.device)
        if use_global:
            query_tokens =  query_tokens  +  global_feat_new 
        else:
            query_tokens = query_tokens 
        
        return query_tokens
