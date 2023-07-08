# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor



class LiDARSemanticFeatureAggregationModule(nn.Module):
    """
    Aggregate the semantic embeddings across all the voxels
    """
    def __init__(self, scale=1):
        super(LiDARSemanticFeatureAggregationModule, self).__init__()

    def forward(self, feats, probs, batch_idx, batch_size):
        """
        feats: [N1+N2+..., c]
        probs: [N1+N2+..., num_cls] 
        batch_idx: [N1+N2+..., ]
        batch_size

        semantic_embeddings: [batch_size, c, num_cls, 1]
        """
        total_n, num_classes = probs.size()
        channels = feats.size(1)

        semantic_embedding_list = [] 
        for i in range(batch_size):
            bs_mask = batch_idx == i
            # (Ni, C)
            cur_feats = feats[bs_mask]
            # (Ni, num_cls) -> (num_cls, Ni)
            cur_probs = probs[bs_mask].permute(1,0).contiguous()
            cur_probs = F.softmax(cur_probs, dim=1)
            # (num_cls, Ni) x (Ni, C) -> (num_cls, C)
            cur_semantic_embeddings = torch.matmul(cur_probs, cur_feats)
            semantic_embedding_list.append(cur_semantic_embeddings)

        # batch_size (num_cls, C) -> (batch, num_cls, C) -> (batch, C, num_cls, 1)
        semantic_embeddings = torch.stack(semantic_embedding_list, dim=0).permute(0,2,1).contiguous().unsqueeze(3)


        return semantic_embeddings


class SemanticFeatureFusionModule(nn.Module):
    """
    SFFM based on the transformer decoder
    """
    def __init__(self, d_input_point, d_input_embeddings1, d_input_embeddings2, embeddings_proj_kernel_size=1, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 ):
        super().__init__()

        self.input_proj_point = nn.Linear(d_input_point, d_model)
        self.input_proj_embeddings1 = nn.Conv1d(d_input_embeddings1, d_model, embeddings_proj_kernel_size)
        self.input_proj_embeddings2 = nn.Conv1d(d_input_embeddings2, d_model, embeddings_proj_kernel_size)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, embeddings_proj_kernel_size, 
                                                activation, normalize_before)
        decoder_norm_tgt = nn.LayerNorm(d_model)
        # decoder_norm_mem = nn.LayerNorm(d_model)
        decoder_norm_mem = None 
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm_tgt, decoder_norm_mem)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, input_point_features, input_sem_embeddings1, input_sem_embeddings2, batch_idx, batch_size, return_context=False):
        """"
        Input:
            input_point_features: [N1+N2+..., C_point]
            input_sem_embeddings1: [batch, C_image, num_cls, 1]
            input_sem_embeddings2: [batch, C_voxel, num_cls, 1]
            batch_idx: [N1+N2+..., C_point]
            batch_size: int
        Output: 
            output_feat_point: [N1+N2+..., d_model]
            output_feat_context: [L=2num_cls, B, E]
        """
        # input proj & prepare as [L, B, E]
        # [N1+N2+..., C_point] -> [N1+N2+..., E]
        point_feats = self.input_proj_point(input_point_features)
        # [batch, C_image, num_cls, 1] -> [batch, C_image, num_cls] -> [L=num_cls, B, E]
        sem_embeddings1 = self.input_proj_embeddings1(input_sem_embeddings1.squeeze(-1)).permute(2,0,1).contiguous()
        sem_embeddings2 = self.input_proj_embeddings2(input_sem_embeddings2.squeeze(-1)).permute(2,0,1).contiguous()
        # 2[L=num_cls, B, E] -> [L=2num_cls, B, E]
        sem_embeddings = torch.cat([sem_embeddings1, sem_embeddings2], dim=0)

        output_feat_point, output_feat_context = self.decoder(
            tgt=point_feats, memory=sem_embeddings, batch_idx=batch_idx, batch_size=batch_size,
        )

        if return_context:
            return output_feat_point, output_feat_context
        
        return output_feat_point




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm_tgt=None, norm_mem=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm_tgt = norm_tgt
        self.norm_mem = norm_mem

    def forward(self, tgt, memory, batch_idx, batch_size,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                mem_pos: Optional[Tensor] = None):
        output_tgt = tgt
        output_mem = memory

        for layer in self.layers:
            output_tgt, output_mem = layer(output_tgt, output_mem, batch_idx, batch_size,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, mem_pos=mem_pos)

        if self.norm_tgt is not None:
            output_tgt = self.norm_tgt(output_tgt)
        if self.norm_mem is not None:
            output_mem = self.norm_tgt(output_mem)

        return output_tgt, output_mem



class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, embeddings_proj_kernel_size=1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # normal MHA
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # modified MHA for points
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.crossocr_attn = SparsePointCorssAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout,
            kv_proj_kernel_size=embeddings_proj_kernel_size,
        )

        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, batch_idx, batch_size,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     mem_pos: Optional[Tensor] = None):
        """
        Input:
            tgt: point feature, [N, E]
            memory: cls emb, [L,B,E] 

        Output:
            tgt:
            memory:
        """


        q = k = self.with_pos_embed(memory, mem_pos)
        memory2 = self.self_attn(q, k, value=memory, attn_mask=memory_mask,
                              key_padding_mask=memory_key_padding_mask)[0]
        memory = memory + self.dropout1(memory2)
        memory = self.norm1(memory)


        tgt2 = self.crossocr_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            batch_idx=batch_idx, 
            batch_size=batch_size,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, memory

    def forward_pre(self, tgt, memory, batch_idx, batch_size,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    mem_pos: Optional[Tensor] = None, ):
        """
        tgt: point feature
        memory: cls emb 
        """
        memory2 = self.norm1(memory)
        q = k = self.with_pos_embed(memory2, mem_pos)
        memory2 = self.self_attn(q, k, value=memory2, attn_mask=memory_mask,
                              key_padding_mask=memory_key_padding_mask)[0]
        memory = memory + self.dropout1(memory2)
        
        tgt2 = self.norm2(tgt)
        tgt2 = self.crossocr_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            batch_idx=batch_idx, 
            batch_size=batch_size,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)


        return tgt, memory

    def forward(self, tgt, memory, batch_idx, batch_size,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                mem_pos: Optional[Tensor] = None, ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, batch_idx, batch_size, 
                                    tgt_mask, memory_mask, 
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, mem_pos)
        return self.forward_post(tgt, memory, batch_idx, batch_size,
                                 tgt_mask, memory_mask, 
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, mem_pos)



class SparsePointCorssAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., kv_proj_kernel_size=1, bias=True, matmul_norm=True, **kwargs):
        super().__init__()
        self.d_embed = embed_dim
        self.n_head = num_heads
        self.d_head = embed_dim // num_heads

        self.matmul_norm = matmul_norm
        if self.matmul_norm:
            self.mat_norm = self.d_head**-.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Conv1d(embed_dim, embed_dim, kv_proj_kernel_size, bias=bias)
        self.v_proj = nn.Conv1d(embed_dim, embed_dim, kv_proj_kernel_size, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


    def forward(self, query, key, value, batch_idx, batch_size):
        """Forward function."""
        """
        query_feats: [N1+N2+..., E]
        key_feats: [L=#cls, B, E]
        val_feats: [L=#cls, B, E]
        batch_idx: [N1+N2+..., ]

        return:    
            atted: [N1+N2+..., E]
        """

        # [N1+N2+..., E] -> [N1+N2+..., E]
        q = self.q_proj(query)
        # [L=#cls, B, E] -> [B, E, L=#cls]
        k = self.k_proj(key.permute(1,2,0))
        v = self.v_proj(value.permute(1,2,0))


        # q: [N1+N2+..., E] -> [N1+N2+..., #H, E//#H]
        # k: [B, E, L=#cls] -> [L=B, #H, E//#H, #cls]
        # v: [B, E, L=#cls] -> [L=B, #H, E//#H, #cls]
        # multihead format
        q = q.reshape(-1, self.n_head, self.d_head)
        k = k.reshape(batch_size, self.n_head, self.d_head, -1)
        v = v.reshape(batch_size, self.n_head, self.d_head, -1)
        atted_list = []

        # MHA in each frame
        for i in range(batch_size):
            cur_mask = batch_idx == i
            # [Ni, #H, E//#H] -> [#H, Ni, E//#H]
            cur_q = q[cur_mask].permute(1,0,2).contiguous()
            # [#H, E//#H, #cls]
            cur_k = k[i]
            # [#H, E//#H, #cls] -> [#H, #cls, E//#H]
            cur_v = v[i].permute(0,2,1).contiguous()
            
            # [#H, Ni, E//#H] x [#H, E//#H, #cls] ->  [#H, Ni, #cls]
            cur_sim_map = torch.bmm(cur_q, cur_k)
            if self.matmul_norm:
                cur_sim_map = self.mat_norm * cur_sim_map
            cur_sim_map = F.softmax(cur_sim_map, dim=-1)

            # [#H, Ni, #cls] x [#H, #cls, E//#H] -> [#H, Ni, E//#H]
            cur_atted = torch.bmm(cur_sim_map, cur_v)
            
            # reshape back: [#H, Ni, E//#H] -> [Ni, #H, E//#H]
            cur_atted = cur_atted.permute(1,0,2)
            atted_list.append(cur_atted)

        # cat: [N1+N2+..., #H, E//#H]
        atted = torch.cat(atted_list, dim=0)
        # reshape back: [N1+N2+..., #H, E//#H] -> [N1+N2+..., E]
        atted = self.out_proj( atted.reshape(-1, self.n_head*self.d_head) )

        return atted
    

