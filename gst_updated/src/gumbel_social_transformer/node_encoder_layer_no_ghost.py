import torch
import torch.nn as nn
from gst_updated.src.gumbel_social_transformer.mha import VanillaMultiheadAttention
from gst_updated.src.gumbel_social_transformer.utils import _get_activation_fn

class NodeEncoderLayer(nn.Module):
    r"""No ghost version"""
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu", attn_mech='vanilla'):
        super(NodeEncoderLayer, self).__init__()
        self.attn_mech = attn_mech
        if self.attn_mech == 'vanilla':
            self.self_attn = VanillaMultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm_node = nn.LayerNorm(d_model)
        else:
            raise RuntimeError('NodeEncoderLayer currently only supports vanilla mode.')
        self.norm1_node = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.nhead = nhead
    
    def forward(self, x, sampled_edges, attn_mask, device="cuda:0"):
        """
        Encode pedestrian edge with node information.
        inputs:
            - x: vertices representing pedestrians of one sample. 
                # bsz is batch size corresponding to Transformer setting. it corresponds to time steps in pedestrian setting.
                # (bsz, nnode, d_model)
            - sampled_edges: sampled adjacency matrix with ghost at the last column.
                # (time_step, nnode <target>, nhead_edges, neighbor_node)
                # where neighbor_node = nnode <neighbor>
            - attn_mask: attention mask provided in advance.
                # (bsz, nnode <target>, nnode <neighbor>)
                # row -> target, col -> neighbor
                # 1. means yes, i.e. attention exists.  0. means no.
            - device: 'cuda:0' or 'cpu'.
        outputs:
            - x: encoded vertices. # (bsz, nnode, d_model)
            - attn_weights: attention weights. # (bsz, nhead, nnode <target>, neighbor_node)
                # where neighbor_node = nnode <neighbor>
        """
        if self.attn_mech == 'vanilla':
            bsz = x.shape[0]
            attn_mask_ped = (attn_mask.sum(-1) > 0).float().unsqueeze(-1).to(device) # (bsz, nnode, 1)
            x = self.norm_node(x)
            x = x * attn_mask_ped
            x_perm = x.permute(1, 0, 2) # (nnode, bsz, d_model)
            adj_mat = sampled_edges.sum(2) # (bsz, nnode, nnode)
            # adj_mat = torch.cat([adj_mat for _ in range(self.nhead)], dim=0) # (nhead*bsz, target_node, neighbor_node)
            adj_mat = torch.stack([adj_mat for _ in range(self.nhead)], dim=1) # (bsz, nhead, target_node, neighbor_node)
            adj_mat = adj_mat.view(bsz*self.nhead, adj_mat.shape[2], adj_mat.shape[3]) # (bsz*nhead, target_node, neighbor_node) # ! really important bug fix
            x2, attn_weights, _ = self.self_attn(x_perm, x_perm, x_perm, attn_mask=adj_mat) 
            # inputs: (L, N, E), (S, N, E), (S, N, E), (N*nhead, L, S)
            # x2: (nnode, bsz, d_model); attn_weights: (bsz, nhead, target_node, neighbor_node)
            x2 = x2.permute(1, 0, 2) # (bsz, node, d_model)
            x = x + self.dropout(x2)
        else:
            raise RuntimeError('NodeEncoderLayer currently only supports vanilla mode.')
        
        x2 = self.norm1_node(x)
        x2 = self.dropout1(self.activation(self.linear1(x2)))
        x2 = self.dropout2(self.linear2(x2))
        x = x + x2
        return x, attn_weights