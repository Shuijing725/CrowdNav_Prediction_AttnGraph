# import pathhack
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from src.gumbel_social_transformer.mha import VanillaMultiheadAttention
from src.gumbel_social_transformer.utils import _get_activation_fn, gumbel_softmax

class EdgeSelector(nn.Module):
    r"""Ghost version."""
    def __init__(self, d_motion, d_model, nhead=4, dropout=0.1, activation="relu"):
        super(EdgeSelector, self).__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.augmented_edge_embedding = nn.Linear(3*d_motion, d_model)
        self.self_attn = VanillaMultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm_augmented_edge = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(self.head_dim, self.head_dim)
        self.linear2 = nn.Linear(self.head_dim, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.d_model = d_model
        self.d_motion = d_motion
        print("new edge selector")
        
    
    def forward(self, x, A, attn_mask, tau=1., hard=False, device='cuda:0'):
        """
        Encode pedestrian edge with node information.
        inputs:
            # * done: x, A need to be masked first before processing.
            - x: vertices representing pedestrians of one sample. 
                # * bsz is batch size corresponding to Transformer setting.
                # * In pedestrian setting, bsz = batch_size*time_step
                # (bsz, node, d_motion)
            - A: edges representation relationships between pedestrians of one sample.
                # (bsz, node, node, d_motion)
                # row -> neighbor, col -> target
            - attn_mask: attention mask provided in advance.
                # (bsz, target_node, neighbor_node)
                # 1. means yes, i.e. attention exists.  0. means no.
            - tau: temperature hyperparameter of gumbel softmax. 
                # ! Need annealing though training.
            - hard: hard or soft sampling.
                # True means one-hot sample for evaluation.
                # False means soft sample for reparametrization.
            - device: 'cuda:0' or 'cpu'.
        outputs:
            - edge_multinomial: The categorical distribution over the connections from targets to the neighbors
                # (time_step, target_node, num_heads, neighbor_node)
                # neighbor_node = nnode + 1 in ghost mode
            - sampled_edges: The edges sampled from edge_multinomial
                # (time_step, target_node, num_heads, neighbor_node)
                # neighbor_node = nnode + 1 in ghost mode
        """
        bsz, nnode, d_motion = x.shape
        assert d_motion == self.d_motion
        attn_mask = attn_mask.to("cpu")
        attn_mask_ped = (attn_mask.sum(-1) > 0).float().unsqueeze(-1) # (bsz, nnode, 1)
        x = x * attn_mask_ped.to(device)
        x_ghost = torch.zeros(bsz, 1, d_motion).to(device)
        x_neighbor = torch.cat((x, x_ghost), dim=1) # (bsz, nnode+1, d_motion)
        x_neighbor = torch.ones(bsz,nnode+1,nnode,d_motion).to(device)*x_neighbor.view(bsz,nnode+1,1,d_motion) # row -> neighbor
        x_target = torch.ones(bsz,nnode+1,nnode,d_motion).to(device)*x.view(bsz,1,nnode,d_motion) # col -> target
        x_neighbor_target = torch.cat((x_neighbor, x_target), dim=-1) # (bsz,nnode+1,nnode,2*d_motion)

        A = A * attn_mask.permute(0,2,1).unsqueeze(-1).to(device) # (bsz, neighbor_node, target_node, d_motion)
        A_ghost = torch.zeros(bsz, 1, nnode, d_motion).to(device)
        A = torch.cat((A, A_ghost), dim=1) # (bsz,nnode+1,nnode,d_motion)
        A = torch.cat((x_neighbor_target, A), dim=-1) # (bsz,nnode+1,nnode,3*d_motion) # n_node==t_node+1==nnode+1
        A = self.augmented_edge_embedding(A) # (bsz,nnode+1,nnode,d_model)
        A = self.norm_augmented_edge(A)
        A_perm = A.permute(0,2,1,3) # (bsz,nnode,nnode+1,d_model)
        A_perm = A_perm.reshape(A_perm.shape[0]*A_perm.shape[1], A_perm.shape[2], A_perm.shape[3]) # (time_step*target_node, neighbor_node, d_model)
        A_perm = A_perm.permute(1,0,2) # (neighbor_node, time_step*target_node, d_model)
        
        attn_mask_ghost = torch.ones(bsz, nnode, 1) # ghost exists all the time, not missing
        attn_mask = torch.cat((attn_mask, attn_mask_ghost), dim=2) #(bsz, nnode, nnode+1) # n_node==t_node+1==nnode+1
        attn_mask_neighbors = attn_mask.view(bsz, nnode, nnode+1, 1) * attn_mask.view(bsz, nnode, 1, nnode+1) # (bsz, nnode, nnode+1, nnode+1)
        attn_mask_neighbors = attn_mask_neighbors.view(bsz*nnode, nnode+1, nnode+1) # (time_step*target_node, neighbor_node, neighbor_node)
        # attn_mask_neighbors = torch.cat([attn_mask_neighbors for _ in range(self.nhead)], dim=0) # (nhead*time_step*target_node, neighbor_node, neighbor_node) # ! bug fixed
        attn_mask_neighbors = torch.stack([attn_mask_neighbors for _ in range(self.nhead)], dim=1) # (time_step*target_node, nhead, neighbor_node, neighbor_node)
        attn_mask_neighbors = attn_mask_neighbors.view(attn_mask_neighbors.shape[0]*attn_mask_neighbors.shape[1], \
            attn_mask_neighbors.shape[2], attn_mask_neighbors.shape[3]) # (time_step*target_node*nhead, neighbor_node, neighbor_node)   
        
        _, _, A2 = self.self_attn(A_perm, A_perm, A_perm, attn_mask=attn_mask_neighbors.to(device))
        # inputs: (L, N, E), (S, N, E), (S, N, E), (N*nhead, L, S)
        # bsz, num_heads, tgt_len, head_dim # A2 # (time_step*target_node, num_heads, neighbor_node, 4*d_model/nhead) # we use head_dim = 4*d_model/nhead
        A2 = A2.reshape(bsz, nnode, self.nhead, nnode+1, self.head_dim) # (time_step, target_node, num_heads, neighbor_node, head_dim)

        A2 = self.linear2(self.dropout1(self.activation(self.linear1(A2)))).squeeze(-1) # (time_step, target_node, num_heads, neighbor_node)
        edge_multinomial = softmax(A2, dim=-1) # (time_step, target_node, num_heads, neighbor_node)
        
        edge_multinomial = edge_multinomial * attn_mask.unsqueeze(2).to(device) # (time_step, target_node, num_heads, neighbor_node)
        edge_multinomial = edge_multinomial / (edge_multinomial.sum(-1).unsqueeze(-1)+1e-10)
        sampled_edges = self.edge_sampler(edge_multinomial, tau=tau, hard=hard)
        
        return edge_multinomial, sampled_edges


    def edge_sampler(self, edge_multinomial, tau=1., hard=False):
        r"""
        Sample from edge_multinomial using gumbel softmax for differentiable search.
        """
        logits = torch.log(edge_multinomial+1e-10) # (time_step, target_node, num_heads, neighbor_node)
        sampled_edges = gumbel_softmax(logits, tau=tau, hard=hard, eps=1e-10) # (time_step, target_node, num_heads, neighbor_node)
        return sampled_edges

if __name__ == "__main__":
    device = "cuda:0"
    edge_selector = EdgeSelector(2, 32, nhead=2, dropout=0.1).to(device)
    x = torch.randn(8, 3, 2)
    position = torch.randn(8,3,2)
    A = position.unsqueeze(2)-position.unsqueeze(1) # (8,3,3,2)
    loss_mask = torch.ones(3,8)
    loss_mask[0,:3] = 0.
    loss_mask[1,:5] = 0.
    attn_mask = []
    x, A = x.to(device), A.to(device)
    for i in range(8):
        attn_mask.append(torch.outer(loss_mask[:,i], loss_mask[:,i]))
    attn_mask = torch.stack(attn_mask, dim=0)
    attn_mask = attn_mask.to(device)
    print(attn_mask[3])
    edge_multinomial, sampled_edges = edge_selector(x, A, attn_mask, tau=1., hard=False, device=device)
    print("hello world.")