"""
st_model.py
A multi-pedestrian trajectory prediction model 
that follows spatial -> temporal encoding manners.
"""

import torch
import torch.nn as nn

# from src.social_transformer.social_transformer import SpatialSocialTransformerEncoder
from gst_updated.src.gumbel_social_transformer.gumbel_social_transformer import GumbelSocialTransformer
from gst_updated.src.gumbel_social_transformer.temporal_convolution_net import TemporalConvolutionNet


def offset_error_square_full_partial(x_pred, x_target, loss_mask_ped, loss_mask_pred_seq):
    """
    Offset Error Square between positions.
    # * average_offset_error and final_offset_error in utils.py are computed for full pedestrians.
    inputs:
        - x_pred
            # prediction on pedestrian displacements in prediction period.
            # (batch, pred_seq_len, node, motion_dim)
            # batch = 1
        - x_target
            # ground truth pedestrian displacements in prediction period.
            # (batch, pred_seq_len, node, motion_dim)
        - loss_mask_ped
            # loss mask on each pedestrian. 1 means the pedestrian is valid, and 0 means not valid.
            # * equivalent as loss_mask_rel_full_partial in st_model.
            # * Used to filter out the ones we do not predict. (disappear early, not appear until prediction period.)
            # (batch, node)
        - loss_mask_pred_seq
            # loss_mask_rel in prediction sequence. float32 tensor: (batch, num_peds, pred_seq_len)
    outputs:
        - offset_error_sq: offset error for each pedestrians. 
            # Already times eventual_loss_mask before output. shape: (pred_seq_len, node)
        - eventual_loss_mask: eventual loss mask on each pedestrian and each prediction step. 
            # shape: (pred_seq_len, node)
    """
    assert x_pred.shape[0] == loss_mask_ped.shape[0] == loss_mask_pred_seq.shape[0] == 1 # batch
    assert x_pred.shape[1] == x_target.shape[1] == loss_mask_pred_seq.shape[2] # pred_seq_len
    assert x_pred.shape[2] == x_target.shape[2] == loss_mask_ped.shape[1] == loss_mask_pred_seq.shape[1] # num_peds
    assert x_pred.shape[3] == x_target.shape[3] == 2 # motion dim
    # mask out invalid values
    loss_mask_rel_pred = loss_mask_pred_seq.permute(0, 2, 1).unsqueeze(-1) # (batch, pred_seq_len, num_peds, 1)
    x_pred_m = x_pred * loss_mask_rel_pred # (batch, pred_seq_len, node, motion_dim)
    x_target_m = x_target * loss_mask_rel_pred
    x_pred_m = x_pred_m * loss_mask_ped.unsqueeze(1).unsqueeze(-1) # (batch, pred_seq_len, node, motion_dim)
    x_target_m = x_target_m * loss_mask_ped.unsqueeze(1).unsqueeze(-1) # (batch, pred_seq_len, node, motion_dim)

    pos_pred = torch.cumsum(x_pred_m, dim=1)
    pos_target = torch.cumsum(x_target_m, dim=1)
    offset_error_sq = (((pos_pred-pos_target)**2.).sum(3))[0] # (pred_seq_len, node)

    eventual_loss_mask = loss_mask_rel_pred[0,:,:,0] * loss_mask_ped[0] # (pred_seq_len, node)
    offset_error_sq = offset_error_sq * eventual_loss_mask

    return offset_error_sq, eventual_loss_mask



def negative_log_likelihood_full_partial(gaussian_params, x_target, loss_mask_ped, loss_mask_pred_seq):
    """
    Compute negative log likelihood of gaussian parameters.
    inputs:
        - gaussian_params: tuple.
            - mu: (batch, pred_seq_len, node, 2)
            - sx: (batch, pred_seq_len, node, 1)
            - sy: (batch, pred_seq_len, node, 1)
            - corr: (batch, pred_seq_len, node, 1)
        - x_target
            # ground truth pedestrian displacements in prediction period.
            # (batch, pred_seq_len, node, motion_dim)
        - loss_mask_ped
            # loss mask on each pedestrian. 1 means the pedestrian is valid, and 0 means not valid.
            # * equivalent as loss_mask_rel_full_partial in st_model.
            # * Used to filter out the ones we do not predict. (disappear early, not appear until prediction period.)
            # (batch, node)
        - loss_mask_pred_seq
            # loss_mask_rel in prediction sequence. float32 tensor: (batch, num_peds, pred_seq_len)
    outputs:
        - prob_loss: (pred_seq_len, node)
        - eventual_loss_mask: eventual loss mask on each pedestrian and each prediction step. 
            # shape: (pred_seq_len, node)
    """

    mu, sx, sy, corr = gaussian_params
    
    loss_mask_rel_pred = loss_mask_pred_seq.permute(0, 2, 1).unsqueeze(-1) # (batch, pred_seq_len, num_peds, 1)
    mu = mu * loss_mask_rel_pred # (batch, pred_seq_len, node, 2)
    corr = corr * loss_mask_rel_pred # (batch, pred_seq_len, node, 1)
    x_target = x_target * loss_mask_rel_pred
    mu = mu * loss_mask_ped.unsqueeze(1).unsqueeze(-1) # (batch, pred_seq_len, node, 2)
    corr = corr * loss_mask_ped.unsqueeze(1).unsqueeze(-1) # (batch, pred_seq_len, node, 1)
    x_target = x_target * loss_mask_ped.unsqueeze(1).unsqueeze(-1) # (batch, pred_seq_len, node, 2)
    
    sx = sx * loss_mask_rel_pred + (1.-loss_mask_rel_pred) # sigma should not be zero, so mask with one
    sy = sy * loss_mask_rel_pred + (1.-loss_mask_rel_pred) # sigma should not be zero, so mask with one
    sx = sx * loss_mask_ped.unsqueeze(1).unsqueeze(-1) + (1.-loss_mask_ped.unsqueeze(1).unsqueeze(-1)) # (batch, pred_seq_len, node, 1)
    sy = sy * loss_mask_ped.unsqueeze(1).unsqueeze(-1) + (1.-loss_mask_ped.unsqueeze(1).unsqueeze(-1)) # (batch, pred_seq_len, node, 1)
    sigma = torch.cat((sx, sy), dim=3)

    x_target_norm = (x_target-mu)/sigma
    nx, ny = x_target_norm[:,:,:,0:1], x_target_norm[:,:,:,1:2]
    loss_term_1 = torch.log(1.-corr**2.)/2.+torch.log(sx)+torch.log(sy)
    loss_term_2 = (nx**2.-2.*corr*nx*ny+ny**2.)/(2.*(1.-corr**2.))
    prob_loss = (loss_term_1+loss_term_2).squeeze(3).squeeze(0) # (pred_seq_len, node)
    eventual_loss_mask = loss_mask_rel_pred[0,:,:,0] * loss_mask_ped[0] # (pred_seq_len, node)
    prob_loss = prob_loss * eventual_loss_mask

    return prob_loss, eventual_loss_mask



class st_model(nn.Module):

    def __init__(self, args, device='cuda:0'):
        """
        Initialize spatial and temporal encoding components.
        inputs:
            - args: arguments from user input. Here only list arguments used in st_model.
                (in __init__)
                ### * in function __init__() * ###
                - spatial # spatial encoding methods. options: rel_conv.
                - temporal # temporal encoding methods. options: lstm.
                - motion_dim # pedestrian motion is 2D, so motion_dim is always 2.
                - output_dim # 5 means probabilistic output (mu_x, mu_y, sigma_x, sigma_y, corr)
                # 2 means deterministic output (x, y) # ! may not do output_dim=2 in our work
                - embedding_size # size of pedstrian embeddings after spatial encoding.
                - spatial_num_heads # number of heads for multi-head attention
                # mechanism in spatial encoding.
                - spatial_beta # beta used in skip connection as a percentage of original input.
                # default can be None. If beta is not None, beta = 0.9 means
                # out <- 0.9 * x + 0.1 * out
                - lstm_hidden_size # hidden size of lstm.
                - lstm_num_layers # number of layers of lstm.
                - lstm_batch_first # batch first or not for lstm. 
                - lstm_dropout # dropout rate of lstm.
                - decode_style # 'recursive' or 'readout'.
                # 'recursive' means recursively encode and decode.
                # 'readout' means encoding and decoding are separated.
                - detach_sample # bool value on whether detach samples from gaussian_params or not.
                # detach_sample=False is default. It means using reparametrization trick and enable gradient flow.
                # detach_sample=True means to disable reparametrization trick.
                # ! To add
                # ! args.spatial_num_heads_edges
                # ! args.ghost
                ### * in function foward() * ###
                - pred_seq_len # length of prediction period: 12

            - device: 'cuda:0' or 'cpu'.
        """
        super(st_model, self).__init__()
        ## spatial
        if args.spatial == 'gumbel_social_transformer':
            # self.node_embedding = nn.Linear(args.motion_dim, args.embedding_size).to(device)
            # self.edge_embedding = nn.Linear(args.motion_dim, 2 * args.embedding_size).to(device)
            self.gumbel_social_transformer = GumbelSocialTransformer(
                args.motion_dim,
                args.embedding_size,
                args.spatial_num_heads,
                args.spatial_num_heads_edges,
                args.spatial_num_layers,
                dim_feedforward=128,
                dim_hidden=32,
                dropout=0.1,
                activation="relu",
                attn_mech="vanilla",
                ghost=args.ghost,
            ).to(device)
            # self.spatial_social_transformer = SpatialSocialTransformerEncoder(args.embedding_size, args.spatial_num_heads, args.spatial_num_layers, \
            #     dim_feedforward=256, dim_hidden=32, dropout=0.1, activation="relu", attn_mech="vanilla").to(device)
        else:
            raise RuntimeError('The spatial component is not found.')
        ## temporal
        if args.temporal == 'lstm' or args.temporal == 'faster_lstm':
            self.lstm = nn.LSTM(
                    input_size=args.embedding_size,
                    hidden_size=args.lstm_hidden_size,
                    num_layers=args.lstm_num_layers,
                    batch_first=False,
                    dropout=0.,
                    bidirectional=False,
                    ).to(device)
            self.hidden2pos = nn.Linear(args.lstm_num_layers*args.lstm_hidden_size, args.output_dim).to(device)
        else:
            raise RuntimeError('The temporal component is not lstm nor faster_lstm.')
        ## others
        self.args = args
        print("new st model")

    def raw2gaussian(self, prob_raw):
        """
        Turn raw values into gaussian parameters.
        inputs:
            - prob_raw: (batch, time, node, output_dim)
            - device: 'cuda:0' or 'cpu'.
        outputs:
            - gaussian_params: tuple.
                - mu: (batch, time, node, 2)
                - sx: (batch, time, node, 1)
                - sy: (batch, time, node, 1)
                - corr: (batch, time, node, 1)
        """
        mu = prob_raw[:,:,:,:2]
        sx, sy = torch.exp(prob_raw[:,:,:,2:3]), torch.exp(prob_raw[:,:,:,3:4])
        corr = torch.tanh(prob_raw[:,:,:,4:5])
        gaussian_params = (mu, sx, sy, corr)
        return gaussian_params

    def sample_gaussian(self, gaussian_params, device='cuda:0', detach_sample=False, sampling=True):
        """
        Generate a sample from Gaussian.
        inputs:
            - gaussian_params: tuple.
                - mu: (batch, time, node, 2)
                - sx: (batch, time, node, 1)
                - sy: (batch, time, node, 1)
                - corr: (batch, time, node, 1)
            - device: 'cuda:0' or 'cpu'
            - detach_sample: Bool. Default False.
                # Detach is to cut the gradient flow between gaussian_params and the next sample.
                # detach_sample=True means reparameterization trick is disabled.
                # detach_sample=False means reparameterization trick is enabled.
                # ! if it causes error, we need to manually turn detach_sample=False 
                # ! or we have to change args file for val_best before jan 4, 2021.
            - sampling: 
                # True means sampling. # False means using mu.
        outputs:
            - sample: (batch, time, node, 2)
        """
        mu, sx, sy, corr = gaussian_params
        if detach_sample:
            mu, sx, sy, corr = mu.detach(), sx.detach(), sy.detach(), corr.detach()
        if sampling:
            sample_unit = torch.empty(mu.shape).normal_().to(device) # N(0,1) with shape (batch, time, node, 2)
            sample_unit_x, sample_unit_y = sample_unit[:,:,:,0:1], sample_unit[:,:,:,1:2] # (batch, time, node, 1)
            sample_x = sx*sample_unit_x
            sample_y = corr*sy*sample_unit_x+((1.-corr**2.)**0.5)*sy*sample_unit_y
            sample = torch.cat((sample_x, sample_y), dim=3)+mu
        else:
            sample = mu
        return sample


    def edge_evolution(self, xt_plus, At, device='cuda:0'):
        """
        Compute edges at the next time step (At_plus) based on 
        pedestrian displacements at the next time step (xt_plus)
        and edges at the current time step (At).
        inputs:
            - xt_plus: vertices representing pedestrian displacement from t to t+1.
            # (batch, unit_time, node, motion_dim)
            - At: edges representing relative position between pedestrians at time t.
            At(i, j) is the vector pos_i,t - pos_j,t. I.e. the vector from pedestrian j
            to pedestrian i. 
            # (batch, unit_time, node, node, edge_feat)
            # batch = unit_time = 1.
            # edge_feat = 2.
            - device: 'cuda:0' or 'cpu'.
        outputs:
            - At_plus: edges representing relative position between pedestrians at time t.
            # (batch, unit_time, node, node, edge_feat)
        """
        # xt_plus # (batch, unit_time, node, motion_dim)
        # At # (batch, unit_time, node, node, edge_feat)
        # (batch, unit_time, node, 1, motion_dim) - (batch, unit_time, 1, node, motion_dim)
        At_plus = At + (xt_plus.unsqueeze(3) - xt_plus.unsqueeze(2))
        return At_plus

    def forward(self, x, A, attn_mask, loss_mask_rel, tau=1., hard=False, sampling=True, device='cuda:0'):
        """
        Forward function.
        inputs:
            - x
                # vertices representing pedestrians during observation period.
                # (batch, obs_seq_len, node, in_feat)
                # node: number of pedestrians
                # in_feat: motion_dim, i.e. 2.
                # Refer to V_obs in src.mgnn.utils.dataset_format().
            - A
                # edges representation relationships between pedestrians during observation period.
                # (batch, obs_seq_len, node, node, edge_feat)
                # edge_feat: feature dim of edges. if spatial encoding is rel_conv, edge_feat = 2. 
                # Refer to A_obs in src.mgnn.utils.dataset_format().
            - attn_mask
                # attention mask on pedestrian interactions in observation period.
                # row -> neighbor, col -> target
                # Should neighbor affect target?
                # 1 means yes, i.e. attention exists.  0 means no.
                # float32 tensor: (batch, obs_seq_len, neighbor_num_peds, target_num_peds)
            - loss_mask_rel
                # loss mask on displacement in the whole period
                # float32 tensor: (batch, num_peds, seq_len)
                # 1 means the displacement of pedestrian i at time t is valid. 0 means not valid.
                # If the displacement of pedestrian i at time t is valid,
                # then position of pedestrian i at time t and t-1 is valid.
                # If t is zero, then it means position of pedestrian i at time t is valid.
            - tau: temperature hyperparameter of gumbel softmax.
                # ! Need annealing though training. 1 is considered really soft at the beginning.
            - hard: hard or soft sampling.
                # True means one-hot sample for evaluation.
                # False means soft sample for reparametrization.
            - sampling: sample gaussian (True) or use mean for prediction (False).
            - device: 'cuda:0' or 'cpu'.
        outputs:
            # TODO
        """
        # ! Start writing multiple batches
        info = {}
        # processing when missing pedestrians are included
        batch_size, _, num_peds, _ = x.shape
        loss_mask_per_pedestrian = (loss_mask_rel.sum(2)==loss_mask_rel.shape[2]).float() # (batch, num_peds)
        if self.args.only_observe_full_period:
            attn_mask_single_step = torch.bmm(loss_mask_per_pedestrian.unsqueeze(2), loss_mask_per_pedestrian.unsqueeze(1)) # (batch, num_peds, num_peds)
            attn_mask = torch.ones(batch_size, self.args.obs_seq_len, num_peds, num_peds).to(device) * attn_mask_single_step.unsqueeze(1) # (batch, obs_seq_len, num_peds, num_peds)
        ## observation period: spatial
        if self.args.spatial == 'gumbel_social_transformer':
            # x_embedding = self.node_embedding(x)[0] # (obs_seq_len, node, d_model)
            # A_embedding = self.edge_embedding(A)[0] # (obs_seq_len, nnode <neighbor>, nnode <target>, 2*d_model)
            attn_mask = attn_mask.permute(0, 1, 3, 2) # (batch, obs_seq_len, nnode <target>, nnode <neighbor>)
            attn_mask_reshaped = attn_mask.reshape(batch_size*self.args.obs_seq_len, num_peds, num_peds)
            x_reshaped = x.reshape(batch_size*self.args.obs_seq_len, num_peds, -1)
            A_reshaped = A.reshape(batch_size*self.args.obs_seq_len, num_peds, num_peds, -1) #(batch, obs_seq_len, node, node, edge_feat)
            # ! attn_mask = attn_mask[0].permute(0,2,1) # (obs_seq_len, nnode <target>, nnode <neighbor>)
            # ! xs, sampled_edges, edge_multinomial, attn_weights = self.gumbel_social_transformer(x[0], A[0], attn_mask, tau=tau, hard=hard, device=device)
            xs, sampled_edges, edge_multinomial, attn_weights = self.gumbel_social_transformer(x_reshaped, A_reshaped, attn_mask_reshaped, tau=tau, hard=hard, device=device)
            xs = xs.reshape(batch_size, self.args.obs_seq_len, num_peds, -1) # (batch, obs_seq_len, nnode, embedding_size)
            info['sampled_edges'], info['edge_multinomial'], info['attn_weights'] = [], [], []
            sampled_edges = sampled_edges.reshape(batch_size, self.args.obs_seq_len,
                sampled_edges.shape[1], sampled_edges.shape[2], sampled_edges.shape[3])
            edge_multinomial = edge_multinomial.reshape(batch_size, self.args.obs_seq_len,
                edge_multinomial.shape[1], edge_multinomial.shape[2], edge_multinomial.shape[3])
            attn_weights = attn_weights.reshape(attn_weights.shape[0], batch_size, self.args.obs_seq_len,
                attn_weights.shape[2], attn_weights.shape[3], attn_weights.shape[4])
            info['sampled_edges'].append(sampled_edges.detach().to("cpu"))
            info['edge_multinomial'].append(edge_multinomial.detach().to("cpu"))
            info['attn_weights'].append(attn_weights.detach().to("cpu")) 
        else:
            raise RuntimeError("The spatial component is not found.")
        ## observation period: temporal
        ht = torch.zeros(self.args.lstm_num_layers, batch_size*num_peds, self.args.lstm_hidden_size).to(device)
        ct = torch.zeros(self.args.lstm_num_layers, batch_size*num_peds, self.args.lstm_hidden_size).to(device) 
        if self.args.temporal == 'lstm':
            for tt in range(self.args.obs_seq_len):
                loss_mask_rel_tt = loss_mask_rel[:,:,tt:tt+1].reshape(-1,1) # (batch*num_peds, 1)
                xs_tt = xs[:, tt].reshape(batch_size*num_peds, -1).unsqueeze(0)*loss_mask_rel_tt # (1, batch*num_peds, embedding_size)
                _, (htp, ctp) = self.lstm(xs_tt, (ht, ct)) # tp == tplus
                ht = htp * loss_mask_rel_tt + ht * (1.-loss_mask_rel_tt)
                ct = ctp * loss_mask_rel_tt + ct * (1.-loss_mask_rel_tt)
        elif self.args.temporal == 'faster_lstm':
            obs_mask = loss_mask_rel[:,:,:self.args.obs_seq_len].permute(0,2,1).unsqueeze(-1) # (batch, obs_seq_len, num_peds,1)
            xs_masked = xs*obs_mask # (batch, obs_seq_len, num_peds, embedding_size)
            xs_masked = xs_masked.permute(1,0,2,3).reshape(self.args.obs_seq_len, batch_size*num_peds, -1) # (obs_seq_len, batch*num_peds, embedding_size)
            _, (ht, ct) = self.lstm(xs_masked, (ht, ct))
        else:
            raise RuntimeError('The temporal component is not lstm nor faster_lstm.')
        if self.args.only_observe_full_period:
            loss_mask_rel_full_partial = loss_mask_per_pedestrian # (batch, num_peds)
        else:
            # We predict motion of pedestrians that show up in observation period, and did not disappear after last observed time step. <-> equivalent as only predict pedestrians whose relative position is present at the last observed time step.
            loss_mask_rel_full_partial = loss_mask_rel[:,:,self.args.obs_seq_len-1] # (batch, num_peds)
        ht = ht * loss_mask_rel_full_partial.reshape(-1).unsqueeze(-1)
        ct = ct * loss_mask_rel_full_partial.reshape(-1).unsqueeze(-1)
        attn_mask_pred = torch.bmm(loss_mask_rel_full_partial.unsqueeze(2), loss_mask_rel_full_partial.unsqueeze(1)).permute(0,2,1) # (batch*unit_time, nnode <target>, nnode <neighbor>)
        ## prediction period
        if self.args.decode_style == 'recursive':
            if self.args.temporal == 'lstm' or self.args.temporal == 'faster_lstm':
                # ht # (num_layers, node, hidden_size)
                # ht # (num_layers, batch_size*num_peds, hidden_size) # batch_size*num_peds, sth
                prob_raw = self.hidden2pos(ht.permute(1,0,2).reshape(batch_size*num_peds, -1)).reshape(batch_size, num_peds, -1).unsqueeze(1) # (batch, unit_time, node, output_dim)
                gaussian_params = self.raw2gaussian(prob_raw)
                mu, sx, sy, corr = gaussian_params
                x_sample = self.sample_gaussian(gaussian_params, device=device, detach_sample=self.args.detach_sample, sampling=sampling)
                # x_sample: (batch, unit_time, node, motion_dim)
                # loss_mask_rel_full_partial： (batch, num_peds)
                x_sample = x_sample * loss_mask_rel_full_partial.unsqueeze(1).unsqueeze(-1) # mask x_sample here for edge evolution
                # A: (batch, obs_seq_len, node, node, edge_feat)
                A_sample = self.edge_evolution(x_sample, A[:,-1:], device=device) # (batch, unit_time, node, node, edge_feat)
                # starts recursion
                prob_raw_pred, x_sample_pred, A_sample_pred = [], [], []
                mu_pred, sx_pred, sy_pred, corr_pred = [], [], [], []
                prob_raw_pred.append(prob_raw)
                x_sample_pred.append(x_sample)
                A_sample_pred.append(A_sample)
                mu_pred.append(mu)
                sx_pred.append(sx)
                sy_pred.append(sy)
                corr_pred.append(corr)
                for tt in range(1, self.args.pred_seq_len):
                    if self.args.spatial == 'gumbel_social_transformer':
                        # * spatial encoding at prediction step tt
                        # # attn_mask_pred # (batch*unit_time, nnode <target>, nnode <neighbor>)
                        x_sample_reshaped = x_sample.reshape(batch_size, num_peds, -1) # (batch*unit_time, node, motion_dim)
                        A_sample_reshaped = A_sample.reshape(batch_size, num_peds, num_peds, -1) # (batch*unit_time, node, node, edge_feat)
                        xs_tt, sampled_edges, edge_multinomial, attn_weights = \
                            self.gumbel_social_transformer(x_sample_reshaped, A_sample_reshaped, attn_mask_pred, \
                            tau=tau, hard=hard, device=device)
                        sampled_edges = sampled_edges.reshape(batch_size, 1,
                            sampled_edges.shape[1], sampled_edges.shape[2], sampled_edges.shape[3])
                        edge_multinomial = edge_multinomial.reshape(batch_size, 1,
                            edge_multinomial.shape[1], edge_multinomial.shape[2], edge_multinomial.shape[3])
                        attn_weights = attn_weights.reshape(attn_weights.shape[0], batch_size, 1,
                            attn_weights.shape[2], attn_weights.shape[3], attn_weights.shape[4])
                        info['sampled_edges'].append(sampled_edges.detach().to("cpu"))
                        info['edge_multinomial'].append(edge_multinomial.detach().to("cpu"))
                        info['attn_weights'].append(attn_weights.detach().to("cpu"))
                        # xs_tt: # (batch*unit_time, node, d_model)
                        # * temporal encoding at prediction step tt
                        loss_mask_rel_tt = loss_mask_rel_full_partial.reshape(-1,1) # (batch*num_peds,1)
                        xs_tt = xs_tt.reshape(batch_size*num_peds, -1).unsqueeze(0)*loss_mask_rel_tt # (unit_time, batch*num_peds, embedding_size)
                        _, (htp, ctp) = self.lstm(xs_tt, (ht, ct)) # tp == tplus
                        ht = htp * loss_mask_rel_tt + ht * (1.-loss_mask_rel_tt) # (num_layers, batch_size*num_peds, hidden_size)
                        ct = ctp * loss_mask_rel_tt + ct * (1.-loss_mask_rel_tt) # (num_layers, batch_size*num_peds, hidden_size)
                        # * prediction at prediction step tt
                        prob_raw = self.hidden2pos(ht.permute(1,0,2).reshape(batch_size*num_peds, -1)).reshape(batch_size, num_peds, -1).unsqueeze(1) # (batch, unit_time, node, output_dim)
                        gaussian_params = self.raw2gaussian(prob_raw)
                        mu, sx, sy, corr = gaussian_params
                        x_sample = self.sample_gaussian(gaussian_params, device=device, detach_sample=self.args.detach_sample, sampling=sampling)
                        x_sample = x_sample * loss_mask_rel_full_partial.unsqueeze(1).unsqueeze(-1) # mask x_sample here for edge evolution
                        A_sample = self.edge_evolution(x_sample, A_sample, device=device) # (batch, unit_time, node, node, edge_feat)
                        # * append to results
                        prob_raw_pred.append(prob_raw)
                        x_sample_pred.append(x_sample)
                        A_sample_pred.append(A_sample)
                        mu_pred.append(mu)
                        sx_pred.append(sx)
                        sy_pred.append(sy)
                        corr_pred.append(corr)
                    else:
                        raise RuntimeError("The spatial component is not found.")
                
                # concatenate predictions together
                prob_raw_pred = torch.cat(prob_raw_pred, dim=1) # (batch, pred_seq_len, node, output_dim)
                x_sample_pred = torch.cat(x_sample_pred, dim=1) # (batch, pred_seq_len, node, motion_dim)
                A_sample_pred = torch.cat(A_sample_pred, dim=1) # (batch, pred_seq_len, node, node, edge_feat)
                mu_pred = torch.cat(mu_pred, dim=1) # (batch, pred_seq_len, node, 2)
                sx_pred = torch.cat(sx_pred, dim=1) # (batch, pred_seq_len, node, 1)
                sy_pred = torch.cat(sy_pred, dim=1) # (batch, pred_seq_len, node, 1)
                corr_pred = torch.cat(corr_pred, dim=1) # (batch, pred_seq_len, node, 1)
                gaussian_params_pred = (mu_pred, sx_pred, sy_pred, corr_pred)
                # gaussian_params_pred = self.raw2gaussian(prob_raw_pred)
                info['sampled_edges'] = torch.cat(info['sampled_edges'], dim=1) # (batch_size, seq_len-1, nnode <target>, nhead_edges, neighbor_node)
                info['edge_multinomial'] = torch.cat(info['edge_multinomial'], dim=1) # (batch_size, seq_len-1, nnode <target>, nhead_edges, neighbor_node)
                info['attn_weights'] = torch.cat(info['attn_weights'], dim=2) # (batch_size, nlayer, seq_len-1, nhead, nnode <target>, neighbor_node)
                info['A_sample_pred'] = A_sample_pred # (batch, pred_seq_len, node, node, edge_feat)
                info['loss_mask_rel_full_partial'] = loss_mask_rel_full_partial # (batch, num_peds)
                info['loss_mask_per_pedestrian'] = loss_mask_per_pedestrian # (batch, num_peds)
                # pack results
                results = (gaussian_params_pred, x_sample_pred, info)
                return results
            else:
                raise RuntimeError('The temporal component is not lstm nor faster_lstm.')
        else:
            raise RuntimeError("The decoder style is not recursive.")