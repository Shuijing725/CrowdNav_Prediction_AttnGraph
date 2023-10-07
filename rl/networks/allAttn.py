import torch.nn.functional as F
import torch

from .srnn_model import *

class SpatialEdgeSelfAttn(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """
    def __init__(self, args):
        super(SpatialEdgeSelfAttn, self).__init__()
        self.args = args

        # Store required sizes
        # todo: hard-coded for now
        # with human displacement: + 2
        # pred 4 steps + disp: 12
        # pred 4 steps + no disp: 10
        # pred 5 steps + no disp: 12
        # pred 5 steps + no disp + probR: 17
        # Gaussian pred 5 steps + no disp: 27
        # pred 8 steps + no disp: 18
        if args.env_name in ['CrowdSimPred-v0', 'CrowdSimPredRealGST-v0']:
            self.input_size = 12
        elif args.env_name == 'CrowdSimVarNum-v0':
            self.input_size = 2 # 4
        else:
            raise NotImplementedError
        self.num_attn_heads=8
        self.attn_size=512


        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(nn.Linear(self.input_size, 128), nn.ReLU(),
                                             nn.Linear(128, self.attn_size), nn.ReLU()
                                             )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        self.multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)


    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len*nenv, max_human_num+1).cuda()
        mask[torch.arange(seq_len*nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        return mask


    def forward(self, inp, each_seq_len):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        '''
        # inp is padded sequence [seq_len, nenv, max_human_num, 2]
        seq_len, nenv, max_human_num, _ = inp.size()
        if self.args.sort_humans:
            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
            attn_mask = attn_mask.squeeze(1)  # if we use pytorch builtin function
        else:
            # combine the first two dimensions
            attn_mask = each_seq_len.reshape(seq_len*nenv, max_human_num)


        input_emb=self.embedding_layer(inp).view(seq_len*nenv, max_human_num, -1)
        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # if we use pytorch builtin function, v1.7.0 has no batch first option
        q=self.q_linear(input_emb)
        k=self.k_linear(input_emb)
        v=self.v_linear(input_emb)

        #z=self.multihead_attn(q, k, v, mask=attn_mask)
        z,_=self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function
        z=torch.transpose(z, dim0=0, dim1=1) # if we use pytorch builtin function
        return z



class EdgeAttention_M(nn.Module):
    '''
    Class for the robot-human attention module
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention_M, self).__init__()

        self.args = args

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size



        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer=nn.ModuleList()
        self.spatial_edge_layer=nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))



        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        attn = temporal_embed * spatial_embed
        attn = torch.sum(attn, dim=3)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # print(attn[0, 0, 0].cpu().numpy())

        # Compute weighted value
        # weighted_value = torch.mv(torch.t(h_spatials), attn)

        # reshape h_spatials and attn
        # shape[0] = seq_len, shape[1] = num of spatial edges (6*5 = 30), shape[2] = 256
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2,
                                                                                         1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]
        weighted_value = torch.bmm(h_spatials, attn)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn



    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, h_spatials, each_seq_len):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        each_seq_len:
            if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
            else, it is the mask itself
        '''
        seq_len, nenv, max_human_num, _ = h_spatials.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)

            if self.args.sort_humans:
                attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
                attn_mask = attn_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            else:
                attn_mask = each_seq_len
            weighted_value,attn=self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]

class EndRNN(RNNBase):
    '''
    Class for the GRU
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN, self).__init__(args, edge=False)

        self.args = args

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)


        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)



    def forward(self, robot_s, h_spatial_other, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)
        encoded_input = self.relu(encoded_input)

        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)

        x, h_new = self._forward_gru(concat_encoded, h, masks)

        outputs = self.output_linear(x)


        return outputs, h_new

class All_Attn(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(All_Attn, self).__init__()

        print('Using All Attention')
        self.infer = infer
        self.is_recurrent = True
        self.args=args

        self.human_num = obs_space_dict['human_state'].shape[0]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        #self.humanNodeRNN = EndRNN(args)

        # Initialize attention module
        #self.attn = EdgeAttention_M(args)


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        # uplc network for test
        self.relu = nn.ReLU()

        self.msa = torch.nn.MultiheadAttention(128, 8)

        self.qh_linear = nn.Linear(in_features=7, out_features=128)
        self.qg_linear = nn.Linear(in_features=7, out_features=128)

        self.kh_linear = nn.Linear(in_features=5, out_features=128)
        self.kg_linear = nn.Linear(in_features=2, out_features=128)

        self.vh_linear = nn.Linear(in_features=5, out_features=128)
        self.vg_linear = nn.Linear(in_features=2, out_features=128)

        self.qhm_linear = nn.Linear(in_features=128, out_features=128)
        self.khm_linear = nn.Linear(in_features=128, out_features=128)
        self.vhm_linear = nn.Linear(in_features=128, out_features=128)

        self.end_layer = nn.Linear(in_features=135, out_features=256)

        num_inputs = hidden_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))


        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num+1)

        dummy_human_mask = [0] * self.human_num
        dummy_human_mask[0] = 1

        if self.args.no_cuda:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        else:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())

    def forward(self, inputs, infer=False):
        #print(f'forward infer: {infer}')
        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_node = reshapeT(inputs['robot_state'], seq_length, nenv).squeeze(0)
        goal_node = reshapeT(inputs['goal_state'], seq_length, nenv).squeeze(0)
        human_node = reshapeT(inputs['human_state'], seq_length, nenv).squeeze(0)

        # print(f'robot node size: {robot_node.shape}')
        # print(f'goal node size: {goal_node.shape}')
        # print(f'human node size: {human_node.shape}')
        
        q_h = self.qh_linear(robot_node.repeat_interleave(self.human_num, -2))
        q_h = self.relu(q_h)

        q_g = self.qg_linear(robot_node)
        q_g = self.relu(q_g)

        k_h = self.kh_linear(human_node)

        if k_h.shape[0] == 30:
            #print(f'k_h size: {k_h.size()}')
            k_h = k_h.view(-1, 5, 128)
            q_h_m = self.qhm_linear(k_h)
            q_h_m = self.relu(q_h_m)
            k_h_m = self.khm_linear(k_h)
            k_h_m = self.relu(k_h_m)
            v_h_m = self.vhm_linear(k_h)
            v_h_m = self.relu(v_h_m)

            k_h,_ = self.msa(q_h_m, k_h_m, v_h_m)
            k_h = k_h.view(-1, 8, 5, 128)
        else:
            #print(f'k_h size: {k_h.size()}')
            q_h_m = self.qhm_linear(k_h)
            q_h_m = self.relu(q_h_m)
            k_h_m = self.khm_linear(k_h)
            k_h_m = self.relu(k_h_m)
            v_h_m = self.vhm_linear(k_h)
            v_h_m = self.relu(v_h_m)

            k_h,_ = self.msa(q_h_m, k_h_m, v_h_m)

        k_h = self.relu(k_h)

        k_g = self.kg_linear(goal_node)
        k_g = self.relu(k_g)

        v_h = self.vh_linear(human_node)
        v_h = self.relu(v_h)

        v_g = self.vg_linear(goal_node)
        v_g = self.relu(v_g)

        alpha_h = q_h * k_h
        alpha_g = q_g * k_g

        # print(f'alpha_h size: {alpha_h.size()}')
        # print(f'alpha_g size: {alpha_g.size()}')

        attn = torch.cat((alpha_h, alpha_g), -2)
        attn = torch.sum(attn, dim=-1)

        # print(f'attention weights before softmax: {attn[0, :]}')
        
        attn = torch.nn.functional.softmax(attn, dim=-1).unsqueeze(-1)

        # self.attention_weights = attn[0, : ,0]

        # i = 0
        # for weight in self.attention_weights.data.cpu().numpy():
        #     i = i + 1
        #     if i < 6:
        #         print(f'human_{i}: ' + '{:.2f}'.format(weight))
        #     else:
        #         print('goal: ' + '{:.2f}'.format(weight))
        
        joint_v = torch.cat((v_h, v_g), -2)
        #print(f'joint_v size: {joint_v.size()}')

        trans_joint_v = joint_v.transpose(-2, -1)

        if trans_joint_v.shape[0] == 30:
            if trans_joint_v.shape[1] == 8:
                #trans_joint_v = trans_joint_v.squeeze(1)
                #print(f'joint_v size: {trans_joint_v.size()}')
                trans_joint_v = trans_joint_v.view(-1, 128, self.human_num + 1)
                #attn = attn.squeeze(1)
                attn = attn.view(-1, self.human_num + 1, 1)
                weighted_value = torch.bmm(trans_joint_v, attn).transpose(-2, -1)
            else:
                trans_joint_v = trans_joint_v.squeeze(1)
                attn = attn.squeeze(1)
                weighted_value = torch.bmm(trans_joint_v, attn).transpose(-2, -1).unsqueeze(1)
            # print(f'trans_joint_v size: {trans_joint_v.size()}')
            # print(f'attn size: {attn.size()}')
        else:
            weighted_value = torch.bmm(trans_joint_v, attn).transpose(-2, -1)
            #print(f'weighted_value size: {weighted_value.size()}')

        #print(f': {weighted_value.size()}')
        # print(f'robot_node size: {robot_node.size()}')

        if robot_node.shape[1] == 8:
            robot_node = robot_node.view(-1, 1, 7)

        outputs = self.end_layer(torch.cat((weighted_value, robot_node), -1))

        x = outputs

        #print(f'output size: {outputs.size()}')

        # # x is the output and will be sent to actor and critic
        #x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        #print(x.mean([1]).unsqueeze(1).size())
        hidden_actor = self.actor(x)
        #print(f'hidden_actor size: {hidden_actor.size()}')

        #print(f'output size {hidden_actor.view(-1, self.output_size).dim()}')
        #print(f'output action size: {hidden_actor.squeeze(0).view(-1, self.output_size).size()}')
        #print(f'output value size: {self.critic_linear(hidden_critic).squeeze(0).view(-1, 1).size()}')

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0).view(-1, 1), hidden_actor.squeeze(0).view(-1, self.output_size)
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size)


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))