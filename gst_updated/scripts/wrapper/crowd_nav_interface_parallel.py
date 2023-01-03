
from gst_updated.src.gumbel_social_transformer.st_model import st_model
from os.path import join, isdir
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def seq_to_graph(seq_, seq_rel, attn_mech='rel_conv'):
    """
    inputs:
        - seq_ # (n_env, num_peds, 2, obs_seq_len)
        - seq_rel # (n_env, num_peds, 2, obs_seq_len)
    outputs:
        - V # (n_env, obs_seq_len, num_peds, 2)
        - A # (n_env, obs_seq_len, num_peds, num_peds, 2)
    """
    V = seq_rel.permute(0, 3, 1, 2) # (n_env, obs_seq_len, num_peds, 2)
    seq_permute = seq_.permute(0, 3, 1, 2) # (n_env, obs_seq_len, num_peds, 2)
    A = seq_permute.unsqueeze(3)-seq_permute.unsqueeze(2) # (n_env, obs_seq_len, num_peds, 1, 2) - (n_env, obs_seq_len, 1, num_peds, 2)
    return V, A

class CrowdNavPredInterfaceMultiEnv(object):

    def __init__(self, load_path, device, config, num_env):
        # *** Load model
        self.args = config
        self.device = device
        self.nenv = num_env

        # Uncomment if you want a fixed random seed.
        # torch.manual_seed(args_eval.random_seed)
        # np.random.seed(args_eval.random_seed)
        self.args_eval = config
        checkpoint_dir = join(load_path, 'checkpoint')
        self.model = st_model(self.args_eval, device=device).to(device)
        model_filename = 'epoch_'+str(self.args_eval.num_epochs)+'.pt'
        model_checkpoint = torch.load(join(checkpoint_dir, model_filename), map_location=device)
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        print("LOADED MODEL")
        print("device: ", device)
        print()

    def forward(self, input_traj,input_binary_mask, sampling = True):
        """
        inputs:
            - input_traj:
                # numpy
                # (n_env, num_peds, obs_seq_len, 2)
            - input_binary_mask:
                # numpy
                # (n_env, num_peds, obs_seq_len, 1)
                # Zhe: I think we should not just have the binary mask of shape (n_env, number of pedestrains, 1)
                # because some agents are partially detected, and they should not be simply ignored.
            - sampling:
                # bool
                # True means you sample from Gaussian.
                # False means you choose to use the mean of Gaussian as output.
        outputs:
            - output_traj:
                # torch "cpu"
                # (n_env, num_peds, pred_seq_len, 5)
                # where 5 includes [mu_x, mu_y, sigma_x, sigma_y, correlation coefficient]
            - output_binary_mask:
                # torch "cpu"
                # (n_env, num_peds, 1)
                # Zhe: this means for prediction, if an agent does not show up in the last and second
                # last observation time step, then the agent will not be predicted.
        """

        invalid_value = -999.
        # *** Process input data
        obs_traj = input_traj.permute(0,1,3,2) # (n_env, num_peds, 2, obs_seq_len)
        n_env, num_peds = obs_traj.shape[:2]
        loss_mask_obs = input_binary_mask[:,:,:,0] # (n_env, num_peds, obs_seq_len)
        loss_mask_rel_obs = loss_mask_obs[:,:,:-1] * loss_mask_obs[:,:,-1:]
        loss_mask_rel_obs = torch.cat((loss_mask_obs[:,:,:1], loss_mask_rel_obs), dim=2) # (n_env, num_peds, obs_seq_len)
        loss_mask_rel_pred = (torch.ones((n_env, num_peds, self.args_eval.pred_seq_len), device=self.device) * loss_mask_rel_obs[:,:,-1:])
        loss_mask_rel = torch.cat((loss_mask_rel_obs, loss_mask_rel_pred), dim=2) # (n_env, num_peds, seq_len)
        loss_mask_pred = loss_mask_rel_pred
        loss_mask_rel_obs_permute = loss_mask_rel_obs.permute(0,2,1).reshape(n_env*self.args_eval.obs_seq_len, num_peds) # (n_env*obs_seq_len, num_peds)
        attn_mask_obs = torch.bmm(loss_mask_rel_obs_permute.unsqueeze(2), loss_mask_rel_obs_permute.unsqueeze(1)) # (n_env*obs_seq_len, num_peds, num_peds)
        attn_mask_obs = attn_mask_obs.reshape(n_env, self.args_eval.obs_seq_len, num_peds, num_peds)
        
        obs_traj_rel = obs_traj[:,:,:,1:] - obs_traj[:,:,:,:-1]
        obs_traj_rel = torch.cat((torch.zeros(n_env, num_peds, 2, 1, device=self.device), obs_traj_rel), dim=3)
        obs_traj_rel = invalid_value*torch.ones_like(obs_traj_rel)*(1-loss_mask_rel_obs.unsqueeze(2)) \
            + obs_traj_rel*loss_mask_rel_obs.unsqueeze(2)
        v_obs, A_obs = seq_to_graph(obs_traj, obs_traj_rel, attn_mech='rel_conv')
        # *** Perform trajectory prediction
        sampling = False
        with torch.no_grad():
            v_obs, A_obs, attn_mask_obs, loss_mask_rel = \
                v_obs.to(self.device), A_obs.to(self.device), attn_mask_obs.to(self.device), loss_mask_rel.to(self.device)
            results = self.model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=0.03, hard=True, sampling=sampling, device=self.device)
            gaussian_params_pred, x_sample_pred, info = results
        mu, sx, sy, corr = gaussian_params_pred
        mu = mu.cumsum(1)
        sx_squared = sx**2.
        sy_squared = sy**2.
        corr_sx_sy = corr*sx*sy
        sx_squared_cumsum = sx_squared.cumsum(1)
        sy_squared_cumsum = sy_squared.cumsum(1)
        corr_sx_sy_cumsum = corr_sx_sy.cumsum(1)
        sx_cumsum = sx_squared_cumsum**(1./2)
        sy_cumsum = sy_squared_cumsum**(1./2)
        corr_cumsum = corr_sx_sy_cumsum/(sx_cumsum*sy_cumsum)
        mu_cumsum = mu.detach().to(self.device) + obs_traj.permute(0,3,1,2)[:,-1:]# np.transpose(obs_traj[:,:,:,-1:], (0,3,1,2)) # (batch, time, node, 2)
        mu_cumsum = mu_cumsum * loss_mask_pred.permute(0,2,1).unsqueeze(-1) + invalid_value*(1-loss_mask_pred.permute(0,2,1).unsqueeze(-1))
        output_traj = torch.cat((mu_cumsum.detach().to(self.device), sx_cumsum.detach().to(self.device), sy_cumsum.detach().to(self.device), corr_cumsum.detach().to(self.device)), dim=3)
        output_traj = output_traj.permute(0, 2, 1, 3) # (n_env, num_peds, pred_seq_len, 5)
        output_binary_mask = loss_mask_pred[:,:,:1].detach().to(self.device) # (n_env, num_peds, 1) # first step same as following in prediction
        return output_traj, output_binary_mask


def visualize_output_trajectory_deterministic(input_traj, input_binary_mask, output_traj, output_binary_mask, sample_index, obs_seq_len=5, pred_seq_len=5):
    ##### Print Visualization Started #####
    input_traj_i = input_traj[sample_index]
    input_binary_mask_i = input_binary_mask[sample_index]
    output_traj_i = output_traj[sample_index]
    output_binary_mask_i = output_binary_mask[sample_index]
    num_peds, seq_len = input_traj_i.shape[0], obs_seq_len+pred_seq_len
    full_obs_ped_idx = np.where(input_binary_mask_i.sum(1)[:,0]==obs_seq_len)[0]
    full_traj = np.concatenate((input_traj_i, output_traj_i[:,:,:2]), axis=1)
    output_binary_mask_i_pred_len = np.stack([output_binary_mask_i for j in range(pred_seq_len)], axis=1)
    loss_mask = np.concatenate((input_binary_mask_i, output_binary_mask_i_pred_len), axis=1)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    for ped_idx in range(num_peds):
        if ped_idx in full_obs_ped_idx:
            ax.plot(full_traj[ped_idx, obs_seq_len:, 0], full_traj[ped_idx, obs_seq_len:, 1], '.-', c='r')
            ax.plot(full_traj[ped_idx, :obs_seq_len, 0], full_traj[ped_idx, :obs_seq_len, 1], '.-', c='k') # black for obs   
        else:
            for t_idx in range(seq_len):
                if loss_mask[ped_idx,t_idx,0] == 1:
                    if t_idx < obs_seq_len:
                        # obs blue for partially detected pedestrians
                        ax.plot(full_traj[ped_idx, t_idx, 0], full_traj[ped_idx, t_idx, 1], '.', c='b')
                    else:
                        # pred orange for partially detected pedestrians
                        ax.plot(full_traj[ped_idx, t_idx, 0], full_traj[ped_idx, t_idx, 1], '.', c='C1', alpha=0.2)

    ax.set_aspect('equal', adjustable='box')
    ax.plot()
    fig.savefig(str(sample_index)+".png")
    print(str(sample_index)+".png is created.")
    return


if __name__ == '__main__':
    # *** Create an input that aligns with the format of CrowdNav
    obs_seq_len = 5
    pred_seq_len = 5
    invalid_value = -999.
    wrapper_demo_data = torch.load('/home/shuijing/Desktop/CrowdNav_Prediction/gst_updated/datasets/wrapper_demo/wrapper_demo_data.pt')
    print("wrapper_demo_data.pt is loaded.")
    input_traj, input_binary_mask = wrapper_demo_data['input_traj'], wrapper_demo_data['input_mask']
    n_env = input_traj.shape[0]
    assert input_traj.shape[2] == obs_seq_len
    """
    - input_traj:
        # tensor
        # (n_env, num_peds, obs_seq_len, 2)
    - input_binary_mask:
        # tensor
        # (n_env, num_peds, obs_seq_len, 1)
    """
    print()
    print("INPUT DATA")
    print("number of environments: ", n_env)
    print("input_traj shape: ", input_traj.shape)
    print("input_binary_mask shape:", input_binary_mask.shape)
    print()
    load_path = '/home/shuijing/Desktop/CrowdNav_Prediction/gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj'
    # load_path = join(pathhack.pkg_path, 'results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000"
    model = CrowdNavPredInterfaceMultiEnv(load_path=load_path,
                                          device=device, config = args, num_env=n_env)

    input_traj = input_traj.cuda()
    input_binary_mask = input_binary_mask.cuda()
    output_traj, output_binary_mask = model.forward(
        input_traj,
        input_binary_mask,
        sampling = True,
    )
    print()
    print("OUTPUT DATA")
    print("output_traj shape: ", output_traj.shape)
    print("output_binary_mask shape:", output_binary_mask.shape)
    print()
    for sample_index in range(n_env):
        visualize_output_trajectory_deterministic(input_traj, input_binary_mask, output_traj, output_binary_mask, sample_index, obs_seq_len=5, pred_seq_len=5)
