from gst_updated.scripts.data import pathhack
import pickle
from os.path import join, isdir
import torch
import numpy as np
from gst_updated.src.mgnn.utils import seq_to_graph
from gst_updated.src.gumbel_social_transformer.st_model import st_model



class CrowdNavPredInterfaceMultiEnv(object):

    def __init__(self, load_path, device, config, num_env):
        # *** Load model
        self.args = config
        self.device = device
        self.nenv = num_env
        if not isdir(load_path):
            raise RuntimeError('The result directory was not found.')
        checkpoint_dir = join(load_path, 'checkpoint')
        with open(join(checkpoint_dir, 'args.pickle'), 'rb') as f:
            args_eval = pickle.load(f)
        # Uncomment if you want a fixed random seed.
        # torch.manual_seed(args_eval.random_seed)
        # np.random.seed(args_eval.random_seed)
        self.model = st_model(args_eval, device=device).to(device)
        model_filename = 'epoch_'+str(args_eval.num_epochs)+'.pt'
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
        input_traj = input_traj.cpu().numpy()
        input_binary_mask = input_binary_mask.cpu().numpy()
        obs_seq_len = 5
        pred_seq_len = 5
        invalid_value = -999.

        # *** Process input data
        num_peds_batch = 0
        b_num_peds = []
        b_obs_traj, b_obs_traj_rel, b_loss_mask_rel, b_loss_mask, b_v_obs, b_A_obs, b_attn_mask_obs = \
            [],[],[],[],[],[],[]
        for i in range(self.nenv):
            input_traj_i = input_traj[i]
            input_binary_mask_i = input_binary_mask[i]
            obs_traj = np.transpose(input_traj_i, (0,2,1)) # (num_peds, 2, obs_seq_len)
            n_peds = obs_traj.shape[0]
            loss_mask_obs = input_binary_mask_i[:,:,0] # (num_peds, obs_seq_len)
            loss_mask_rel_obs = loss_mask_obs[:,:-1] * loss_mask_obs[:,-1:]
            loss_mask_rel_obs = np.concatenate((loss_mask_obs[:,0:1], loss_mask_rel_obs), axis=1) # (num_peds, obs_seq_len)
            loss_mask_rel_pred = (np.ones((n_peds, pred_seq_len)) * loss_mask_rel_obs[:,-1:]).astype('bool') # (num_peds)
            obs_traj_rel = obs_traj[:,:,1:] - obs_traj[:,:,:-1]
            obs_traj_rel = np.concatenate((np.zeros((n_peds, 2, 1)), obs_traj_rel), axis=2)
            obs_traj_rel = invalid_value*np.ones_like(obs_traj_rel)*(1-loss_mask_rel_obs[:,np.newaxis,:]) \
                + obs_traj_rel*loss_mask_rel_obs[:,np.newaxis,:]
            obs_traj = torch.from_numpy(obs_traj)
            obs_traj_rel = torch.from_numpy(obs_traj_rel)
            # v_obs, A_obs = seq_to_graph(obs_traj, obs_traj_rel, attn_mech='rel_conv')
            # v_obs, A_obs = v_obs.unsqueeze(0), A_obs.unsqueeze(0)
            loss_mask_rel = np.concatenate((loss_mask_rel_obs, loss_mask_rel_pred), axis=1)[np.newaxis,:,:] # (1, num_peds, seq_len)
            loss_mask_rel = torch.from_numpy(loss_mask_rel)
            loss_mask_rel_obs = torch.from_numpy(loss_mask_rel_obs[np.newaxis,:,:])
            attn_mask_obs = []
            for tt in range(obs_seq_len):
                loss_mask_rel_tt = loss_mask_rel_obs[0,:,tt] # (n_peds,)
                attn_mask_obs.append(torch.outer(loss_mask_rel_tt, loss_mask_rel_tt).float()) # (n_peds, n_peds)
            attn_mask_obs = torch.stack(attn_mask_obs, dim=0).unsqueeze(0) # (1,obs_seq_len, n_peds, n_peds)
            obs_traj = obs_traj.unsqueeze(0)
            obs_traj_rel = obs_traj_rel.unsqueeze(0)
            loss_mask_pred = loss_mask_rel_pred
            loss_mask = np.concatenate((loss_mask_obs, loss_mask_pred), axis=1)
            loss_mask = torch.from_numpy(loss_mask).unsqueeze(0)
            num_peds_batch += obs_traj.shape[1]
            b_num_peds.append(obs_traj.shape[1])
            b_obs_traj.append(obs_traj[0].float())
            b_obs_traj_rel.append(obs_traj_rel[0].float())
            b_loss_mask_rel.append(loss_mask_rel[0].float())
            b_loss_mask.append(loss_mask[0].float())
            # b_v_obs.append(v_obs[0].float())
            # b_A_obs.append(A_obs[0].float())
            b_attn_mask_obs.append(attn_mask_obs[0].float())
        b_obs_traj = torch.cat(b_obs_traj, dim=0)
        b_obs_traj_rel = torch.cat(b_obs_traj_rel, dim=0)
        b_loss_mask_rel = torch.cat(b_loss_mask_rel, dim=0)
        b_loss_mask = torch.cat(b_loss_mask, dim=0)
        b_v_obs, b_A_obs = seq_to_graph(b_obs_traj, b_obs_traj_rel, attn_mech='rel_conv')
        b_attn_mask_obs_tensor = torch.zeros(obs_seq_len, num_peds_batch, num_peds_batch).float()
        curr_ped_idx = 0
        for num_peds, attn_mask_obs_single in zip(b_num_peds, b_attn_mask_obs):
            b_attn_mask_obs_tensor[:,curr_ped_idx:curr_ped_idx+num_peds, \
                curr_ped_idx:curr_ped_idx+num_peds] = attn_mask_obs_single
            curr_ped_idx += num_peds
        b_obs_traj = b_obs_traj.unsqueeze(0)
        b_obs_traj_rel = b_obs_traj_rel.unsqueeze(0)
        b_v_obs = b_v_obs.unsqueeze(0)
        b_A_obs = b_A_obs.unsqueeze(0)
        b_loss_mask_rel = b_loss_mask_rel.unsqueeze(0)
        b_attn_mask_obs_tensor = b_attn_mask_obs_tensor.unsqueeze(0)
        b_loss_mask = b_loss_mask.unsqueeze(0)
        # print("PROCESSED DATA")
        # print("b_obs_traj shape: ", b_obs_traj.shape)
        # print("b_obs_traj_rel shape: ", b_obs_traj_rel.shape)
        # print("b_v_obs shape: ", b_v_obs.shape)
        # print("b_A_obs shape: ", b_A_obs.shape)
        # print("b_loss_mask_rel shape: ", b_loss_mask_rel.shape)
        # print("b_attn_mask_obs_tensor shape: ", b_attn_mask_obs_tensor.shape)
        # print("b_loss_mask shape: ", b_loss_mask.shape)
        # print()

        # *** Perform trajectory prediction
        sampling = False
        with torch.no_grad():

            b_v_obs, b_A_obs, b_attn_mask_obs_tensor, b_loss_mask_rel = \
                b_v_obs.to(self.device), b_A_obs.to(self.device), \
                b_attn_mask_obs_tensor.float().to(self.device), b_loss_mask_rel.float().to(self.device)
            results = self.model(b_v_obs, b_A_obs, b_attn_mask_obs_tensor, b_loss_mask_rel, tau=0.03, hard=True, sampling=sampling, device=self.device)
            gaussian_params_pred, x_sample_pred, info = results
        mu, sx, sy, corr = gaussian_params_pred
        # print(mu.shape) # torch.Size([1, 5, 48, 2])
        # print(sx.shape)
        # print(sy.shape)
        # print(corr.shape)
        # print(b_obs_traj.shape)
        # print(b_loss_mask.shape)
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
        mu_cumsum = mu.detach().to("cpu") + np.transpose(b_obs_traj[:,:,:,-1:], (0,3,1,2)) # (batch, time, node, 2)
        loss_mask_pred = b_loss_mask[:,:,obs_seq_len:].permute(0,2,1).unsqueeze(-1).bool()
        mu_cumsum = mu_cumsum * loss_mask_pred + invalid_value*(~loss_mask_pred)
        output_traj = torch.cat((mu_cumsum.detach().to("cpu"), sx_cumsum.detach().to("cpu"), sy_cumsum.detach().to("cpu"), corr_cumsum.detach().to("cpu")), dim=3)
        output_traj = output_traj.permute(0, 2, 1, 3)[0] # (ped, time, 5)
        output_binary_mask = loss_mask_pred[0,0,:,:].detach().to("cpu") # (ped, 1)
        num_total_peds = output_traj.shape[0]
        output_traj = output_traj.reshape(self.nenv, num_total_peds // self.nenv, output_traj.shape[1], output_traj.shape[2]) # (n_env, ped, time, 5)
        output_binary_mask = output_binary_mask.reshape(self.nenv, num_total_peds // self.nenv, 1) # (n_env, ped, 1)

        output_traj, output_binary_mask = output_traj.to(self.device), output_binary_mask.to(self.device)
        # reshape has been tested
        # print("PERFORMED PREDICTION")
        # print("mu_cumsum shape: ", mu_cumsum.shape)
        # print("sx_cumsum shape: ", sx_cumsum.shape)
        # print("sy_cumsum shape: ", sy_cumsum.shape)
        # print("corr_cumsum shape: ", corr_cumsum.shape)
        # print("output_traj device: ", output_traj.device)
        # print("output_binary_mask device: ", output_binary_mask.device)
        # print("output_traj shape: ", output_traj.shape)
        # print("output_binary_mask shape: ", output_binary_mask.shape)
        # print()
        return output_traj, output_binary_mask

if __name__ == '__main__':
    # *** Create an input that aligns with the format of CrowdNav
    obs_seq_len = 5
    pred_seq_len = 5
    invalid_value = -999.
    ped_step = 0.56
    x_init = 2.
    ped_0 = np.stack((x_init+np.arange((obs_seq_len-1)*ped_step, -ped_step, -ped_step),np.zeros(obs_seq_len)), 0)
    ped_1 = -ped_0 # (2,8)
    ped_3 = np.stack((np.zeros(obs_seq_len), x_init+np.arange((obs_seq_len-1)*ped_step, -ped_step, -ped_step)), 0)
    ped_2 = -ped_3
    ped_4 = np.ones((2,obs_seq_len))*invalid_value
    ped_5 = np.ones((2,obs_seq_len))*invalid_value
    obs_traj_undamaged = np.stack((ped_0,ped_1,ped_2,ped_3,ped_4,ped_5), axis=0) # (num_peds, 2, obs_seq_len)
    n_peds = obs_traj_undamaged.shape[0]
    original_obs_traj = obs_traj_undamaged # (num_peds, 2, obs_seq_len)
    # 0: right, 1: left, 2: up, 3: down, 4,5: invalid agent
    original_obs_traj[2,:,:2] = invalid_value
    original_obs_traj[3,:,:2] = invalid_value
    # * input_traj and input_binary_mask are example inputs from CrowdNav
    input_traj = np.transpose(original_obs_traj, (0,2,1)) # (num_peds, obs_seq_len, 2)
    input_binary_mask = (input_traj!=invalid_value).prod(2).astype('bool')[:,:,np.newaxis] # (num_peds, obs_seq_len, 1)

    n_env = 8
    input_traj = np.stack([input_traj for i in range(n_env)], axis=0) # (n_env, num_peds, obs_seq_len, 2)
    input_binary_mask = np.stack([input_binary_mask for i in range(n_env)],axis=0) # (n_env, num_peds, obs_seq_len, 1)
    print()
    print("INPUT DATA")
    print("input_traj shape: ", input_traj.shape)
    print("input_binary_mask shape:", input_binary_mask.shape)
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000"
    model = CrowdNavPredInterfaceMultiEnv(load_path='/home/shuijing/Desktop/CrowdNav_Prediction/gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj',
                                          device=device, config = args, num_env=n_env)
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