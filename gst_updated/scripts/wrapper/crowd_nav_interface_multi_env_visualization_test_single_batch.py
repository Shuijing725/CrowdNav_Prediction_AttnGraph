import pathhack
import pickle
from os.path import join, isdir
import torch
import numpy as np
from src.mgnn.utils import seq_to_graph
from src.gumbel_social_transformer.st_model import st_model
import matplotlib.pyplot as plt

# from gst_updated.src.mgnn.utils import average_offset_error, final_offset_error, args2writername, load_batch_dataset
# from gst_updated.src.gumbel_social_transformer.st_model import st_model, offset_error_square_full_partial,\
#     negative_log_likelihood_full_partial
from torch.utils.data import DataLoader

def load_batch_dataset(pkg_path, subfolder='test', num_workers=4, shuffle=None):
    result_filename = 'sj_dset_'+subfolder+'_trajectories.pt'
    dataset_folderpath = join(pkg_path, 'datasets/shuijing/orca_20humans_fov')
    dset = torch.load(join(dataset_folderpath, result_filename))
    if shuffle is None:
        if subfolder == 'train':
            shuffle = True
        else:
            shuffle = False
    dloader = DataLoader(
        dset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers)
    return dloader

loader_test = load_batch_dataset(pathhack.pkg_path, subfolder='test')
print(len(loader_test))

test_batches = []
max_num_peds = 0
for batch_idx, batch in enumerate(loader_test):
    if batch_idx % 1000 == 0 and len(test_batches) < 4 and batch_idx != 0:
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
        v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt = batch
        test_batches.append(batch)
        print("obs traj shape: ", obs_traj.shape)
        print("loss mask shape: ", loss_mask.shape)
        if max_num_peds < obs_traj.shape[1]:
            max_num_peds = obs_traj.shape[1]
            print("max number of pedestrians: ", max_num_peds)

wrapper_input_traj = []
wrapper_input_binary_mask = []
wrapper_output_traj_gt = []
wrapper_output_binary_mask_gt = []

invalid_value = -999.
for batch in test_batches:
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
        v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt = batch
    input_traj = obs_traj[0].permute(0,2,1) # (n_peds, obs_seq_len,  2)
    input_binary_mask = loss_mask[0,:,:input_traj.shape[1]].unsqueeze(2) # (n_peds, obs_seq_len, 1)
    # print(input_binary_mask.shape)
    output_traj_gt = pred_traj_gt[0].permute(0,2,1) # (num_peds, pred_seq_len, 2)
    output_binary_mask_gt = loss_mask[0,:,input_traj.shape[1]:].unsqueeze(2) # (n_peds, pred_seq_len, 1)
    n_peds, obs_seq_len, pred_seq_len = input_traj.shape[0], input_traj.shape[1], output_traj_gt.shape[1]
    if n_peds < max_num_peds:
        input_traj_complement = torch.ones(max_num_peds-n_peds, obs_seq_len, 2)*invalid_value
        input_binary_mask_complement = torch.zeros(max_num_peds-n_peds, obs_seq_len, 1)
        output_traj_gt_complement = torch.ones(max_num_peds-n_peds, pred_seq_len, 2)*invalid_value
        output_binary_mask_gt_complement = torch.zeros(max_num_peds-n_peds, pred_seq_len, 1)

        input_traj = torch.cat((input_traj, input_traj_complement), dim=0)
        input_binary_mask = torch.cat((input_binary_mask, input_binary_mask_complement), dim=0)
        output_traj_gt = torch.cat((output_traj_gt, output_traj_gt_complement), dim=0)
        output_binary_mask_gt = torch.cat((output_binary_mask_gt, output_binary_mask_gt_complement), dim=0)
    
    wrapper_input_traj.append(input_traj)
    wrapper_input_binary_mask.append(input_binary_mask)
    wrapper_output_traj_gt.append(output_traj_gt)
    wrapper_output_binary_mask_gt.append(output_binary_mask_gt)

wrapper_input_traj = torch.stack(wrapper_input_traj, dim=0)
wrapper_input_binary_mask = torch.stack(wrapper_input_binary_mask, dim=0)
wrapper_output_traj_gt = torch.stack(wrapper_output_traj_gt, dim=0)
wrapper_output_binary_mask_gt = torch.stack(wrapper_output_binary_mask_gt, dim=0)
print("wrapper_input_traj shape: ", wrapper_input_traj.shape)
print("wrapper_input_binary_mask shape: ", wrapper_input_binary_mask.shape)  
print("wrapper_output_traj_gt shape: ", wrapper_output_traj_gt.shape)
print("wrapper_output_binary_mask_gt shape: ", wrapper_output_binary_mask_gt.shape)  
# wrapper_input_traj shape:  torch.Size([4, 15, 5, 2])
# wrapper_input_binary_mask shape:  torch.Size([4, 15, 5, 1])
# wrapper_output_traj_gt shape:  torch.Size([4, 15, 5, 2])
# wrapper_output_binary_mask_gt shape:  torch.Size([4, 15, 5, 1])

n_env = wrapper_input_traj.shape[0]
input_traj = wrapper_input_traj.numpy()
input_binary_mask = wrapper_input_binary_mask.numpy()
print()
print("INPUT DATA")
print("input_traj shape: ", input_traj.shape)
print("input_binary_mask shape:", input_binary_mask.shape)
print()

# *** Visualize input data
def visualize_trajectory(obs_traj, loss_mask, sample_index, obs_seq_len=5, pred_seq_len=5):
    ##### Print Visualization Started #####
    n_peds, seq_len = obs_traj.shape[1], obs_seq_len+pred_seq_len
    full_ped_idx = torch.where(loss_mask.sum(2)[0]==seq_len)[0] # loss_mask tensor: (1, num_peds, seq_len)
    fig, ax = plt.subplots()
    # ax.set_xlim((-8, 8))
    # ax.set_ylim((-5, 5))
    fig.set_tight_layout(True)
    for ped_idx in range(n_peds):
        if ped_idx in full_ped_idx:
            ax.plot(obs_traj[0, ped_idx, 0, :obs_seq_len], obs_traj[0, ped_idx, 1, :obs_seq_len], '.-', c='k') # black for obs
        else:
            for t_idx in range(seq_len):
                if loss_mask[0,ped_idx,t_idx] == 1:
                    if t_idx < obs_seq_len:
                        # obs blue for partially detected pedestrians
                        ax.plot(obs_traj[0, ped_idx, 0, t_idx], obs_traj[0, ped_idx, 1, t_idx], '.', c='b')
    ax.set_aspect('equal', adjustable='box')
    ax.plot()
    fig.savefig("tmp_img_to_be_deleted_"+str(sample_index)+".png")
    print("tmp_img_to_be_deleted_"+str(sample_index)+".png is created.")
    return


# visualize_trajectory(obs_traj, loss_mask, sample_index, obs_seq_len=obs_seq_len, pred_seq_len=obs_seq_len)




def CrowdNavPredInterfaceMultiEnv(
    input_traj,
    input_binary_mask,
    sampling = True,
):
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
    # *** Process input data
    num_peds_batch = 0
    b_num_peds = []
    b_obs_traj, b_obs_traj_rel, b_loss_mask_rel, b_loss_mask, b_v_obs, b_A_obs, b_attn_mask_obs = \
        [],[],[],[],[],[],[]
    n_env = 1
    for i in range(n_env):
        input_traj_i = input_traj[i]
        input_binary_mask_i = input_binary_mask[i]
        obs_traj = np.transpose(input_traj_i, (0,2,1)) # (num_peds, 2, obs_seq_len)
        n_peds = obs_traj.shape[0]
        loss_mask_obs = input_binary_mask_i[:,:,0] # (num_peds, obs_seq_len)
        loss_mask_rel_obs = loss_mask_obs[:,:-1] * loss_mask_obs[:,-1:]
        loss_mask_rel_obs = np.concatenate((loss_mask_obs[:,0:1], loss_mask_rel_obs), axis=1) # (num_peds, obs_seq_len)
        # import pdb; pdb.set_trace()
        loss_mask_rel_pred = (np.ones((n_peds, pred_seq_len)) * loss_mask_rel_obs[:,-1:]).astype('bool') # (num_peds)
        obs_traj_rel = obs_traj[:,:,1:] - obs_traj[:,:,:-1]
        obs_traj_rel = np.concatenate((np.zeros((n_peds, 2, 1)), obs_traj_rel), axis=2)
        obs_traj_rel = invalid_value*np.ones_like(obs_traj_rel)*(1-loss_mask_rel_obs[:,np.newaxis,:]) \
            + obs_traj_rel*loss_mask_rel_obs[:,np.newaxis,:]
        obs_traj = torch.from_numpy(obs_traj)
        obs_traj_rel = torch.from_numpy(obs_traj_rel)
        v_obs, A_obs = seq_to_graph(obs_traj, obs_traj_rel, attn_mech='rel_conv')
        v_obs, A_obs = v_obs.unsqueeze(0), A_obs.unsqueeze(0)
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
        visualize_trajectory(obs_traj, loss_mask, i, obs_seq_len=obs_seq_len, pred_seq_len=obs_seq_len)
        num_peds_batch += obs_traj.shape[1]
        b_num_peds.append(obs_traj.shape[1])
        b_obs_traj.append(obs_traj[0].float())
        b_obs_traj_rel.append(obs_traj_rel[0].float())
        b_loss_mask_rel.append(loss_mask_rel[0].float())
        b_loss_mask.append(loss_mask[0].float())
        b_v_obs.append(v_obs[0].float())
        b_A_obs.append(A_obs[0].float())
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
    # *** Load model
    writername = "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000"
    dataset = "sj" # Shui Jing
    logdir = join(pathhack.pkg_path, 'results', writername, dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not isdir(logdir):
        raise RuntimeError('The result directory was not found.')
    checkpoint_dir = join(logdir, 'checkpoint')
    with open(join(checkpoint_dir, 'args.pickle'), 'rb') as f:
        args_eval = pickle.load(f)
    # Uncomment if you want a fixed random seed.
    # torch.manual_seed(args_eval.random_seed)
    # np.random.seed(args_eval.random_seed)
    model = st_model(args_eval, device=device).to(device)
    model_filename = 'epoch_'+str(args_eval.num_epochs)+'.pt'
    model_checkpoint = torch.load(join(checkpoint_dir, model_filename), map_location=device)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    # print("LOADED MODEL")
    # print("device: ", device)
    # print('Loaded configuration: ', writername)
    # print()
    # *** Perform trajectory prediction
    sampling = False
    with torch.no_grad():
        model.eval()
        b_v_obs, b_A_obs, b_attn_mask_obs_tensor, b_loss_mask_rel = \
            b_v_obs.to(device), b_A_obs.to(device), \
            b_attn_mask_obs_tensor.float().to(device), b_loss_mask_rel.float().to(device)
        results = model(b_v_obs, b_A_obs, b_attn_mask_obs_tensor, b_loss_mask_rel, tau=0.03, hard=True, sampling=sampling, device=device)
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
    output_traj = output_traj.reshape(n_env, num_total_peds // n_env, output_traj.shape[1], output_traj.shape[2]) # (n_env, ped, time, 5)
    output_binary_mask = output_binary_mask.reshape(n_env, num_total_peds // n_env, 1) # (n_env, ped, 1)
    output_traj, output_binary_mask = output_traj.numpy(), output_binary_mask.numpy()
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

output_traj, output_binary_mask = CrowdNavPredInterfaceMultiEnv(
    input_traj[:1],
    input_binary_mask[:1],
    sampling = True,
)
print()
print("OUTPUT DATA")
print("output_traj shape: ", output_traj.shape)
print("output_binary_mask shape:", output_binary_mask.shape)
print()


# wrapper_input_traj shape:  torch.Size([4, 15, 5, 2])
# wrapper_input_binary_mask shape:  torch.Size([4, 15, 5, 1])
# wrapper_output_traj_gt shape:  torch.Size([4, 15, 5, 2])
# wrapper_output_binary_mask_gt shape:  torch.Size([4, 15, 5, 1])
# OUTPUT DATA
# output_traj shape:  torch.Size([4, 15, 5, 5])
# output_binary_mask shape: torch.Size([4, 15, 1])
# n_env = wrapper_input_traj.shape[0]
# input_traj = wrapper_input_traj.numpy()
# input_binary_mask = wrapper_input_binary_mask.numpy()
# print()
# print("INPUT DATA")
# print("input_traj shape: ", input_traj.shape)
# print("input_binary_mask shape:", input_binary_mask.shape)
# print()
def visualize_output_trajectory_deterministic(input_traj, input_binary_mask, output_traj, output_binary_mask, sample_index, obs_seq_len=5, pred_seq_len=5):
    ##### Print Visualization Started #####
    input_traj_i = input_traj[sample_index] #15, 5, 2])
    input_binary_mask_i = input_binary_mask[sample_index] #15, 5, 1])
    output_traj_i = output_traj[sample_index] #15, 5, 5])
    output_binary_mask_i = output_binary_mask[sample_index] #15, 1])
    n_peds, seq_len = input_traj_i.shape[0], obs_seq_len+pred_seq_len
    full_obs_ped_idx = np.where(input_binary_mask_i.sum(1)[:,0]==obs_seq_len)[0]
    full_traj = np.concatenate((input_traj_i, output_traj_i[:,:,:2]), axis=1) # (15, 10, 2)
    output_binary_mask_i_pred_len = np.stack([output_binary_mask_i for j in range(pred_seq_len)], axis=1) # (15, 5, 1)

    loss_mask = np.concatenate((input_binary_mask_i, output_binary_mask_i_pred_len), axis=1) # (15, 10, 1)

    fig, ax = plt.subplots()
    # ax.set_xlim((-8, 8))
    # ax.set_ylim((-5, 5))
    fig.set_tight_layout(True)
    for ped_idx in range(n_peds):
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
    fig.savefig("SINGLE_pred_tmp_img_to_be_deleted_"+str(sample_index)+".png")
    print("SINGLE_pred_tmp_img_to_be_deleted_"+str(sample_index)+".png is created.")
    return

for sample_index in range(n_env):
    visualize_output_trajectory_deterministic(input_traj, input_binary_mask, output_traj, output_binary_mask, sample_index, obs_seq_len=5, pred_seq_len=5)


wrapper_visualization_data = {}
wrapper_visualization_data['input_traj'] = input_traj
wrapper_visualization_data['input_mask'] = input_binary_mask
wrapper_visualization_data['output_traj'] = output_traj
wrapper_visualization_data['output_mask'] = output_binary_mask
wrapper_visualization_data['output_traj_gt'] = wrapper_output_traj_gt.numpy()
wrapper_visualization_data['output_mask_gt'] = wrapper_output_binary_mask_gt.numpy()
with open(join('wrapper_visualization_data.pickle'), 'wb') as f:
    pickle.dump(wrapper_visualization_data, f)
    print("wrapper_visualization_data.pickle is dumped.")
# output_traj, output_binary_mask

# wrapper_input_traj = torch.stack(wrapper_input_traj, dim=0)
# wrapper_input_binary_mask = torch.stack(wrapper_input_binary_mask, dim=0)
# wrapper_output_traj_gt = torch.stack(wrapper_output_traj_gt, dim=0)
# wrapper_output_binary_mask_gt = torch.stack(wrapper_output_binary_mask_gt, dim=0)
# print("wrapper_input_traj shape: ", wrapper_input_traj.shape)
# print("wrapper_input_binary_mask shape: ", wrapper_input_binary_mask.shape)  
# print("wrapper_output_traj_gt shape: ", wrapper_output_traj_gt.shape)
# print("wrapper_output_binary_mask_gt shape: ", wrapper_output_binary_mask_gt.shape)  
# wrapper_input_traj shape:  torch.Size([4, 15, 5, 2])
# wrapper_input_binary_mask shape:  torch.Size([4, 15, 5, 1])
# wrapper_output_traj_gt shape:  torch.Size([4, 15, 5, 2])
# wrapper_output_binary_mask_gt shape:  torch.Size([4, 15, 5, 1])

# n_env = wrapper_input_traj.shape[0]
# input_traj = wrapper_input_traj.numpy()
# input_binary_mask = wrapper_input_binary_mask.numpy()

# with open(join(checkpoint_dir, 'args.pickle'), 'wb') as f:
#     pickle.dump(args, f)

# def visualize_trajectory(obs_traj, loss_mask, sample_index, obs_seq_len=5, pred_seq_len=5):
#     ##### Print Visualization Started #####
#     n_peds, seq_len = obs_traj.shape[1], obs_seq_len+pred_seq_len
#     full_ped_idx = torch.where(loss_mask.sum(2)[0]==seq_len)[0] # loss_mask tensor: (1, num_peds, seq_len)
#     fig, ax = plt.subplots()
#     # ax.set_xlim((-8, 8))
#     # ax.set_ylim((-5, 5))
#     fig.set_tight_layout(True)
#     for ped_idx in range(n_peds):
#         if ped_idx in full_ped_idx:
#             ax.plot(obs_traj[0, ped_idx, 0, :obs_seq_len], obs_traj[0, ped_idx, 1, :obs_seq_len], '.-', c='k') # black for obs
#         else:
#             for t_idx in range(seq_len):
#                 if loss_mask[0,ped_idx,t_idx] == 1:
#                     if t_idx < obs_seq_len:
#                         # obs blue for partially detected pedestrians
#                         ax.plot(obs_traj[0, ped_idx, 0, t_idx], obs_traj[0, ped_idx, 1, t_idx], '.', c='b')
#     ax.set_aspect('equal', adjustable='box')
#     ax.plot()
#     fig.savefig("tmp_img_to_be_deleted_"+str(sample_index)+".png")
#     print("tmp_img_to_be_deleted_"+str(sample_index)+".png is created.")
#     return


# # lstm 
# torch.manual_seed(args.random_seed)
# np.random.seed(args.random_seed)
# target_ped_idx_from_full = 1#2#0#1#2 # 0-2 for zara1
# # print(full_ped_idx[target_ped_idx_from_full])
# with torch.no_grad():
#     model.eval()
#     v_obs, A_obs, attn_mask_obs, loss_mask_rel = \
#         v_obs.to(device), A_obs.to(device), \
#         attn_mask_obs.float().to(device), loss_mask_rel.float().to(device)

#     results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=True, sampling=False, device=device)
#     # gaussian_params_pred, x_sample_pred, _, loss_mask_per_pedestrian = results
#     gaussian_params_pred, x_sample_pred, info = results
    
#     pred_traj_rel = torch.cat((obs_traj[:,:,:,-1:], x_sample_pred.permute(0,2,3,1)), dim=3) # (1, num_peds, 2, 1+pred_seq_len)
#     pred_traj = pred_traj_rel.cumsum(dim=3).to('cpu')
    
    
#     # full_traj = torch.cat((obs_traj, pred_traj_gt), dim=3) # (1, num_peds, 2, seq_len)  
#     full_traj = torch.cat((obs_traj[:,:,:,:-1], pred_traj), dim=3) # (1, num_peds, 2, seq_len)  
#     n_peds, seq_len = full_traj.shape[1], full_traj.shape[3]
#     # full_ped_idx = torch.where(loss_mask.sum(2)[0]==seq_len)[0] # # loss_mask tensor: (1, num_peds, seq_len)
# #     full_ped_idx = torch.where(info['loss_mask_rel_full_partial']==1)[1]
# #     info['loss_mask_rel_full_partial']
#     full_ped_idx = torch.where(info['loss_mask_per_pedestrian']==1)[1]
#     partial_full_ped_idx = torch.where(info['loss_mask_rel_full_partial']==1)[1]

#     fig, ax = plt.subplots()
#     ax.set_xlim((-8, 8))
#     ax.set_ylim((-5, 5))
#     fig.set_tight_layout(True)
#     for ped_idx in range(n_peds):
#         if ped_idx in partial_full_ped_idx:
#             if ped_idx in full_ped_idx:
# #                 ax.plot(full_traj[0, ped_idx, 0, 7:], full_traj[0, ped_idx, 1, 7:], '.-', c='C2') # green for pred gt
#                 ax.plot(full_traj[0, ped_idx, 0, :8], full_traj[0, ped_idx, 1, :8], '.-', c='k') # black for obs
#                 ax.plot(pred_traj[0, ped_idx, 0], pred_traj[0, ped_idx, 1], '.-', c='C1',alpha=0.2) # orange for pred           
#         if ped_idx not in full_ped_idx:
#             for t_idx in range(seq_len):
#                 if loss_mask[0,ped_idx,t_idx] == 1:
#                     if t_idx < 8:
#                         ax.plot(full_traj[0, ped_idx, 0, t_idx], full_traj[0, ped_idx, 1, t_idx], '.', c='b')
#                         # obs blue
#                     else:
#                         ax.plot(full_traj[0, ped_idx, 0, t_idx], full_traj[0, ped_idx, 1, t_idx], '.', c='r')
#                         # pred red
#     ax.set_aspect('equal', adjustable='box')
#     ax.plot()
    