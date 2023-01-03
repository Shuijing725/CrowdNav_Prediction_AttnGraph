from os.path import join
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader


def average_offset_error(x_pred, x_target, loss_mask=None):
    assert x_pred.shape[0] == 1
    pos_pred = torch.cumsum(x_pred, dim=1)
    pos_target = torch.cumsum(x_target, dim=1)
    offset_error = torch.sqrt(((pos_pred-pos_target)**2.).sum(3))[0]
    aoe = offset_error.mean(0)
    if loss_mask is not None:
        aoe = aoe * loss_mask[0]
    return aoe

def final_offset_error(x_pred, x_target, loss_mask=None):
    assert x_pred.shape[0] == 1
    pos_pred = torch.cumsum(x_pred, dim=1)
    pos_target = torch.cumsum(x_target, dim=1)
    offset_error = torch.sqrt(((pos_pred-pos_target)**2.).sum(3))[0]
    foe = offset_error[-1]
    if loss_mask is not None:
        foe = foe * loss_mask[0]
    return foe


def negative_log_likelihood(gaussian_params, x_target, loss_mask=None):
    mu, sx, sy, corr = gaussian_params
    assert mu.shape[0] == 1
    sigma = torch.cat((sx, sy), dim=3)
    x_target_norm = (x_target-mu)/sigma
    nx, ny = x_target_norm[:,:,:,0:1], x_target_norm[:,:,:,1:2]
    loss_term_1 = torch.log(1.-corr**2.)/2.+torch.log(sx)+torch.log(sy)
    loss_term_2 = (nx**2.-2.*corr*nx*ny+ny**2.)/(2.*(1.-corr**2.))
    loss = loss_term_1+loss_term_2
    loss = loss.squeeze(3).mean(1)
    loss = (loss[0] * loss_mask[0]).sum()/loss_mask[0].sum()
    return loss


def seq_to_graph(seq_, seq_rel, attn_mech='glob_kip'):
    if len(seq_.shape) == 4:
        seq_ = seq_[0]
        seq_rel = seq_rel[0]
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))

    for s in range(seq_len):
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_rel)):
            V[s, h, :] = step_rel[h]

    if attn_mech == 'plain':
        A = np.ones((seq_len, max_nodes, max_nodes))*(1./max_nodes)
    elif attn_mech == 'rel_conv':
        A = []
        for tt in range(seq_len):
            x = seq_[:,:,tt]
            x_row = torch.ones(max_nodes,max_nodes,2)*x.view(max_nodes,1,2)
            x_col = torch.ones(max_nodes,max_nodes,2)*x.view(1,max_nodes,2)
            edge_attr = x_row-x_col
            A.append(edge_attr.numpy())
        A = np.stack(A, axis=0)
    else:
        raise RuntimeError('Wrong attention mechanism.')

    return torch.from_numpy(V).type(torch.float),\
        torch.from_numpy(A).type(torch.float)


def rotate_graph(vtx, adj, theta):
    vtx_rotated_x = vtx[:,:,:,0:1]*np.cos(theta)-vtx[:,:,:,1:2]*np.sin(theta)
    vtx_rotated_y = vtx[:,:,:,0:1]*np.sin(theta)+vtx[:,:,:,1:2]*np.cos(theta)
    vtx_rotated = torch.cat((vtx_rotated_x, vtx_rotated_y), dim=3)

    adj_rotated_x = adj[:,:,:,:,0:1]*np.cos(theta)-adj[:,:,:,:,1:2]*np.sin(theta)
    adj_rotated_y = adj[:,:,:,:,0:1]*np.sin(theta)+adj[:,:,:,:,1:2]*np.cos(theta)
    adj_rotated = torch.cat((adj_rotated_x, adj_rotated_y), dim=4)

    return vtx_rotated, adj_rotated

def random_rotate_graph(args, vtx_obs, adj_obs, vtx_pred_gt, adj_pred_gt):
    if args.rotation_pattern is None:
        raise RuntimeError('random_rotate_seq should not be called when rotation_pattern is None.')

    if args.rotation_pattern == 'right_angle':
        theta = (torch.randint(0, 4, ()).float()/2.*np.pi).item()
    elif args.rotation_pattern == 'random':
        theta = (torch.rand(())*2.*np.pi).item()
    else:
        raise RuntimeError('Rotation pattern is not found in args.')
    vtx_obs_rotated, adj_obs_rotated = rotate_graph(vtx_obs, adj_obs, theta)
    vtx_pred_gt_rotated, adj_pred_gt_rotated = rotate_graph(vtx_pred_gt, adj_pred_gt, theta)
    return (vtx_obs_rotated, adj_obs_rotated, vtx_pred_gt_rotated, adj_pred_gt_rotated), theta

def load_batch_dataset(args, pkg_path, subfolder='train', num_workers=4, shuffle=None):
    result_filename = args.dataset+'_dset_'+subfolder+'_batch_trajectories.pt'
    if args.dataset == 'sdd':
        dataset_folderpath = join(pkg_path, 'datasets/sdd/social_pool_data')
    elif args.dataset == 'real' or args.dataset == 'synth' or args.dataset == 'all':
        dataset_folderpath = join(pkg_path, 'datasets/trajnet++/train/')
    elif args.dataset == 'deathCircle' or args.dataset == 'hyang':
        dataset_folderpath = join(pkg_path, 'datasets/sdd', args.dataset)
    elif args.dataset == 'sj':
        dataset_folderpath = join(pkg_path, 'datasets/shuijing/orca_20humans_fov')
    else:
        # dataset_folderpath = join(pkg_path, 'datasets/eth_ucy', args.dataset)
        dataset_folderpath = join(pkg_path, 'datasets/self_eth_ucy', args.dataset)
        print("self_eth_ucy")
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

def args2writername(args):
    writername = str(args.temp_epochs)+'-'+str(args.spatial)\
        +'-'+str(args.temporal)+'-lr_'+str(args.lr)
    if args.deterministic:
        writername = writername + '-deterministic'
    # if args.init_temp != 1.:
    writername = writername + '-init_temp_'+str(args.init_temp)
    if args.clip_grad is not None:
        writername = writername + '-clip_grad_'+str(args.clip_grad)
    # if args.spatial_num_heads_edges != 4:
    writername = writername + '-edge_head_'+str(args.spatial_num_heads_edges)
    if args.only_observe_full_period:
        writername = writername + '-only_full'
    if args.detach_sample:
        writername = writername + '-detach'
    # if args.embedding_size != 64:
    writername = writername + '-ebd_'+str(args.embedding_size)
    # if args.spatial_num_layers != 3:
    writername = writername + '-snl_'+str(args.spatial_num_layers)
    # if args.spatial_num_heads != 8:
    writername = writername + '-snh_'+str(args.spatial_num_heads)
    if args.ghost:
        writername = writername + '-ghost'
    writername = writername + '-seed_'+str(args.random_seed)
    return writername


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial', default='rel_conv',
                        help='spatial encoding methods: rel_conv, plain')
    parser.add_argument('--temporal', default='lstm',
                        help='temporal encoding methods: lstm, temporal_convolution_net, faster_lstm.')                    
    parser.add_argument('--output_dim', type=int, default=5,
                        help='5 for mu_x, mu_y, sigma_x, sigma_y, corr')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--spatial_num_heads', type=int, default=8,
                        help='number of heads for multi-head \
                        attention mechanism in spatial encoding')
    parser.add_argument('--lstm_hidden_size', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=1)          
    parser.add_argument('--decode_style', default='recursive',
                        help='recursive, readout')
    parser.add_argument('--detach_sample', action='store_true',
                        help='Default False means using reparameterization trick. \
                        True means disable the trick and cut gradient flow between \
                        gaussian parameters and samples in prediction period.')
    parser.add_argument('--motion_dim', type=int, default=2,
                        help='pedestrian motion is 2D')
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2')
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--rotation_pattern', default=None,
                        help='rotation pattern used for data augmentation in training \
                        phase: right_angle, or random. None means no rotation.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gradient clipping')
    parser.add_argument('--organize_csv', action='store_true',
                        help='Organize test_performance.csv towards performance_organized.csv.')
    parser.add_argument('--spatial_num_layers', type=int, default=3)
    parser.add_argument('--only_observe_full_period', action='store_true',
                        help='Only observe pedestrians that appear in the full period.')
    parser.add_argument('--spatial_num_heads_edges', type=int, default=4)
    parser.add_argument('--ghost', action='store_true')
    parser.add_argument('--random_seed', type=int, default=1000)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--init_temp', type=float, default=1.)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--temp_epochs', type=int, default=200)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='the index of epoch where we resume training.')
    return parser.parse_args()
