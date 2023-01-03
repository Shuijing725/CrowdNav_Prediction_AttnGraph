import pathhack
import pickle
from os.path import join, isdir
import torch
import csv
import numpy as np
from src.mgnn.utils import arg_parse, average_offset_error, final_offset_error, args2writername, load_batch_dataset
from src.gumbel_social_transformer.st_model import st_model, offset_error_square_full_partial,\
    negative_log_likelihood_full_partial
from torch.utils.data import DataLoader
import ndjson

# def load_single_batch_dataset_eth_ucy(args, pkg_path, subfolder='train', num_workers=4, shuffle=None):
shuffle = None
subfolder = 'test'
# dset_name = 'orca_synth'
# dset_name = 'collision_test'
writername = "100-gumbel_social_transformer-faster_lstm-lr_0.001-deterministic-init_temp_0.5-edge_head_4-ebd_64-snl_1-snh_8-ghost-seed_1000"
# "100-gumbel_social_transformer-faster_lstm-lr_0.001-deterministic-init_temp_0.5-edge_head_1-ebd_64-snl_1-snh_8-ghost-seed_1000"

# "100-gumbel_social_transformer-faster_lstm-lr_0.001-deterministic-init_temp_0.5-edge_head_8-ebd_64-snl_1-snh_8-ghost-seed_1000"
for dset_name in ['orca_synth', 'collision_test']:
    result_filename = dset_name+'_dset_test_trajectories.pt'
    # result_filename = 'orca_synth_dset_test_trajectories.pt'
    # result_filename = 'collision_test_dset_test_trajectories.pt'
    dataset_folderpath = join(pathhack.pkg_path, 'datasets/trajnet++/test')
    dset = torch.load(join(dataset_folderpath, result_filename))


    dataset_folderpath = join(pathhack.pkg_path, 'datasets/trajnet++/test')
    ndjson_filepath = join(dataset_folderpath, 'synth_data', dset_name+'.ndjson')
    # ndjson_filepath = join(dataset_folderpath, 'synth_data', 'orca_synth.ndjson')
    # ndjson_filepath = join(dataset_folderpath, 'synth_data', 'collision_test.ndjson')
    scene_posts = []
    with open(ndjson_filepath) as f:
        reader = ndjson.reader(f)
        for post in reader:
            if "scene" in post.keys():
                scene_posts.append(post)

    if shuffle is None:
        if subfolder == 'train':
            shuffle = True
        else:
            shuffle = False
    dloader = DataLoader(
        dset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=4)

    frame_diff = 1
    for batch_idx, batch in enumerate(dloader):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
        v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt, frame_id_seq, ped_id_list = batch
        print(v_obs)
        print(frame_id_seq)
        print(ped_id_list)
        print(pred_traj_gt)
        break
    device = "cuda:0"
    
    logdir = join(pathhack.pkg_path, 'results', writername, "synth") 
    checkpoint_dir = join(logdir, 'checkpoint')
    model_filename = 'epoch_'+str(100)+'.pt'
    with open(join(checkpoint_dir, 'args.pickle'), 'rb') as f:
        args_eval = pickle.load(f)

    model = st_model(args_eval, device=device).to(device)
    model_checkpoint = torch.load(join(checkpoint_dir, model_filename), map_location=device)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    print('Loaded configuration: ', writername)
    print('The best validation losses printed below should be the same.')
    print('Validation loss in the checkpoint: ', model_checkpoint['val_loss_epoch'])

    ndjson_lines = []
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(dloader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
            v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt, frame_id_seq, ped_id_list = batch
            v_obs, A_obs, attn_mask_obs, loss_mask_rel = \
                v_obs.to(device), A_obs.to(device), \
                attn_mask_obs.to(device), loss_mask_rel.to(device)
            
            results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=0.03, hard=True, sampling=False, device=device)
            gaussian_params_pred, x_sample_pred, info = results

            obs_traj = obs_traj.to("cpu")
            x_sample_pred = x_sample_pred.to("cpu").permute(0, 2, 3, 1) #[1, 6, 2, 12]
            pred_traj = torch.cat((obs_traj[:,:,:,-1:], x_sample_pred), dim=3) # [1, 6, 2, 13]
            pred_traj = torch.cumsum(pred_traj, dim=3)

            start_frame_id = int(frame_id_seq.item())
            pred_traj = pred_traj[:,:,:,1:] # [1, 6, 2, 12]
            loss_mask_rel_full_partial = info['loss_mask_rel_full_partial'][0]
            # {"track": {"f": 370, "p": 3, "x": -1.17, "y": 4.24, "prediction_number": 0, "scene_id": 0}}
            for i in range(len(loss_mask_rel_full_partial)):
                if loss_mask_rel_full_partial[i]:
                    for tt in range(9, 21): # 9-20
                        ndjson_line = {}
                        ndjson_line["track"] = {}
                        ndjson_line["track"]["f"] = start_frame_id + int(frame_diff)*tt
                        ndjson_line["track"]["p"] = int(ped_id_list[i].item())
                        ndjson_line["track"]["x"] = round(pred_traj[0,i,0,tt-9].item(), 2)
                        ndjson_line["track"]["y"] = round(pred_traj[0,i,1,tt-9].item(), 2)
                        ndjson_line["track"]["prediction_number"] = 0
                        ndjson_line["track"]["scene_id"] = batch_idx
                        ndjson_lines.append(ndjson_line)
        print("final scene id: ", batch_idx)

    # with open('./collision_test_predictions.ndjson', 'w') as f:
    # with open('./orca_synth_predictions.ndjson', 'w') as f:
    with open('./'+dset_name+'_predictions.ndjson', 'w') as f:
        writer = ndjson.writer(f, ensure_ascii=False)
        for post in scene_posts:
            writer.writerow(post)
        for post in ndjson_lines:
            writer.writerow(post)

# import pdb; pdb.set_trace()
"""     
# import pdb; pdb.set_trace()
# break
ndjson_lines = []
# ndjson_line = {}
# ndjson_line["track"] = {}
start_frame_id = int(frame_id_seq.item())
# ndjson_line["track"]["p"] = int(frame_id_seq.item()[0])


# {"track": {"f": 113, "p": 19973, "x": -0.95, "y": -1.86}}
pred_traj = pred_traj[:,:,:,1:] # [1, 6, 2, 12]
loss_mask_rel_full_partial = info['loss_mask_rel_full_partial'][0]
# {"track": {"f": 370, "p": 3, "x": -1.17, "y": 4.24, "prediction_number": 0, "scene_id": 0}}
for i in range(len(loss_mask_rel_full_partial)):
    if loss_mask_rel_full_partial[i]:
        for tt in range(9, 21): # 9-20
            ndjson_line = {}
            ndjson_line["track"] = {}
            ndjson_line["track"]["f"] = start_frame_id + int(frame_diff)*tt
            ndjson_line["track"]["p"] = int(ped_id_list[i].item())
            ndjson_line["track"]["x"] = round(pred_traj[0,i,0,tt-9], 2)
            ndjson_line["track"]["y"] = round(pred_traj[0,i,0,tt-9], 2)
            ndjson_line["track"]["prediction_number"] = 0
            ndjson_line["track"]["scene_id"] = batch_idx
"""


# import matplotlib .pyplot as plt
# (Pdb) fig, ax = plt.subplots()
# (Pdb) [ax.plot(pred_traj[0,i,0,:], pred_traj[0,i,1,:], '.-') for i in range(6)]
# [[<matplotlib.lines.Line2D object at 0x7f9317e107b8>], [<matplotlib.lines.Line2D object at 0x7f93bc3eab38>], [<matplotlib.lines.Line2D object at 0x7f93bc3eada0>], [<matplotlib.lines.Line2D object at 0x7f93bc403048>], [<matplotlib.lines.Line2D object at 0x7f93bc403320>], [<matplotlib.lines.Line2D object at 0x7f93bc4035f8>]]
# (Pdb) fig.savefig("tmp4.jpg")


# import ndjson

# # Streaming lines from ndjson file:
# with open('./posts.ndjson') as f:
#     reader = ndjson.reader(f)

#     for post in reader:
#         print(post)

# # Writing items to a ndjson file
# with open('./posts.ndjson', 'w') as f:
#     writer = ndjson.writer(f, ensure_ascii=False)

#     for post in posts:
#         writer.writerow(post)


# print(pred_traj.shape)
# obs_traj
# obs_traj # (1,6,2,8)
# x_sample_pred torch.Size([1, 12, 6, 2])
# pos_pred = torch.cumsum(x_sample_pred, dim=1)
# pos_target = torch.cumsum(x_target_m, dim=1)

# self.obs_traj[start:end, :], self.pred_traj[start:end, :],
#             self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
#             self.loss_mask_rel[start:end, :], self.loss_mask[start:end, :],
#             self.v_obs[index], self.A_obs[index],
#             self.v_pred[index], self.A_pred[index],
#             self.attn_mask_obs[index], self.attn_mask_pred[index],
#         ]
#     for mode in ['test']:
#         dataset_filepath = join(dataset_folderpath, 'biwi_eth.ndjson')
#         dset = TrajectoriesDataset(
#             dataset_filepath,
#             obs_seq_len=args.obs_seq_len,
#             pred_seq_len=args.pred_seq_len,
#             skip=1,
#             invalid_value=args.invalid_value,
#             mode=mode,
#         )
#         result_filename = 'biwi_eth_dset_'+mode+'_trajectories.pt'
#         torch.save(dset, join(dataset_folderpath, '..', result_filename))
#         print(join(dataset_folderpath, '..', result_filename)+' is created.')




# def main(args):
#     print('\n\n')
#     print('-'*50)
#     torch.manual_seed(args.random_seed)
#     np.random.seed(args.random_seed)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     loader_test = load_batch_dataset(args, pathhack.pkg_path, subfolder='test')
#     if args.dataset == 'sdd':
#         loader_val = load_batch_dataset(args, pathhack.pkg_path, subfolder='test') # no val for sdd
#     else:
#         loader_val = load_batch_dataset(args, pathhack.pkg_path, subfolder='val')
#     print('dataset: ', args.dataset)
#     writername = args2writername(args)
#     logdir = join(pathhack.pkg_path, 'results', writername, args.dataset)
#     if not isdir(logdir):
#         raise RuntimeError('The result directory was not found.')
#     checkpoint_dir = join(logdir, 'checkpoint')
#     model_filename = 'epoch_'+str(args.num_epochs)+'.pt'
#     with open(join(checkpoint_dir, 'args.pickle'), 'rb') as f:
#         args_eval = pickle.load(f)
#     model = st_model(args_eval, device=device).to(device)
#     model_checkpoint = torch.load(join(checkpoint_dir, model_filename), map_location=device)
#     model.load_state_dict(model_checkpoint['model_state_dict'])
#     print('Loaded configuration: ', writername)
#     print('The best validation losses printed below should be the same.')
#     print('Validation loss in the checkpoint: ', model_checkpoint['val_loss_epoch'])
#     val_loss_epoch, val_aoe_epoch, val_foe_epoch = inference(loader_val, model, args, mode='val', tau=0.03, device=device)
#     print('Validation loss from loaded model: ', val_loss_epoch)
#     test_loss_epoch, test_aoe, test_foe, test_aoe_std, test_foe_std, test_aoe_min, test_foe_min = inference(loader_test, model, args, mode='test', tau=0.03, device=device)
#     print('Test loss from loaded model: ', test_loss_epoch)
#     print('dataset: {0} | test aoe: {1:.4f} | test aoe std: {2:.4f} | test foe: {3:.4f} | test foe std: {4:.4f} | min aoe: {5:.4f}, min foe: {6:.4f}'\
#                         .format(args.dataset, test_aoe, test_aoe_std, test_foe, test_foe_std, test_aoe_min, test_foe_min))
#     dataset_list = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
#     dataset_idx = dataset_list.index(args.dataset)
#     csv_row_data = np.ones((4,5))*999999.
#     csv_row_data[:,dataset_idx] = np.array([test_aoe, test_foe, test_aoe_std, test_foe_std])
#     csv_row_data = csv_row_data.reshape(-1)
#     csv_filename = join(checkpoint_dir, '../..', 'test_performance.csv')
#     with open(csv_filename, mode='a') as test_file:
#         test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         test_writer.writerow(csv_row_data)
#         print(csv_filename+' is written.')


# def inference(loader, model, args, mode='val', tau=1., device='cuda:0'):
#     with torch.no_grad():
#         model.eval()
#         loss_epoch, aoe_epoch, foe_epoch, loss_mask_epoch = [], [], [], []
#         aoe_mean_epoch, aoe_std_epoch, foe_mean_epoch, foe_std_epoch = [], [], [], []
#         aoe_min_epoch, foe_min_epoch = [], []

#         for batch_idx, batch in enumerate(loader):
#             obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
#             v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt = batch
#             v_obs, A_obs, v_pred_gt, attn_mask_obs, loss_mask_rel = \
#                 v_obs.to(device), A_obs.to(device), v_pred_gt.to(device), \
#                 attn_mask_obs.to(device), loss_mask_rel.to(device)
#             if mode == 'val':
#                 results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=False, sampling=False, device=device)
#                 gaussian_params_pred, x_sample_pred, info = results
#                 loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
#                 loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
#                 loss_mask_rel_pred = loss_mask_rel[:,:,-args.pred_seq_len:]
#                 if args.deterministic:
#                     offset_error_sq, eventual_loss_mask = offset_error_square_full_partial(x_sample_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
#                     loss = offset_error_sq.sum()/eventual_loss_mask.sum()
#                 else:
#                     prob_loss, eventual_loss_mask = negative_log_likelihood_full_partial(gaussian_params_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
#                     loss = prob_loss.sum()/eventual_loss_mask.sum()
#                 aoe = average_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
#                 foe = final_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
            
#             elif mode == 'test':
#                 if args.deterministic:
#                     sampling = False
#                 else:
#                     sampling = True
#                 best_aoe_mean = 999999.
#                 aoe, foe, loss = [], [], []
#                 for _ in range(20):
#                     results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=True, sampling=sampling, device=device)
#                     gaussian_params_pred, x_sample_pred, info = results
#                     loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
#                     loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
#                     loss_mask_rel_pred = loss_mask_rel[:,:,-args.pred_seq_len:]
#                     if args.deterministic:
#                         offset_error_sq, eventual_loss_mask = offset_error_square_full_partial(x_sample_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
#                         loss_tmp = offset_error_sq.sum()/eventual_loss_mask.sum()
#                     else:
#                         prob_loss, eventual_loss_mask = negative_log_likelihood_full_partial(gaussian_params_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
#                         loss_tmp = prob_loss.sum()/eventual_loss_mask.sum()
#                     aoe_tmp = average_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
#                     foe_tmp = final_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
#                     aoe.append(aoe_tmp)
#                     foe.append(foe_tmp)
#                     loss.append(loss_tmp)
#                 aoe = torch.stack(aoe, dim=0)
#                 foe = torch.stack(foe, dim=0)
#                 aoe = aoe.sum(1)
#                 foe = foe.sum(1)
#                 aoe_mean = aoe.mean()
#                 aoe_std = aoe.std()
#                 foe_mean = foe.mean()
#                 foe_std = foe.std()
#                 aoe_min, foe_min = aoe.min(), foe.min()
#                 loss = sum(loss)/len(loss)
#             else:
#                 raise RuntimeError('We now only support val and test mode.')
            
#             if mode == 'val':
#                 loss_epoch.append(loss.detach().to('cpu').item())
#                 aoe_epoch.append(aoe.detach().to('cpu').numpy())
#                 foe_epoch.append(foe.detach().to('cpu').numpy())
#                 loss_mask_epoch.append(loss_mask_per_pedestrian[0].detach().to('cpu').numpy())
#             elif mode == 'test':
#                 loss_epoch.append(loss.detach().to('cpu').item())
#                 aoe_mean_epoch.append(aoe_mean.item())
#                 foe_mean_epoch.append(foe_mean.item())
#                 aoe_std_epoch.append(aoe_std.item())
#                 foe_std_epoch.append(foe_std.item())
#                 aoe_min_epoch.append(aoe_min.item())
#                 foe_min_epoch.append(foe_min.item())
#                 loss_mask_epoch.append(loss_mask_per_pedestrian[0].detach().to('cpu').numpy())
#             else:
#                 raise RuntimeError('We only support val and test mode.')
        
#         if mode == 'val':
#             aoe_epoch, foe_epoch, loss_mask_epoch = \
#                 np.concatenate(aoe_epoch, axis=0), \
#                 np.concatenate(foe_epoch, axis=0), \
#                 np.concatenate(loss_mask_epoch, axis=0)
#             loss_epoch, aoe_epoch, foe_epoch = \
#                 np.mean(loss_epoch), \
#                 aoe_epoch.sum()/loss_mask_epoch.sum(), \
#                 foe_epoch.sum()/loss_mask_epoch.sum()
#             return loss_epoch, aoe_epoch, foe_epoch
#         elif mode == 'test':
#             loss_mask_epoch = np.concatenate(loss_mask_epoch, axis=0)
#             aoe = sum(aoe_mean_epoch)/loss_mask_epoch.sum()
#             foe = sum(foe_mean_epoch)/loss_mask_epoch.sum()
#             aoe_std = sum(aoe_std_epoch)/loss_mask_epoch.sum()
#             foe_std = sum(foe_std_epoch)/loss_mask_epoch.sum()
#             aoe_min = sum(aoe_min_epoch)/loss_mask_epoch.sum()
#             foe_min = sum(foe_min_epoch)/loss_mask_epoch.sum()
#             loss_epoch = np.mean(loss_epoch)
#             return loss_epoch, aoe, foe, aoe_std, foe_std, aoe_min, foe_min
#         else:
#             raise RuntimeError('We now only support val mode.')    


# if __name__ == "__main__":
#     args = arg_parse()
#     main(args)