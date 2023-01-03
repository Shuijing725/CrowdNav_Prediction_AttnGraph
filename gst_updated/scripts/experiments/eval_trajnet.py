import pathhack
import pickle
from os.path import join, isdir
import torch
import csv
import numpy as np
from src.mgnn.utils import arg_parse, average_offset_error, final_offset_error, args2writername, load_batch_dataset
from src.gumbel_social_transformer.st_model import st_model, offset_error_square_full_partial,\
    negative_log_likelihood_full_partial


def main(args):
    print('\n\n')
    print('-'*50)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader_test = load_batch_dataset(args, pathhack.pkg_path, subfolder='test')
    if args.dataset == 'sdd':
        loader_val = load_batch_dataset(args, pathhack.pkg_path, subfolder='test') # no val for sdd
    else:
        loader_val = load_batch_dataset(args, pathhack.pkg_path, subfolder='val')
    print('dataset: ', args.dataset)
    writername = args2writername(args)
    logdir = join(pathhack.pkg_path, 'results', writername, args.dataset)
    if not isdir(logdir):
        raise RuntimeError('The result directory was not found.')
    checkpoint_dir = join(logdir, 'checkpoint')
    model_filename = 'epoch_'+str(args.num_epochs)+'.pt'
    with open(join(checkpoint_dir, 'args.pickle'), 'rb') as f:
        args_eval = pickle.load(f)
    model = st_model(args_eval, device=device).to(device)
    model_checkpoint = torch.load(join(checkpoint_dir, model_filename), map_location=device)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    print('Loaded configuration: ', writername)
    print('The best validation losses printed below should be the same.')
    print('Validation loss in the checkpoint: ', model_checkpoint['val_loss_epoch'])
    val_loss_epoch, val_aoe_epoch, val_foe_epoch = inference(loader_val, model, args, mode='val', tau=0.03, device=device)
    print('Validation loss from loaded model: ', val_loss_epoch)
    test_loss_epoch, test_aoe, test_foe, test_aoe_std, test_foe_std, test_aoe_min, test_foe_min = inference(loader_test, model, args, mode='test', tau=0.03, device=device)
    print('Test loss from loaded model: ', test_loss_epoch)
    print('dataset: {0} | test aoe: {1:.4f} | test aoe std: {2:.4f} | test foe: {3:.4f} | test foe std: {4:.4f} | min aoe: {5:.4f}, min foe: {6:.4f}'\
                        .format(args.dataset, test_aoe, test_aoe_std, test_foe, test_foe_std, test_aoe_min, test_foe_min))
    dataset_list = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    dataset_idx = dataset_list.index(args.dataset)
    csv_row_data = np.ones((4,5))*999999.
    csv_row_data[:,dataset_idx] = np.array([test_aoe, test_foe, test_aoe_std, test_foe_std])
    csv_row_data = csv_row_data.reshape(-1)
    csv_filename = join(checkpoint_dir, '../..', 'test_performance.csv')
    with open(csv_filename, mode='a') as test_file:
        test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_writer.writerow(csv_row_data)
        print(csv_filename+' is written.')


def test(loader, model, tau=0.03, device='cuda:0'):
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(loader):
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
            v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt, frame_id_seq, ped_id_list = batch
            v_obs, A_obs, v_pred_gt, attn_mask_obs, loss_mask_rel = \
                v_obs.to(device), A_obs.to(device), v_pred_gt.to(device), \
                attn_mask_obs.to(device), loss_mask_rel.to(device)
            
            results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=False, sampling=False, device=device)
            gaussian_params_pred, x_sample_pred, info = results
            import pdb; pdb.set_trace()
            # loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
            # loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
            # pos_pred = torch.cumsum(x_sample_pred, dim=1)
            # pos_target = torch.cumsum(x_target_m, dim=1)


                # loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
                # loss_mask_rel_pred = loss_mask_rel[:,:,-args.pred_seq_len:]


            # gaussian_params_pred, x_sample_pred, info = results
            #     loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
            #     loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
    
            break

def inference(loader, model, args, mode='val', tau=1., device='cuda:0'):
    with torch.no_grad():
        model.eval()
        loss_epoch, aoe_epoch, foe_epoch, loss_mask_epoch = [], [], [], []
        aoe_mean_epoch, aoe_std_epoch, foe_mean_epoch, foe_std_epoch = [], [], [], []
        aoe_min_epoch, foe_min_epoch = [], []

        for batch_idx, batch in enumerate(loader):
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
            v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt = batch
            v_obs, A_obs, v_pred_gt, attn_mask_obs, loss_mask_rel = \
                v_obs.to(device), A_obs.to(device), v_pred_gt.to(device), \
                attn_mask_obs.to(device), loss_mask_rel.to(device)
            if mode == 'val':
                results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=False, sampling=False, device=device)
                gaussian_params_pred, x_sample_pred, info = results
                loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
                loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
                loss_mask_rel_pred = loss_mask_rel[:,:,-args.pred_seq_len:]
                if args.deterministic:
                    offset_error_sq, eventual_loss_mask = offset_error_square_full_partial(x_sample_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
                    loss = offset_error_sq.sum()/eventual_loss_mask.sum()
                else:
                    prob_loss, eventual_loss_mask = negative_log_likelihood_full_partial(gaussian_params_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
                    loss = prob_loss.sum()/eventual_loss_mask.sum()
                aoe = average_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
                foe = final_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
            
            elif mode == 'test':
                if args.deterministic:
                    sampling = False
                else:
                    sampling = True
                best_aoe_mean = 999999.
                aoe, foe, loss = [], [], []
                for _ in range(20):
                    results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=True, sampling=sampling, device=device)
                    gaussian_params_pred, x_sample_pred, info = results
                    loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
                    loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
                    loss_mask_rel_pred = loss_mask_rel[:,:,-args.pred_seq_len:]
                    if args.deterministic:
                        offset_error_sq, eventual_loss_mask = offset_error_square_full_partial(x_sample_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
                        loss_tmp = offset_error_sq.sum()/eventual_loss_mask.sum()
                    else:
                        prob_loss, eventual_loss_mask = negative_log_likelihood_full_partial(gaussian_params_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
                        loss_tmp = prob_loss.sum()/eventual_loss_mask.sum()
                    aoe_tmp = average_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
                    foe_tmp = final_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
                    aoe.append(aoe_tmp)
                    foe.append(foe_tmp)
                    loss.append(loss_tmp)
                aoe = torch.stack(aoe, dim=0)
                foe = torch.stack(foe, dim=0)
                aoe = aoe.sum(1)
                foe = foe.sum(1)
                aoe_mean = aoe.mean()
                aoe_std = aoe.std()
                foe_mean = foe.mean()
                foe_std = foe.std()
                aoe_min, foe_min = aoe.min(), foe.min()
                loss = sum(loss)/len(loss)
            else:
                raise RuntimeError('We now only support val and test mode.')
            
            if mode == 'val':
                loss_epoch.append(loss.detach().to('cpu').item())
                aoe_epoch.append(aoe.detach().to('cpu').numpy())
                foe_epoch.append(foe.detach().to('cpu').numpy())
                loss_mask_epoch.append(loss_mask_per_pedestrian[0].detach().to('cpu').numpy())
            elif mode == 'test':
                loss_epoch.append(loss.detach().to('cpu').item())
                aoe_mean_epoch.append(aoe_mean.item())
                foe_mean_epoch.append(foe_mean.item())
                aoe_std_epoch.append(aoe_std.item())
                foe_std_epoch.append(foe_std.item())
                aoe_min_epoch.append(aoe_min.item())
                foe_min_epoch.append(foe_min.item())
                loss_mask_epoch.append(loss_mask_per_pedestrian[0].detach().to('cpu').numpy())
            else:
                raise RuntimeError('We only support val and test mode.')
        
        if mode == 'val':
            aoe_epoch, foe_epoch, loss_mask_epoch = \
                np.concatenate(aoe_epoch, axis=0), \
                np.concatenate(foe_epoch, axis=0), \
                np.concatenate(loss_mask_epoch, axis=0)
            loss_epoch, aoe_epoch, foe_epoch = \
                np.mean(loss_epoch), \
                aoe_epoch.sum()/loss_mask_epoch.sum(), \
                foe_epoch.sum()/loss_mask_epoch.sum()
            return loss_epoch, aoe_epoch, foe_epoch
        elif mode == 'test':
            loss_mask_epoch = np.concatenate(loss_mask_epoch, axis=0)
            aoe = sum(aoe_mean_epoch)/loss_mask_epoch.sum()
            foe = sum(foe_mean_epoch)/loss_mask_epoch.sum()
            aoe_std = sum(aoe_std_epoch)/loss_mask_epoch.sum()
            foe_std = sum(foe_std_epoch)/loss_mask_epoch.sum()
            aoe_min = sum(aoe_min_epoch)/loss_mask_epoch.sum()
            foe_min = sum(foe_min_epoch)/loss_mask_epoch.sum()
            loss_epoch = np.mean(loss_epoch)
            return loss_epoch, aoe, foe, aoe_std, foe_std, aoe_min, foe_min
        else:
            raise RuntimeError('We now only support val mode.')    


if __name__ == "__main__":
    args = arg_parse()
    main(args)