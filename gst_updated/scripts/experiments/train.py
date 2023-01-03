import pathhack
import pickle
import time
from os.path import join, isdir
from os import makedirs
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from src.mgnn.utils import arg_parse, average_offset_error, final_offset_error, random_rotate_graph, args2writername, load_batch_dataset
from src.gumbel_social_transformer.temperature_scheduler import Temp_Scheduler
from scripts.experiments.eval import inference
from datetime import datetime



def main(args):
    print('\n\n')
    print('-'*50)
    print('arguments: ', args)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if args.batch_size != 1:
        raise RuntimeError("Batch size must be 1 for BatchTrajectoriesDataset.")
    if args.dataset == 'sdd' and args.rotation_pattern is not None:
        raise RuntimeError("SDD should not allow rotation since it uses pixels.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    loader_train = load_batch_dataset(args, pathhack.pkg_path, subfolder='train')
    if args.dataset == 'sdd':
        loader_val = load_batch_dataset(args, pathhack.pkg_path, subfolder='test') # no val for sdd
    else:
        loader_val = load_batch_dataset(args, pathhack.pkg_path, subfolder='val')
    train_data_loaders = [loader_train, loader_val]
    print('dataset: ', args.dataset)
    writername = args2writername(args)
    print('Config: ', writername)
    logdir = join(pathhack.pkg_path, 'results', writername, args.dataset)
    if isdir(logdir) and not args.resume_training:
        print('Error: The result directory was already created and used.')
        print('-'*50)
        print('\n\n')
        return
    writer = SummaryWriter(logdir=logdir)
    print('-'*50)
    print('\n\n')
    train(args, train_data_loaders, writer, logdir, device=device)
    writer.close()

def train(args, data_loaders, writer, logdir, device='cuda:0'):
    print('-'*50)
    print('Training Phase')
    print('-'*50, '\n')
    loader_train, loader_val = data_loaders
    model = st_model(args, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=int(args.temp_epochs/4), gamma=0.3)
    checkpoint_dir = join(logdir, 'checkpoint')
    if args.resume_training:
        if not isdir(checkpoint_dir):
            raise RuntimeError("Checkpoint folder does not exist.")
        with open(join(checkpoint_dir, 'train_hist.pickle'), 'rb') as f:
            hist = pickle.load(f)
            print(join(checkpoint_dir, 'train_hist.pickle')+' is loaded.')
        if args.resume_epoch is None:
            checkpoint_epoch = hist['epoch']
        else:
            checkpoint_epoch = args.resume_epoch
            hist['train_loss_task'], hist['val_loss_task'] = \
                hist['train_loss_task'][:checkpoint_epoch//args.save_epochs], hist['val_loss_task'][:checkpoint_epoch//args.save_epochs]
            hist['train_aoe_task'], hist['val_aoe_task'] = \
                hist['train_aoe_task'][:checkpoint_epoch//args.save_epochs], hist['val_aoe_task'][:checkpoint_epoch//args.save_epochs]
            hist['train_foe_task'], hist['val_foe_task'] = \
                hist['train_foe_task'][:checkpoint_epoch//args.save_epochs], hist['val_foe_task'][:checkpoint_epoch//args.save_epochs]
        checkpoint = torch.load(join(checkpoint_dir, 'epoch_'+str(checkpoint_epoch)+'.pt'))
        # load checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        temperature_scheduler = Temp_Scheduler(args.temp_epochs, args.init_temp, args.init_temp, \
            temp_min=0.03, last_epoch=checkpoint_epoch-1)
        start_epoch = checkpoint_epoch+1
        print('Model, optimizer, lr_scheduler, and temperature scheduler are loaded.')
        print('EPOCHS: '+str(start_epoch)+' to '+str(args.num_epochs))
    else:
        if not isdir(checkpoint_dir):
            makedirs(checkpoint_dir)
        with open(join(checkpoint_dir, 'args.pickle'), 'wb') as f:
            pickle.dump(args, f)
        temperature_scheduler = Temp_Scheduler(args.temp_epochs, args.init_temp, args.init_temp, temp_min=0.03)
        hist = {}
        hist['epoch'] = 0
        hist['train_loss_task'], hist['val_loss_task'] = [], []
        hist['train_aoe_task'], hist['val_aoe_task'] = [], []
        hist['train_foe_task'], hist['val_foe_task'] = [], []
        start_epoch = 1
        print('Model, optimizer, lr_scheduler, and temperature scheduler are initialized.')
        print('EPOCHS: '+str(start_epoch)+' to '+str(args.num_epochs))
    print('Training started.\n')
    for epoch in range(start_epoch, args.num_epochs+1):
        model.train()
        epoch_start_time = time.time()
        tau = temperature_scheduler.step()
        train_loss_epoch, train_aoe_epoch, train_foe_epoch, train_loss_mask_epoch = [], [], [], []
        batch_len = 0
        valid_batch_len = 0
        for batch_idx, batch in enumerate(loader_train):
            batch_len+=1
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
            v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt = batch
            # print(v_obs.shape)
            if v_obs.shape[2]>128:
                continue
            valid_batch_len += 1
            if args.rotation_pattern is not None:
                (v_obs, A_obs, v_pred_gt, A_pred_gt), _ = \
                    random_rotate_graph(args, v_obs, A_obs, v_pred_gt, A_pred_gt)
            v_obs, A_obs, v_pred_gt, attn_mask_obs, loss_mask_rel = \
                v_obs.to(device), A_obs.to(device), v_pred_gt.to(device), \
                attn_mask_obs.to(device), loss_mask_rel.to(device)
            results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=False, sampling=False, device=device)
            gaussian_params_pred, x_sample_pred, info = results
            loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
            loss_mask_rel_full_partial = info['loss_mask_rel_full_partial'] # value depends on only_observe_full_period
            loss_mask_rel_pred = loss_mask_rel[:,:,-args.pred_seq_len:]
            if args.deterministic:
                offset_error_sq, eventual_loss_mask = offset_error_square_full_partial(x_sample_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
                loss = offset_error_sq.sum()/eventual_loss_mask.sum()
            else:
                prob_loss, eventual_loss_mask = negative_log_likelihood_full_partial(gaussian_params_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
                loss = prob_loss.sum()/eventual_loss_mask.sum()

            train_loss_epoch.append(loss.detach().to('cpu').item())
            loss = loss / args.batch_size
            loss.backward()
            # only consider the fully detected pedestrians
            aoe = average_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
            foe = final_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
            train_aoe_epoch.append(aoe.detach().to('cpu').numpy())
            train_foe_epoch.append(foe.detach().to('cpu').numpy())
            train_loss_mask_epoch.append(loss_mask_per_pedestrian[0].detach().to('cpu').numpy())
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()
        lr_scheduler.step()
        print("valid batch ratio: ", valid_batch_len/batch_len)
        train_aoe_epoch, train_foe_epoch, train_loss_mask_epoch = \
            np.concatenate(train_aoe_epoch, axis=0), \
            np.concatenate(train_foe_epoch, axis=0), \
            np.concatenate(train_loss_mask_epoch, axis=0)
        train_loss_epoch, train_aoe_epoch, train_foe_epoch = \
            np.mean(train_loss_epoch), \
            train_aoe_epoch.sum()/train_loss_mask_epoch.sum(), \
            train_foe_epoch.sum()/train_loss_mask_epoch.sum()
        hist['train_loss_task'].append(train_loss_epoch)
        hist['train_aoe_task'].append(train_aoe_epoch)
        hist['train_foe_task'].append(train_foe_epoch)
        training_epoch_period = time.time() - epoch_start_time
        training_epoch_period_per_sample = training_epoch_period/len(loader_train)

        val_loss_epoch, val_aoe_epoch, val_foe_epoch = inference(loader_val, model, args, mode='val', tau=tau, device=device)
        hist['val_loss_task'].append(val_loss_epoch)
        hist['val_aoe_task'].append(val_aoe_epoch)
        hist['val_foe_task'].append(val_foe_epoch)
        hist['epoch'] = epoch
        print('Epoch: {0} | train loss: {1:.4f} | val loss: {2:.4f} | train aoe: {3:.4f} | val aoe: {4:.4f} | train foe: {5:.4f} | val foe: {6:.4f} | period: {7:.2f} sec | time per sample: {8:.4f} sec'\
                        .format(epoch, train_loss_epoch, val_loss_epoch,\
                        train_aoe_epoch, val_aoe_epoch,\
                        train_foe_epoch, val_foe_epoch,\
                        training_epoch_period, training_epoch_period_per_sample))
        if epoch % args.save_epochs == 0:
            model_filename = join(checkpoint_dir, 'epoch_'+str(epoch)+'.pt')
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_loss_epoch': train_loss_epoch,
                    'val_loss_epoch': val_loss_epoch,
                    'train_aoe_epoch': train_aoe_epoch,
                    'val_aoe_epoch': val_aoe_epoch, 
                    'train_foe_epoch': train_foe_epoch,
                    'val_foe_epoch': val_foe_epoch,
                    'training_date': datetime.today().strftime('%y%m%d'),
                    }, model_filename)
            print('epoch_'+str(epoch)+'.pt is saved.')
            with open(join(checkpoint_dir, 'train_hist.pickle'), 'wb') as f:
                pickle.dump(hist, f)
                print(join(checkpoint_dir, 'train_hist.pickle')+' is saved.')
        writer.add_scalars('loss', {'train': hist['train_loss_task'][-1], 'val': hist['val_loss_task'][-1]}, epoch)
        writer.add_scalars('aoe', {'train': hist['train_aoe_task'][-1], 'val': hist['val_aoe_task'][-1]}, epoch)
        writer.add_scalars('foe', {'train': hist['train_foe_task'][-1], 'val': hist['val_foe_task'][-1]}, epoch)
    return

if __name__ == "__main__":
    args = arg_parse()
    if args.temporal == "lstm" or args.temporal == "faster_lstm":
        from src.gumbel_social_transformer.st_model import st_model, offset_error_square_full_partial, \
            negative_log_likelihood_full_partial
    else:
        raise RuntimeError('The temporal component is not lstm nor faster_lstm.')
    main(args)