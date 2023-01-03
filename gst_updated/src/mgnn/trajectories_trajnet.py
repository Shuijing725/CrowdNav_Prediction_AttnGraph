from torch.utils.data import Dataset
import os
import numpy as np
import math
import torch
from src.mgnn.utils import seq_to_graph
import ndjson

class TrajectoriesDataset(Dataset):
    def __init__(
        self,
        data_dir,
        obs_seq_len=8,
        pred_seq_len=12,
        skip=1,
        delim='\t',
        invalid_value=-999.,
        mode=None,
        ):
        super(TrajectoriesDataset, self).__init__()
        self.data_dir = data_dir
        self.obs_seq_len = obs_seq_len
        self.pred_seq_len = pred_seq_len
        self.skip = skip
        self.seq_len = self.obs_seq_len + self.pred_seq_len
        self.delim = delim
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        loss_mask_rel_list = []
        frame_id_seq = []
        print("Files to be written into the dataset: ")
        print(all_files)
        for path in all_files:
            print("current sequence length: ", len(seq_list))
            print(path)
            data, frame_diff, start_frames_considered, filename_str = read_trajnet_file(path)
            if len(data) == 0:
                continue
            # data = read_file(path, delim)
            frames_np = np.unique(data[:, 0])
            frames = frames_np.tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[data[:, 0]==frame, :])
            if filename_str[:3]=="cff": # too big
                skip = 100
            else:
                skip = self.skip
            print("skip: ", skip)
            print("total length: ", len(start_frames_considered[::skip]))
            if mode is None:
                start_frames_np = start_frames_considered[::skip]
            elif mode == 'train':
                start_frames_np = start_frames_considered[:int(0.8*len(start_frames_considered)):skip]
                # idx_range = range(0, int((num_sequences * self.skip + 1)*0.8), skip)
            elif mode == 'val':
                start_frames_np = start_frames_considered[int(0.8*len(start_frames_considered)):int(0.9*len(start_frames_considered)):skip]
                # idx_range = range(int((num_sequences * self.skip + 1)*0.8), num_sequences * self.skip + 1, skip)
            elif mode == 'test':
                start_frames_np = start_frames_considered[int(0.9*len(start_frames_considered))::skip]
                # idx_range = range(int((num_sequences * self.skip + 1)*0.8), num_sequences * self.skip + 1, skip)
            else:
                raise RuntimeError("Wrong mode for TrajectoriesDataset.")

            # num_sequences = math.floor((len(frames)-self.seq_len)/self.skip)+1
            # if mode is None:
            #     idx_range = range(0, num_sequences * self.skip + 1, skip)
            # elif mode == 'train':
            #     idx_range = range(0, int((num_sequences * self.skip + 1)*0.8), skip)
            # elif mode == 'val':
            #     idx_range = range(int((num_sequences * self.skip + 1)*0.8), num_sequences * self.skip + 1, skip)
            # elif mode == 'test':
            #     idx_range = range(int((num_sequences * self.skip + 1)*0.8), num_sequences * self.skip + 1, skip)
            # else:
            #     raise RuntimeError("Wrong mode for TrajectoriesDataset.")
            # idx_range = range(0, num_sequences * self.skip + 1, skip)
            # print(len(idx_range))
            # for idx in idx_range:
            for start_frame in start_frames_np:
                idx, = np.where(frames_np==start_frame)
                idx = idx[0]
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                start_frame_id = curr_seq_data[0, 0]
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                ped_survive_all_time = False
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    frames_curr_ped_seq = np.unique(curr_ped_seq[:,0])
                    if len(frames_curr_ped_seq) == self.seq_len and \
                        np.all(frames_curr_ped_seq[1:]-frames_curr_ped_seq[:-1]==frame_diff):
                            ped_survive_all_time = True
                            break
                # import pdb; pdb.set_trace()
                if not ped_survive_all_time:
                    continue
                curr_seq = np.ones((len(peds_in_curr_seq), 2, self.seq_len)) * invalid_value
                curr_seq_rel = np.ones((len(peds_in_curr_seq), 2, self.seq_len)) * invalid_value
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                curr_loss_mask_rel = np.zeros((len(peds_in_curr_seq), self.seq_len))
                for ped_id_curr_seq, ped_id in enumerate(peds_in_curr_seq):
                    for tt in range(self.seq_len):
                        frame_id = start_frame_id + tt * frame_diff
                        frame_ped_id = (curr_seq_data[:,0]==frame_id) * (curr_seq_data[:,1]==ped_id)
                        if len(curr_seq_data[frame_ped_id]) == 0:
                            curr_loss_mask[ped_id_curr_seq, tt] = 0
                            curr_loss_mask_rel[ped_id_curr_seq, tt] = 0
                        elif len(curr_seq_data[frame_ped_id]) == 1:
                            curr_seq[ped_id_curr_seq,:,tt] = curr_seq_data[frame_ped_id, 2:]
                            curr_loss_mask[ped_id_curr_seq, tt] = 1
                            if tt == 0:
                                curr_seq_rel[ped_id_curr_seq,:,tt] = np.zeros((2,))
                                curr_loss_mask_rel[ped_id_curr_seq, tt] = 1
                            else:
                                if curr_loss_mask[ped_id_curr_seq, tt-1] == 1:
                                    curr_seq_rel[ped_id_curr_seq,:,tt] = curr_seq[ped_id_curr_seq,:,tt] - curr_seq[ped_id_curr_seq,:,tt-1]
                                    curr_loss_mask_rel[ped_id_curr_seq, tt] = 1
                                else:
                                    curr_loss_mask_rel[ped_id_curr_seq, tt] = 0
                        else:
                            raise RuntimeError("The same pedestrian has multiple locations in the same frame.")

                num_peds_in_seq.append(len(peds_in_curr_seq))
                seq_list.append(curr_seq)
                seq_list_rel.append(curr_seq_rel)
                loss_mask_list.append(curr_loss_mask)
                loss_mask_rel_list.append(curr_loss_mask_rel)
                frame_id_seq.append(start_frame_id)

        self.num_seq = len(seq_list)
        print("total sequence length: ", len(seq_list))
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        loss_mask_rel_list = np.concatenate(loss_mask_rel_list, axis=0)
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_seq_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_seq_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_seq_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_seq_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.loss_mask_rel = torch.from_numpy(loss_mask_rel_list).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx[:-1], cum_start_idx[1:])]
        self.frame_id_seq = frame_id_seq
        self.v_obs = []
        self.A_obs = []
        self.attn_mask_obs = []
        self.v_pred = []
        self.A_pred = []
        self.attn_mask_pred = []
        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            v_, a_ = seq_to_graph(
                self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], attn_mech='rel_conv')
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(
                self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], attn_mech='rel_conv')
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
            attn_mask = []
            for tt in range(self.seq_len):
                loss_mask_rel_tt = self.loss_mask_rel[start:end, tt]
                attn_mask.append(torch.outer(loss_mask_rel_tt, loss_mask_rel_tt).float())
            attn_mask = torch.stack(attn_mask, dim=0)
            self.attn_mask_obs.append(attn_mask[:self.obs_seq_len])
            self.attn_mask_pred.append(attn_mask[self.obs_seq_len:])


    def __len__(self):
        return self.num_seq


    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.loss_mask_rel[start:end, :], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.attn_mask_obs[index], self.attn_mask_pred[index],
        ]
        return out
    
    

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def read_trajnet_file(_path):
    _, filename = os.path.split(_path)
    print(filename)
    filename_str, filename_ext = filename.split('.')
    if filename_ext != "ndjson":
        print("Not ndjson file for trajnet++.")
        data = []
        frame_diff = 0.
        start_frames_considered = []
        return data, frame_diff, start_frames_considered, filename_str
        # raise RuntimeError("Not ndjson file for trajnet++.")
    lines = []
    frame_diff = 0.
    start_frames_considered = []
    with open(_path) as f:
        reader = ndjson.reader(f)
        for post in reader:
            # print(type(post))
            # print(post.keys())
            # print(post['scene'])
            if "scene" in post.keys():# and len(start_frames_considered) < 20:
                start_frame = post["scene"]["s"]
                if frame_diff==0.:
                    frame_diff = (post["scene"]["e"]-post["scene"]["s"])/20
                start_frames_considered.append(start_frame)

                # np.unique(data[:, 0]).tolist()
                # {"scene": {"id": 0, "p": 24, "s": 500, "e": 700, "fps": 2.5, "tag": [3, [2]]}}
                # {"scene": {"id": 1, "p": 24, "s": 520, "e": 720, "fps": 2.5, "tag": [4, []]}}
                # {"scene": {"id": 2, "p": 24, "s": 540, "e": 740, "fps": 2.5, "tag": [4, []]}}
                # {"scene": {"id": 3, "p": 24, "s": 560, "e": 760, "fps": 2.5, "tag": [4, []]}}
            if "track" in post.keys():
                f = post["track"]["f"]
                p = post["track"]["p"]
                x = post["track"]["x"]
                y = post["track"]["y"]
                line = [f,p,x,y] # str(f)+'\t'+str(p)+'\t'+str(x)+'\t'+str(y)+'\n'
                lines.append(line)
                # if len(lines)==3000:
                #     break
    data = np.asarray(lines)
    start_frames_considered = np.unique(start_frames_considered)
    return data, frame_diff, start_frames_considered, filename_str





#     import pathhack
# import ndjson
# import os

# raw_folder = 'real_data'
# output_folder = raw_folder+'_txt'
# data_dir = os.path.join(pathhack.pkg_path, "datasets/trajnet++/train", raw_folder)
# output_dir = os.path.join(pathhack.pkg_path, "datasets/trajnet++/train", output_folder)
# all_files = os.listdir(data_dir)
# all_files = [os.path.join(data_dir, _path) for _path in all_files]

# for path in all_files:
#     print(path)
#     _, filename = os.path.split(path)
#     filename_str, filename_ext = filename.split('.')
#     frame = None
#     if filename_ext == "ndjson":
#         lines = []
#         with open(path) as f:
#             reader = ndjson.reader(f)
#             for post in reader:
#                 # print(type(post))
#                 # print(post.keys())
#                 # print(post['scene'])
#                 if "track" in post.keys():
#                     f = post["track"]["f"]
#                     if frame is None:
#                         frame = f
#                     else:
#                         print(f-frame)
#                         break
#                     p = post["track"]["p"]
#                     x = post["track"]["x"]
#                     y = post["track"]["y"]
#                     line = str(f)+'\t'+str(p)+'\t'+str(x)+'\t'+str(y)+'\n'
#                     lines.append(line)
