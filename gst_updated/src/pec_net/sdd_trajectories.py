from torch.utils.data import Dataset
import numpy as np

class SDDTrajectoriesDataset(Dataset):
    def __init__(
        self,
        ):
        super(SDDTrajectoriesDataset, self).__init__()
        self.num_seq = 0
        self.obs_traj = []
        self.pred_traj = []
        self.obs_traj_rel = []
        self.pred_traj_rel = []
        self.loss_mask_rel = []
        self.loss_mask = []
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        self.attn_mask_obs = []
        self.attn_mask_pred = []

    def add_batch(
        self,
        obs_traj,
        pred_traj_gt,
        obs_traj_rel,
        pred_traj_rel_gt,
        loss_mask_rel,
        loss_mask,
        v_obs, 
        A_obs,
        v_pred,
        A_pred,
        attn_mask_obs,
        attn_mask_pred,
    ):
        self.num_seq += 1
        self.obs_traj.append(obs_traj[0].float())
        self.pred_traj.append(pred_traj_gt[0].float())
        self.obs_traj_rel.append(obs_traj_rel[0].float())
        self.pred_traj_rel.append(pred_traj_rel_gt[0].float())
        self.loss_mask_rel.append(loss_mask_rel[0].float())
        self.loss_mask.append(loss_mask[0].float())
        self.v_obs.append(v_obs[0].float())
        self.A_obs.append(A_obs[0].float())
        self.v_pred.append(v_pred[0].float())
        self.A_pred.append(A_pred[0].float())
        self.attn_mask_obs.append(attn_mask_obs[0].float())
        self.attn_mask_pred.append(attn_mask_pred[0].float())

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        out = [
            self.obs_traj[index], self.pred_traj[index],
            self.obs_traj_rel[index], self.pred_traj_rel[index],
            self.loss_mask_rel[index], self.loss_mask[index],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.attn_mask_obs[index], self.attn_mask_pred[index],
        ]
        return out