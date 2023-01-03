from torch.utils.data import Dataset

class BatchTrajectoriesDataset(Dataset):
    def __init__(
        self,
        ):
        super(BatchTrajectoriesDataset, self).__init__()
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
        self.obs_traj.append(obs_traj.float())
        self.pred_traj.append(pred_traj_gt.float())
        self.obs_traj_rel.append(obs_traj_rel.float())
        self.pred_traj_rel.append(pred_traj_rel_gt.float())
        self.loss_mask_rel.append(loss_mask_rel.float())
        self.loss_mask.append(loss_mask.float())
        self.v_obs.append(v_obs.float())
        self.A_obs.append(A_obs.float())
        self.v_pred.append(v_pred.float())
        self.A_pred.append(A_pred.float())
        self.attn_mask_obs.append(attn_mask_obs.float())
        self.attn_mask_pred.append(attn_mask_pred.float())

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
