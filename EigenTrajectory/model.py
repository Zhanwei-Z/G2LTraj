import torch
import torch.nn as nn
from .anchor import ETAnchor
from .descriptor import ETDescriptor


class EigenTrajectory(nn.Module):
    r"""The EigenTrajectory model

    Args:
        baseline_model (nn.Module): The baseline model
        hook_func (dict): The bridge functions for the baseline model
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, baseline_model, hook_func, hyper_params):
        super().__init__()

        self.baseline_model = baseline_model
        self.hook_func = hook_func
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.static_dist = hyper_params.static_dist
        self.out_dim = 2

        self.ET_m_descriptor = ETDescriptor(hyper_params=hyper_params, norm_sca=True)
        self.ET_s_descriptor = ETDescriptor(hyper_params=hyper_params, norm_sca=False)
        self.ET_m_anchor = ETAnchor(hyper_params=hyper_params)
        self.ET_s_anchor = ETAnchor(hyper_params=hyper_params)
        if self.hyper_params.use_rnn:
            self.MLP1 = MLP(self.t_pred*self.out_dim, self.out_dim, hidden=self.out_dim, bias=True, activation="prelu", norm=False)
            self.pe = PositionalEncoding(num_hiddens=self.out_dim, max_len=self.t_pred+1)

            self.MLP2 = MLP(self.out_dim*3, self.out_dim, hidden=self.out_dim, bias=True, activation="prelu", norm=False)
            self.output_next2 = nn.Linear(self.out_dim, self.out_dim)
            torch.nn.init.constant_(self.output_next2.weight, 0)
            torch.nn.init.constant_(self.output_next2.bias, 0)

            self.MLP4 = MLP(self.out_dim*3, self.out_dim, hidden=self.out_dim, bias=True, activation="prelu", norm=False)
            self.output_next4 = nn.Linear(self.out_dim, self.out_dim)
            torch.nn.init.constant_(self.output_next4.weight, 0)
            torch.nn.init.constant_(self.output_next4.bias, 0)

            self.MLP8 = MLP(self.out_dim*3, self.out_dim, hidden=self.out_dim, bias=True, activation="prelu", norm=False)
            self.output_next8 = nn.Linear(self.out_dim, self.out_dim)
            torch.nn.init.constant_(self.output_next8.weight, 0)
            torch.nn.init.constant_(self.output_next8.bias, 0)

            self.MLP_w = MLP(self.out_dim, 3, hidden=self.out_dim, bias=True, activation="prelu", norm=False)
            self.output_next_w = nn.Linear(3, 3)
            self.output_next_w.weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))\
                .to(self.output_next_w.weight.device)
            torch.nn.init.constant_(self.output_next_w.bias, 0)
            initial_value = torch.zeros(self.t_pred+1, self.out_dim)
            self.pos_embedding = nn.Parameter(initial_value).to(self.output_next_w.weight.device)





    def calculate_parameters(self, obs_traj, pred_traj):
        r"""Calculate the ET descriptors of the EigenTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Note:
            This function should be called once before training the model.
        """

        # Mask out static trajectory
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        obs_m_traj, pred_m_traj = obs_traj[mask], pred_traj[mask]
        obs_s_traj, pred_s_traj = obs_traj[~mask], pred_traj[~mask]

        # Descriptor initialization
        data_m = self.ET_m_descriptor.parameter_initialization(obs_m_traj, pred_m_traj)
        data_s = self.ET_s_descriptor.parameter_initialization(obs_s_traj, pred_s_traj)

        # Anchor generation
        self.ET_m_anchor.anchor_generation(*data_m)
        self.ET_s_anchor.anchor_generation(*data_s)

    def inter_value(self, pred_traj_step_start, step, encoder_traj, mlp, output_next, pred_traj_recon):
        p1, p2 = pred_traj_step_start.shape[0], pred_traj_step_start.shape[1]  ## 0为
        if self.hyper_params.use_dynamic_pe:
            pred_traj_step_start = self.pos_embedding[::step, :].to(pred_traj_step_start.device) \
                                       .expand(p1, p2, -1, -1) * 0.01 + pred_traj_step_start
        else:
            pred_traj_step_start = self.pe.P[::step, :].to(pred_traj_step_start.device)\
                                       .expand(p1, p2, -1, -1)*0.01 + pred_traj_step_start
        long_inter = pred_traj_step_start.shape[2] - 1
        pred_traj_step_inter = torch.cat([pred_traj_step_start[:, :, :-1, :],
                                          pred_traj_step_start[:, :, 1:, :],
                                          encoder_traj.unsqueeze(2).repeat(1, 1, long_inter, 1)], dim=-1)

        pred_traj_step_inter = mlp(pred_traj_step_inter)
        pred_traj_step_inter = output_next(pred_traj_step_inter)\
                               + (pred_traj_step_start[:, :, :-1, :] + pred_traj_step_start[:, :, 1:, :])/2
        pred_traj_recon1 = pred_traj_recon.clone()
        if step == 8:
            pred_traj_recon1[:, :, step // 2:step // 2+1, :] = pred_traj_step_inter
        else:
            pred_traj_recon1[:, :, step//2::step, :] = pred_traj_step_inter
        return pred_traj_recon1

    @staticmethod
    def error_single(pred_traj_recon,pred_traj, flag='minade'):
        if flag=='minade':
            error_recon = (pred_traj_recon - pred_traj.unsqueeze(dim=0)).norm(p=2, dim=-1)
            error_recon = error_recon.mean(dim=-1).min(dim=0)[0]
        elif flag=='minfde':
            error_recon = (pred_traj_recon - pred_traj.unsqueeze(dim=0)).norm(p=2, dim=-1)
            error_recon = error_recon[:, :, -1].min(dim=0)[0]
        return error_recon


    def forward(self, obs_traj, pred_traj=None, addl_info=None):
        r"""The forward function of the EigenTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (optional, for training only)
            addl_info (dict): The additional information (optional, if baseline model requires)

        Returns:
            output (dict): The output of the model (recon_traj, loss, etc.)
        """
        n_ped = obs_traj.size(0)
        # Filter out static trajectory
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        obs_m_traj = obs_traj[mask]
        obs_s_traj = obs_traj[~mask]
        pred_m_traj_gt = pred_traj[mask] if pred_traj is not None else None
        pred_s_traj_gt = pred_traj[~mask] if pred_traj is not None else None


        # Projection
        C_m_obs, C_m_pred_gt = self.ET_m_descriptor.projection(obs_m_traj, pred_m_traj_gt)
        C_s_obs, C_s_pred_gt = self.ET_s_descriptor.projection(obs_s_traj, pred_s_traj_gt)
        C_obs = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
        C_obs[:, mask], C_obs[:, ~mask] = C_m_obs, C_s_obs  # KN

        # Absolute coordinate
        obs_m_ori = self.ET_m_descriptor.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_s_ori = self.ET_s_descriptor.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_ori = torch.zeros((2, n_ped), dtype=torch.float, device=obs_traj.device)
        obs_ori[:, mask], obs_ori[:, ~mask] = obs_m_ori, obs_s_ori
        obs_ori -= obs_ori.mean(dim=1, keepdim=True)  # move scene to origin

        # Trajectory prediction
        input_data = self.hook_func.model_forward_pre_hook(C_obs, obs_ori, addl_info)
        output_data = self.hook_func.model_forward(input_data, self.baseline_model)
        C_pred_refine = self.hook_func.model_forward_post_hook(output_data, addl_info)

        # Anchor refinement
        C_m_pred = self.ET_m_anchor(C_pred_refine[:, mask])
        C_s_pred = self.ET_s_anchor(C_pred_refine[:, ~mask])

        # Reconstruction
        pred_m_traj_recon = self.ET_m_descriptor.reconstruction(C_m_pred)
        pred_s_traj_recon = self.ET_s_descriptor.reconstruction(C_s_pred)
        pred_traj_recon = torch.zeros((self.s, n_ped, self.t_pred, self.dim), dtype=torch.float, device=obs_traj.device)
        pred_traj_recon[:, mask], pred_traj_recon[:, ~mask] = pred_m_traj_recon, pred_s_traj_recon

        if self.hyper_params.use_rnn:
            encoder_traj = self.MLP1(pred_traj_recon.reshape(pred_traj_recon.shape[0], pred_traj_recon.shape[1], -1))
            # encoder_traj = encoder_traj.unsqueeze(2).repeat(1, 1, 6, 1)
            in_end = pred_traj_recon.shape[2]
            pred_traj_recon_extra = 2*pred_traj_recon[:, :, in_end-1:in_end, :] \
                                    - pred_traj_recon[:, :, in_end-2:in_end-1, :]
            pred_traj_recon_1 = torch.cat([pred_traj_recon, pred_traj_recon_extra], dim=-2)

            if self.hyper_params.get("step_2", True):
                step = 2
                pred_traj_step_start = pred_traj_recon_1[:, :, ::step, :]

                pred_traj_recon_2 = self.inter_value(pred_traj_step_start, step, encoder_traj,
                                                     self.MLP2, self.output_next2, pred_traj_recon_1)
                output = {"recon_traj": pred_traj_recon_2[:, :, :-1, :]}

            if self.hyper_params.get("step_4", True):
                step = 4
                pred_traj_step_start = pred_traj_recon_1[:, :, ::step, :]
                pred_traj_recon_4 = self.inter_value(pred_traj_step_start, step, encoder_traj,
                                                    self.MLP4, self.output_next4, pred_traj_recon_1)
                step = 2
                pred_traj_step_start = pred_traj_recon_4[:, :, ::step, :]
                pred_traj_recon_4 = self.inter_value(pred_traj_step_start, step, encoder_traj,
                                                    self.MLP2, self.output_next2, pred_traj_recon_4)

                output = {"recon_traj": pred_traj_recon_4[:, :, :-1, :]}

            if self.hyper_params.get("step_8", True):
                step = 8
                pred_traj_step_start = pred_traj_recon_1[:, :, ::step, :]
                pred_traj_recon_8 = self.inter_value(pred_traj_step_start, step, encoder_traj,
                                                     self.MLP8, self.output_next8, pred_traj_recon_1)
                step = 4
                pred_traj_step_start = pred_traj_recon_8[:, :, ::step, :]
                pred_traj_recon_8 = self.inter_value(pred_traj_step_start, step, encoder_traj,
                                                     self.MLP4, self.output_next4, pred_traj_recon_8)

                step = 2
                pred_traj_step_start = pred_traj_recon_8[:, :, ::step, :]
                pred_traj_recon_8 = self.inter_value(pred_traj_step_start, step, encoder_traj,
                                                     self.MLP2, self.output_next2, pred_traj_recon_8)

                output = {"recon_traj": pred_traj_recon_8[:, :, :-1, :]}

            if self.hyper_params.get("fusion", True):
                # assert (self.hyper_params.get("step_2", True) + self.hyper_params.get("step_4", True)
                #         + self.hyper_params.get("step_8", True)) == 3,\
                #     "The number of fusion heads should be equal to 3."
                head_weights = self.MLP_w(encoder_traj)
                head_weights = self.output_next_w(head_weights)
                head_weights = torch.softmax(head_weights, dim=-1)
                output["head_weights"] = head_weights
                if pred_traj is not None:
                    error_recon_2 = self.error_single(pred_traj_recon_2[:, :, :-1, :], pred_traj, flag='minade')
                    error_recon_4 = self.error_single(pred_traj_recon_4[:, :, :-1, :], pred_traj, flag='minade')
                    error_recon_8 = self.error_single(pred_traj_recon_8[:, :, :-1, :], pred_traj, flag='minade')
                    error_recon = torch.cat([error_recon_2.unsqueeze(dim=-1),error_recon_4.unsqueeze(dim=-1),
                              error_recon_8.unsqueeze(dim=-1)], dim=-1)
                    min_indices = torch.argmin(error_recon, dim=1)
                    gt_error = torch.zeros_like(error_recon).to(error_recon.device).to(error_recon.dtype)
                    gt_error.scatter_(1, min_indices.unsqueeze(1), 1)
                    gt_error = gt_error.unsqueeze(dim=0).expand(head_weights.shape[0],-1,-1)
                    output["gt_error"] = gt_error

                max_indices = torch.argmax(head_weights, dim=-1)
                head_weights = torch.zeros_like(head_weights).to(head_weights.device).to(head_weights.dtype)
                head_weights.scatter_(-1, max_indices.unsqueeze(-1), 1)


                pred_traj_recon_2f = pred_traj_recon_2[:, :, :-1, :].unsqueeze(2)
                pred_traj_recon_4f = pred_traj_recon_4[:, :, :-1, :].unsqueeze(2)
                pred_traj_recon_8f = pred_traj_recon_8[:, :, :-1, :].unsqueeze(2)

                pred_traj_recon_fusion = torch.cat([pred_traj_recon_2f, pred_traj_recon_4f, pred_traj_recon_8f], dim=2)
                ep1, ep2 = pred_traj_recon_fusion.shape[-2], pred_traj_recon_fusion.shape[-1]
                head_weights = head_weights.unsqueeze(dim=-1).unsqueeze(dim=-1)
                pred_traj_recon_fusion = (pred_traj_recon_fusion * head_weights.expand(-1, -1, -1, ep1, ep2)).sum(dim=2)
                output["recon_traj"] = pred_traj_recon_fusion

            output["encoder_traj"] = encoder_traj


        else:
            output = {"recon_traj": pred_traj_recon}


        if pred_traj is not None:
            C_pred = torch.zeros((self.k, n_ped, self.s), dtype=torch.float, device=obs_traj.device)
            C_pred[:, mask], C_pred[:, ~mask] = C_m_pred, C_s_pred

            # Low-rank approximation for gt trajectory
            C_pred_gt = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
            C_pred_gt[:, mask], C_pred_gt[:, ~mask] = C_m_pred_gt, C_s_pred_gt
            C_pred_gt = C_pred_gt.detach()

            # Loss calculation
            error_coefficient = (C_pred - C_pred_gt.unsqueeze(dim=-1)).norm(p=2, dim=0)
            error_displacement = (output["recon_traj"] - pred_traj.unsqueeze(dim=0)).norm(p=2, dim=-1)
            output["loss_eigentraj"] = error_coefficient.min(dim=-1)[0].mean()
            output["loss_euclidean_ade"] = error_displacement.mean(dim=-1).min(dim=0)[0].mean()
            output["loss_euclidean_fde"] = error_displacement[:, :, -1].min(dim=0)[0].mean()

            if self.hyper_params.use_rnn:
                error_displacement_v = (torch.diff(output["recon_traj"], dim=-2) -
                                        torch.diff(pred_traj.unsqueeze(dim=0), dim=-2)).norm(p=2, dim=-1)
                output["loss_euclidean_ade_v"] = error_displacement_v.mean(dim=-1).min(dim=0)[0].mean()
                output["loss_euclidean_fde_v"] = error_displacement_v[:, :, -1].min(dim=0)[0].mean()

        return output



class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden=64, bias=True, activation="relu", norm='layer'):
        super(MLP, self).__init__()

        # define the activation function
        self.norm = norm
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "relu6":
            act_layer = nn.ReLU6
        elif activation == "leaky":
            act_layer = nn.LeakyReLU
        elif activation == "prelu":
            act_layer = nn.PReLU
        else:
            raise NotImplementedError

        # define the normalization function
        if norm == "layer":
            norm_layer = nn.LayerNorm
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d
        # else:
        #     raise NotImplementedError

        # insert the layers
        self.linear1 = nn.Linear(in_channel, hidden, bias=bias)
        if out_channel != 3:
            self.linear1.apply(self._init_weights)
        self.linear2 = nn.Linear(hidden, out_channel, bias=bias)
        if out_channel != 3:
            self.linear2.apply(self._init_weights)
        if self.norm:
            self.norm1 = norm_layer(hidden)
            self.norm2 = norm_layer(out_channel)
        if activation == "prelu":
            self.act1 = act_layer()
            self.act2 = act_layer()
        else:
            self.act1 = act_layer(inplace=True)
            self.act2 = act_layer(inplace=True)

        # self.shortcut = None
        # if in_channel != out_channel:
        #     if self.norm:
        #         self.shortcut = nn.Sequential(
        #             nn.Linear(in_channel, out_channel, bias=bias),
        #             norm_layer(out_channel),
        #             nn.Dropout(0.1, inplace=True)
        #         )
        #     else:
        #         self.shortcut = nn.Sequential(
        #             nn.Linear(in_channel, out_channel, bias=bias),
        #             nn.Dropout(0.1, inplace=True)
        #         )


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.constant_(m.weight, 0)
            torch.nn.init.constant_(m.bias, 0)
            # m.bias.data.fill_(0)

    def forward(self, x):
        out = self.linear1(x)
        if self.norm:
            out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        if self.norm:
            out = self.norm2(out)

        # if self.shortcut:
        #     out += self.shortcut(x)
        # else:
        #     out += x
        return self.act2(out)

class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    # num_hiddens:Input data's embedding dimension，max_len：Input sequence's maximum length
    def __init__(self, num_hiddens=2, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((max_len, num_hiddens))
        pe = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, 0::2] = torch.sin(pe)
        self.P[:, 1::2] = torch.cos(pe)

