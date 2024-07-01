import torch
import torch.nn.functional as F
import numpy as np
def choose_disp_weighted_map(gt, map_shape, device, mode = 'motion_status'):

    gt_clone = gt.clone()
    gt_clone[gt < 0] = 0
    gt_clone = gt_clone.numpy()
    disp_weight_map = gt_clone
    
    bs, _, _, bev_w, bev_h = map_shape
    # [320, 2, 256, 256]
    disp_weight_map = disp_weight_map.reshape(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])
    # disp_weight_map = np.mean(disp_weight_map, axis=1).unsqueeze(1)

    if mode == 'norm1':
        disp_weight_map = F.normalize(disp_weight_map, dim=[-2, -1], p=1) * 5
    elif mode == 'norm2':
        disp_weight_map = F.normalize(disp_weight_map, dim=[-2, -1], p=2) * 5
    elif mode == 'softmax':
        disp_weight_map = F.softmax(disp_weight_map, dim=-2)
    elif mode == 'cls':
        # [bs, 1 , 1]
        max_disp_map = disp_weight_map.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        
        # print(max_disp_map.shape)
        # print(disp_weight_map.max(dim=-1)[0].shape)
        
        # print(max_disp_map[0])
        disp_weight_map[disp_weight_map > 0.5 * max_disp_map] = 2
        # disp_weight_map[torch.where(disp_weight_map > 0.5 * max_disp_map, disp_weight_map, torch.zeros_like(disp_weight_map))] = 2
    elif mode == 'motion_status':
        last_frame_disp_norm = np.linalg.norm(gt_clone, ord=2, axis=-1)

        # [bs, 256, 256]
        # last_frame_disp_norm = last_frame_disp_norm[:, -1, :, :]
        upper_thresh = 0.2
        frame_skip = 3
        upper_bound = (frame_skip + 1) / 20 * upper_thresh
        selected_future_sweeps = np.arange(0, 20 + 1, frame_skip + 1)
        selected_future_sweeps = selected_future_sweeps[1:]

        future_sweeps_disp_field_gt_norm = last_frame_disp_norm[:, -len(selected_future_sweeps):, ...]
        static_cell_mask = future_sweeps_disp_field_gt_norm <= upper_bound
        static_cell_mask = np.all(static_cell_mask, axis=1)  # along the sequence axis
        moving_cell_mask = np.logical_not(static_cell_mask)

        # speed_intervals = np.array([[0, 5.0], [5.0, 20.0], [20.0, np.inf]])
        speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0], [20.0, np.inf]])
        # Next, compute the speed level mask
        last_future_sweep_id = selected_future_sweeps[-1]
        distance_intervals = speed_intervals * (last_future_sweep_id / 20.0)

        
        last_frame_disp_norm = last_frame_disp_norm[:, -1, :, :]
        
        weight_vector = [0.95, 1.25, 1.35, 0.75]
        disp_weight_map = np.zeros((bs, bev_w, bev_h), dtype=np.float32)

        for weight, s, d in zip(weight_vector, range(0, speed_intervals.shape[0]), distance_intervals):
            if s == 0:
                mask = static_cell_mask
            else:
                mask = np.logical_and(d[0] <= last_frame_disp_norm, last_frame_disp_norm < d[1])
                mask = np.logical_and(mask, moving_cell_mask)
            disp_weight_map[mask] = weight

        disp_weight_map = disp_weight_map[:, np.newaxis, np.newaxis, ...]
        disp_weight_map = torch.from_numpy(disp_weight_map).to(device)
    elif mode == "motion_status_withoutS":
        last_frame_disp_norm = np.linalg.norm(gt_clone, ord=2, axis=-1)

        # [bs, 256, 256]
        last_frame_disp_norm = last_frame_disp_norm[:, -1, :, :]

        speed_intervals = np.array([[0, 5.0], [5.0, 20.0], [20.0, np.inf]])
        # speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0], [20.0, np.inf]])
        # Next, compute the speed level mask
        # last_future_sweep_id = selected_future_sweeps[-1]
        distance_intervals = speed_intervals * (20.0 / 20.0)
        
        weight_vector = [1.2, 0.9, 0.4]
        disp_weight_map = np.zeros((bs, bev_w, bev_h), dtype=np.float32)

        for weight, s, d in zip(weight_vector, range(0, speed_intervals.shape[0]), distance_intervals):

            mask = np.logical_and(d[0] <= last_frame_disp_norm, last_frame_disp_norm < d[1])
            disp_weight_map[mask] = weight

        disp_weight_map = disp_weight_map[:, np.newaxis, np.newaxis, ...]
        disp_weight_map = torch.from_numpy(disp_weight_map).to(device)
    else:
        AssertionError,'wrong loss name'
    # loss_disp = torch.sum(loss_disp * cat_weight_map * disp_weight_map) / valid_pixel_num
    return disp_weight_map

if __name__ == '__main__':
    map_shape = [16, 1, 1, 256, 256]
    gt = torch.rand(16, 20, 256, 256, 2)
    disp_weight_map = choose_disp_weighted_map(gt, map_shape, 'motion_status')