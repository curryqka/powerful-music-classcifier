"""
Train WeakMotionNet in Stage2
Some of the code are modified based on 'train_single_seq.py' in MotionNet.

Reference:
MotionNet (https://www.merl.com/research/?research=license-request&sw=MotionNet)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from model.weak_model import WeakMotionNet
from model.be_sti import MotionNet
from data.weak_nuscenes_dataloader import DatasetSingleSeq_Stage2
from data.weak_waymo_dataloader import DatasetSingleSeq_Stage2 as DatasetSingleSeq_Stage2_waymo

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from loss_utils import FGBG_seg_loss, CCD_loss
from evaluation_utils import evaluate_FGBG_prediction, evaluate_motion_prediction, \
    evaluate_FGBG_prediction_general, evaluate_motion_prediction_general    


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

out_seq_len = 1  # The number of future frames we are going to predict
height_feat_size = 13  # The size along the height dimension

parser = argparse.ArgumentParser()
parser.add_argument('-md', '--motiondata', default='/path_to/nuScenes/input-data/train/', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('-wd', '--weakdata', default='/path_to/nuScenes/weak-data/train/', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('-FBd', '--FBdata', default='/path_to/nuScenes/FGBG-data/nuscenes_seg_0-01/', type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--datatype', default='nuScenes', type=str, choices=['Waymo', 'nuScenes'])

parser.add_argument('-t', '--evaldata', default='/path_to/nuScenes/input-data/val/', type=str, help='The path to the preprocessed sparse BEV training data')

parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--batch', default=8, type=int, help='Batch size')
parser.add_argument('--nepoch', default=60, type=int, help='Number of epochs')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')
parser.add_argument('--log', default=True, action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')
parser.add_argument('--gpu', default='0')
parser.add_argument('--annotation_ratio', default=0.01, type=float)

parser.add_argument('--model', default='WeakMotionNet', type=str, help='The backbone network:[WeakMotionNet/be-sti]')
parser.add_argument('--loss', action='store_true', help='Whether to use weak supervised loss function')

args = parser.parse_args()
print(args)

num_epochs = args.nepoch
need_log = args.log
BATCH_SIZE = args.batch
num_workers = args.nworker
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
datatype = args.datatype
annotation_ratio = args.annotation_ratio

def main():
    start_epoch = 1
    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'Stage2'))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()


        else:
            model_save_path = args.resume

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
    if datatype == 'nuScenes':
        area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
    elif datatype == 'Waymo':
        area_extents = np.array([[-32., 32.], [-32., 32.], [-1., 4.]])

    tmp = args.motiondata
    # load folder name "train" "trainval" "val", etc
    trainset_split = tmp.split('/')[-1] if tmp.split('/')[-1] is not '' else tmp.split('/')[-2]
    if datatype == 'nuScenes':
        trainset = DatasetSingleSeq_Stage2(dataset_root=args.motiondata, weakdata_root=args.weakdata,
                                           FBdata_root=args.FBdata, split=trainset_split,
                                           annotation_ratio=annotation_ratio,
                                           voxel_size=voxel_size, area_extents=area_extents)
    elif datatype == 'Waymo':
        trainset = DatasetSingleSeq_Stage2_waymo(dataset_root=args.motiondata, weakdata_root=args.weakdata,
                                                 FBdata_root=args.FBdata, split=trainset_split,
                                                 annotation_ratio=annotation_ratio,
                                                 voxel_size=voxel_size, area_extents=area_extents)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    print("Training dataset size:", len(trainset))


    tmp = args.evaldata
    evalset_split = tmp.split('/')[-1] if tmp.split('/')[-1] is not '' else tmp.split('/')[-2]
    if datatype == 'nuScenes':
        evalset = DatasetSingleSeq_Stage2(dataset_root=args.evaldata, split=evalset_split,
                                          voxel_size=voxel_size, area_extents=area_extents)
    elif datatype == 'Waymo':
        evalset = DatasetSingleSeq_Stage2_waymo(dataset_root=args.evaldata, split=evalset_split,
                                                voxel_size=voxel_size, area_extents=area_extents)

    evalloader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=num_workers)
    print("Eval dataset size:", len(evalset))


    model_dict = {
        'WeakMotionNet' : WeakMotionNet,
        'be-sti' : MotionNet,
    }
    # model = WeakMotionNet(out_seq_len=out_seq_len, FGBG_category_num=2, height_feat_size=height_feat_size)
    model = model_dict[args.model](out_seq_len, 2, height_feat_size=height_feat_size)

    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        model.train()
        loss_FGBG_seg, loss_disp = train(model, trainloader, optimizer, device, epoch, voxel_size, area_extents)

        model.eval()
        me_static, me_slow, me_fast, acc_bg, acc_fg = eval(model, evalloader, device)

        scheduler.step()


        if need_log:
            saver.write("loss_FGBG_seg:{}\t loss_disp:{}\n".format(loss_FGBG_seg, loss_disp))
            saver.write("me_static:{}\t me_slow:{}\t me_fast:{}\n".format(me_static, me_slow, me_fast))
            saver.write("acc_bg:{}\t acc_fg:{}\n".format(acc_bg, acc_fg))
            saver.flush()

        # save model
        if need_log and (epoch >= 30) and (epoch % 3 == 0):
            save_dict = {'epoch': epoch,
                         'model_state_dict': model.state_dict(),
                         'loss_FGBG_seg': loss_FGBG_seg.avg,
                         'loss_disp': loss_disp.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '_%.3f_%.3f_%.3f_%.3f_%.3f'%(me_static, me_slow, me_fast, acc_bg, acc_fg) + '.pth'))

    if need_log:
        saver.close()


def train(model, trainloader, optimizer, device, epoch, voxel_size, area_extents):
    running_loss_FGBG_seg = AverageMeter('FGBG_Seg', ':.6f')  # for cell FG/BG segmentation error
    running_loss_disp= AverageMeter('disp', ':.6f')  # for cell motion prediction error

    # for i, data in enumerate(trainloader, 0):
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader), smoothing=0.9):
        padded_voxel_points, _, _, \
        non_empty_map, _, _, \
        pc_seg, point_FGBG_gt_mask_seg, curr_seg_num, \
        FG_point_0, FG_point_num_0, FG_point_1, FG_point_num_1, FG_point_2, FG_point_num_2  = data
        # print(pc_seg,pc_seg.size())
        optimizer.zero_grad()

        # Move to GPU/CPU
        padded_voxel_points = padded_voxel_points.to(device)
        non_empty_map = non_empty_map.to(device)

        # Make prediction
        if args.model == 'be-sti':
            disp_pred, _, FGBG_pred = model(padded_voxel_points)

        elif args.model == 'WeakMotionNet':
            disp_pred, FGBG_pred = model(padded_voxel_points)
        else:
            assert False,"The model contains [WeakMotionNet/be-sti]"

        # Compute and back-propagate the losses
        # print(args.loss)
        if not args.loss:
            # print('12')
            loss_FGBG_seg = FGBG_seg_loss(FGBG_pred, point_FGBG_gt_mask_seg, pc_seg, curr_seg_num, voxel_size, area_extents)
            loss_disp = CCD_loss(disp_pred, FG_point_0, FG_point_num_0, FG_point_1, FG_point_num_1, FG_point_2, FG_point_num_2,
                                non_empty_map, voxel_size, area_extents, epoch, epoch_threshold=10, theta2=1)

            total_loss = loss_FGBG_seg + loss_disp
            running_loss_FGBG_seg.update(loss_FGBG_seg.item())
            
        else:
            # print('1')
            loss_disp = CCD_loss(disp_pred, FG_point_0, FG_point_num_0, FG_point_1, FG_point_num_1, FG_point_2, FG_point_num_2,
                                non_empty_map, voxel_size, area_extents, epoch, epoch_threshold=10, theta2=1)

            total_loss = loss_disp
        total_loss.backward()
        optimizer.step()

        # running_loss_FGBG_seg.update(loss_FGBG_seg.item())
        running_loss_disp.update(loss_disp.item())

    print("{}, \t {}, \tat epoch {}, \titerations {}".
          format(running_loss_FGBG_seg, running_loss_disp, epoch, i))

    return running_loss_FGBG_seg, running_loss_disp


def eval(model, evalloader, device):

    # Motion
    if datatype == 'nuScenes':
        num_future_sweeps = 20
        frequency = 20.0
        speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])
    elif datatype == 'Waymo':
        num_future_sweeps = 10
        frequency = 10.0
        speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 40.0]])

    selected_future_sweeps = np.arange(0, num_future_sweeps + 1, num_future_sweeps)  # We evaluate predictions at 1s
    selected_future_sweeps = selected_future_sweeps[1:]
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / frequency)  # "20" is because the LIDAR scanner is 20Hz

    cell_groups = list()  # grouping the cells with different speeds
    for i in range(distance_intervals.shape[0]):
        cell_statistics = list()

        for j in range(len(selected_future_sweeps)):
            # corresponds to each row, which records the MSE, median etc.
            cell_statistics.append([])
        cell_groups.append(cell_statistics)


    # Foreground/Background Classification
    overall_cls_pred = list()  # to compute FG/BG classification accuracy for each object category
    overall_cls_gt = list()  # to compute FG/BG classification accuracy for each object category

    # nongeneral Classification
    overall_nongeneral_gt = list()
    # for i, data in enumerate(evalloader, 0):
    for i, data in tqdm(enumerate(evalloader, 0), total=len(evalloader), smoothing=0.9):
        padded_voxel_points, all_disp_field_gt, pixel_cat_map_gt, \
        non_empty_map, all_valid_pixel_maps, future_steps,\
        _, _, _, \
        _, _, _, _, _, _ = data

        padded_voxel_points = padded_voxel_points.to(device)

        with torch.no_grad():
            if args.model == 'WeakMotionNet':
                disp_pred, FGBG_pred = model(padded_voxel_points)
            elif args.model == 'be-sti':
                disp_pred, _, FGBG_pred = model(padded_voxel_points)
            else:
                assert False,"The model contains [WeakMotionNet/be-sti]"

            disp_pred = disp_pred * 2.0

            non_empty_map_numpy = non_empty_map.numpy()
            pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
            '''
            overall_cls_gt, overall_cls_pred = evaluate_FGBG_prediction(FGBG_pred, non_empty_map_numpy, pixel_cat_map_gt_numpy,
                                                                        overall_cls_gt, overall_cls_pred, datatype)
            cell_groups = evaluate_motion_prediction(disp_pred, FGBG_pred,
                                                     all_disp_field_gt, all_valid_pixel_maps, future_steps,
                                                     distance_intervals, selected_future_sweeps, cell_groups, datatype)
            '''
            overall_cls_gt, overall_cls_pred, overall_nongeneral_gt, cat_nongeneral_map = evaluate_FGBG_prediction_general(FGBG_pred, non_empty_map_numpy, pixel_cat_map_gt_numpy,
                                                                        overall_cls_gt, overall_cls_pred, overall_nongeneral_gt, datatype)
            
            cell_groups_general = evaluate_motion_prediction_general(disp_pred, FGBG_pred, 
                                                     all_disp_field_gt, pixel_cat_map_gt_numpy, all_valid_pixel_maps, future_steps,
                                                     distance_intervals, selected_future_sweeps, cell_groups, datatype)
            
    me_list = np.zeros([3])

    def get_speed_error(cell):
        for i, d in enumerate(speed_intervals):
            group = cell[i]
            print("--------------------------------------------------------------")
            print("For cells within speed range [{}, {}]:\n".format(d[0], d[1]))

            dump_error = []
            dump_error_quantile_50 = []

            for s in range(len(selected_future_sweeps)):
                row = group[s]

                errors = np.concatenate(row) if len(row) != 0 else row

                if len(errors) == 0:
                    mean_error = None
                    error_quantile_50 = None
                else:
                    mean_error = np.average(errors)
                    error_quantile_50 = np.quantile(errors, 0.5)

                dump_error.append(mean_error)
                dump_error_quantile_50.append(error_quantile_50)

                msg = "Frame {}:\nThe mean error is {}\nThe 50% error quantile is {}". \
                    format(selected_future_sweeps[s], mean_error, error_quantile_50)
                print(msg)
            me_list[i] = mean_error
        return me_list
    
    # me_list = get_speed_error(cell_groups)
    me_list = get_speed_error(cell_groups_general)
    # print(me_list == me_list_general)
    
    '''
    for i, d in enumerate(speed_intervals):
        group = cell_groups[i]
        print("--------------------------------------------------------------")
        print("For cells within speed range [{}, {}]:\n".format(d[0], d[1]))

        dump_error = []
        dump_error_quantile_50 = []

        for s in range(len(selected_future_sweeps)):
            row = group[s]

            errors = np.concatenate(row) if len(row) != 0 else row

            if len(errors) == 0:
                mean_error = None
                error_quantile_50 = None
            else:
                mean_error = np.average(errors)
                error_quantile_50 = np.quantile(errors, 0.5)

            dump_error.append(mean_error)
            dump_error_quantile_50.append(error_quantile_50)

            msg = "Frame {}:\nThe mean error is {}\nThe 50% error quantile is {}". \
                format(selected_future_sweeps[s], mean_error, error_quantile_50)
            print(msg)
        me_list[i] = mean_error
    '''

    # Compute the mean FG/BG classification accuracy for each category
    overall_cls_gt = np.concatenate(overall_cls_gt)
    overall_cls_pred = np.concatenate(overall_cls_pred)
    cm = confusion_matrix(overall_cls_gt, overall_cls_pred)
    cm_sum = np.sum(cm, axis=1)

    mean_cat = cm[np.arange(2), np.arange(2)] / cm_sum
    cat_map = {0: 'Background', 1: 'Foreground'}
    for i in range(len(mean_cat)):
        print("mean cat accuracy of {}: {}".format(cat_map[i], mean_cat[i]))

    # Compute the statistics of mean pixel classification accuracy
    pixel_acc = np.sum(cm[np.arange(2), np.arange(2)]) / np.sum(cm_sum)
    print("Mean pixel classification accuracy: {}".format(pixel_acc))

    return me_list[0], me_list[1], me_list[2], cm[0, 0] / cm_sum[0], cm[1, 1] / cm_sum[1]

if __name__ == "__main__":
    main()
