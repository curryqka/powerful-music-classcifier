# net-v5-2
# baseline: contexture encoder add seq_mixture

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: B, T, C, H, W
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W
        x = self.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        return x

class Seq_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Seq_Conv, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = Conv3D(out_channel, out_channel, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.bn1_1 = nn.BatchNorm2d(out_channel)
        self.bn1_2 = nn.BatchNorm2d(out_channel)
    
    def forward(self, x):
        # x: B, T, C1, H, W
        B, T, C, H, W = x.size()
        x = x.view(-1, C, H, W)    # BT, C1, H, W
        x = self.relu(self.bn1_1(self.conv1_1(x)))    # BT, C2, H, W
        x = self.relu(self.bn1_2(self.conv1_2(x)))    # BT, C2, H, W
        x = x.view(B, -1, x.size(1), x.size(2), x.size(3))    # B, T, C2, H, W
        x = self.conv1_3(x)    # B, T-2, C2, H, W
        return x
        
class Seq_Mixture1(nn.Module):
    def __init__(self, in_channel):
        super(Seq_Mixture1, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.seq_conv1 = Seq_Conv(in_channel, 3*in_channel)
        self.seq_conv2 = Seq_Conv(3*in_channel, 5*in_channel)
        self.conv2d = nn.Conv2d(2*in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(in_channel)

    
    def forward(self, x):
        # x: B, T, C, H, W
        B, T, C, H, W = x.size()
        x_ = self.seq_conv1(x)   # B, T-2, 3*C, H, W
        x_ = self.seq_conv2(x_)  # B, 1, 5*C, H, W
        x_ = x_.view(-1, x_.size(2), x_.size(3), x_.size(4))   # B, 5C, H, W
        x_ = x_.view(x_.size(0), 5, x_.size(1) // 5, x_.size(2), x_.size(3))   # B, 5, C, H, W
        x = torch.cat((x, x_), dim=2)    # B, 5, 2c, h, w
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()     # BT, 2c, h, w
        x = self.relu(self.bn2d(self.conv2d(x)))    # BT, c, h, w
        x = x.view(B, T, C, H, W)

        return x

class Seq_Mixture2(nn.Module):
    def __init__(self, in_channel):
        super(Seq_Mixture2, self).__init__()        
        self.relu = nn.ReLU(inplace=False)
        self.seq_conv = Seq_Conv(in_channel, 3*in_channel)
        self.conv2d = nn.Conv2d(2*in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(in_channel)
    def forward(self, x):
        # x: B, 3, C, H, W
        B, T, C, H, W = x.size()
        x_ = self.seq_conv(x)  # B, 1, 3C, H, W
        x_ = x_.view(-1, x_.size(2), x_.size(3), x_.size(4))   # B, 3C, H, W
        x_ = x_.view(x_.size(0), 3, x_.size(1) // 3, x_.size(2), x_.size(3))   # B, 3, C, H, W
        x = torch.cat((x, x_), dim=2)    # B, 3, 2c, h, w
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()     # B*3, 2c, h, w
        x = self.relu(self.bn2d(self.conv2d(x)))    # B3, c, h, w   
        x = x.view(B, T, C, H, W)    
        return x 


class CellClassification(nn.Module):
    # qka 20240129 : category_num = 2, 
    def __init__(self, category_num=2):
        super(CellClassification, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class StateEstimation(nn.Module):
    def __init__(self, motion_category_num=2):
        super(StateEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, motion_category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class MotionPrediction(nn.Module):
    def __init__(self, seq_len):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x


class STPN(nn.Module):
    def __init__(self, height_feat_size=13):
        super(STPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

        self.seq_mixture1 = Seq_Mixture1(32)
        self.seq_mixture2 = Seq_Mixture2(64)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))   # B*seq, z=13, h, w
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))        # B*seq, 32, h, w
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))        # B*seq, 32, h, w

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))   # B, 5, 32, H, W
        x = self.seq_mixture1(x)   # B, 5, C, H, W
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()    # BT, 32, H, W


        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))            # B*seq, 64, h/2, w/2
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))          # B*seq, 64, h/2, w/2

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # B, seq=5, 64, h/2, w/2
        x_1 = self.conv3d_1(x_1)                                                       # B, seq=3, 64, h/2, w/2

        x_1 = self.seq_mixture2(x_1)   # B, 3, 64, H, W
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()         # B*seq=3,  64, h/2, w/2

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))                                    # B*seq=3, 128, h/4, w/4
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))                                    # B*seq=3, 128, h/4, w/4

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # B, seq=3, 128, h/4, w/4
        x_2 = self.conv3d_2(x_2)                                                       # B, seq=1, 128, h/4, w/4
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()         # B*seq=1,  128, h/4, w/4

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))                                    # B*seq=1, 256, h/8, w/8
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))                                    # B*seq=1, 256, h/8, w/8

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))                                    # B*seq=1, 512, h/16, w/16
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))                                    # B*seq=1, 512, h/16, w/16


        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))    # B*seq=1, 512 + 256 ->256, h/8, w/8
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))                                                                 # B*seq=1, 256, h/8, w/8

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))                                            # B, seq=1, 128, h/4, w/4
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()                                                               # B, 128, 1, h/4, w/4
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))                                                           # B, 128, 1, h/4, w/4
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()                                                               # B, 1, 128, h/4, w/4
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()                                      # B*1,  128, h/4, w/4

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))    # B*seq=1, 256 + 128 ->128, h/4, w/4
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))                                                                 # B*seq=1, 128, h/4, w/4

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))                                            # B, seq=3, 64, h/2, w/2
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()                                                               # B, 64, 3, h/2, w/2
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))                                                           # B, 64, 1, h/2, w/2
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()                                                               # B, 1, 64, h/2, w/2
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()                                      # B*1,  64, h/2, w/2

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))    # B*seq=1, 128 + 64 ->64, h/2, w/2
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))                                                                 # B*seq=1, 64, h/2, w/2

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))                                                      # B, seq=5, 32, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()                                                                   # B, 32, seq=5, h, w
        x = F.adaptive_max_pool3d(x, (1, None, None))                                                               # B, 32, 1, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()                                                                   # B, seq=1, 32, h, w
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()                                                # B*seq=1,  32, h, w

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))      # B*seq=1, 64+32->32, h, w
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))                                                               # B*seq=1, 32, h, w

        return res_x                                                                                                # B, 32, h, w


class MotionNet(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13):
        super(MotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn = STPN(height_feat_size=height_feat_size)


    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs)

        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred


# For MGDA loss computation
class FeatEncoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(FeatEncoder, self).__init__()
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.stpn(bevs)

        return x


class MotionNetMGDA(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2):
        super(MotionNetMGDA, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

    def forward(self, stpn_out):
        # Cell Classification head
        cell_class_pred = self.cell_classify(stpn_out)

        # Motion State Classification head
        state_class_pred = self.state_classify(stpn_out)

        # Motion Displacement prediction
        disp = self.motion_pred(stpn_out)
        disp = disp.view(-1, 2, stpn_out.size(-2), stpn_out.size(-1))

        return disp, cell_class_pred, state_class_pred
