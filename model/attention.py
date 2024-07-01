import torch.nn.functional as F
import torch.nn as nn
import torch,math

# modified by SE model
class SElayer(nn.Module):
    """
    reduction is the hyper-parameter
    """
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# activate function in CAlayer
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CAlayer(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CAlayer, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        # ReLU can be replaced by h_swish
        self.act = h_swish()

        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x): # torch.size(2, 64, 5, 5)
        bs, c, h, w = x.size()

        # (b, c, h, w) -> (b, c, h, 1) -> (b, c, 1, h)
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)

        # (b, c, h, w) -> (b, c, 1, w) 
        x_w = self.avg_pool_y(x)

        # cat and conv
        # (b, c, 1, w) cat (b, c, 1, h) -> (b, c, 1, w + h)
        # (b, c, 1, w + h) -> (b, c/r, 1, w + h)
        x_cat_conv_relu = self.act(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # split
        # (b, c/r, 1, w + h) -> (b, c/r, 1, h) + (b, c/r, 1, w)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        # (b, c/r, 1, h) -> (b, c, h, 1)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))

        # (b, c/r, 1, w) -> (b, c, 1, w)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        # out = x * s_h.expand(x.shape) * s_w.expand_as(x.shape)

        return out

class ECAlayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):  #64
        super(ECAlayer, self).__init__()
        kernel_size = int(abs((math.log(channel,2)+  b)/gamma))  #3
        kernel_size = kernel_size if kernel_size % 2  else kernel_size+1  #3
        padding = kernel_size//2
        self.avg_pool =nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()  #torch.Size([2, 64, 26, 26])
        #变成序列的形式
        avg = self.avg_pool(x).view([b, 1, c])   #torch.Size([2, 64, 1, 1])   torch.Size([2, 1, 64])
        out = self.conv(avg)                     #torch.Size([2, 1, 64])
        out = self.sigmoid(out).view([b, c, 1, 1])  #torch.Size([2, 64, 1, 1])
        return  out * x

# channel attention
class channel_attention(nn.Module):
    def __init__(self, channel, ration=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel//ration, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):           #torch.Size([2, 64, 5, 5])
        b, c, h, w = x.size()
        avg_pool = self.avg_pool(x).view([b, c])  #torch.Size([2, 64])
        max_pool = self.max_pool(x).view([b, c])  #torch.Size([2, 64])

        avg_fc = self.fc(avg_pool)  #torch.Size([2, 64])
        max_fc = self.fc(max_pool)  #torch.Size([2, 64])

        out = self.sigmoid(max_fc+avg_fc).view([b, c, 1, 1])  ##torch.Size([2, 64, 1, 1])
        return x * out

# spatial attention
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # channel maxPooling
        max_pool = torch.max(x, dim=1, keepdim=True).values  #torch.Size([2, 1, 5, 5])
        avg_pool = torch.mean(x, dim=1, keepdim=True)        #torch.Size([2, 1, 5, 5])
        pool_out = torch.cat([max_pool, avg_pool], dim=1)    #torch.Size([2, 2, 5, 5])
        conv = self.conv(pool_out)                           #torch.Size([2, 1, 5, 5])
        out = self.sigmoid(conv)

        return out * x

# combine channel and spatial
class CBAM(nn.Module):
    def __init__(self, channel, ration=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(channel, ration)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)    #torch.Size([2, 64, 5, 5])
        out = self.spatial_attention(out)  #torch.Size([2, 64, 5, 5])

        return out