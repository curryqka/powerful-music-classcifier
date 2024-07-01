import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorInfo(nn.Module):
    r"""An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1704.00028v3>`_ paper.
    """

    def __init__(self, in_channel=32, out_channel=128) -> None:
        super(DiscriminatorInfo, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, \
            kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(64, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.QHead = QHead()
        self.DHead = DHead()
        # self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1 , inplace=True)

        # discriminator
        true_prob = self.DHead(x)

        # info
        disp, cell_class_pred, state_class_pred, mu, var = self.QHead(x)
        return disp, cell_class_pred, state_class_pred, true_prob, mu, var
class DiscriminatorAC(nn.Module):
    r"""An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1704.00028v3>`_ paper.
    """
    def __init__(self, in_channel=32, out_channel=32) -> None:
        super(DiscriminatorAC, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, \
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(32, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

        self.AHead = AHead()
        self.DHead = DHead()
        # self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        # x shape: (bs, 32, w, h)
        res = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1 , inplace=True)

        x = self.bn2(self.conv2(x))
        x = x + res
        x = F.leaky_relu(x, 0.1, inplace=True)
        # discriminator
        true_prob = self.DHead(x)

        # info
        disp, cell_class_pred, state_class_pred= self.AHead(x)
        return disp, cell_class_pred, state_class_pred, true_prob

class GeneratorAC(nn.Module):
    # TODO: get the input dim
    def __init__(self, input_channel=100, mid_channel=67, out_channel=32):
        super(GeneratorAC, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(input_channel, 512, 4, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        # self.tconv5 = nn.ConvTranspose2d(64, out_channel, 4, 2, padding=1, bias=False)

        self.conv = nn.Conv2d(mid_channel+64, out_channel, 3, 1, 1)
    def forward(self, x, labels: torch.tensor=None):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))

        # img = F.relu(self.tconv5(x))
        img = torch.concat([x, labels], dim=1)
        img = self.conv(img)
        return img

class DHead(nn.Module):
    def __init__(self, hidden_dim = 32,):
        super(DHead, self).__init__()

        self.conv = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x):
        # output = torch.sigmoid(self.conv(x))
        # WGAN
        output = torch.flatten(self.conv(x))
        return output

class QHead(nn.Module):
    def __init__(self, hidden_dim = 32,out_seq_len=20, motion_category_num=2,) -> None:
        super(QHead, self).__init__()

        self.out_seq_len = out_seq_len
        self.conv1 = nn.Conv2d(hidden_dim, 32, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

        self.conv_mu = nn.Conv2d(32, 2, 1)
        self.conv_var = nn.Conv2d(32, 2, 1)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        
        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disp, cell_class_pred, state_class_pred, mu, var

class AHead(nn.Module):
    def __init__(self, hidden_dim = 32, hidden_=False, out_seq_len=20, motion_category_num=2,) -> None:
        super(AHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.out_seq_len = out_seq_len
        
        if hidden_:
            self.conv1 = nn.Conv2d(hidden_dim, 32, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

    def forward(self, x):
        if x.shape[1] != 32:
            x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        
        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred


class CellClassification(nn.Module):
    def __init__(self, category_num=5):
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
    r""" An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1610.09585>`_ paper.
    """
    def __init__(self, seq_len):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x

class DiscriminatorDisp(nn.Module):

    def __init__(self, in_channel=40, out_channel=512):
        super(DiscriminatorDisp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, \
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=out_channel)

        self.AHead = AHead(hidden_dim=512, hidden_=True)
        self.DHead = DHead(hidden_dim=512)

    def forward(self, x: torch.Tensor):
        if len(x.shape) > 4:
            x = x.view(x.shape[0], -1, x.shape[2], x.shape[3])
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1 , inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1 , inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1 , inplace=True)
        # discriminator
        true_prob = self.DHead(x)

        disp, cell_class_pred, state_class_pred= self.AHead(x)

        return disp, cell_class_pred, state_class_pred, true_prob

class GeneratorDisp(nn.Module):
    r""" An Generator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1610.09585>`_ paper.
    """
    def __init__(self, grid_size:int=256, input_channel=100, mid_channel=38, out_channel=40) -> None:
        super().__init__()
    
        self.tconv1 = nn.ConvTranspose2d(input_channel, 512, 4, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        # self.tconv5 = nn.ConvTranspose2d(64, out_channel, 4, 2, padding=1, bias=False)

        self.conv = nn.Conv2d(mid_channel+64, out_channel, 3, 1, 1)

    def forward(self, x, labels: torch.tensor=None):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))

        # img = F.relu(self.tconv5(x))
        img = torch.concat([x, labels], dim=1)
        img = self.conv(img)
        out = img.reshape(img.shape[0], -1, img.shape[2], img.shape[3], 2)

        return out