import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diff // 2, diff - diff //2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


### add RNN to C2F_Encoder
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class C2F_Encoder(nn.Module):
    '''
        Features are extracted at the last layer of decoder. 
    '''
    def __init__(self, n_channels):
        super(C2F_Encoder, self).__init__()
        self.inc = inconv(n_channels, 256)
        self.down1 = down(256, 256)
        self.down2 = down(256, 256)
        self.down3 = down(256, 128)
        self.down4 = down(128, 128)
        self.down5 = down(128, 128)
        self.up = up(256, 128)
        self.up1 = up(256, 128)
        self.up2 = up(384, 128)
        self.up3 = up(384, 128)
        self.up4 = up(384, 128)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(128, 128, 128),
            BidirectionalLSTM(128, 128, 128))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x_5 = self.up(x6, x5)
        x_5_out = x_5.permute(2, 0, 1)  #  permute to T, b, h
        x_5_out = self.rnn(x_5_out)
        x_5_out = x_5_out.permute(1, 2, 0)

        x_4 = self.up1(x_5, x4)
        x_4_out = x_4.permute(2, 0, 1)
        x_4_out = self.rnn(x_4_out)
        x_4_out = x_4_out.permute(1, 2, 0)

        x_3 = self.up2(x_4, x3)
        x_3_out = x_3.permute(2, 0, 1)
        x_3_out = self.rnn(x_3_out)
        x_3_out = x_3_out.permute(1, 2, 0)

        x_2 = self.up3(x_3, x2)
        x_2_out = x_2.permute(2, 0, 1)
        x_2_out = self.rnn(x_2_out)
        x_2_out = x_2_out.permute(1, 2, 0)

        x_1 = self.up4(x_2, x1)
        x_1_out = x_1.permute(2, 0, 1)
        x_1_out = self.rnn(x_1_out)
        x_1_out = x_1_out.permute(1, 2, 0)
        return [x_5_out, x_4_out, x_3_out, x_2_out, x_1_out, x]

class C2F_Classifier(nn.Module):
    '''
        Linear layers. 
    '''
    def __init__(self, n_classes, num_f_maps):
        super(C2F_Classifier, self).__init__()
        self.outcc1 = outconv(num_f_maps, n_classes)
        self.outcc2 = outconv(num_f_maps, n_classes)
        self.outcc3 = outconv(num_f_maps, n_classes)
        self.outcc4 = outconv(num_f_maps, n_classes)
        self.outcc = outconv(num_f_maps, n_classes)

    def forward(self, embedding_list):
        x_5, x_4, x_3, x_2, x_1, x = embedding_list
        y1 = self.outcc1(F.relu(x_5))
        y2 = self.outcc2(F.relu(x_4))
        y3 = self.outcc3(F.relu(x_3))
        y4 = self.outcc4(F.relu(x_2))
        y = self.outcc(x_1)
        return y, [y4, y3, y2, y1], x