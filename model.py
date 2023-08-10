import torch
nn = torch.nn
F = torch.nn.functional
import thop


def conv_bn_relu_maxpool(in_ch, out_ch, kernel_size, stride, padding):
    squ = nn.Sequential()
    squ.append(nn.Conv2d(in_ch, out_ch,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding))
    squ.append(nn.BatchNorm2d(out_ch))
    squ.append(nn.LeakyReLU(0.1))
    squ.append(nn.MaxPool2d((2, 2), (2, 2)))

    return squ

def conv3x3(in_ch, out_ch, stride, padding):
    squ = nn.Sequential()
    squ.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding))
    squ.append(nn.BatchNorm2d(out_ch))
    squ.append(nn.LeakyReLU(0.1)) 

    return squ

def conv1x1_conv3x3(in_ch1x1, out_ch1x1, in_ch3x3, out_ch3x3):
    squ = nn.Sequential()
    squ.append(nn.Conv2d(in_ch1x1, out_ch1x1, kernel_size=1, stride=1, padding='same'))
    squ.append(nn.BatchNorm2d(out_ch1x1))
    squ.append(nn.LeakyReLU(0.1))
    squ.append(conv3x3(in_ch3x3, out_ch3x3, stride=1, padding='same'))

    return squ

class YoloModel(nn.Module):
    def __init__(self, S, B, C):
        super(YoloModel, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.conv_block1 = conv_bn_relu_maxpool(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv_block2 = conv_bn_relu_maxpool(64, 192, kernel_size=(3, 3), stride=(1, 1), padding='same')
    
        self.conv_block3 = nn.Sequential()
        self.conv_block3.append(conv1x1_conv3x3(192, 128, 128, 256))
        self.conv_block3.append(conv1x1_conv3x3(256, 256, 256, 512))
        self.conv_block3.append(nn.MaxPool2d((2, 2), (2, 2)))

        self.conv_block4 = nn.Sequential()
        for i in range(4):
            self.conv_block4.append(conv1x1_conv3x3(512, 256, 256, 512))
        self.conv_block4.append(conv1x1_conv3x3(512, 512, 512, 1024))
        self.conv_block4.append(nn.MaxPool2d((2, 2), (2, 2)))

        self.conv_block5 = nn.Sequential()
        for i in range(2):
            self.conv_block5.append(conv1x1_conv3x3(1024, 512, 512, 1024))
        self.conv_block5.append(conv3x3(1024, 1024, stride=1, padding='same'))
        self.conv_block5.append(conv3x3(1024, 1024, stride=2, padding=(1, 1)))

        self.conv_block6 = nn.Sequential()
        for i in range(2):
            self.conv_block6.append(conv3x3(1024, 1024, stride=1, padding='same'))

        self.fc1 = nn.Linear(S*S*1024, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, S*S*(B*5 + C))

    def forward(self, x):
        for i in range(1, 7):
            x = getattr(self, 'conv_block'+str(i))(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.contiguous().view(-1, self.S, self.S, (self.B*5 + self.C))

        return x

if __name__ == "__main__":
    noo_pred = torch.randn(2,30)
    noo_pred_mask = torch.zeros(noo_pred.size()).type(torch.bool).to(noo_pred.device)
    noo_pred_mask[:, 4] = 1
    noo_pred_mask[:, 9] = 1
    noo_pred_c = noo_pred[noo_pred_mask]
    x = torch.normal(0, 1, size=(1, 3, 448, 448)).type(torch.float32)

    model = YoloModel(7, 2, 20)
    ops, params = thop.profile(model, inputs=(x,), verbose=False)
    macs, params = thop.clever_format([ops, params], '%.3f')
    print(model(x).shape)
    print("macs: {} params: {}".format(macs, params))
