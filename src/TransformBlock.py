import torch
import torch.nn as nn

# Lightweight pointwise convolution (1x1)
class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x):
        return self.block(x)

# Lightweight depthwise separable convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=6, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))

# Channel-wise Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# YOLOv8 head with transformation layers
class YOLOv8HeadWithTransform(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nl = len(anchors)
        self.na = anchors[0].shape[0] // 2 if anchors is not None and isinstance(anchors[0], torch.Tensor) and anchors[0].numel() > 0 else (len(anchors[0]) // 2 if anchors is not None and isinstance(anchors[0], list) and len(anchors[0]) > 0 else 0)
        self.no = nc + 5
        self.stride = torch.tensor([8., 16., 32.])
        self.export = False
        self.nc = nc
        self.reg_max = 16  
        self._init_weights()


        self.convs = nn.ModuleList(nn.Conv2d(x, 256, 1) for x in ch[:-1])
        self.nl_convs = nn.ModuleList(nn.Conv2d(256 + ch[-1] if i > 0 else ch[-1], 256, 3, 1, 1) for i in range(self.nl))

        self.transform = nn.Sequential(
            PointwiseConv(256, 256, hidden_channels=128, dropout=0.2),
            DepthwiseSeparableConv(256, 256),
            SEBlock(256)
        )

        self.pred = nn.ModuleList(nn.Conv2d(256, self.no * self.na, 1) for _ in range(self.nl))
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = []
        for i in range(self.nl):
            if i > 0:
                xi = self.convs[i - 1](x[-i - 1])
                x = self.nl_convs[i](torch.cat((x[-1], xi), 1))
            else:
                x = self.nl_convs[i](x[-1])

            x = self.transform(x)

            out.append(self.pred[i](x))

        return out if self.training else (torch.cat(out, 1),) if self.export else (torch.cat(out, 1),)
