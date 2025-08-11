import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class MLPBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.mlp(x)


class ModifiedYOLOv8Head(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nl = len(anchors)
        self.na = anchors[0].shape[0] // 2 if anchors is not None and len(anchors) > 0 and isinstance(anchors[0], torch.Tensor) and anchors[0].numel() > 0 else (len(anchors[0]) // 2 if anchors is not None and len(anchors) > 0 and isinstance(anchors[0], list) and len(anchors[0]) > 0 else 0)
        self.no = nc + 5
        self.stride = torch.tensor([8., 16., 32.])
        self.export = False
        self.nc = nc
        self.reg_max = 16  # Add a dummy reg_max attribute (or your actual value if applicable)

        self.convs = nn.ModuleList(nn.Conv2d(x, 256, 1) for x in ch[:-1])
        self.nl_convs = nn.ModuleList(nn.Conv2d(256 + ch[-1] if i > 0 else ch[-1], 256, 3, 1, 1) for i in range(self.nl))
        self.pred = nn.ModuleList(nn.Conv2d(256, self.no * self.na, 1) for _ in range(self.nl))

        first_stride = self.stride[0].item() if isinstance(self.stride, torch.Tensor) else self.stride[0]
        feature_map_size = 80 // int(first_stride)
        mlp_in_features = 256 * feature_map_size ** 2
        self.mlp_block = MLPBlock(mlp_in_features, 512, 256)

    def forward(self, x):
        out = []
        for i in range(self.nl):
            if i > 0:
                xi = self.convs[i - 1](x[-i - 1])
                x = self.nl_convs[i](torch.cat((x[-1], xi), 1))
            else:
                x = self.nl_convs[i](x[-1])

            if i == 0:
                bs, _, h, w = x.shape
                x_flat = x.flatten(start_dim=1)
                x_mlp = self.mlp_block(x_flat)
                x = x_mlp.view(bs, 256, h, w)

            out.append(self.pred[i](x))

        return out if self.training else (torch.cat(out, 1),) if self.export else (torch.cat(out, 1),)
