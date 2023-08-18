import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50
from torchvision.models.vgg import vgg19_bn
# feature extraction
class SimCLRBase(nn.Module):
    def __init__(self, arch='resnet18'):
        super(SimCLRBase, self).__init__()
        self.f = []

        if arch == 'resnet18':
            model_name = resnet18()
        elif arch == 'resnet34':
            model_name = resnet34()
        elif arch == 'resnet50':
            model_name = resnet50()
        elif arch == 'vgg19':
            model_name = vgg19_bn()
        else:
            raise NotImplementedError

        if arch != 'vgg19':
            for name, module in model_name.named_children():  # 返回包含子模块的迭代器，同时产生模块的名称以及模块本身
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            self.f = nn.Sequential(*self.f)  # layers of model
        else:
            for name, module in model_name.named_children():
                if name == 'avgpool':
                    module = nn.AdaptiveAvgPool2d((1, 1))
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

            self.f = copy.deepcopy(model_name.features)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature

# projection
class SimCLR(nn.Module):
    def __init__(self, feature_dim=128, arch='resnet18'):
        super(SimCLR, self).__init__()
        self.f = SimCLRBase(arch)
        if arch == 'resnet18':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'resnet34':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'resnet50':
            projection_model = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'vgg19':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        else:
            raise NotImplementedError

        self.g = projection_model

    def forward(self, x):
        feature = self.f(x)    # feture extraction
        out = self.g(feature)  # projection head
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


