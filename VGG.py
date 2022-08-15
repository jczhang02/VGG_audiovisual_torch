"""
Author: Chengrui Zhang, jczhang@live.it
Backbone file, contains VGG16 and VGGish architectures
"""

import rich
import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    """
    Pytorch implementation of the VGG16 model.
    Adapted from Pytorch official hub.
    """

    def __init__(self):
        super(VGG16, self).__init__()

        self.features = models.vgg16(pretrained=False).features

    def forward(self, x):
        """
        x: (1, 512, 31, 4) -> (1, 124, 512)

        :param x Tensor: Input variable
        """
        x = self.features(x).permute(0, 2, 3, 1)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
        return x


class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.
    Adapted from: https://github.com/harritaylor/torch-vggish.
    """

    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        # remove full-connect
        # self.fc = nn.Sequential(
        #     nn.Linear(512 * 24, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 128),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        """
        x: (1, 512, 8, 8) -> (1, 64, 512)

        :param x Tensor: Input variable
        """
        x = self.features(x).permute(0, 2, 3, 1)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
        # .permute(0, 2, 3, 1).contiguous()
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


def load_partial_parameters(model, pth_path):
    """
    Load partial parameters from pretrained weight file.
    :param model torch model: implemented model
    :param pth_path str: pretrained weight file path
    """
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pth_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    return model_dict


def __vgg16_getitem__(pretrained=False, pth_path=None):
    """
    Return vgg16 model

    :param pretrained bool: Load pretrained model or not
    :param pth_path str: pretrained weight file path
    """
    model = VGG16()
    if not pretrained:
        model.eval()
        # model.to(device)
        return model
    else:
        model.load_state_dict(load_partial_parameters(model=model, pth_path=pth_path))
        model.eval()
        # model.to(device)
        return model


def __vggish_getitem__(pretrained=False, pth_path=None):
    """
    Return vggish model

    :param pretrained bool: Load pretrained model or not
    :param pth_path str: pretrained weight file path
    """
    model = VGGish()
    if not pretrained:
        model.eval()
        # model.to(device)
        return model
    else:
        model.load_state_dict(load_partial_parameters(model=model, pth_path=pth_path))
        model.eval()
        # model.to(device)
        return model
