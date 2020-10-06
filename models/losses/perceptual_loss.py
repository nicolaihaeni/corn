import torch.nn as nn
from models.perceptual_model import PerceptualVGG19


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.perceptual_loss_module = PerceptualVGG19(feature_layers=[0, 5, 10, 15], use_normalization=False)

    def forward(self, input, target):
        fake_features = self.perceptual_loss_module(input)
        real_features = self.perceptual_loss_module(target)
        vgg_tgt = ((fake_features - real_features) ** 2).mean()
        return vgg_tgt
