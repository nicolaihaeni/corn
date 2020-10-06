import torch
import torch.nn as nn
from torchvision import models


# VGG19 network used for perceptual loss computation
class PerceptualVGG19(nn.Module):
    def __init__(self, feature_layers, use_normalization=True,
                 path=None):
        super(PerceptualVGG19, self).__init__()
        if path != '' and path is not None:
            print('Loading pretrained model')
            model = models.vgg19(pretrained=False)
            model.load_state_dict(torch.load(path))
        else:
            model = models.vgg19(pretrained=True)
        model.float()
        model.eval()

        self.model = model
        self.feature_layers = feature_layers

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.mean_tensor = None

        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.std_tensor = None

        self.use_normalization = use_normalization

        for param in self.model.features.parameters():
            param.requires_grad = False

    def normalize(self, x):
        if self.mean_tensor is None:
            self.mean_tensor = self.mean.view(1, 3, 1, 1).expand(x.size()).to(x.device)
            self.std_tensor = self.std.view(1, 3, 1, 1).expand(x.size()).to(x.device)

        x = (x + 1) / 2
        return (x - self.mean_tensor) / self.std_tensor

    def run(self, x):
        features = []

        h = x
        for f in range(max(self.feature_layers) + 1):
            h = self.model.features[f](h)
            if f in self.feature_layers:
                not_normed_features = h.clone().view(h.size(0), -1)
                features.append(not_normed_features)
        return torch.cat(features, dim=1)

    def forward(self, x):
        h = self.normalize(x)
        return self.run(h)
