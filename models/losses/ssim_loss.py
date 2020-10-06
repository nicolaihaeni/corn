import torch.nn as nn
import pytorch_ssim


# Defines the SSIM (Structural similarity)
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.pyssim_loss = pytorch_ssim.SSIM(window_size=11)

    def transform_gray_scale(self, x):
        # Transform an RGB image batch to grayscale with scaled average
        return ((0.3 * x[:, 0, :, :] + 0.59 * x[:, 1, :, :] + 0.11 * x[:, 2, :, :]) / 3.0).unsqueeze(1)

    def forward(self, input, target):
        ssim = self.pyssim_loss(self.transform_gray_scale(input),
                                self.transform_gray_scale(target))
        return (1.0 - ssim)
