import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.blocks import Conv2d
from .losses.gan_loss import GANLoss


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, opt, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        norm_type = opt.norm
        sequence = [Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw,
                           norm_type=None),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                                padding=padw, norm_type=norm_type),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                            padding=padw, norm_type=norm_type),
        ]

        sequence += [Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw,
                            norm_type=None, nonlinear_type=None)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class BaseDiscriminator(nn.Module):
    '''
    Abstracting away computation of generator / discriminator losses
    '''
    def __init__(self, opt):
        super(BaseDiscriminator, self).__init__()

        self.opt = opt
        self.criterion_GAN = GANLoss(opt.gan_mode)
        self.netD = NLayerDiscriminator(opt, input_nc=3, ndf=opt.n_filters)

    # Given fake and real images, return the prediction of discriminator for
    # each image
    def discriminate(self, fake_image, real_image):
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        fake = pred[: pred.size(0) // 2]
        real = pred[pred.size(0) // 2 :]
        return fake, real

    def compute_discriminator_loss(self, fake_image, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = fake_image.detach()
            # fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(fake_image, real_image)

        D_losses['D_Fake'] = self.criterion_GAN(pred_fake, False, for_discriminator=True)
        D_losses['D_Real'] = self.criterion_GAN(pred_real, True, for_discriminator=True)
        D_losses['Sum'] = sum(D_losses.values()).mean()
        return D_losses

    def compute_generator_loss(self, fake_image, real_image):
        G_losses = {}
        pred_fake, pred_real = self.discriminate(fake_image, real_image)
        G_losses['GAN'] = self.criterion_GAN(pred_fake, True, for_discriminator=False)
        G_losses['Sum'] = sum(G_losses.values()).mean()
        return G_losses

    def forward(self, fake_image, real_image, mode='generator'):
        if mode == 'generator':
            g_loss = self.compute_generator_loss(fake_image, real_image)
            return g_loss
        else:
            d_loss = self.compute_discriminator_loss(fake_image, real_image)
            return d_loss
