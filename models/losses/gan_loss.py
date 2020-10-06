import torch
import torch.nn as nn
import torch.nn.functional as F

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.zero_tensor = None
        self.real_label_tensor = None
        self.fake_label_tensor = None

        if gan_mode not in ['ls', 'original', 'w', 'hinge']:
            raise ValueError(f'Unexpected gan_mode {gan_mode}')

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = torch.Tensor(1).fill_(self.real_label).to(input.device)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = torch.Tensor(1).fill_(self.fake_label).to(input.device)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        self.zero_tensor = self.zero_tensor.to(input.device)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original': # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert(target_is_real) # The generators hinge loss must aim to be real
                loss = -torch.mean(input)
        else:
            # WGAN
            if target_is_real:
                loss = -input.mean()
            else:
                loss = input.mean()
        return loss

    def __call__(self, input, target_is_real, for_discriminator=True):
        return self.loss(input, target_is_real, for_discriminator)
