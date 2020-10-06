import random
import torch
from .base_model import BaseModel
from .generator import Generator
from .discriminator import BaseDiscriminator
from .layers.helpers import initialize_model, data_parallel
from .losses.perceptual_loss import PerceptualLoss
from .losses.ssim_loss import SSIMLoss
from .transform.point_sampler import sample_uniformly, sample_normally


class CORNModel(BaseModel):
    def name(self):
        return 'CORNModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = opt.device
        self.points_3d = None

        # specify the training loss names
        self.loss_names = self.get_loss_names()

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = []
        for ii in range(opt.num_views):
            self.visual_names += [f'src_{ii}', f'src_{ii}_rec', f'src_{ii}_pred', f'src_{ii}_mask', f'src_{ii}_mask_rec', f'src_{ii}_mask_pred']
        self.visual_names += ['tgt', 'tgt_0_pred','tgt_1_pred',  'tgt_mask',  'tgt_0_mask_pred', 'tgt_1_mask_pred']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if opt.isTrain and opt.lambda_GAN > 0.0:
            self.model_names = ['G_NVS', 'D_NVS']
        else:
            self.model_names = ['G_NVS']

        # Define the Generator network
        self.netG_NVS = Generator(opt=opt)
        self.netG_NVS = initialize_model(self.netG_NVS, opt)
        self.netG_NVS = data_parallel(self.netG_NVS, self.opt)

        # Define the Discriminator network
        if opt.phase == 'train':
            if opt.lambda_GAN > 0.0:
                self.netD_NVS = BaseDiscriminator(opt)
                self.netD_NVS = initialize_model(self.netD_NVS, opt)
                self.netD_NVS = data_parallel(self.netD_NVS, self.opt)

            # define loss functions
            self.criterion_BCE = data_parallel(torch.nn.BCELoss(), opt)
            self.criterion_L1 = data_parallel(torch.nn.L1Loss(), opt)
            self.criterion_SSIM = data_parallel(SSIMLoss(), opt)
            if opt.lambda_VGG > 0.0:
                self.criterion_VGG = data_parallel(PerceptualLoss(), opt)

            # initialize the optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_NVS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = {}
            self.optimizers['optimizer_G'] = self.optimizer_G
            if opt.lambda_GAN > 0.0:
                self.optimizer_D = torch.optim.Adam(self.netD_NVS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers['optimizer_D'] = self.optimizer_D

            # Lambda factors
            self.lambda_L1 = opt.lambda_L1
            self.lambda_VGG = opt.lambda_VGG
            self.lambda_GAN = opt.lambda_GAN
            self.lambda_3D = opt.lambda_3D
            self.lambda_SSIM = opt.lambda_SSIM
            self.lambda_BCE = opt.lambda_BCE
            self.lambda_TGT = opt.lambda_TGT

    def set_input(self, input):
        # Transfer data to device
        self.src_0 = input['src_0'].to(self.device)
        self.src_0_mask = input['src_0_mask'].to(self.device)
        self.src_0_cam_pose = input['src_0_cam_pose'].to(self.device)
        self.src_1 = input['src_1'].to(self.device)
        self.src_1_mask = input['src_1_mask'].to(self.device)
        self.src_1_cam_pose = input['src_1_cam_pose'].to(self.device)
        self.tgt = input['tgt'].to(self.device)
        self.tgt_mask = input['tgt_mask'].to(self.device)
        self.tgt_cam_pose = input['tgt_cam_pose'].to(self.device)

    def sample_points(self, batch_size):
        if self.opt.point_sampling == 'uniform':
            return sample_uniformly(self.opt.num_points, batch_size=batch_size)
        else:
            return sample_normally(self.opt.num_points, batch_size=batch_size)

    def forward(self, points=None):
        # Either receive 3D points or sample them
        self.points_3d = self.sample_points(self.src_0.shape[0]).to(self.device) if points is None else points.to(self.device)
        output = self.netG_NVS(self.src_0, self.src_0_cam_pose, self.tgt_cam_pose, self.points_3d)
        self.tgt_0_pred = output['novel_views']
        self.tgt_0_mask_pred = output['novel_masks']
        self.src_0_feat = output['3D_features']

    def backward_G(self):
        # Decode source 0 features to get src_1 prediction
        output = self.netG_NVS(None, None, self.src_1_cam_pose, self.points_3d, self.src_0_feat,
                               'decode')
        self.src_1_rec, self.src_1_mask_rec = output['novel_views'], output['novel_masks']

        # Source 1 to target
        output = self.netG_NVS(self.src_1, self.src_1_cam_pose, self.tgt_cam_pose, self.points_3d)
        self.tgt_1_pred = output['novel_views']
        self.tgt_1_mask_pred = output['novel_masks']
        self.src_1_feat = output['3D_features']

        # Decode source 0 features to get src_1 prediction
        output = self.netG_NVS(None, None, self.src_0_cam_pose, self.points_3d, self.src_1_feat,
                               'decode')
        self.src_0_rec, self.src_0_mask_rec = output['novel_views'], output['novel_masks']

        # Transformation chain src0 -> tgt - > src1
        output = self.netG_NVS(self.tgt_0_pred, self.tgt_cam_pose, self.src_1_cam_pose, self.points_3d)
        self.src_1_pred = output['novel_views']
        self.src_1_mask_pred = output['novel_masks']
        self.tgt_0_feat = output['3D_features']

        # Transformation chain src1 -> tgt - > src0
        output = self.netG_NVS(self.tgt_1_pred, self.tgt_cam_pose, self.src_0_cam_pose, self.points_3d)
        self.src_0_pred = output['novel_views']
        self.src_0_mask_pred = output['novel_masks']
        self.tgt_1_feat = output['3D_features']

        # Compute all the losses
        zero_loss = torch.zeros(1).to(self.device)
        self.loss_G_L1 = zero_loss if self.lambda_L1 <= 0 else \
                self.criterion_L1(self.src_0_rec, self.src_0).mean() + \
                self.criterion_L1(self.src_1_rec, self.src_1).mean() + \
                self.criterion_L1(self.src_0_pred, self.src_0).mean() + \
                self.criterion_L1(self.src_1_pred, self.src_1).mean()

        # Occupancy loss
        self.loss_G_BCE = zero_loss if self.lambda_BCE <= 0 else \
                self.criterion_BCE(self.src_0_mask_rec, self.src_0_mask).mean() + \
                self.criterion_BCE(self.src_1_mask_rec, self.src_1_mask).mean() + \
                self.criterion_BCE(self.src_0_mask_pred, self.src_0_mask).mean() + \
                self.criterion_BCE(self.src_1_mask_pred, self.src_1_mask).mean()

        # SSIM Loss
        self.loss_G_SSIM = zero_loss if self.lambda_SSIM <= 0 else \
                self.criterion_SSIM(self.src_0_rec, self.src_0).mean() + \
                self.criterion_SSIM(self.src_1_rec, self.src_1).mean() + \
                self.criterion_SSIM(self.src_0_pred, self.src_0).mean() + \
                self.criterion_SSIM(self.src_1_pred, self.src_1).mean()

        # Feature loss
        self.loss_G_3D = zero_loss if self.lambda_3D <= 0 else \
                self.criterion_L1(self.tgt_0_feat, self.src_0_feat).mean() + \
                self.criterion_L1(self.tgt_0_feat, self.src_1_feat).mean() + \
                self.criterion_L1(self.tgt_1_feat, self.src_0_feat).mean() + \
                self.criterion_L1(self.tgt_1_feat, self.src_1_feat).mean() + \
                self.criterion_L1(self.src_0_feat, self.src_1_feat).mean() + \
                self.criterion_L1(self.tgt_1_feat, self.tgt_0_feat).mean()

        # GAN loss
        self.loss_G_GAN = zero_loss if self.lambda_GAN <= 0 else \
                self.netD_NVS(self.tgt_0_pred, self.src_0, mode='generator')['Sum'].mean() + \
                self.netD_NVS(self.tgt_1_pred, self.src_1, mode='generator')['Sum'].mean()

        # Perceptual Loss
        self.loss_G_VGG = zero_loss if self.lambda_VGG <= 0 else \
                self.criterion_VGG(self.src_0_rec, self.src_0).mean() + \
                self.criterion_VGG(self.src_1_rec, self.src_1).mean() + \
                self.criterion_VGG(self.src_0_pred, self.src_0).mean() + \
                self.criterion_VGG(self.src_1_pred, self.src_1).mean()

        # Target consistency loss
        self.loss_G_TGT = zero_loss if self.lambda_TGT <= 0 else \
                self.criterion_L1(self.tgt_0_pred, self.tgt_1_pred).mean()

        self.loss_G = self.lambda_L1 * self.loss_G_L1 + self.lambda_3D * self.loss_G_3D \
                      + self.lambda_SSIM * self.loss_G_SSIM + self.lambda_BCE * self.loss_G_BCE\
                      + self.lambda_VGG * self.loss_G_VGG + self.lambda_GAN * self.loss_G_GAN \
                      + self.lambda_TGT * self.loss_G_TGT

        self.loss_G.backward()

    def backward_D(self):
        self.loss_D = self.netD_NVS(self.tgt_0_pred, self.src_0, mode='discriminator')['Sum'].mean() +\
                      self.netD_NVS(self.tgt_1_pred, self.src_1, mode='discriminator')['Sum'].mean()
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        # Optimize the Generator
        if self.lambda_GAN > 0.0:
            self.set_requires_grad([self.netD_NVS], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Optimize the Discriminator
        if self.lambda_GAN > 0.0:
            self.set_requires_grad([self.netD_NVS], True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

    # Helper function to compute validation error on batch
    def validate(self, input):
        with torch.no_grad():
            self.src = input['src_0'].to(self.device)
            self.src_cam_pose = input['src_0_cam_pose'].to(self.device)
            self.tgt_0 = input['tgt'].to(self.device)
            self.tgt_0_cam_pose = input['tgt_cam_pose'].to(self.device)
            self.forward()
            return self.criterion_L1(self.tgt_0_pred, self.tgt_0).mean()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self, input):
        self.src_0 = input['src_0'].to(self.device)
        self.src_0_mask = input['src_0_mask'].to(self.device)
        self.src_0_cam_pose = input['src_0_cam_pose'].to(self.device)
        self.tgt = input['tgt'].to(self.device)
        self.tgt_mask = input['tgt_mask'].to(self.device)
        self.tgt_cam_pose = input['tgt_cam_pose'].to(self.device)
        with torch.no_grad():
            self.forward()
