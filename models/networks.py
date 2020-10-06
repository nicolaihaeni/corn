import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

from .layers.blocks import Conv1d, Conv2d, UpConv2d, ResnetBlock
from .transform.projection import ProjectionHelper
from .transform.point_embedder import get_point_embedder
from .renderer.get_renderer import get_renderer


class ImageFeatureExtractor(nn.Module):
    ''' Resnet 18 Local/Global Feature Extractor
        Takes as input an image of size BxCxHxW and returns local feature map
        z_local of size BxCxHxW and global feature z_global of size Bxn_global_features
    '''
    def __init__(self, opt):
        super(ImageFeatureExtractor, self).__init__()

        # Pretrained Resnet-18 for feature extraction
        n_global_features = 128
        norm = opt.norm
        n_filters = opt.n_filters
        model = models.resnet18(pretrained=True)
        model = model.float()

        self.prep_layer = nn.Sequential(*list(model.children())[:3])
        self.layer_1 = nn.Sequential(*list(model.children())[3:5])
        self.layer_2 = nn.Sequential(*list(model.children())[5])
        self.layer_3 = nn.Sequential(*list(model.children())[6])
        self.layer_4 = nn.Sequential(*list(model.children())[7])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_global_features)

        # Upsampling layers
        self.upconv_4 = UpConv2d(8*n_filters, 4*n_filters, norm_type=norm)
        self.iconv_4 = Conv2d(8*n_filters, 4*n_filters, padding=1, norm_type=norm)
        self.upconv_3 = UpConv2d(4*n_filters, 2*n_filters, norm_type=norm)
        self.iconv_3 = Conv2d(4*n_filters, 2*n_filters, padding=1, norm_type=norm)
        self.upconv_2 = UpConv2d(2*n_filters, n_filters, norm_type=norm)
        self.iconv_2 = Conv2d(2*n_filters, n_filters, padding=1, norm_type=norm)
        self.upconv_1 = UpConv2d(n_filters, n_filters, norm_type=norm)
        self.iconv_1 = Conv2d(2*n_filters, n_filters, padding=1, norm_type=norm)

    def forward(self, x):
        skip_1 = self.prep_layer(x)
        skip_2 = self.layer_1(skip_1)
        skip_3 = self.layer_2(skip_2)
        skip_4 = self.layer_3(skip_3)
        base_2d = self.layer_4(skip_4)

        # Global feature descriptor
        x = self.avgpool(base_2d)
        x = torch.flatten(x, 1)
        z_global = self.fc(x)

        # Upsampling with skip connections to get local feature map
        out = self.upconv_4(base_2d)
        out = self.iconv_4(torch.cat((out, skip_4), 1))
        out = self.upconv_3(out)
        out = self.iconv_3(torch.cat((out, skip_3), 1))
        out = self.upconv_2(out)
        out = self.iconv_2(torch.cat((out, skip_2), 1))
        out = self.upconv_1(out)
        z_local = self.iconv_1(torch.cat((out, skip_1), 1))
        return z_global, z_local


class ContinuousFunctionNet(nn.Module):
    ''' Multi-layer perceptron for continuous function representation
        Takes as input a number of features BxKxX, where K is the feature
        dimension and X is the number of sample points.
    '''
    def __init__(self, opt):
        in_channels = 255
        self.opt = opt
        super(ContinuousFunctionNet, self).__init__()
        self.conv_1 = Conv1d(in_channels, 512, kernel_size=1, norm_type=opt.norm)
        self.conv_2 = Conv1d(512, 256, kernel_size=1, norm_type=opt.norm)
        self.conv_3 = Conv1d(256, 128, kernel_size=1, norm_type=opt.norm)
        self.conv_4 = Conv1d(128, opt.n_filters, kernel_size=1, norm_type=opt.norm)

        # Occupancy prediction
        self.conv_5 = Conv1d(opt.n_filters, 1, kernel_size=1, norm_type=opt.norm)
        self.act_fn = nn.Sigmoid()

    def forward(self, features):
        out = self.conv_1(features)
        out = self.conv_2(out)
        out = self.conv_3(out)
        feature = self.conv_4(out)

        occ = self.act_fn(self.conv_5(feature))
        return feature, occ


class UNet(nn.Module):
    ''' UNet to render pixels from projected features.
        Takes as input BxKxHxW feature maps and output images at novel
        views of size Bx3xHxW
    '''
    def __init__(self, in_channels, out_channels, n_filters, opt, use_tanh=True):
        super(UNet, self).__init__()

        # Conv layers for downsampling
        self.norm = opt.norm
        self.out_channels = out_channels
        self.conv_1 = ResnetBlock(in_channels, n_filters, kernel_size=3, padding=1, stride=2,
                                  norm_type=self.norm)
        self.conv_2 = ResnetBlock(n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2,
                                  norm_type=self.norm)
        self.conv_3 = ResnetBlock(2 * n_filters, 4 * n_filters, kernel_size=3, padding=1,
                                  stride=2, norm_type=self.norm)
        self.conv_4 = ResnetBlock(4 * n_filters, 8 * n_filters, kernel_size=3, padding=1,
                                  stride=2, norm_type=self.norm)

        # Upconv layers
        self.dconv_1 = UpConv2d(8 * n_filters, 4 * n_filters, kernel_size=3, scale=2, padding=1,
                                norm_type=self.norm)
        self.iconv_1 = Conv2d(8 * n_filters, 4 * n_filters, kernel_size=3, padding=1,
                              norm_type=self.norm)
        self.dconv_2 = UpConv2d(4 * n_filters, 2 * n_filters, kernel_size=3, scale=2, padding=1,
                                norm_type=self.norm)
        self.iconv_2 = Conv2d(4 * n_filters, 2 * n_filters, kernel_size=3, padding=1,
                              norm_type=self.norm)
        self.dconv_3 = UpConv2d(2 * n_filters, n_filters, kernel_size=3, scale=2, padding=1,
                                norm_type=self.norm)
        self.iconv_3 = Conv2d(2 * n_filters, n_filters, kernel_size=3, padding=1,
                              norm_type=self.norm)
        self.dconv_4 = UpConv2d(n_filters, n_filters, kernel_size=3, scale=2, padding=1,
                                norm_type=self.norm)
        self.iconv_4 = Conv2d(n_filters, n_filters, kernel_size=3, padding=1,
                              norm_type=self.norm)

        # Output layers
        output = [nn.Conv2d(n_filters, out_channels, kernel_size=3, stride=1, padding=1)]

        # Sigmoid or tanh output
        self.img_out = nn.Tanh()
        self.mask_out = nn.Sigmoid()
        self.output = nn.Sequential(*output)

    def forward(self, x):
        # Downsampling resblocks
        skip_1 = self.conv_1(x)
        skip_2 = self.conv_2(skip_1)
        skip_3 = self.conv_3(skip_2)
        out = self.conv_4(skip_3)

        # Upsampling and convs
        out = self.dconv_1(out)
        out = self.iconv_1(torch.cat((out, skip_3), 1))

        out = self.dconv_2(out)
        out = self.iconv_2(torch.cat((out, skip_2), 1))

        out = self.dconv_3(out)
        out = self.iconv_3(torch.cat((out, skip_1), 1))

        out = self.dconv_4(out)
        out = self.iconv_4(out)

        out = self.output(out)

        # Output only occupancy mask
        if self.out_channels == 1:
            return self.mask_out(out)
        # Output only the image
        elif self.out_channels == 3:
            return self.img_out(out)
        # Output occupancy mask and image
        else:
            return torch.cat([self.img_out(out[:, :3, :, :]),
                              self.mask_out(out[:, 3:, :, :])], 1)


class Encoder(nn.Module):
    ''' Encoder model that encodes an image and given pose into a 3D scene
        Input:
        x: images of size Bx3xHxW
        src_T_cam_pose: camera poses Bx4x4
        points_3d: K 3D points of shape BxKx3
        Output:
        features_3D: 3D scene representation BxNxK with N being number of
        filter channels and K number of 3D points.
        z_global: global feature descriptor of size Bx128
    '''
    def __init__(self, opt):
        super(Encoder, self).__init__()
        n_filters = opt.n_filters
        self.opt = opt

        # Feature Extranction network
        self.img_feature_extractor = ImageFeatureExtractor(opt)

        # Initialize projection helper
        self.proj = ProjectionHelper(image_height=opt.final_img_size//2)

        # Periodic point embedding function
        self.embed_fn, _ = get_point_embedder()

        # Continous function network
        self.cont_rep = ContinuousFunctionNet(opt)

    def forward(self, imgs, src_T_cam_pose, points_3d):
        batch_size = imgs.shape[0]

        # Embedd the points in higher dim space
        embedded_points = self.embed_fn(points_3d)

        # Extract global and local features
        z_global, z_local = self.img_feature_extractor(imgs)
        point_features = self.proj.pointset_to_local_features(points_3d,
                                                              z_local,
                                                              src_T_cam_pose)
        features = torch.cat((point_features,
                              z_global.unsqueeze(-1).repeat(1, 1, point_features.shape[-1]),
                              embedded_points.permute(0, 2, 1)), 1)

        # Continuous Function representation
        features_3d, occupancy = self.cont_rep(features)

        # Concat occupancy and features for projection
        features_3d = torch.cat([features_3d, occupancy], 1)
        return features_3d, z_global


class Decoder(nn.Module):
    ''' Decoder model that decodes a scene representation and given pose into novel views
        Input:
        features_3D: 3D scene representation BxNxK with N being number of
        tgt_T_cam_pose: camera poses Bx4x4
        Output:
        novel_views: novel_views of size Bx3xHxW
    '''
    def __init__(self, opt):
        super(Decoder, self).__init__()

        # neural renderer
        self.opt = opt
        opt.n_filters = opt.n_filters + 1
        self.nr = get_renderer(renderer=opt.renderer_type, options=opt)
        opt.n_filters = opt.n_filters - 1

        # Rendering UNet
        self.render_net = UNet(in_channels=opt.n_filters + 1, out_channels=4,
                               n_filters=opt.n_filters, opt=opt)

        if opt.occupancy_renderer:
            self.render_net = UNet(in_channels=opt.n_filters, out_channels=3,
                               n_filters=opt.n_filters, opt=opt)

            self.render_occupancy_net = UNet(in_channels=1, out_channels=1,
                               n_filters=opt.n_filters, opt=opt)

        self.up_sample = False
        if opt.final_img_size != opt.img_size:
            self.up_sample = True
            self.up = nn.functional.interpolate


    def forward(self, features_3d, tgt_T_cam_pose, points_3d):
        # render features to novel view
        rendered_features = self.nr(points_3d, features_3d.permute(0, 2, 1), tgt_T_cam_pose)

        # use unet to render novel views
        if not self.opt.occupancy_renderer:
            novel_views = self.render_net(rendered_features)
            novel_img = novel_views[:, :3, :, :]
            novel_mask = novel_views[:, 3:, :, :]
        else:
            novel_img = self.render_net(rendered_features[:,:-1])
            novel_mask = self.render_occupancy_net(rendered_features[:,-1:])

        if self.up_sample:
            novel_img = self.up(novel_img, (self.opt.final_img_size, self.opt.final_img_size), mode='bilinear')
            novel_mask = self.up(novel_mask, (self.opt.final_img_size, self.opt.final_img_size), mode='nearest')
        return novel_img, novel_mask
