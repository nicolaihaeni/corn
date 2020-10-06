import torch
import torch.nn as nn
from .networks import Encoder, Decoder


class Generator(nn.Module):
    ''' Novel view synthesis generator.
    '''
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def encode(self, img, scr_cam, points):
        return self.encoder(img, scr_cam, points)

    def decode(self, feat, tgt_cam, points):
        return self.decoder(feat, tgt_cam, points)

    # Forward function that let's you choose if you want to run full model, only encoding
    # or only decoding. Necessary workaround for DataParallel
    def forward(self, img, src_T_cam_pose, tgt_T_cam_pose, points_3d,
                features=None, function='full'):

        # Only run the encoder
        if function == 'encode':
            features, z = self.encode(img, src_T_cam_pose, points_3d)
            return {'3D_features': features,
                    'z': z}

        # Only run the decoder
        if function == 'decode':
            assert(features is not None)
            nvs_imgs, nvs_masks = self.decode(features, tgt_T_cam_pose, points_3d)
            return {'novel_views': nvs_imgs,
                    'novel_masks': nvs_masks}

        # Run the full model
        else:
            features, z = self.encode(img, src_T_cam_pose, points_3d)
            nvs_imgs, nvs_masks = self.decode(features, tgt_T_cam_pose, points_3d)
            return {'novel_views': nvs_imgs,
                    'novel_masks': nvs_masks,
                    '3D_features': features,
                    'z': z}
