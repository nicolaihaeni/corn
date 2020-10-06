import torch
import torch.nn as nn

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points


class RasterizePointsXYsBlending(nn.Module):
    """
    Code inspired fromSynSin: End-to-end View Synthesis from a Single Image (CVPR 2020)
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected (BxNx3)
    - src: the corresponding features (BxNxK) where K is # of features
    - n_filters: size of feature
    - radius: radius of where pixels project to (in pixels)
    - img_size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """
    def __init__(self, n_filters=64, radius=1.5, img_size=256,
                 points_per_pixel=8, gamma=1.0):
        super().__init__()

        self.radius = radius
        self.img_size = img_size
        self.points_per_pixel = points_per_pixel
        self.gamma = gamma

    def forward(self, pts, features):
        # Make sure pts and features are equal
        assert pts.size(2) == 3
        assert pts.size(1) == features.size(1)

        pts[:, :, 0] = - pts[:, :, 0]
        pts[:, :, 1] = - pts[:, :, 1]

        radius = float(self.radius) / float(self.img_size) * 2.0
        params = compositing.CompositeParams(radius=radius)

        pointcloud = Pointclouds(points=pts, features=features)
        points_idx, _, dist = rasterize_points(pointcloud, self.img_size, radius, self.points_per_pixel)

        dist = dist / pow(radius, 2)

        alphas = (1 - dist.clamp(max=1, min=1e-3).pow(0.5).pow(self.gamma).permute(0, 3, 1, 2))

        transformed_feature_alphas = compositing.alpha_composite(points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pointcloud.features_packed().permute(1, 0),
                params
                )

        return transformed_feature_alphas


class SynSinRenderer(nn.Module):
    """
    Code inspired from SynSin: End-to-end View Synthesis from a Single Image (CVPR 2020)
    Differentiable rendering of 3D points and features
    Inputs:
    - pts3D: the 3D points to be projected (BxNx3)
    - src: the corresponding features (BxNxK) where K is # of features
    - obj_T_cam_pose: Extrinsic camera matrix to project points to (Bx4x4)
    - n_filters: size of feature
    - radius: radius of where pixels project to (in pixels)
    - img_size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - gamma: factor for alpha compositing
    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """
    def __init__(self, n_filters=64, radius=4, img_size=256, points_per_pixel=128, gamma=1.0):
        super(SynSinRenderer, self).__init__()
        self.n_filters = n_filters
        self.radius = radius
        self.img_size = img_size
        self.points_per_pixel = points_per_pixel

        self.rasterizer = RasterizePointsXYsBlending(n_filters=n_filters,
                                                     radius=radius,
                                                     img_size=img_size,
                                                     points_per_pixel=points_per_pixel,
                                                     gamma=gamma)

        K = torch.Tensor([[2, 0, 0, 0],
                          [0, 2, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]).float()
        self.register_buffer('K', K)

    def forward(self, points, features, obj_T_cam_pose):
        # Homogenous coordinates
        if points.shape[-1] == 3:
            points = torch.cat((points, torch.ones_like(points[:, :, 0:1])),-1)
        point_set_homog = points.permute(0, 2, 1)

        # Transform into a new view
        cam_points = (torch.inverse(obj_T_cam_pose).float() @ point_set_homog)
        im_coords = self.K @ cam_points

        # Normalize
        z = im_coords[:, 2:3, :]
        im_coords_homog = (im_coords / im_coords[:, -1:, :])[:, :3, :]
        im_coords_homog = torch.cat((im_coords[:, 0:2, :] / z, im_coords[:, 2:3, :]), 1)
        points = im_coords_homog.permute(0, 2, 1)

        # Rasterize points
        feature_images = self.rasterizer(points, features)
        return feature_images
