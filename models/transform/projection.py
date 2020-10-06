import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as la


class ProjectionHelper(nn.Module):
    def __init__(self, image_height=256):
        super(ProjectionHelper, self).__init__()
        self.image_height = image_height

        # Intrinsic camera parameters
        intrinsics = torch.Tensor([[1., 0., 0.5],
                                   [0., 1., 0.5],
                                   [0., 0., 1.]]).float()

        intrinsics[:2, :] *= image_height
        self.register_buffer("intrinsics", intrinsics)

    def interpolate_projections(self, im_coords, feature_image):
        ''' Takes the image coordinates (B, 2, N) and feature image (B, K, H, W) as input
            returns the bilinear interpolation of the nearest 4 pixel feature for each N projection
        '''
        batch_size = feature_image.shape[0]

        # TODO: check the boundary conditions
        coord_floors = im_coords.floor().long()
        coord_ceils = coord_floors + 1

        ind = torch.arange(batch_size).to(im_coords.device)

        # features at each nearest corner
        f_x0y0 = feature_image[:,:,coord_floors[:,1], coord_floors[:,0]][ind, :, ind]
        f_x0y1 = feature_image[:,:,coord_floors[:,1], coord_ceils[:,0]][ind, :, ind]
        f_x1y0 = feature_image[:,:,coord_ceils[:,1], coord_floors[:,0]][ind, :, ind]
        f_x1y1 = feature_image[:,:,coord_ceils[:,1], coord_ceils[:,0]][ind, :, ind]

        # floor and ceil weights
        to_floor_wgh = im_coords - coord_floors
        to_ceil_wgh = coord_ceils - im_coords

        f_xy = f_x0y0 * to_ceil_wgh[:,[1]] * to_ceil_wgh[:,[0]] + \
               f_x0y1 * to_ceil_wgh[:,[1]] * to_floor_wgh[:,[0]] + \
               f_x1y0 * to_floor_wgh[:,[1]] * to_ceil_wgh[:,[0]] + \
               f_x1y1 * to_floor_wgh[:,[1]] * to_floor_wgh[:,[0]]

        return f_xy


    def pointset_to_local_features(self, point_set, feature_image, extrinsics, outlier_robustness='zeros'):
        ''' Takes as input a point set (tensor of dim BxNx4 or BxNx3), a feature image (tensor of dim BxKxWxH)
            and extrinsics as obj_T_cam (Bx4x4), and outlier_robustness option {'zeros', 'nn'}: fill out of canvas points with zeros or nn feature
            If the outlier_robustness flag is True then the points whose projections are outside the image canvas are eliminated
            Returns a per point local feature fetched from the feature image according to the point's projection (BxNxK)
            Check experimental/projection_helper.py to see an example usage
        '''

        # make the point set have homog coordinates
        # (B, N, 3) -> (B, 4, N)
        if point_set.shape[-1] == 3:
            point_set = torch.cat((point_set, torch.ones_like(point_set[:, :, 0:1])),-1)
        point_set_homog = point_set.permute(0,2,1)

        # point set in camera frame (B, 3, N)
        # extrinsics is a batch of cam_T_obj
        cam_points = (torch.inverse(extrinsics).float() @ point_set_homog)[:,:3]

        # point projections to image plane
        im_coords = self.intrinsics @ cam_points

        # normalized coordinates of image pixel coordinates. shape: (B, 2, N)
        im_coords_homog = (im_coords / im_coords[:,-1:,:])[:,:2,:]

        valid_ind_mask = (torch.ge(im_coords_homog[:, 0], 0) *
                          torch.ge(im_coords_homog[:, 1], 0) *
                          torch.lt(im_coords_homog[:, 0], self.image_height) *
                          torch.lt(im_coords_homog[:, 1], self.image_height))

        # if the point projection lies outside the image canvas then use the features of its nearest neighbor or zeros instead.
        # assumes that image width = height
        if outlier_robustness == 'zeros' or outlier_robustness == 'nn':
            im_coords_homog = im_coords_homog.clamp(1, self.image_height-2)

        per_point_features = self.interpolate_projections(im_coords_homog, feature_image)

        if outlier_robustness == 'zeros':
            per_point_features[~valid_ind_mask.unsqueeze(1).repeat(1, feature_image.size(1) ,1)] = 0

        return per_point_features


    def evaluate_projection_occupancy(self, point_coords, source_mask, src_T_cam_pose):
        """ Computes a binary tensor indicating whether a point at the corresponding index has a projection inside
        the mask or not (for each batch)
        point_coords (B,N,3), source_mask (B,1,H,W), src_T_cam_pose (B,4,4)
        """
        source_mask = source_mask.squeeze(1)
        batch_size = src_T_cam_pose.shape[0]

        # homogeneous coordinates for the point positions (B,N,4)
        if point_coords.shape[-1] == 3:
            point_coords = torch.cat((point_coords, torch.ones_like(point_coords[:, :, 0:1])), -1)

        # K * [cam_R_obj | cam_t_obj] (B,3,4)
        cam_T_obj_poses = torch.inverse(src_T_cam_pose)
        proj_mat = self.intrinsics @ cam_T_obj_poses[:,:3]

        # projection to image plane (B,3,N) and normalization (B,2,N)
        point_projections = proj_mat @ point_coords.permute(0,2,1)
        point_projections = (point_projections / point_projections[:,-1:])[:,:2]
        point_projections = point_projections.round().long()

        # binary matrix (B,N) indicating if point projection is within the image canvas
        within_canvas = (torch.ge(point_projections[:, 0], 0) *
                        torch.ge(point_projections[:, 1], 0) *
                        torch.lt(point_projections[:, 0], self.image_height) *
                        torch.lt(point_projections[:, 1], self.image_height))

        # clip values to avoid out of bounds error -- we already know their indices!
        point_projections = point_projections.clamp(min=0, max=self.image_height-1)

        # check if point projection lies within the mask of the object in the image
        t = torch.arange(batch_size)
        within_mask = source_mask[:, point_projections[:,1], point_projections[:,0]][t, t]
        within_mask = within_mask * within_canvas

        return within_mask.float()
