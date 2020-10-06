import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def sample_uniformly(num_points, voxel_len=1.0, batch_size=1):
    # Generates a point set of size (B,N,3) sampled from a zero-mean uniform distribution
    point_set = torch.zeros(batch_size, num_points, 3).data.uniform_(-voxel_len/2, voxel_len/2)
    return point_set

def sample_normally(num_points, variance=0.05, batch_size=1):
    # Generates a point set of size (B,N,3) sampled from a zero-mean normal distribution
    distrib = MultivariateNormal(torch.zeros(3), variance*torch.eye(3))
    point_set = distrib.sample([batch_size, num_points])
    return point_set
