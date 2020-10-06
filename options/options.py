import torch
import argparse


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--name', type=str, default='novelpose', help='name of the model')
        parser.add_argument('--data_dir', type=str, default='./datasets/shapenet/', help='data directory')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_workers', type=int, default=4, help='number of threads for data loading')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--n_filters', type=int, default=64, help='number of conv filters in first layer')
        parser.add_argument('--num_points', type=int, default=10000, help='number of points to sample')
        parser.add_argument('--point_sampling', type=str, default='uniform', help='way to sample points, either uniform or gaussian')
        parser.add_argument('--img_size', type=int, default=128, help='size of image height/width')
        parser.add_argument('--final_img_size', type=int, default=128, help='size of final output image height/width')
        parser.add_argument('--device', type=str, default='cpu', help='device on which to run model')
        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: eg. 0   0,1,2 use -1 for CPU')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--use_spectral', type=bool, default=True, help='use spectral normalization')
        parser.add_argument('--num_views', type=int, default=2, help='number of source views for novel view prediction')
        parser.add_argument('--load_epoch', type=str, default='latest', help='epoch to load for testing. Can either be an integer or latest')
        parser.add_argument('--init_type', type=str, default='normal', help='network weight initialization type')
        parser.add_argument('--init_gain', type=float, default=0.02, help='network weight initialization gain')
        parser.add_argument('--renderer_type', type=str, default='synsin', help='renderer type: only synsin at the moment')
        parser.add_argument('--occupancy_renderer', type=bool, default=True, help='create a separate silhouette renderer if true')
        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            opt, _ = parser.parse_known_args()
            self.parser = parser
            parser.parse_args()
            self.print_options(opt)

        opt.isTrain = self.isTrain

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        self.opt = opt
        return self.opt
