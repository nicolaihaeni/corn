import os
import torch
from collections import OrderedDict
from .layers.helpers import get_scheduler


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.phase == 'train'
        self.device = opt.device
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def setup(self, opt):
        opt.start_epoch = 0
        if not self.isTrain or opt.continue_train:
            opt.start_epoch = self.load_networks(opt.load_epoch)
        if self.isTrain:
            self.schedulers = [get_scheduler(self.optimizers[key], opt) for key in self.optimizers]

        self.print_networks(opt.verbose)
        return opt.start_epoch

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net = net.eval()

    # make models eval mode during test time
    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net = net.train()

    def optimize_parameters(self):
        pass

    def get_loss_names(self):
        if self.opt.isTrain:
            loss_names = ['G']
            opts = vars(self.opt)
            for opt in opts.keys():
                if 'lambda_' in str(opt):
                    string = opt.split('_')
                    loss_names += [f'G_{string[1]}']
            if self.opt.lambda_GAN > 0.0:
                loss_names += ['D']
            return loss_names
        else:
            return None

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                img = getattr(self, name)
                visual_ret[name] = img.detach()
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name).item())
        return errors_ret

    # return visualization 3d data
    def get_current_3d_visuals(self):
        point_coords = self.points_3d
        point_features = self.tgt_feat
        return point_coords, point_features

    # save models to the disk
    def save_networks(self, epoch, name=None):
        if name:
            save_filename = '%s_net_%s.pth' % (name, self.opt.name)
        else:
            save_filename = '%s_net_%s.pth' % (epoch, self.opt.name)
        save_path = os.path.join(self.save_dir, save_filename)

        data = {'epoch': epoch}
        for key in self.optimizers:
            data[f'{key}_state_dict'] = self.optimizers[key].state_dict()

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 1 and torch.cuda.is_available:
                    data[f'net{name}_state_dict'] = net.module.cpu().state_dict()
                else:
                    data[f'net{name}_state_dict'] = net.cpu().state_dict()

        torch.save(data, save_path)

        if len(self.gpu_ids) >= 1 and torch.cuda.is_available():
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    net.cuda(self.opt.gpu_ids[0])

    # load models from the disk
    def load_networks(self, epoch):
        load_filename = '%s_net_%s.pth' % (epoch, self.opt.name)
        load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)
        checkpoint = torch.load(load_path, map_location=self.device)
        if hasattr(checkpoint, '_metadata'):
            del checkpoint._metadata

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net.module.load_state_dict(checkpoint[f'net{name}_state_dict'])
                else:
                    net.load_state_dict(checkpoint[f'net{name}_state_dict'])

        # Discriminator and optimizers are only used in training phase
        if self.opt.phase == 'train':
            for key in self.optimizers:
                optim = getattr(self, key)
                optim.load_state_dict(checkpoint[f'{key}_state_dict'])
                self.optimizers[key] = optim

        return checkpoint['epoch']

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers['optimizer_G'].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
