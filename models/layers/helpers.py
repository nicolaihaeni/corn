import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


##########################################################
# Helper Functions
##########################################################
def get_norm_layer_1D(norm_type='instance'):
    if 'batch' in norm_type:
        return nn.BatchNorm1d
    elif 'instance' in norm_type:
        return nn.InstanceNorm1d
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def get_norm_layer_2D(norm_type='instance'):
    if 'batch' in norm_type:
        return nn.BatchNorm2d
    elif 'instance' in norm_type:
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def get_nonlinear_layer(nonlinear_type='LeakyReLU'):
    if nonlinear_type == 'ReLU':
        return nn.ReLU(inplace=True)
    elif nonlinear_type == 'LeakyReLU':
        return nn.LeakyReLU(inplace=True)
    else:
        raise NotImplementedError('nonlinear layer [%s] is not found' % nonlinear_type)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.start_epoch - opt.niter) / float((opt.num_epochs - opt.niter) + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('nn.Conv') != -1 or classname.find('nn.Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def initialize_model(model, options):
    model = model.to(options.device)
    init_weights(model, options.init_type, gain=options.init_gain)
    return model


def data_parallel(model, options):
    if options.device == 'cuda':
        model = model.to(options.device)
    if len(options.gpu_ids) > 1:
        assert(torch.cuda.is_available())
        model = torch.nn.DataParallel(model)
    return model
