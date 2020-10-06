import os
import h5py
import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data.dataloader import DataLoader
from data.shapenet_img_data_loader import ShapenetDataset
from models.corn_model import CORNModel
from options.test_options import TestOptions
from utils.visualizer import save_images
from utils import html
from models.transform.pose_utils import pose_from_filename


# Set random seed
random_seed = 1234
np.random.seed(random_seed)
torch.manual_seed(random_seed)


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_workers = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1

        # ShapeNet dataset
    test_dataset = ShapenetDataset(opt, os.path.join(opt.data_dir, opt.test_data_filename),
                                   is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                 num_workers=opt.num_workers, drop_last=False)

    model = CORNModel()
    model.initialize(opt)
    epoch = model.setup(opt)
    model.visual_names = ['src_0', 'tgt', 'tgt_0_pred']

    if opt.phase == 'test':
        model.eval()

    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{epoch}')
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {epoch}')

    h5_file = h5py.File(os.path.join(opt.data_dir, opt.test_data_filename), 'r')
    for ii, batch in enumerate(test_dataloader):
        if ii >= opt.n_test:
            break

        model.test(batch)

        # Get predicted images and save
        temp = model.get_current_visuals()
        visuals = {'src': temp['src_0'], 'tgt': temp['tgt'], 'pred': temp['tgt_0_pred']}
        modelname = batch['model_name'][0]
        image_name = [f'{str(ii).zfill(5)}.png']
        save_images(webpage, visuals, image_name)
    webpage.save()
