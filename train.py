import os
import random
import time
import json
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.shapenet_img_data_loader import ShapenetDataset
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
from models.corn_model import CORNModel


# Set random seed
random_seed = 1234
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

########################################################
# Train
########################################################
if __name__ == '__main__':
    opt = TrainOptions().parse()
    train_dataset = ShapenetDataset(opt, os.path.join(opt.data_dir, opt.train_data_filename))
    val_dataset = ShapenetDataset(opt, os.path.join(opt.data_dir, opt.val_data_filename), is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=False, drop_last=False)
    visualizer = Visualizer(opt)

    # Store the arguments in a file
    with open(os.path.join(opt.checkpoints_dir, opt.name, 'session_args.txt'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    # Tensorboard summary writer
    writer = SummaryWriter(f'runs/{opt.name}')

    # Create and initialize the model
    model = CORNModel()
    model.initialize(opt)
    opt.start_epoch = model.setup(opt)

    print('Training Network')
    print('-' * 10)

    # If we continue training, load latest model
    total_steps = 0
    dataset_size = len(train_dataset)

    for epoch in range(opt.start_epoch, opt.num_epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for ii, batch in enumerate(train_dataloader):
            iter_start_time = time.time()
            if total_steps % opt.display_freq == 0:
                t_data = iter_start_time - iter_data_time

            visualizer.reset()
            batch_size = len(batch['src_0'])
            total_steps += batch_size
            epoch_iter += batch_size

            if epoch_iter > opt.max_dataset_size:
                break

            # Optimize the model
            model.set_input(batch)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                # Visualize examples
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / batch_size
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                writer.add_scalar('generator_loss', losses['G'], total_steps)
                writer.add_scalar('discriminator_loss', losses['D'], total_steps)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch {}, total_steps {})'.format(epoch, total_steps))
                model.save_networks(epoch, 'latest')

            iter_data_time = time.time()

        # Save the model to checkpoint directory
        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch: {}'.format(epoch))
            model.save_networks(epoch, 'latest')
            model.save_networks(epoch)

        # Run evaluation on the validation set
        if epoch % opt.validation_freq == opt.validation_freq-1:
            print('Running validation at the end of epoch: {}'.format(epoch))
            validation_losses = []
            model.eval()
            for ii, val_batch in enumerate(val_dataloader):
                validation_losses.append(model.validate(val_batch).item())

            model.train()
            t = (time.time() - iter_start_time) / batch_size
            # Only print validation losses if we have a validation dataset
            if validation_losses:
                validation_loss = {'val_loss': sum(validation_losses) / len(validation_losses)}
                visualizer.print_current_losses(epoch, epoch_iter, validation_loss, t, t_data)
                writer.add_scalar('val_loss', validation_loss['val_loss'], epoch)

        print('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, opt.num_epochs, time.time() - epoch_start_time))
        model.update_learning_rate()

    print('Finished training!')
