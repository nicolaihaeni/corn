from .options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='train', help='phase to use (either test or train)')
        parser.add_argument('--train_data_filename', type=str, default='chair_train.h5', help='file containing the training data')
        parser.add_argument('--val_data_filename', type=str, default='chair_val.h5', help='file containing the validation data')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--num_epochs', type=int, default=10, help='the number of epochs to train the model')
        parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'), help='the number of samples to use for training of the model')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='choose gan loss mode from [ls, original, w, hinge]')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for Adam')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--lambda_VGG', type=float, default=1.0, help='weight of perceptual loss')
        parser.add_argument('--lambda_3D', type=float, default=1.0, help='weight of 3D feature loss')
        parser.add_argument('--lambda_SSIM', type=float, default=1.0, help='weight of structural similarity loss')
        parser.add_argument('--lambda_BCE', type=float, default=1.0, help='weight of occupancy mask loss')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weigth of GAN loss')
        parser.add_argument('--lambda_TGT', type=float, default=1.0, help='weigth of target prediction consistency')

        # Display and saving options
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=3, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--validation_freq', type=int, default=1, help='frequency of running the validation at the end of epochs')
        self.isTrain = True
        return parser
