from .options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='phase to use (either test or train)')
        parser.add_argument('--n_test', type=int, default=float('inf'), help='# of test images to use')
        parser.add_argument('--results_dir', type=str, default='./results', help='saves results here')
        parser.add_argument('--test_data_filename', type=str, default='chair_test.h5', help='test dataset file name')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.isTrain = False
        return parser
