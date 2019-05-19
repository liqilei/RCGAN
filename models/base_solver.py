import torch


class BaseSolver(object):
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models']
        self.is_train = opt['is_train']
        self.use_gpu = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        self.checkpoint_dir = opt['path']['epoch']
        self.results_dir = opt['path']['results']
        self.vis_dir = opt['path']['vis']
        if opt['is_train']:
            self.val_step = opt['train']['val_step']
            self.training_loss = 0.0
            self.val_loss = 0.0
            self.best_prec = 0.0
            self.skip_threshold = opt['train']['skip_threshold']
            self.last_epoch_loss = 1e8  # for skip threshold

        self.model_pth = opt['model_path']

    def name(self):
        return 'BaseSolver'

    def feed_data(self, batch):
        pass

    def summary(self, input_g_size, input_d_size=None):
        """print network summary"""
        pass

    def train_step(self):
        pass

    def validate(self, val_crop, crop_size):
        pass

    def test(self, use_chop):
        pass

    def _exact_crop_forward(self, upscale, crop_size):
        pass

    def _overlap_crop_forward(self, upscale):
        pass

    def save(self, epoch, is_best):
        pass

    def load(self):
        pass

    def current_loss(self):
        pass

    def current_visual(self):
        pass

    def current_learning_rate(self):
        pass

    def update_learning_rate(self, epoch):
        pass

    def tf_log(self, epoch):
        pass
