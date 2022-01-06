from os import name
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch

from torch.utils.tensorboard import SummaryWriter
import domainext.data.loader as Loader
from domainext.data.transforms.transforms import build_transform
from domainext.utils.common import ( tolist_if_not, count_num_param,load_checkpoint, save_checkpoint, resume_from_checkpoint, load_pretrained_weights)
from domainext.optim import build_optimizer,build_lr_scheduler
from domainext.utils.common import build_evaluator
from domainext.utils.common.build import build_dataset

from domainext.data.datawrapper.classification import DomainWrapper
from domainext.models.classification.network.basenet import BaseMlpNet


class TrainerBase:
    "Base class from iterative trainer."

    def __init__(self) -> None:
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )
        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )
        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        assert name not in self._models, 'Found duplicate model names'

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())

        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()
        for name in names:
            model_dict = self._models(name).state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    'state_dict': model_dict,
                    'epoch': epoch + 1,
                    'optimizer': optim_dict,
                    'scheduler': sched_dict
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name
            )

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict)

    def resume_model_if_exit(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print('No checkpoint found, train from scratch')
            return 0

        print(
            'Found checkpoint in "{}". Will resume training'.format(directory)
        )

        start_epoch = resume_from_checkpoint(
            path, self._models[name], self._optims[name],
            self._scheds[name]
        )

        return start_epoch

    def set_model_mode(self, mode='train', names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def init_writer(self, log_dir):
        if self.__dict__.get('_writer') is None or self._writer is None:
            print(
                'Initializing summary writer for tensorboard '
                'with log_dir={}'.format(log_dir)
            )
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)
        

class SimpleClassTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self,dataset=None):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize dataset
        2) initialize transform: tfm_train, tfm_test
        3) initialize datawrapper
        4) assign as attributes the data loaders
        5) assign as attribute the number of classes and the lab2cname
        """
        if dataset is None:
            self.dataset = build_dataset(self.cfg,show_data=True)                                   # 1)
        tfm_train = build_transform(self.cfg, is_train=True)                                            
        tfm_test = build_transform(self.cfg, is_train=False) 
        self.build_data_wrappers(self.dataset,tfm_train,tfm_test)

        self.train_loader_x = Loader.build_train_loader_x(self.cfg,self.datawrappers['wrapper_x'])
        self.train_loader_u = Loader.build_train_loader_u(self.cfg,self.datawrappers['wrapper_u'])
        self.val_loader = Loader.build_val_loader(self.cfg,self.datawrappers['wrapper_val'])
        self.test_loader = Loader.build_val_loader(self.cfg,self.datawrappers['wrapper_test'])

        self.num_classes = self.dataset.num_classes
        self.lab2cname = self.dataset.lab2cname
        

    def build_data_wrappers(self, dataset,tfm_train,tfm_test):
        wrapper_x = DomainWrapper(self.cfg, dataset.train_x, tfm_train) if dataset.train_x else None
        wrapper_u = DomainWrapper(self.cfg, dataset.train_u, tfm_train) if dataset.train_u else None
        wrapper_val = DomainWrapper(self.cfg, dataset.val, tfm_test, False) if dataset.val else None
        wrapper_test = DomainWrapper(self.cfg, dataset.test, tfm_test, False) if dataset.test else None

        self.datawrappers = {
            'wrapper_x': wrapper_x,
            'wrapper_u': wrapper_u,
            'wrapper_test':wrapper_test,
            'wrapper_val':wrapper_val
        }

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg
        print('Buildimg model')
        self.model = BaseMlpNet(cfg,cfg.MODEL,self.num_classes,self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)
    
    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        self.init_writer(self.output_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()
    
    def after_train(self):
        print('Finished training')

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == 'best_val':
                print('Deploy the model with the best val performance')
                self.load_model(self.output_dir)
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

        # Close writer
        self.close_writer()
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            self.epoch + 1
        ) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False

        if do_test:
            if self.cfg.TEST.FINAL_MODEL == 'best_val':
                curr_result = self.test(split='val')
                is_best = curr_result > self.best_result
                if is_best:
                    self.best_result = curr_result
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        model_name='model-best.pth.tar'
                    )

            elif self.cfg.TEST.FINAL_MODEL == 'last_step':
                self.test()

            else:
                raise NotImplementedError

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == 'val' and self.val_loader is not None:
            data_loader = self.val_loader
            print('Do evaluation on {} set'.format(split))
        else:
            data_loader = self.test_loader
            print('Do evaluation on test set')

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results['accuracy']

    def model_inference(self, input,return_feature=False,freeze=False):
        return self.model(input,return_feature=return_feature,freeze=freeze)

    def parse_batch_test(self, batch):
        input = batch['img']
        label = batch['label']

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]['lr']

class SimpleSegTrainer(TrainerBase):
    pass