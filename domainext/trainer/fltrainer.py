import numpy as np
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import time
import datetime

from domainext.utils.common import ( MetricMeter, AverageMeter)
from domainext.optim.lr_scheduler import build_lr_scheduler
from domainext.optim.optimizer import build_optimizer
from .xtrainer import ClassTrainerX,SegTrainerX
import domainext.data.loader as Loader
from .basetrainer import TrainerBase
from domainext.utils.common.data import ForeverDataIterator
from domainext.utils.federated_learning import fedavg

class ClassLocalTrainer(TrainerBase):
    def __init__(self, cfg, wrapper_x):
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.FL.LOCAL.OPTIM.MAX_EPOCH
        self.output_dir = osp.join(cfg.OUTPUT_DIR,'local')

        self.cfg = cfg
        self.wrapper_x = wrapper_x
        self.best_result = -np.inf

        self.clients_nums = len(self.clients)

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_model(self,client_net):
        assert isinstance(client_net,dict)
        for key in client_net:
            setattr(key,copy.deepcopy(client_net[key]))
            optim = build_optimizer(client_net[key],self.cfg.FL.LOCAL.OPTIM)
            sched = build_lr_scheduler(optim,self.cfg.FL.LOCAL.OPTIM)
            setattr(key+'optim',optim)
            setattr(key+'sched',sched)
            self.register_model(key,client_net[key],optim,sched)

    def train(self,server_epoch):
        super().train(self.start_epoch,self.max_epoch)
        self.server_epoch = server_epoch
    
    def before_train(self):
        self.init_writer(osp.join(self.output_dir,'log_dir_%d'%self.server_epoch))
        self.time_start = time.time()
        self.train_loader_x = Loader.build_train_loader_x(self.cfg,self.wrapper_x)

    def after_train(self):
        print('Finished local training')
        self.model_dict = {}
        for key in self.client_net:
            self.model_dict[key] = self.client_net[key].state_net()
        
    @torch.no_grad()
    def test(self):
        pass

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        if self.cfg.TRAIN.ITER_TYPE == 'static':
            self.num_batches = len(self.train_loader_x)
        else:
            self.num_batches = self.cfg.TRAIN.ITER_NUM
            
        self.train_loader_x = ForeverDataIterator(self.train_loader_x)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch = next(self.train_loader_x)
    
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        domain = batch['domain']

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

class FLClassTrainer(ClassTrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.local_trainer = ClassLocalTrainer(self.datawrappers['wrapper_x'])

    def fed_model(self,client_state_dicts,fed_func,**kwargs):

        if not isinstance(client_state_dicts,(list,tuple)):
            client_state_dicts = [client_state_dicts]

        for name in self._models:
            name_state_dict = [client[name] for client in client_state_dicts]
            self._models[name].load_state_dict(fed_func(name_state_dict,**kwargs))
        
class FLSegTrainer(SegTrainerX):
    pass
