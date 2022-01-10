import numpy as np

import torch.nn.functional as F
from .xutrainer import ClassTrainerXU,SegTrainerXU
from domainext.utils.common import build_strategy

def cal_num_active(cfg,epoch,update_epochs,dataset):
    if cfg.ACTIVELEARNING.BUDGET_AVG == 0:
        assert cfg.ACTIVELEARNING.BUDGET_ALL != 0
        num_labeled = cfg.ACTIVELEARNING.BUDGET_ALL

        if cfg.ACTIVELEARNING.BUDGET_ALL < 0:
            if cfg.ACTIVELEARNING.PERCENT.lower() in ['u','un','unlabeled','unlabel']:
                num_labeled = int(-num_labeled * len(dataset.train_u)/100)
            elif cfg.ACTIVELEARNING.PERCENT.lower() in ['x','label','labeled']:
                num_labeled = int(-num_labeled * len(dataset.train_x)/100)
            elif cfg.ACTIVELEARNING.PERCENT.lower() in ['a','all']:
                num_labeled = int(-num_labeled * (len(dataset.train_u)+len(dataset.train_x))/100)

        every_num = np.math.ceil(num_labeled/len(update_epochs))
        if epoch == update_epochs[-1]:
            num_active = num_labeled - (every_num)*(len(update_epochs)-1)
        else:
            num_active = every_num
    else:
        num_active = cfg.ACTIVELEARNING.BUDGET_AVG
    return num_active

class ALClassTrainer(ClassTrainerXU):
    """
    A base trainer for Class Active Learning.
    """
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.stategy = build_strategy(cfg,
            wrapper_labeled=self.datawrappers['wrapper_x'],
            wrapper_unlabeled = self.datawrappers['wrapper_u'],
            net = self.model,
            nclass = self.num_classes,
            embedding_dim = self.model.embedding_dim,
            inference_func = self.model_inference,
            device = self.device,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            loss_func = F.cross_entropy
        )

    def select_data(self,budget,wrapper_labeled,wrapper_unlabeled):
        self.set_model_mode(mode='eval')
        self.stategy.update_data(wrapper_labeled,wrapper_unlabeled)
        indexs = self.stategy.select(budget)
        return np.array(wrapper_unlabeled.data)[indexs]

    def active_learning(self):
        print(f'Before active-learning, Unlabeled Data Num: %d; Labeled Data Num: %d'%(len(self.dataset.train_x),len(self.dataset.train_u)))

        if len(self.cfg.ACTIVELEARNING.UPDATE_EPOCHS) == 0:
            update_epochs = [i for i in range(self.ACTIVELEARNING.START_EPOCH,self.max_epoch-1)]
        else:
            update_epochs = self.cfg.ACTIVELEARNING.UPDATE_EPOCHS
        
        if self.epoch in update_epochs:
            num_active = cal_num_active(self.cfg,self.epoch,update_epochs)
            dataitems = self.select_data(num_active,self.datawrappers['wrapper_x',self.datawrappers['wrapper_u']])
            self.dataset.add_items(dataitems,'train_x')
            self.dataset.remove_items(dataitems,'train_u')

            self.build_data_loader(self.dataset)
        
        print(f'After active-learning, Unlabeled Data Num: %d; Labeled Data Num: %d'%(len(self.dataset.train_x),len(self.dataset.train_u)))

    def after_epoch(self):
        super().after_epoch()
        self.active_learning()

class ALSegTrainer(SegTrainerXU):
    pass