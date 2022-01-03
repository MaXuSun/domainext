"""
@author: Xu Ma
@email: maxu@zju.edu.cn
"""

from torch.utils.data import DataLoader
import torch
from .samplers import build_sampler

__all__ = [
    'build_train_loader_x',
    'build_train_loader_u',
    'build_test_loader',
    'build_val_loader'
    'fast_build_test_loader',
    'build_all_loader'
]

def _build_data_loader(
    cfg,
    wrapper,
    batch_size=32,
    is_train=True,
    sampler_type='SequentialSampler',
    n_domain=0,
    n_ins=2,
):
    sampler = build_sampler(sampler_type,cfg,wrapper.data,batch_size,n_domain,n_ins)

    data_loader = DataLoader(wrapper,batch_size=batch_size,sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,drop_last=is_train,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )

    assert len(data_loader) > 0
    return data_loader

def build_train_loader_x(cfg,wrapper_x):
    sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER
    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE
    n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN
    n_ins=cfg.DATALOADER.TRAIN_X.N_INS
    return _build_data_loader(cfg,wrapper_x,batch_size,True,sampler_type,n_domain,n_ins)

def build_train_loader_u(cfg,wrapper_u):
    sampler_type = cfg.DATALOADER.TRAIN_U.SAMPLER
    batch_size = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
    n_domain = cfg.DATALOADER.TRAIN_U.N_DOMAIN
    n_ins = cfg.DATALOADER.TRAIN_U.N_INS

    return _build_data_loader(cfg,wrapper_u,batch_size,True,sampler_type,n_domain,n_ins)

def build_test_loader(cfg,wrapper_test):
    sampler_type=cfg.DATALOADER.TEST.SAMPLER
    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE

    return _build_data_loader(cfg,wrapper_test,batch_size,False,sampler_type)

def build_val_loader(cfg,wrapper_val):
    return build_test_loader(cfg,wrapper_val)

def fast_build_test_loader(wrapper):
    data_loader = DataLoader(wrapper,batch_size=100,shuffle=False,num_workers=4,drop_last=False,pin_memory=True)
    return data_loader

def build_all_loader(cfg,wrapper_x,wrapper_test,wrapper_u=None,wrapper_val=None):
    rets = []
    rets.append(build_train_loader_x(cfg,wrapper_x))
    rets.append(build_train_loader_x(cfg,wrapper_test))

    if wrapper_u is not None:
        rets.append(build_train_loader_u(cfg,wrapper_u))
    if wrapper_val is not None:
        rets.append(build_val_loader(cfg,wrapper_val))

    return rets

    