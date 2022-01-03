"""
@author: Xu Ma
@email: maxu@zju.edu.cn
"""

from .registry import Registry, check_availability

__all__ = [
    'build_network',
    'build_backbone',
    'build_bottleneck',
    'build_encoder',
    'build_decoder',
    'build_trainer',
    'build_evaluator',
    'build_dataset'
]

##################
###  1. model  ###
##################

# 1.1 common
"""Network Registry"""
NETWORK_REGISTRY = Registry("NETWORK")
def build_network(name, verbose=True, **kwargs):
    avai_models = NETWORK_REGISTRY.registered_names()
    check_availability(name, avai_models)
    if verbose:
        print("Network: {}".format(name))
    return NETWORK_REGISTRY.get(name)(**kwargs)


# 1.2 for classification
"""Backbone Registry"""
BACKBONE_REGISTRY = Registry("BACKBONE")
def build_backbone(name, verbose=True, **kwargs):
    avai_backbones = BACKBONE_REGISTRY.registered_names()
    check_availability(name, avai_backbones)
    if verbose:
        print("Backbone: {}".format(name))
    return BACKBONE_REGISTRY.get(name)(**kwargs)

"""BOTTLENECK Registry"""
BOTTLENECK_REGISTRY = Registry("BOTTLENECK")
def build_bottleneck(name, verbose=True, **kwargs):
    avai_heads = BOTTLENECK_REGISTRY.registered_names()
    check_availability(name, avai_heads)
    if verbose:
        print("BOTTLENECK: {}".format(name))
    return BOTTLENECK_REGISTRY.get(name)(**kwargs)

# 1.3 for segmentation
"""Encoder"""
ENCODER_REGISTRY = Registry("ENCODER")
def build_encoder(name,verbose=True,**kwargs):
    avai_encoder = ENCODER_REGISTRY.registered_names()
    check_availability(name,avai_encoder)
    if verbose:
        print("Encoder: {}".format(name))
    return ENCODER_REGISTRY.get(name)(**kwargs)

"""Decoder"""
DECODER_REGISTRY = Registry("DECODER")
def build_decoder(name,verbose=True,**kwargs):
    avai_decoder = DECODER_REGISTRY.registered_names()
    check_availability(name,avai_decoder)
    if verbose:
        print("Decoder: {}".format(name))
    return DECODER_REGISTRY.get(name)(**kwargs)

##################
###  2. train  ###
##################

"""Trainer Registry"""
TRAINER_REGISTRY = Registry("TRAINER")
def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)

"""Evaluator Registry"""
EVALUATOR_REGISTRY = Registry("EVALUATOR")
def build_evaluator(cfg, **kwargs):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TEST.EVALUATOR, avai_evaluators)
    if cfg.VERBOSE:
        print("Loading evaluator: {}".format(cfg.TEST.EVALUATOR))
    return EVALUATOR_REGISTRY.get(cfg.TEST.EVALUATOR)(cfg, **kwargs)

##################
###  3. data  ###
##################

"""Dataset Registry"""
DATASET_REGISTRY = Registry("DATASET")
def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)