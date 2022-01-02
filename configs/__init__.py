from yacs.config import CfgNode as CN

from . import defaults
from .federatedlearning import FEDLEARN
from .activelearning import ACTIVELEARN

__all__ = [
    'get_default_cfg'
]

def get_default_cfg():
    _C = CN()

    for key in defaults.__all__:
        setattr(_C,key,defaults.__dict__[key])
    _C.FEDLEARN = FEDLEARN
    _C.ACTIVELEARN = ACTIVELEARN
    
    return _C.clone()
