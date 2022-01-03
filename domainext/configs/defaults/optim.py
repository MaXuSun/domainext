from yacs.config import CfgNode as CN

OPTIM = CN()
OPTIM.NAME = "adam"
OPTIM.LR = 0.0003
OPTIM.WEIGHT_DECAY = 5e-4
OPTIM.MOMENTUM = 0.9
OPTIM.SGD_DAMPNING = 0
OPTIM.SGD_NESTEROV = False
OPTIM.RMSPROP_ALPHA = 0.99
OPTIM.ADAM_BETA1 = 0.9
OPTIM.ADAM_BETA2 = 0.999

""" STAGED_LR allows different layers to have different lr, 
e.g. pre-trained base layers can be assigned a smaller lr than the new classification layer 
"""
OPTIM.STAGED_LR = False
OPTIM.NEW_LAYERS = ()
OPTIM.BASE_LR_MULT = 0.1
OPTIM.LR_SCHEDULER = "single_step"   # Learning rate scheduler
OPTIM.STEPSIZE = (-1, )              # -1 or 0 means the stepsize is equal to max_epoch
OPTIM.GAMMA = 0.1
OPTIM.MAX_EPOCH = 10
OPTIM.WARMUP_EPOCH = -1              # Set WARMUP_EPOCH larger than 0 to activate warmup training
OPTIM.WARMUP_TYPE = "linear"         # Either linear or constant
OPTIM.WARMUP_CONS_LR = 1e-5          # Constant learning rate when type=constant
OPTIM.WARMUP_MIN_LR = 1e-5           # Minimum learning rate when type=linear
OPTIM.WARMUP_RECOUNT = True          # Recount epoch for the next scheduler (last_epoch=-1), Otherwise last_epoch=warmup_epoch