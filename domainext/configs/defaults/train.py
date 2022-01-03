from yacs.config import CfgNode as CN

TRAIN = CN()
TRAIN.CHECKPOINT_FREQ = 0      # How often (epoch) to save model during training. Set to 0 or negative value to only save the last one
TRAIN.PRINT_FREQ = 10          # How often (batch) to print training information
TRAIN.COUNT_ITER = "train_x"   # Use 'train_x', 'train_u' or 'smaller_one' to count, the number of iterations in an epoch (for DA and SSL)
TRAIN.ITER_TYPE = "static"     # static or dynamic. If you choose "static", you mast set ITER_NUM. If you choose "dynamic", you must set COUNT_ITER
TRAIN.ITER_NUM = 100
