from yacs.config import CfgNode as CN

DATALOADER = CN()
DATALOADER.NUM_WORKERS = 4
DATALOADER.K_TRANSFORMS = 1                  # Apply transformations to an image K times (during training)
DATALOADER.RETURN_IMG0 = False               # img0 denotes image tensor without augmentation, Useful for consistency learning

DATALOADER.TRAIN_X = CN()                    # Setting for the train_x data-loader
DATALOADER.TRAIN_X.SAMPLER = "RandomSampler" # RandomSampler, SequentialSampler, RandomDomainSampler, SeqDomainSampler, RandomClassSampler,
DATALOADER.TRAIN_X.BATCH_SIZE = 32           
DATALOADER.TRAIN_X.N_DOMAIN = 0              # Parameter for RandomDomainSampler, 0 or -1 means sampling from all domains                 
DATALOADER.TRAIN_X.N_INS = 16                # Parameter of RandomClassSampler, Number of instances per class

DATALOADER.TRAIN_U = CN()                    # Setting for the train_u data-loader
DATALOADER.TRAIN_U.SAME_AS_X = True          # Set to false if you want to have unique data loader params for train_u
DATALOADER.TRAIN_U.SAMPLER = "RandomSampler" 
DATALOADER.TRAIN_U.BATCH_SIZE = 32
DATALOADER.TRAIN_U.N_DOMAIN = 0
DATALOADER.TRAIN_U.N_INS = 16

DATALOADER.TEST = CN()                       # Setting for the test data-loader
DATALOADER.TEST.SAMPLER = "SequentialSampler"
DATALOADER.TEST.BATCH_SIZE = 32