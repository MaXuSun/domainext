from yacs.config import CfgNode as CN

DATASET = CN()
DATASET.ROOT = ""                    # Directory where datasets are stored
DATASET.NAME = ""                    # Dataset name
DATASET.SOURCE_DOMAINS = ()          # List of names of source domains
DATASET.TARGET_DOMAINS = ()          # List of names of target domains

"""
Number of labeled instances in total.
'>0': Represent the number of images, '<0': Represent the percentage of images. 
Useful for the semi-supervised learning
"""
DATASET.NUM_LABELED = -1             
"""
Number of images per class. 
Useful when one wants to evaluate a model in a few-shot learning setting where each class only contains a few number of images.
"""
DATASET.NUM_SHOTS = -1 

DATASET.VAL_PERCENT = 0.1            # Percentage of validation data. Set to 0 if do not want to use val data. Using val data for hyperparameter.
DATASET.STL10_FOLD = -1              # Fold index for STL-10 dataset (normal range is 0 - 9), Negative number means None
DATASET.CIFAR_C_TYPE = ""            # CIFAR-10/100-C's corruption type and intensity level
DATASET.CIFAR_C_LEVEL = 1
DATASET.ALL_AS_UNLABELED = False     # Use all data in the unlabeled data set (e.g. FixMatch)