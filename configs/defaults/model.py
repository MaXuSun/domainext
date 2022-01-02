from yacs.config import CfgNode as CN

MODEL = CN()
MODEL.INIT_WEIGHTS = ""              # Path to model weights (for initialization)

MODEL.BACKBONE = CN()
MODEL.BACKBONE.NAME = ""
MODEL.BACKBONE.PRETRAINED = True

MODEL.BOTTLENECK = CN()              # Definition of bottleneck
MODEL.BOTTLENECK.NAME = ""           # If none, do not construct embedding layers, the, backbone's output will be passed to the classifier
MODEL.BOTTLENECK.HIDDEN_LAYERS = ()  # Structure of hidden layers (a list), e.g. [512, 512]. If undefined, no embedding layer will be constructed
MODEL.BOTTLENECK.ACTIVATION = "relu"
MODEL.BOTTLENECK.BN = True
MODEL.BOTTLENECK.DROPOUT = 0.0