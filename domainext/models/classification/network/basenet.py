from torch import nn
import torch
from domainext.utils.common.build import build_backbone,build_bottleneck

class BaseNet(nn.Module):
    def __init__(self,cfg,model_cfg,num_classes,**kwargs) -> None:
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )

        self._fdim = self.backbone.out_features
        self.bottleneck = self.build_bottleneck(cfg,model_cfg,**kwargs)
        self.classifier = self.build_classifier(num_classes)

        if self.bottleneck is None:
            self._embedding_dim = self._fdim
        else:
            self._embedding_dim = model_cfg.BOTTLENECK.HIDDEN_LAYERS[-1]

    @property
    def fdim(self):
        return self._fdim
    
    @property
    def embedding_dim(self):
        return self._embedding_dim
    
    def forward(self,x,return_feature=False,freeze=False,**kwargs):
        if freeze:
            with torch.no_grad():
                f = self.backbone(x)
                if self.bottleneck is not None:
                    f = self.bottleneck(f)
        else:
            f = self.backbone(x)
            if self.bottleneck is not None:
                f = self.bottleneck(f)
        
        if self.classifier is None:
            return f
        
        y = self.classifier(f)

        if return_feature:
            return y,f
        
        return y

    def build_bottleneck(self,cfg,model_cfg,**kwargs):
        pass

    def build_classifier(self,num_classes):
        pass

class BaseMlpNet(BaseNet):
    def __init__(self, cfg, model_cfg, num_classes, **kwargs) -> None:
        super().__init__(cfg, model_cfg, num_classes, **kwargs)
    
    def build_bottleneck(self,cfg,model_cfg,**kwargs):
        assert cfg.BOTTLENECK.NAME == 'mlp'

        if model_cfg.BOTTLENECK.NAME and model_cfg.BOTTLENECK.HIDDEN_LAYERS:
            return build_bottleneck(
                model_cfg.BOTTLENECK.NAME,
                verbose=cfg.VERBOSE,
                in_features = self.fdim,
                hidden_layers=model_cfg.BOTTLENECK.HIDDEN_LAYERS,
                activation=model_cfg.BOTTLENECK.ACTIVATION,
                bn=model_cfg.BOTTLENECK.BN,
                dropout=model_cfg.BOTTLENECK.DROPOUT,
                **kwargs,
            )

    def build_classifier(self,num_classes):
        if num_classes > 0:
            return nn.Linear(self.fdim,num_classes)

