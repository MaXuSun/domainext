import torch

from .score_streaming_strategy import ScoreStreamingStrategy
from domainext.utils.common.build import STRATEGY_REGISTRY

class Margin(ScoreStreamingStrategy):
    
    """
    Implements the Margin Sampling Strategy a active learning strategy similar to Least Confidence 
    Sampling Strategy. While least confidence only takes into consideration the maximum probability, 
    margin sampling considers the difference between the confidence of first and the second most 
    probable labels.  
    
    Suppose the model has `nclasses` output nodes denoted by :math:`\\overrightarrow{\\boldsymbol{z}}` 
    and each output node is denoted by :math:`z_j`. Thus, :math:`j \\in [1, nclasses]`. 
    Then for a output node :math:`z_i` from the model, the corresponding softmax would be 
    
    .. math::
        \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}} 
        
    Let,
    
    .. math::
        m = \\mbox{argmax}_j{(\\sigma(\\overrightarrow{\\boldsymbol{z}}))}
        
    Then using softmax, Margin Sampling Strategy would pick `budget` no. of elements as follows, 
    
    .. math::
        \\mbox{argmin}_{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\sum_S(\\mbox{argmax}_j {(\\sigma(\\overrightarrow{\\boldsymbol{z}}))}) - (\\mbox{argmax}_{j \\ne m} {(\\sigma(\\overrightarrow{\\boldsymbol{z}}))})}  
    
    where :math:`\\mathcal{U}` denotes the Data without lables i.e. `unlabeled_x` and :math:`k` is the `budget`.
    
    Parameters
    ----------
    wrapper_labeled: torch.utils.data.Dataset
        The labeled training dataset
    wrapper_unlabeled: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    embedding_dim: int
        The embedding dimensionality of model.
    kwargs: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
    """
    
    def __init__(self, wrapper_labeled, wrapper_unlabeled, model, num_classes, embedding_dim, **kwargs):
        super().__init__(wrapper_labeled, wrapper_unlabeled, model, num_classes, embedding_dim, **kwargs)
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob(unlabeled_buffer)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:,1] - probs_sorted[:, 0] # Margin negated => Largest score corresponds to smallest margin
        return U

class MarginDropout(ScoreStreamingStrategy):
    
    """
    Implements the Margin Sampling Strategy with dropout a active learning strategy similar to Least Confidence 
    Sampling Strategy with dropout. While least confidence only takes into consideration the maximum probability, 
    margin sampling considers the difference between the confidence of first and the second most 
    probable labels.  
    
    Suppose the model has `nclasses` output nodes denoted by :math:`\\overrightarrow{\\boldsymbol{z}}` 
    and each output node is denoted by :math:`z_j`. Thus, :math:`j \\in [1, nclasses]`. 
    Then for a output node :math:`z_i` from the model, the corresponding softmax would be 
    
    .. math::
        \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}} 
        
    Let,
    
    .. math::
        m = \\mbox{argmax}_j{(\\sigma(\\overrightarrow{\\boldsymbol{z}}))}
        
    Then using softmax, Margin Sampling Strategy would pick `budget` no. of elements as follows, 
    
    .. math::
        \\mbox{argmin}_{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\sum_S(\\mbox{argmax}_j {(\\sigma(\\overrightarrow{\\boldsymbol{z}}))}) - (\\mbox{argmax}_{j \\ne m} {(\\sigma(\\overrightarrow{\\boldsymbol{z}}))})}  
    
    where :math:`\\mathcal{U}` denotes the Data without lables i.e. `unlabeled_x` and :math:`k` is the `budget`.
    
    The drop out version uses the predict probability dropout function from the base strategy class to find the hypothesised labels.
    User can pass n_drop argument which denotes the number of times the probabilities will be calculated.
    The final probability is calculated by averaging probabilities obtained in all iteraitons.    
    
    Parameters
    ----------
    wrapper_labeled: torch.utils.data.Dataset
        The labeled training dataset
    wrapper_unlabeled: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    embedding_dim: int
        The embedding dimensionality of model.
    kwargs: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
        - **n_drop**: Number of dropout runs (int, optional)
    """
        
    def __init__(self, wrapper_labeled, wrapper_unlabeled, model, num_classes, embedding_dim, **kwargs):
        super().__init__(wrapper_labeled, wrapper_unlabeled, model, num_classes, embedding_dim, **kwargs)
        
        if 'n_drop' in kwargs:
            self.n_drop = kwargs['n_drop']
        else:
            self.n_drop = 10
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob_dropout(unlabeled_buffer, self.n_drop)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:,1] - probs_sorted[:, 0] # Margin negated => Largest score corresponds to smallest margin
        return U

@STRATEGY_REGISTRY.register()
def margin(**kwargs):
    return Margin(**kwargs)

@STRATEGY_REGISTRY.register()
def margin_dropout(**kwargs):
    return MarginDropout(**kwargs)