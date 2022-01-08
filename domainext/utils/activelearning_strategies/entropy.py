import torch

from .score_streaming_strategy import ScoreStreamingStrategy
from domainext.utils.common.build import STRATEGY_REGISTRY

class Entropy(ScoreStreamingStrategy):
    
    """
    Implements the Entropy Sampling Strategy, one of the most basic active learning strategies,
    where we select samples about which the model is most uncertain. To quantify the uncertainity 
    we use entropy and therefore select points which have maximum entropy. 
    Suppose the model has `nclasses` output nodes and each output node is denoted by :math:`z_j`. Thus,  
    :math:`j \\in [1,nclasses]`. Then for a output node :math:`z_i` from the model, the corresponding
    softmax would be 
    
    .. math::
        \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}
    
    Then entropy can be calculated as,
    
    .. math:: 
        ENTROPY = -\\sum_j \\sigma(z_j)*\\log(\\sigma(z_j))
        
    The algorithm then selects `budget` no. of elements with highest **ENTROPY**.
    
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
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(1)
        return U

class EntropyDropout(ScoreStreamingStrategy):
    
    """
    Implements the Entropy Sampling Strategy with dropout. Entropy Sampling Strategy is one 
    of the most basic active learning strategies, where we select samples about which the model 
    is most uncertain. To quantify the uncertainity we use entropy and therefore select points 
    which have maximum entropy. 
    Suppose the model has `nclasses` output nodes and each output node is denoted by :math:`z_j`. Thus,  
    :math:`j \in [1,nclasses]`. Then for a output node :math:`z_i` from the model, the corresponding 
    softmax would be 
    
    .. math::
        \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}
    
    Then entropy can be calculated as,
    
    .. math:: 
        ENTROPY = -\\sum_j \\sigma(z_j)*\\log(\\sigma(z_j))
        
    The algorithm then selects `budget` no. of elements with highest **ENTROPY**.
    
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
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(1)
        return U

@STRATEGY_REGISTRY.register()
def entropy(**kwargs):
    return Entropy(**kwargs)

@STRATEGY_REGISTRY.register()
def entropy_dropout(**kwargs):
    return EntropyDropout(**kwargs)