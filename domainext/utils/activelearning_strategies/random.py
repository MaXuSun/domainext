import numpy as np
from .strategy import Strategy
from domainext.utils.common.build import STRATEGY_REGISTRY

class Random(Strategy):

    """
    Implementation of Random Sampling Strategy. This strategy is often used as a baseline, 
    where we pick a set of unlabeled points randomly.
    
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
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
    """    
    def __init__(self, wrapper_labeled, wrapper_unlabeled, model, num_classes, embedding_dim, **kwargs):
        super().__init__(wrapper_labeled, wrapper_unlabeled, model, num_classes, embedding_dim, **kwargs)
        
    def select(self, budget):

        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to wrapper_unlabeled
        """	        

        rand_idx = np.random.permutation(len(self.wrapper_unlabeled))[:budget]
        rand_idx = rand_idx.tolist()
        return rand_idx

@STRATEGY_REGISTRY.register()
def random(**kwargs):
    return Random(**kwargs)