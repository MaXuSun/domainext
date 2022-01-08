import torch
from .strategy import Strategy
from domainext.utils.common.build import STRATEGY_REGISTRY

class CoreSet(Strategy):
    
    """
    Implementation of CoreSet :footcite:`sener2018active` Strategy. A diversity-based 
    approach using coreset selection. The embedding of each example is computed by the network’s 
    penultimate layer and the samples at each round are selected using a greedy furthest-first 
    traversal conditioned on all labeled examples.
    
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
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
    """
    
    def __init__(self, wrapper_labeled, wrapper_unlabeled, net, nclasses, args={}):
        
        super(CoreSet, self).__init__(wrapper_labeled, wrapper_unlabeled, net, nclasses, args)
  
    def furthest_first(self, unlabeled_embeddings, labeled_embeddings, n):
        
        unlabeled_embeddings = unlabeled_embeddings.to(self.device)
        labeled_embeddings = labeled_embeddings.to(self.device)
        
        m = unlabeled_embeddings.shape[0]
        if labeled_embeddings.shape[0] == 0:
            min_dist = torch.tile(float("inf"), m)
        else:
            dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
            min_dist = torch.min(dist_ctr, dim=1)[0]
                
        idxs = []
        
        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])
            min_dist = torch.minimum(min_dist, dist_new_ctr[:,0])
                
        return idxs
  
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
        
        
        self.set_model_mode()
        embedding_unlabeled = self.get_embedding(self.wrapper_unlabeled)
        embedding_labeled = self.get_embedding(self.wrapper_labeled)
        chosen = self.furthest_first(embedding_unlabeled, embedding_labeled, budget)

        return chosen        

@STRATEGY_REGISTRY.register()
def coreset(**kwargs):
    return CoreSet(**kwargs)