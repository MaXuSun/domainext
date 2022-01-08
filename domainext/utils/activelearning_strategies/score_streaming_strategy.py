from .strategy import Strategy
from torch.utils.data import Subset

def merge(list_1, list_2, key, merged_list_size_cap):
    
    list_1_index = 0
    list_2_index = 0
    
    merged_list = list()
    
    while list_1_index < len(list_1) and list_2_index < len(list_2) and len(merged_list) < merged_list_size_cap:
        
        list_1_elem = list_1[list_1_index]
        list_2_elem = list_2[list_2_index]
        
        if key(list_1_elem) >= key(list_2_elem):
            merged_list.append(list_1_elem)
            list_1_index += 1
        else:
            merged_list.append(list_2_elem)
            list_2_index += 1
            
    while list_1_index < len(list_1) and len(merged_list) < merged_list_size_cap:
        
        list_1_elem = list_1[list_1_index]
        merged_list.append(list_1_elem)
        list_1_index += 1
        
    while list_2_index < len(list_2) and len(merged_list) < merged_list_size_cap:
        
        list_2_elem = list_2[list_2_index]
        merged_list.append(list_2_elem)
        list_2_index += 1
            
    return merged_list

class ScoreStreamingStrategy(Strategy):
    
    """
    Provides a framework for AL strategies wherein each data point in the unlabeled set is 
    attributed a 'score' in a streaming manner. The largest score is then selected.
    
    Parameters
    ----------
    wrapper_labeled: torch.utils.data.Dataset
        The labeled training dataset
    wrapper_unlabeled: torch.utils.data.Dataset
        The unlabeled pool dataset
    model: torch.nn.Module
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
        - **stream_buffer_size**: The buffer size of the stream used in calculating scores (int, optional)
    """
    
    def __init__(self, wrapper_labeled, wrapper_unlabeled, model, num_classes, embedding_dim, **kwargs):
        super().__init__(wrapper_labeled, wrapper_unlabeled, model, num_classes, embedding_dim, **kwargs)
        
        if 'stream_buffer_size' not in kwargs:
            self.stream_buffer_size = 10000
        else:
            self.stream_buffer_size = kwargs['stream_buffer_size']
        
    def acquire_scores(self, unlabeled_batch):
        raise NotImplemented()
    
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
        
        evaluated_points = 0
        
        selected_buffer = list()
        
        while evaluated_points < len(self.wrapper_unlabeled):
            
            buffered_stream = Subset(self.wrapper_unlabeled, list(range(evaluated_points, min(len(self.wrapper_unlabeled), evaluated_points + self.stream_buffer_size))))
            batch_scores = self.acquire_scores(buffered_stream)
            batch_scores = [(x, i + evaluated_points) for i,x in enumerate(batch_scores)]
            batch_scores = sorted(batch_scores, key=lambda x: x[0], reverse=True)
            selected_buffer = merge(selected_buffer, batch_scores, lambda x: x[0], budget)
            evaluated_points += len(batch_scores)
            
        return [i for (_,i) in selected_buffer[:budget]]
            
            