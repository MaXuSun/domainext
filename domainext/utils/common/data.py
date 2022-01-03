from .torchtools import send_to_device
from torch.utils.data import DataLoader
from collections import defaultdict
import random

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

def split_dataset_by_domain(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into domain-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        output[item.domain].append(item)

    return output

def split_dataset_by_label(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into class-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        output[item.label].append(item)

    return output

def generate_fewshot_dataset(*data_sources, num_shots=-1, repeat=False):
    """Generate a few-shot dataset (typically for the training set).

    This function is useful when one wants to evaluate a model
    in a few-shot learning setting where each class only contains
    a few number of images.

    Args:
        data_sources: each individual is a list containing Datum objects.
        num_shots (int): number of instances per class to sample.
        repeat (bool): repeat images if needed (default: False).
    """
    if num_shots < 1:
        if len(data_sources) == 1:
            return data_sources[0]
        return data_sources

    print(f"Creating a {num_shots}-shot dataset")

    output = []

    for data_source in data_sources:
        tracker = split_dataset_by_label(data_source)
        dataset = []

        for label, items in tracker.items():
            if len(items) >= num_shots:
                sampled_items = random.sample(items, num_shots)
            else:
                if repeat:
                    sampled_items = random.choices(items, k=num_shots)
                else:
                    sampled_items = items
            dataset.extend(sampled_items)

        output.append(dataset)

    if len(output) == 1:
        return output[0]

    return output