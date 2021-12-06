import numbers
import torch

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def random_split(dataset, percentages):
    train_split = 0.0
    test_split = 0.0

    if isinstance(percentages, numbers.Number):
        train_split = percentages
        test_split = 1 - percentages
    else:
        train_split = percentages[0]
        test_split = percentages[1]
    
    train_split = int(round(len(dataset)*train_split))
    test_split = int(round(len(dataset)*test_split))  
    
    train_set, test_set = torch.utils.data.random_split(dataset, [train_split, test_split])
    return train_set, test_set

def stratified_random_split(dataset, percentages, targets):
    train_split = 0.0
    test_split = 0.0

    if isinstance(percentages, numbers.Number):
        train_split = percentages
        test_split = 1 - percentages
    else:
        train_split = percentages[0]
        test_split = percentages[1]
    
    train_indxs, test_indxs = train_test_split(list(range(len(targets))), test_size=test_split, train_size=train_split, stratify=targets)
    train_set = Subset(dataset, train_indxs)
    test_set = Subset(dataset, test_indxs)
    return train_set, test_set


