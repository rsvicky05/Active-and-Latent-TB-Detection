import torch
import numpy as np

def preprocess_clinical(data_dict):
    values = np.array(list(data_dict.values()), dtype=np.float32)
    return torch.tensor(values)