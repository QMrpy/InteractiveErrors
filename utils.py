import math
import random
from itertools import zip_longest
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_data(data_fp, shuffle=False):
    """Loads data from file with format '<label>\t<query>\n'."""

    data = []
    with open(data_fp, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            label, query = line.split("\t")
            data.append((query, int(label)))
            
    if shuffle:
        random.shuffle(data)
        
    return data


def split_data(data, ratio, seed=42):
    """Splits data into train, val and prod data."""

    if sum(ratio) != 1 or len(ratio) != 3:
        raise ValueError("Weights must list of 3 numbers which sum to 1.")

    train_data, prod_data = train_test_split(data, test_size=ratio[2], random_state=seed)
    train_data, val_data = train_test_split(train_data, test_size=(ratio[1] / (ratio[0] + ratio[1])), random_state=seed)
    
    return train_data, val_data, prod_data


class TextDataset(Dataset):
    """PyTorch Dataset object for training classification model."""

    def __init__(self, args, tokenizer, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=args.cls_max_seq_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        
        return item

    def __len__(self):
        key = list(self.encodings.keys())[0]
        
        return len(self.encodings[key])


def compute_metrics(output):
    """Computes evaluation metrics for classifcation model."""

    predicted_labels = np.argmax(output.predictions, axis=1)
    accuracy = accuracy_score(output.label_ids, predicted_labels)
    f1 = f1_score(output.label_ids, predicted_labels)

    return {"accuracy": accuracy, "f1_score": f1}


def separate_labels(data):
    """Separates data with format (text, label).. into two lists."""

    texts = [x[0] for x in data]
    labels = [x[1] for x in data]
    
    return texts, labels


def move_dict_to_device(dict, device):
    """Moves all values of a dictionary to given device."""

    for key in dict:
        dict[key] = dict[key].to(device)


class OneLayerNN(nn.Module):
    """Single layer neural net."""

    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)

        return x


class TwoLayerNN(nn.Module):
    """Two layer neural net."""

    def __init__(self, input_size):
        super().__init__()
        hidden_size = math.ceil(math.sqrt(input_size))
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)

        return x


def batch(list_, batch_size, shuffle=True):
    """Generator for reading lists in chunks of batch_size."""

    if shuffle:
        random.shuffle(list_)
    n = len(list_)
    for i in range(0, n, batch_size):
        yield list_[i : i + batch_size]


def clean_args(args):
    """Cleans args for summary_writer.add_hparams."""

    cleaned_dict = {}
    allowed_types = (bool, str, float, int)

    for key, value in vars(args).items():
        if isinstance(value, allowed_types):
            cleaned_dict[key] = value

    return cleaned_dict


def read_file(file_name, shuffle=False):
    """Reads \n separated file."""

    result = []
    with open(file_name, "r") as file:
        result = file.read().splitlines()
    if shuffle:
        random.shuffle(result)

    return result


def write_file(list_, file_name):
    """Writes \n separated file."""

    with open(file_name, "w") as file:
        for e in list_:
            file.write(e.strip() + "\n")


def grouper(iterable, n, fillvalue=None):
    """ 
    Collect data into fixed-length chunks or blocks, for e.g.,
    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx. 
    """
    args = [iter(iterable)] * n

    return zip_longest(*args, fillvalue=fillvalue)
