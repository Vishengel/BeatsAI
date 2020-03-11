import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config
from torch.utils.data import Dataset, DataLoader

class BeatsData(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=0, index_col=0)
        self.n_features = len(self.data.iloc[0]) - 1
        self.n_classes = len(self.data['class'].unique())
        self.label_to_idx = self.make_label_to_idx()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def count_labels(self, pd_data):
        count = {}

        for row in pd_data['class']:
            if not row in count.keys():
                count[row] = 1
            else:
                count[row] += 1

        return count

    def make_label_to_idx(self):
        label_to_idx = {}
        idx_to_label = {}

        for idx, label in enumerate(self.data['class'].unique()):
            label_to_idx[label] = idx
            idx_to_label[str(idx)] = label

        return label_to_idx

    def encode_label(self, label):
        one_hot_vector = np.zeros(self.n_classes)
        one_hot_vector[self.label_to_idx[label]] = 1.0

        return one_hot_vector

def get_dataloader(csv_file):
    dataset = BeatsData(csv_file)
    train_split, val_split = train_test_split(dataset.data, train_size=0.8, stratify=dataset.data['class'])
    train_loader = DataLoader(train_split, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=config.batch_size, shuffle=False)

    return dataset, train_loader, val_loader

