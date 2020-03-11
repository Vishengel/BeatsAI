from src import config, models
from src.dataset import BeatsData, get_dataloader

def main():
    dataset, train_loader, val_loader = get_dataloader(config.data_path)
    config.input_size = dataset.n_features
    config.n_classes = dataset.n_classes

if __name__ == "__main__":
    main()