import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.features: torch.Tensor = X
        self.labels: torch.Tensor = y

    def __getitem__(self, index: int):
        one_x: torch.Tensor = self.features[index]
        one_y: torch.Tensor = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


def get_dataloaders(
    train_ds: Dataset, test_ds: Dataset
) -> tuple[DataLoader, DataLoader]:
    train_loader: DataLoader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    test_loader: DataLoader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    X_train: torch.Tensor = torch.tensor(
        [
            [-1.2, 3.1],
            [-0.9, 2.9],
            [-0.5, 2.6],
            [2.3, -1.1],
            [2.7, -1.5],
        ]
    )
    y_train: torch.Tensor = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )
    y_test: torch.Tensor = torch.tensor([0, 1])

    train_ds: Dataset = CustomDataset(X=X_train, y=y_train)
    test_ds: Dataset = CustomDataset(X=X_test, y=y_test)

    torch.manual_seed(123)
    train_loader, test_loader = get_dataloaders(train_ds, test_ds)

    for index, (x, y) in enumerate(train_loader):
        print(f"Entry {index}:\nx = {x}\ny = {y}")
