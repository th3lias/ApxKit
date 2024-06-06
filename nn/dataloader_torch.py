from torch.utils.data import DataLoader, Dataset


# write Dataset that takes grid from GridProvider

class LS_Dataset(Dataset):
    def __init__(self, basis, y):
        self.grid = basis
        self.y = y
        self.combin = zip(basis, y)

    def __len__(self):
        return self.grid.shape[0]

    def __getitem__(self, idx):
        x = self.grid[idx]
        y = self.y[idx]
        return (x,y)

