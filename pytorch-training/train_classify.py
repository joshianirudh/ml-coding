import torch
from torch import nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_iris

class Iris(Dataset):

    def __init__(self):

        iris = load_iris()
        X, y = iris.data, iris.target

        self.features = torch.tensor(X)
        self.labels = torch.tensor(y)

    def __len__(self):

        return self.features.shape[0]

    def __getitem__(self, idx):

        x, y = self.features[idx], self.labels[idx]
        return x, y


class Model(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.l1 = nn.Linear(input_shape, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 512)
        self.l4 = nn.Linear(512, 256)
        self.l5 = nn.Linear(256, output_shape)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.softmax(self.l5(x))
        return x

def train(model, train_loader):
    pass


def evaluate(model, test_loader):
    pass

if __name__ == "__main__":

    dataset = Iris()
    size = int(len(dataset) * 0.8)
    size2 = len(dataset) - size
    train_ds, test_ds = random_split(dataset, [size, size2])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    model = Model()
    train(model, train_loader)
    evaluate(model, test_loader)






    