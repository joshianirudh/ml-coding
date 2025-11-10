import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_iris

class Iris(Dataset):

    def __init__(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):

        return self.features.shape[0]

    def __getitem__(self, idx):

        x, y = self.features[idx], self.labels[idx]
        return x, y


class Model(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.l1 = nn.Linear(input_shape, 128)
        self.drop1 = nn.Dropout(0.25)
        self.l2 = nn.Linear(128, 256)
        self.drop2 = nn.Dropout(0.25)
        self.l3 = nn.Linear(256, 512)
        self.drop3 = nn.Dropout(0.25)
        self.l4 = nn.Linear(512, 256)
        self.drop4 = nn.Dropout(0.25)
        self.l5 = nn.Linear(256, output_shape)

    def forward(self, x):

        x = F.relu(self.drop1(self.l1(x)))
        x = F.relu(self.drop2(self.l2(x)))
        x = F.relu(self.drop3(self.l3(x)))
        x = F.relu(self.drop4(self.l4(x)))
        x = self.l5(x)
        return x

def train(model, train_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    EPOCHS = 50
    for epoch in range(EPOCHS):
        print(f"epoch {epoch + 1}")
        # enable train mode
        model.train(True)
        running_loss, correct, total = 0.0, 0, 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = outputs.argmax(dim=1)
            correct = (outputs == labels).float().sum()
            total = inputs.size(0)
            running_loss += loss.item() 

            optimizer.zero_grad()
        
        print(f"Loss {running_loss}, Acc {correct/total}")


def evaluate(model, test_loader):
    
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        loss , correct, total = 0.0, 0, 0

        for i, data in enumerate(test_loader):
            inputs, labels = data
            logits = model(inputs)
            loss += loss_fn(logits, labels).item()
            logits = logits.argmax(dim=1)
            correct += (logits == labels).float().sum()
            total += inputs.size(0)
    

        print(f"Eval Loss: {loss}, Eval acc: {correct/total}")


if __name__ == "__main__":
    dataset = Iris()
    size = int(len(dataset) * 0.8)
    size2 = len(dataset) - size
    train_ds, test_ds = random_split(dataset, [size, size2])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    model = Model(4, 3)
    train(model, train_loader)
    evaluate(model, test_loader)






    