import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Dataloader


class CNN(nn.Module):
    def __init__(self, num_class=10, input_channels=3):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2))
        self.layer4 = nn.Conv2d(in_channels=64, out_channels=100, kernel_size=(2, 2))
        self.ann = nn.Linear(in_features=900, out_features=num_class)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.pool(x)
        x = F.relu(self.layer2(x))
        x = self.pool(x)
        x = F.relu(self.layer4(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.ann(x)

        return F.softmax(x)


model = CNN()
# sample = torch.randn(32, 3, 32, 32)
# print(model(sample).shape)
trainset = Datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=False)
testset = Datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=False)
trainload = Dataloader.DataLoader(trainset, batch_size=32, shuffle=True)
testload = Dataloader.DataLoader(testset, batch_size=32, shuffle=True)
epochs = 12
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
a = 0
for epoch in range(epochs):
    a += 1
    count = 0
    print("Epoch", a)
    for data, label in trainload:
        count += 1
        if int(count/len(trainload)*100) % 5 == 0:
            print("Epoch", a, ":", int(100*count/len(trainload)), "%")
        score = model(data)
        loss = criterion(score, label)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def acc_check(loader, model):
    num_samples = 0
    num_correct = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            score = model(x)
            ret, pred = score.max(1)
            num_correct += (y == pred).sum()
            num_samples += pred.size(0)

        print("Accuracy = ", num_correct/num_samples)


acc_check(testload, model)
