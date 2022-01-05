import torch
import os
import cv2
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
transformer = transforms.Compose([transforms.Resize((60, 60)),
                                  transforms.ToTensor(),
                                  transforms.ColorJitter(12, 12, 12),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                  transforms.RandomHorizontalFlip()])

train_path = "C:/Users/harsh/Downloads/ASL"
# test_path = "C:/Users/harsh/PycharmProjects/Harshvir_S/Traffic Signs DATA/Test"

#  PLOTTING HISTOGRAM TO SEE DISTRIBUTION OF SHAPE OF MOST IMAGES
"""
photo_list = []
for imgs in os.listdir((train_path+"/0")):
    path = train_path+"/0/"+imgs
    photo = cv2.imread(path)
    x = photo.shape[1]
    photo_list.append(x)

plt.hist(photo_list, bins=10)
plt.show()
"""

train_loader = DataLoader(torchvision.datasets.ImageFolder(train_path, transform=transformer),
                          batch_size=100, shuffle=True)
# test_loader = DataLoader(torchvision.datasets.ImageFolder(test_path, transform=transformer),
#                          batch_size=64, shuffle=True)


class Classifier(nn.Module):
    def __init__(self, num_classes=17, in_channel=3):
        super(Classifier, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=in_channel, out_channels=10, kernel_size=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.layer2 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(2, 2))
        self.layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2))
        self.layer4 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(2, 2))
        self.ann = nn.Linear(in_features=320, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.pool(x)
        x = F.relu(self.layer2(x))
        x = self.pool(x)
        x = F.relu(self.layer3(x))
        x = self.pool(x)
        x = F.relu(self.layer4(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.ann(x)

        return F.softmax(x, dim=1)


class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()

    def forward(self, x):
        return x


sample = torch.randn((1, 3, 60, 60))
model = Classifier()
transfer_model = torchvision.models.vgg16(pretrained=True)
transfer_model.avgpool = Transfer()
transfer_model.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=43, bias=True),
                                          nn.Softmax(dim=1))
print(transfer_model)
# print(sum(transfer_model(sample)[0]), transfer_model(sample)[0], transfer_model(sample).shape)

img = cv2.imread("C:/Users/harsh/PycharmProjects/Harshvir_S/Traffic Signs DATA/Test/00001.png")
img = cv2.resize(img, dsize=(60, 60))
img = img.reshape((1, 3, 60, 60))
img = (img/255)
img = torch.from_numpy(img).type(torch.float)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for param in model.parameters():
    param.requires_grad = False

epochs = 3
criterion = nn.NLLLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)
loss_list = []
for epoch in range(epochs):
    count = 0
    epoch_loss = 0
    for data, label in train_loader:
        score = transfer_model(data)
        count += 1
        loss = criterion(score, label)
        epoch_loss += loss.item() * data.size(0)
        print("Epoch", epoch+1, ":(", count, "/", len(train_loader), ")   ", loss)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print("Epoch", epoch+1, "Loss: ", epoch_loss/len(train_loader))
    loss_list.append(epoch_loss/len(train_loader))
print(loss_list)


pred = transfer_model.forward(img)
print(pred, len(pred[0]))
print(torch.sum(pred[0]), torch.argmax(pred))