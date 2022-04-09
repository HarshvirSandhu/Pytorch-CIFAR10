import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.4),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x


learning_rate = 3e-4
batch_size = 32
num_epochs = 100
z_dim = 64

disc = Discriminator()
gen = Generator(z_dim)
optimizer_disc = optim.Adam(disc.parameters(), lr=learning_rate)
optimizer_gen = optim.Adam(gen.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
Mnist_dataset = Dataset.MNIST(root="dataset/", transform=transform, train=True, download=True)
loader = DataLoader(Mnist_dataset, batch_size, shuffle=True)

for epoch in range(num_epochs):
    print(epoch)
    for real, label in loader:
        # Discriminator Training

        # print(real.shape)  # [batch size, 1, 28, 28]
        batch_size = real.shape[0]
        real = real.view(-1, 784)
        noise = torch.randn(batch_size, z_dim)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake).view(-1)

        lossD_real = criterion(disc_real, torch.ones_like(disc_real))  # torch.ones_like is used as discriminator
        # wants to maximise probability of predicting real data as real

        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # torch.zeros_like is used as discriminator
        # wants to maximise probability of predicting fake data as fake

        lossD = (lossD_fake + lossD_real) / 2
        optimizer_disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()

        # Training The Generator

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))  # torch.ones_like is used as Generator wants discriminator
        # wants to maximise probability of predicting real data as real
        optimizer_gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()

    if (epoch+3) % 1 == 0:
        check = gen(torch.randn(batch_size, z_dim))
        img = check.view(batch_size, 1, 28, 28)
        # print(img)
        grid = make_grid(img)
        print(grid.shape)
        # print(grid)
        save_image(grid, "output.jpg")