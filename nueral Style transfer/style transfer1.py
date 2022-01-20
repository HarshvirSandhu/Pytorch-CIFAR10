import torch
from torchvision.transforms import transforms
import torchvision
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
# model = torchvision.models.vgg19(pretrained=True).features
# print(model)


class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.layers = [0, 5, 10, 19, 28]
        self.model = torchvision.models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if layer_num in self.layers:
                features.append(x)
        return features


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

image_size = 480

load = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10)
    ])


def format_img(image):
    img = Image.open(image)
    img = load(img).unsqueeze(0)
    return img.to(device)


style_img = format_img("Sunflower.jpg")
content = format_img("starry_night_full.jpg")

generated_img = content.clone().requires_grad_(True)
model = Vgg().to(device).eval()  # Freezing model weights

total_steps = 2000
learning_rate = 1e-3
optimizer = Adam([generated_img], lr=learning_rate)
alpha = 1
beta = 0.03

for steps in range(total_steps):
    print(steps)
    generated_features = model(generated_img)
    content_features = model(content)
    style_features = model(style_img)

    style_loss = 0
    content_loss = 0

    for gen_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        content_loss += torch.mean((gen_feature - content_feature) ** 2)

        # GRAM MATRIX
        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G - A) ** 2)
    total_loss = alpha*content_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if steps % 100 == 0:
        print(total_loss)
        save_image(generated_img, "generated.jpg")
