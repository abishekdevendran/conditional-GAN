import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Custom Dataset (same as before)


class PixelArtDataset(Dataset):
    def __init__(self, input_folder, target_folder, transform=None):
        self.input_files = sorted(
            [f for f in os.listdir(input_folder) if f.endswith('.png')])
        self.target_files = sorted(
            [f for f in os.listdir(target_folder) if f.endswith('.png')])
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):

        input_img_path = os.path.join(self.input_folder, self.input_files[idx])
        target_img_path = os.path.join(
            self.target_folder, self.target_files[idx])

        input_img = Image.open(input_img_path)
        target_img = Image.open(target_img_path)

        if self.transform:

            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img


# Residual Block for better gradient flow
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return nn.ReLU()(x)

# Generator (same as before)
# Modified Generator with pixel-art specific adjustments
class Generator(nn.Module):
    def __init__(self, channels=128):  # Increased base channels
        super(Generator, self).__init__()

        # Encoder with smaller kernels for better detail preservation
        self.encoder = nn.Sequential(
            nn.Conv2d(4, channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2),
        )

        # Residual blocks for better feature preservation
        self.residual = nn.Sequential(
            ResidualBlock(channels * 4),
            ResidualBlock(channels * 4),
            ResidualBlock(channels * 4)
        )

        # Decoder with pixel shuffle for better upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels * 4, channels *
                               2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            nn.ConvTranspose2d(channels, 4, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x
# Discriminator (same as before)


class Discriminator(nn.Module):
    def __init__(self, channels=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            SpectralNorm(nn.Conv2d(8, channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(channels, channels *
                         2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(channels * 2, channels *
                         4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            SpectralNorm(nn.Conv2d(channels * 4, 1, 7, stride=1, padding=0)),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        combined = torch.cat([x, condition], dim=1)
        return self.model(combined)

# Spectral Normalization for stability


def SpectralNorm(module):
    return nn.utils.spectral_norm(module)

# Updated training function with fixed dimensions


def train_cgan(input_folder, target_folder, num_epochs=2000, batch_size=8, lr=0.0001, beta1=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Ensure all images are the same size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    ])

    dataset = PixelArtDataset(input_folder, target_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)  # Added drop_last=True

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    l1_loss = nn.L1Loss()

    g_optimizer = optim.Adam(generator.parameters(),
                             lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(beta1, 0.999))

    # Training loop with fixed dimensions
    for epoch in range(num_epochs):
        for i, (input_imgs, target_imgs) in enumerate(dataloader):
            # Ensure consistent batch size
            current_batch_size = input_imgs.size(0)
            real_label = torch.ones(current_batch_size, 1, 10, 10).to(device)
            fake_label = torch.zeros(current_batch_size, 1, 10, 10).to(device)

            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()

            d_real = discriminator(target_imgs, input_imgs)
            d_real_loss = criterion(d_real, real_label)

            fake_imgs = generator(input_imgs)
            d_fake = discriminator(fake_imgs.detach(), input_imgs)
            d_fake_loss = criterion(d_fake, fake_label)

            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            fake_imgs = generator(input_imgs)
            g_fake = discriminator(fake_imgs, input_imgs)
            g_gan_loss = criterion(g_fake, real_label)

            g_l1_loss = l1_loss(fake_imgs, target_imgs) * 100

            g_loss = g_gan_loss + g_l1_loss
            g_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

        # if (epoch + 1) % 10 == 0:
        #     torch.save({
        #         'generator_state_dict': generator.state_dict(),
        #         'discriminator_state_dict': discriminator.state_dict(),
        #         'g_optimizer_state_dict': g_optimizer.state_dict(),
        #         'd_optimizer_state_dict': d_optimizer.state_dict(),
        #     }, f'checkpoint_epoch_{epoch+1}.pth')
        # at the end of every 100 epochs, generate an image
        if (epoch + 1) % 100 == 0:
            if not os.path.exists("sprites/epochs"):
                os.makedirs("sprites/epochs")
            generated = generate_image(
                generator, "sprites/input/1.png", device)
            generated.save(f"sprites/epochs/generated_epoch_{epoch+1}.png")
        # at the last epoch, save the model
        if epoch == num_epochs - 1:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth')

    return generator, discriminator

# Generate image function (same as before)


def generate_image(generator, input_image_path, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    ])

    input_img = Image.open(input_image_path)
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = generator(input_tensor)

    generated = generated.cpu().squeeze(0)
    generated = generated * 0.5 + 0.5
    generated = transforms.ToPILImage()(generated)

    return generated


input_folder = "sprites/input"
target_folder = "sprites/output"
generator, discriminator = train_cgan(input_folder, target_folder)
