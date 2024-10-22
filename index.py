import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path


class SpriteDataset(Dataset):
    def __init__(self, spritesheet_path):
        self.spritesheet = Image.open(spritesheet_path)
        self.num_rows = 24
        self.num_cols = 26

        # Calculate sprite dimensions
        width, height = self.spritesheet.size
        self.sprite_width = width // self.num_cols
        self.sprite_height = height // self.num_rows

        # Determine padding for square sprites
        self.max_dim = max(self.sprite_width, self.sprite_height)
        self.pad_width = (self.max_dim - self.sprite_width) // 2
        self.pad_height = (self.max_dim - self.sprite_height) // 2

        print(f"Sprite dimensions: {self.max_dim}x{self.max_dim}")

        # Extract and process sprites
        self.sprites = []
        for row in range(self.num_rows):
            row_sprites = []
            for col in range(5, self.num_cols):  # Skip first 5 poses
                sprite = self.extract_sprite(row, col)
                row_sprites.append(sprite)
            self.sprites.append(row_sprites)

        self.transform = transforms.Compose([
            transforms.Resize(64),  # Resize to fixed size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def extract_sprite(self, row, col):
        left = col * self.sprite_width
        top = row * self.sprite_height
        right = left + self.sprite_width
        bottom = top + self.sprite_height

        sprite = self.spritesheet.crop((left, top, right, bottom))

        # Pad to square if necessary
        if self.pad_width > 0 or self.pad_height > 0:
            padded = Image.new('RGB', (self.max_dim, self.max_dim), (0, 0, 0))
            padded.paste(sprite, (self.pad_width, self.pad_height))
            return padded
        return sprite

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        row_sprites = self.sprites[idx]
        # 6th pose (index 0 after skipping 5)
        input_sprite = self.transform(row_sprites[0])
        target_sprites = torch.stack(
            [self.transform(sprite) for sprite in row_sprites[1:]])
        return input_sprite, target_sprites


class Generator(nn.Module):
    def __init__(self, latent_dim, num_poses=20):
        super(Generator, self).__init__()

        # Initial dense layer
        self.fc = nn.Linear(latent_dim + 3 * 64 * 64, 4 * 4 * 512)

        # Main convolutional layers
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Output layers for each pose
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(64, 3, 4, 2, 1),
                nn.Tanh()
            ) for _ in range(num_poses)
        ])

    def forward(self, z, condition):
        batch_size = z.size(0)
        condition_flat = condition.view(batch_size, -1)
        z = torch.cat([z, condition_flat], 1)

        x = self.fc(z)
        x = x.view(batch_size, 512, 4, 4)
        x = self.main(x)

        # Generate all poses
        outputs = [layer(x) for layer in self.output_layers]
        return torch.stack(outputs, dim=1)


class Discriminator(nn.Module):
    def __init__(self, num_poses=20):
        super(Discriminator, self).__init__()

        # Calculate input channels (condition + all poses)
        input_channels = 3 * (num_poses + 1)

        self.main = nn.Sequential(
            # Input layer: 64x64 -> 32x32
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, poses, condition):
        batch_size = poses.size(0)
        # Reshape poses to (batch_size, channels * num_poses, height, width)
        poses_flat = poses.view(batch_size, -1, poses.size(-2), poses.size(-1))
        x = torch.cat([condition, poses_flat], 1)
        return self.main(x)


def train_gan(spritesheet_path, num_epochs=200, batch_size=32, latent_dim=100):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = SpriteDataset(spritesheet_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    # Initialize models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Setup optimizers
    g_optimizer = optim.Adam(generator.parameters(),
                             lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for i, (input_sprites, target_sprites) in enumerate(dataloader):
            batch_size = input_sprites.size(0)
            real_label = torch.ones(batch_size, 1, 1, 1).to(device)
            fake_label = torch.zeros(batch_size, 1, 1, 1).to(device)

            # Move data to device
            input_sprites = input_sprites.to(device)
            target_sprites = target_sprites.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()

            # Real loss
            d_real = discriminator(target_sprites, input_sprites)
            d_real_loss = criterion(d_real, real_label)

            # Fake loss
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_poses = generator(z, input_sprites)
            d_fake = discriminator(fake_poses.detach(), input_sprites)
            d_fake_loss = criterion(d_fake, fake_label)

            # Combined D loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            # Generator loss
            g_fake = discriminator(fake_poses, input_sprites)
            g_loss = criterion(g_fake, real_label)

            g_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')

    return generator, discriminator


def generate_poses(generator, input_sprite, latent_dim=100):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(next(generator.parameters()).device)
        input_sprite = input_sprite.unsqueeze(0).to(
            next(generator.parameters()).device)
        generated_poses = generator(z, input_sprite)
        return generated_poses.squeeze(0)


# Example usage
if __name__ == "__main__":
    spritesheet_path = "main.png"
    generator, discriminator = train_gan(spritesheet_path)

    # Save the trained model
    torch.save(generator.state_dict(), "sprite_generator.pth")
