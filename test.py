import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn


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


def create_generated_spritesheet(model_path, original_spritesheet_path, output_path="generated_spritesheet.png"):
    # Load the trained generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=100).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Load and process original spritesheet
    spritesheet = Image.open(original_spritesheet_path)
    width, height = spritesheet.size
    sprite_width = width // 26
    sprite_height = height // 24

    # Calculate dimensions for the new spritesheet
    # We'll generate 20 poses for each row (skipping first 5 poses)
    new_width = sprite_width * 21  # 1 input sprite + 20 generated poses
    new_height = height  # Same number of rows
    new_spritesheet = Image.new('RGB', (new_width, new_height))

    # Transform for input sprites
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Inverse transform for generated sprites
    inverse_transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),  # Inverse of normalize
        transforms.ToPILImage(),
        transforms.Resize((sprite_height, sprite_width))
    ])

    print("Generating poses for each row...")

    # Process each row
    for row in range(24):
        # Extract the first valid sprite (6th pose) from the row
        left = 5 * sprite_width  # Skip first 5 poses
        top = row * sprite_height
        sprite = spritesheet.crop(
            (left, top, left + sprite_width, top + sprite_height))

        # Place the input sprite in the new spritesheet
        new_spritesheet.paste(sprite, (0, row * sprite_height))

        # Prepare input for generator
        sprite_tensor = transform(sprite).unsqueeze(0).to(device)

        # Generate poses
        with torch.no_grad():
            z = torch.randn(1, 100).to(device)  # latent vector
            generated_poses = generator(z, sprite_tensor)
            generated_poses = generated_poses.squeeze(
                0)  # Remove batch dimension

            # Process and place each generated pose
            for i, pose in enumerate(generated_poses):
                # Convert generated tensor to PIL Image
                pose_image = inverse_transform(pose)

                # Place in the new spritesheet
                new_spritesheet.paste(
                    pose_image,
                    ((i + 1) * sprite_width, row * sprite_height)
                )

        print(f"Completed row {row + 1}/24")

    # Save the generated spritesheet
    new_spritesheet.save(output_path)
    print(f"Generated spritesheet saved to {output_path}")
    return new_spritesheet


def generate_specific_row(model_path, input_sprite_path, num_poses=20):
    """Generate poses for a single input sprite"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=100).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Load and process input sprite
    input_sprite = Image.open(input_sprite_path)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    sprite_tensor = transform(input_sprite).unsqueeze(0).to(device)

    # Generate poses
    with torch.no_grad():
        z = torch.randn(1, 100).to(device)
        generated_poses = generator(z, sprite_tensor)
        generated_poses = generated_poses.squeeze(0)

    # Convert to PIL images
    inverse_transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage(),
        transforms.Resize(input_sprite.size)
    ])

    generated_images = [inverse_transform(pose) for pose in generated_poses]

    # Create a combined image
    total_width = input_sprite.width * (num_poses + 1)
    combined_image = Image.new('RGB', (total_width, input_sprite.height))

    # Place input sprite
    combined_image.paste(input_sprite, (0, 0))

    # Place generated poses
    for i, img in enumerate(generated_images):
        combined_image.paste(img, ((i + 1) * input_sprite.width, 0))

    return combined_image


# Example usage
if __name__ == "__main__":
    # Generate complete spritesheet
    model_path = "sprite_generator.pth"
    original_spritesheet_path = "main.png"

    # Generate full spritesheet
    create_generated_spritesheet(
        model_path=model_path,
        original_spritesheet_path=original_spritesheet_path,
        output_path="generated_spritesheet.png"
    )

    # Or generate poses for a single sprite
    # generate_specific_row(
    #     model_path=model_path,
    #     input_sprite_path="path_to_single_sprite.png"
    # ).save("generated_row.png")
