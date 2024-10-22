import os
from PIL import Image

# load a spritesheet
PATH = "main.png"

n, m = 24, 26
spritesheet = Image.open(PATH)
width, height = spritesheet.size
sprite_width = width // m
sprite_height = height // n

# calculate padding to be square and multiple of 8
max_dim = max(sprite_width, sprite_height)
extra_padding = max_dim % 8
new_w, new_h = max_dim + extra_padding, max_dim + extra_padding

INPUT_COL, OUTPUT_COL = 5, 25

# create output directory
if not os.path.exists("sprites"):
    os.makedirs("sprites")
if not os.path.exists("sprites/input"):
    os.makedirs("sprites/input")
if not os.path.exists("sprites/output"):
    os.makedirs("sprites/output")

# extract and process sprites
sprites = []
for row in range(n):
    row_sprites = []
    for col in [INPUT_COL, OUTPUT_COL]:
        x0, y0 = col * sprite_width, row * sprite_height
        x1, y1 = x0 + sprite_width, y0 + sprite_height
        sprite = spritesheet.crop((x0, y0, x1, y1))
        # pad as a square, evenly on all sides
        pad_width = (new_w - sprite.width) // 2
        pad_height = (new_h - sprite.height) // 2
        sprite_temp = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
        sprite_temp.paste(sprite, (pad_width, pad_height))
        sprite = sprite_temp
        row_sprites.append(sprite)
        # save sprite
        if col == OUTPUT_COL:
            sprite.save(f"sprites/output/{row}.png")
        else:
            sprite.save(f"sprites/input/{row}.png")
    sprites.append(row_sprites)
