from PIL import Image

# Open image
img = Image.open("testImg.jpeg")

img.crop((
    0,
    0,
    16,
    16,
    )
)

# Resize smoothly down to 16x16 pixels
imgSmall = img.resize((16,16), resample=Image.Resampling.BILINEAR)

# Scale back up using NEAREST to original size
result = imgSmall.resize(img.size, Image.Resampling.NEAREST)

# Save
result.save('result.png')