
import imgaug.augmenters as iaa

# Image augmentation: Apply slight brightness, contrast, noise, scaling, and rotation adjustments
seq = iaa.Sequential([
    iaa.Multiply((0.8, 1.2)),  # Adjust brightness
    iaa.LinearContrast((0.9, 1.1)),  # Adjust contrast
    iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),  # Add noise
    iaa.Affine(
        scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
        rotate=(-3, 3),
        order=[0, 1],
        mode='reflect'
    )
])

def augment_images(images):
    return seq(images=images)
