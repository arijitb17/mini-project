import os
import cv2
import imgaug.augmenters as iaa
import numpy as np

# Paths
INPUT_PATH = "mini-project-main/dataset"
OUTPUT_PATH = "mini-project-main/augmented-dataset"
NUM_AUGMENTATIONS = 20

# Augmentation pipeline with color transformations
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip
    iaa.Affine(
        rotate=(-10, 10),
        scale=(0.9, 1.1),
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
    ),
    iaa.GammaContrast((0.8, 1.2)),               # Contrast variation
    iaa.AdditiveGaussianNoise(scale=(5, 15)),    # Sensor noise
    iaa.GaussianBlur(sigma=(0.0, 1.0)),          # Slight blur
    iaa.Multiply((0.8, 1.2)),                    # Brightness variation

    # Color-related augmentations
    iaa.AddToHueAndSaturation((-20, 20)),        # Hue/saturation changes
    iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                       children=iaa.WithChannels(2, iaa.Multiply((0.5, 1.5)))),  # Saturation in HSV
    iaa.ChangeColorTemperature((4000, 10000)),   # Warm to cool light
    iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.1, 1.0)))  # Occasionally grayscale
])

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Process each person's folder
for person_name in os.listdir(INPUT_PATH):
    person_path = os.path.join(INPUT_PATH, person_name)
    output_person_path = os.path.join(OUTPUT_PATH, person_name)
    os.makedirs(output_person_path, exist_ok=True)

    if os.path.isdir(person_path):
        images = [img for img in os.listdir(person_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for idx, image_name in enumerate(images):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            # Save original image
            orig_save_path = os.path.join(output_person_path, f"orig_{idx}.jpg")
            cv2.imwrite(orig_save_path, image)

            # Create augmentations
            for i in range(NUM_AUGMENTATIONS):
                augmented = augmenter(image=image)
                save_path = os.path.join(output_person_path, f"aug_{idx}_{i}.jpg")
                cv2.imwrite(save_path, augmented)

print(f"Augmented dataset with color transformations saved to '{OUTPUT_PATH}'")
