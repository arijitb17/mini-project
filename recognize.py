import os
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Paths
TEST_FOLDER = "test-images"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load trained face embeddings
with open("face_embeddings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Convert known embeddings to unit vectors (normalize for cosine similarity)
for name in known_faces:
    known_faces[name] /= np.linalg.norm(known_faces[name])

# Initialize ArcFace Model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Loop through all test images
for image_name in os.listdir(TEST_FOLDER):
    image_path = os.path.join(TEST_FOLDER, image_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Skipping {image_name}, cannot read file.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)

    # Convert image to PIL for annotation
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", 50)  # Larger, clearer font
    except:
        font = ImageFont.load_default()

    recognized_names = set()  # Track names to avoid duplicates

    print(f"\nProcessing {image_name}:")
    for face in faces:
        embedding = face.normed_embedding
        embedding /= np.linalg.norm(embedding)  # Normalize for cosine similarity
        bbox = face.bbox.astype(int)

        # Compare with stored embeddings using cosine similarity
        best_match = "Unknown"
        best_similarity = 0.00  # Default for unknown

        for name, known_emb in known_faces.items():
            similarity = cosine_similarity(embedding, known_emb)
            if similarity > best_similarity and similarity > 0.55 and name not in recognized_names:
                best_match = name
                best_similarity = similarity

        recognized_names.add(best_match)  # Add to set to prevent repeats

        # Draw bounding box & name label
        box_color = "red" if best_match == "Unknown" else "green"
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=box_color, width=5)

        # Adjust text position
        text_position = (bbox[0], bbox[1] - 60)
        text_color = "red" if best_match == "Unknown" else "white"
        
        # Ensure 0.00 for unknowns
        similarity_display = f"{best_similarity:.2f}" if best_match != "Unknown" else "0.00"
        
        draw.text(text_position, f"{best_match} ({similarity_display})", fill=text_color, font=font)

        # Print to console with adjusted similarity score
        print(f" - Detected: {best_match} (Similarity: {similarity_display})")

    # Save the processed image
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{image_name}")
    pil_img.save(output_path)
    print(f"Processed {image_name}, saved to {output_path}")

print("Recognition complete! Check the 'output' folder.")
