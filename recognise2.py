import os
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from insightface.app import FaceAnalysis
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns

# Paths
TEST_FOLDER = "mini-project-main\\test-images"
OUTPUT_FOLDER = "output2"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Augmentation setup (not used here but imported)
augmenter = iaa.Sequential([
    iaa.GammaContrast((0.8, 1.2))
])

# Load trained face embeddings
with open("face_embeddings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Normalize known embeddings
for name in known_faces:
    known_faces[name] /= np.linalg.norm(known_faces[name])

# Initialize ArcFace Model
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compute inter-class similarity stats and per-person thresholds
def compute_similarity_distribution(known_faces):
    all_names = list(known_faces.keys())
    similarities = []
    per_person_thresholds = {}

    for i, name_i in enumerate(all_names):
        sim_list = []
        emb_i = known_faces[name_i]
        for j, name_j in enumerate(all_names):
            if name_i == name_j:
                continue
            emb_j = known_faces[name_j]
            sim = cosine_similarity(emb_i, emb_j)
            similarities.append(sim)
            sim_list.append(sim)

        if sim_list:
            # Use stricter 85th percentile and enforce a minimum threshold of 0.35
            per_person_thresholds[name_i] = max(np.percentile(sim_list, 85), 0.35)

    return np.array(similarities), per_person_thresholds

# Get similarity data and thresholds
all_similarities, person_thresholds = compute_similarity_distribution(known_faces)

# Visualize similarity distribution
plt.figure(figsize=(8, 5))
sns.histplot(all_similarities, bins=30, kde=True, color='skyblue')
plt.axvline(np.percentile(all_similarities, 85), color='red', linestyle='--', label='85th Percentile Threshold')
plt.title("Inter-class Cosine Similarity Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("similarity_distribution.png")
plt.show()

print("\n[Info] Similarity histogram saved as 'similarity_distribution.png'")

# Recognition process
for image_name in os.listdir(TEST_FOLDER):
    image_path = os.path.join(TEST_FOLDER, image_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Skipping {image_name}, cannot read file.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)

    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()

    recognized_names = set()
    print(f"\nProcessing {image_name}:")

    for face in faces:
        embedding = face.normed_embedding
        embedding /= np.linalg.norm(embedding)
        bbox = face.bbox.astype(int)

        best_match = "Unknown"
        best_similarity = 0.00

        for name, known_emb in known_faces.items():
            similarity = cosine_similarity(embedding, known_emb)
            if similarity > best_similarity and name not in recognized_names:
                best_match = name
                best_similarity = similarity

        # Apply per-person threshold with failsafe minimum
        if best_match != "Unknown":
            threshold = person_thresholds.get(best_match, 0.35)
            if best_similarity < threshold:
                best_match = "Unknown"
                best_similarity = 0.00

        recognized_names.add(best_match)

        # Draw box and label
        box_color = "red" if best_match == "Unknown" else "green"
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=box_color, width=5)

        text_position = (bbox[0], bbox[1] - 60)
        text_color = "red" if best_match == "Unknown" else "white"
        similarity_display = f"{best_similarity:.2f}" if best_match != "Unknown" else "0.00"
        draw.text(text_position, f"{best_match} ({similarity_display})", fill=text_color, font=font)

        print(f" - Detected: {best_match} (Similarity: {similarity_display})")

    output_path = os.path.join(OUTPUT_FOLDER, f"output_{image_name}")
    pil_img.save(output_path)
    print(f"Processed {image_name}, saved to {output_path}")

print("Recognition complete! Check the 'output' folder.")