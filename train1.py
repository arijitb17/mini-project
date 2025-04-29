import os
import cv2
import numpy as np
import pickle
import torch
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from insightface.app import FaceAnalysis

# Initialize ArcFace Model
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)
# print("Antelopev2 model loaded successfully!")

# Path to dataset
DATASET_PATH = "mini-project-main\dataset"
OUTPUT_FILE = "face_embeddings.pkl"

# Dictionary to store embeddings
face_db = {}
embedding_vectors = []
labels = []

# Data augmentation pipeline
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip 50% of images
    iaa.Affine(rotate=(-10, 10)),  # Rotate between -10 and 10 degrees
    iaa.GammaContrast((0.8, 1.2))  # Adjust brightness
])

# Process each person's folder
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if os.path.isdir(person_path):
        embeddings = []
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Augment and extract face embeddings
            faces = app.get(img)
            if len(faces) > 0:
                embeddings.append(faces[0].normed_embedding)

            # Generate augmented images
            aug_img = augmenter.augment_image(img)
            faces_aug = app.get(aug_img)
            if len(faces_aug) > 0:
                embeddings.append(faces_aug[0].normed_embedding)

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            face_db[person_name] = avg_embedding
            
            embedding_vectors.append(avg_embedding)
            labels.append(person_name)

# Save embeddings to file
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(face_db, f)

print(f"Training complete! Embeddings saved to {OUTPUT_FILE}")

# Convert embeddings to numpy array
embedding_vectors = np.array(embedding_vectors)

# Perform t-SNE for better visualization
tsne = TSNE(n_components=2, perplexity=min(5, len(embedding_vectors) - 1),learning_rate=50, random_state=42)

reduced_embeddings = tsne.fit_transform(embedding_vectors)

# Plot face clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='deep', s=100)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Face Embeddings Clustering using t-SNE")
plt.legend()
plt.show()
