import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from insightface.app import FaceAnalysis

# Initialize Face Recognition Model  
# Using InsightFace's ArcFace model ("buffalo_l") with CPU processing.  
# The model is prepared to detect and extract facial embeddings from images.  
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Define Dataset Path and Output File  
# The dataset contains multiple folders, each representing a different person.  
# Extracted face embeddings will be saved to a file for future use.  
DATASET_PATH = "dataset"
OUTPUT_FILE = "face_embeddings.pkl"

# Initialize Embeddings and Labels Storage  
# This list stores feature vectors representing facial embeddings.  
# Labels correspond to the folder names (individual identities).  
embedding_vectors = []
labels = []

# Apply Data Augmentation Techniques  
# This pipeline introduces variations such as horizontal flips,  
# small rotations, and brightness adjustments to improve robustness.  
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-10, 10)),
    iaa.GammaContrast((0.8, 1.2))
])

# Process Images in Each Folder  
# Each folder represents a different person. For each image,  
# facial embeddings are extracted and augmented variations are also processed.  
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract facial embeddings using the ArcFace model  
            faces = app.get(img)
            if len(faces) > 0:
                embedding_vectors.append(faces[0].normed_embedding)
                labels.append(person_name)

            # Generate and process augmented versions of the image  
            aug_img = augmenter.augment_image(img)
            faces_aug = app.get(aug_img)
            if len(faces_aug) > 0:
                embedding_vectors.append(faces_aug[0].normed_embedding)
                labels.append(person_name)

# Normalize and Transform Embeddings  
# Convert embeddings into a NumPy array and standardize them  
# to improve the performance of dimensionality reduction algorithms.  
embedding_vectors = np.array(embedding_vectors)
embedding_vectors = StandardScaler().fit_transform(embedding_vectors)

# Apply t-SNE for Dimensionality Reduction  
# Reduces high-dimensional face embeddings into a 2D space  
# for visualization while maintaining meaningful cluster relationships.  
tsne = TSNE(n_components=2, perplexity=min(10, len(embedding_vectors) - 1),
            learning_rate=50, random_state=42)
reduced_embeddings = tsne.fit_transform(embedding_vectors)

# Visualize Face Embedding Clusters  
# Each point represents a unique image, and colors indicate  
# different individuals based on folder names.  
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='tab10', s=80, alpha=0.8)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Face Embeddings Clustering using t-SNE")
plt.legend(title="Folders", loc="best", bbox_to_anchor=(1, 1))
plt.show()
