import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Define Paths for Input and Output  
# The `test-images` folder contains images to be processed.  
# The `output` folder stores the annotated images after face recognition.  
TEST_FOLDER = "test-images"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load Stored Face Embeddings  
# Precomputed face embeddings are loaded from a pickle file.  
# The embeddings are normalized to unit vectors to allow cosine similarity comparisons.  
with open("face_embeddings.pkl", "rb") as f:
    known_faces = pickle.load(f)

for name in known_faces:
    known_faces[name] /= np.linalg.norm(known_faces[name])  # Normalize vectors

# Initialize Face Recognition Model  
# The InsightFace ArcFace model (`buffalo_l`) is used for face detection and feature extraction.  
# The model runs on the CPU for compatibility.  
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Define Function for Cosine Similarity Calculation  
# Cosine similarity measures the similarity between two face embeddings.  
# The score ranges from -1 to 1, where 1 means identical faces.  
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Process Each Image in the Test Folder  
# For each image, faces are detected, embeddings are extracted, and they are  
# compared against stored embeddings to identify known individuals.  
for image_name in os.listdir(TEST_FOLDER):
    image_path = os.path.join(TEST_FOLDER, image_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Skipping {image_name}, cannot read file.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)

    # Convert Image to PIL Format for Annotation  
    # The image is converted from OpenCV format to PIL to allow easier drawing of bounding boxes and text.  
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Load Font for Name Labels  
    # A clear font (Arial) is used to display recognized names. If unavailable, a default font is used.  
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()

    recognized_names = set()  # Track recognized names to prevent duplicate annotations

    print(f"\nProcessing {image_name}:")
    for face in faces:
        embedding = face.normed_embedding
        embedding /= np.linalg.norm(embedding)  # Normalize embedding vector
        bbox = face.bbox.astype(int)

        # Compare with Stored Embeddings to Identify the Person  
        # Each detected face is compared with all stored embeddings using cosine similarity.  
        best_match = "Unknown"
        best_similarity = 0.00  

        for name, known_emb in known_faces.items():
            similarity = cosine_similarity(embedding, known_emb)

            # Accept match if similarity is above 0.55 and the person is not already detected  
            if similarity > best_similarity and similarity > 0.55 and name not in recognized_names:
                best_match = name.capitalize()
                best_similarity = similarity

        recognized_names.add(best_match)  # Prevent multiple detections of the same person

        # Draw Bounding Box and Name Label  
        # Faces are highlighted with bounding boxes, and their names are displayed above them.  
        box_color = "red" if best_match == "Unknown" else "green"
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=box_color, width=5)

        # Define Text Position and Color  
        text_position = (bbox[0], bbox[1] - 60)
        text_color = "red" if best_match == "Unknown" else "white"

        # Ensure similarity is displayed as 0.00 for unknown faces  
        similarity_display = f"{best_similarity:.2f}" if best_match != "Unknown" else "0.00"

        # Draw the Recognized Name and Similarity Score on the Image  
        draw.text(text_position, f"{best_match} ({similarity_display})", fill=text_color, font=font)

        # Print Recognition Result in Console  
        print(f" - Detected: {best_match} (Similarity: {similarity_display})")

    # Save the Annotated Image  
    # The processed image is saved in the `output` folder with the same filename.  
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{image_name}")
    pil_img.save(output_path)
    print(f"Processed {image_name}, saved to {output_path}")

print("Recognition complete! Check the 'output' folder.")
