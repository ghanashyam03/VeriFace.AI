from flask import Flask, render_template, request, Response, jsonify
from flask import send_from_directory
from transformers import pipeline
from PIL import Image
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
import threading

app = Flask(__name__, static_url_path='/static')

# Initialize the image classification pipeline
image_classification_pipe = pipeline("image-classification", model="NehaBardeDUKE/autotrain-ai-generated-image-classification-3250490787")

# Load FaceNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

# Function to extract face embeddings
def is_ai_generated(image_path):
    # Perform image classification
    result = image_classification_pipe(image_path)
    artificial_score = next(item['score'] for item in result if item['label'] == 'Artificial')
    # Check if the prediction indicates AI-generated image
    if artificial_score > 0.50 :
        return True
    else:
        return False

def extract_face_embeddings(image):
    faces = mtcnn(image)
    if faces is not None:
        embeddings = resnet(faces)
        return embeddings
    else:
        return None

# Function to calculate cosine similarity between embeddings
def cosine_similarity(embedding1, embedding2):
    embedding1_np = embedding1.detach().numpy().flatten()  # Flatten the embedding
    embedding2_np = embedding2.detach().numpy().flatten()  # Flatten the embedding
    dot_product = np.dot(embedding1_np, embedding2_np)
    norm1 = np.linalg.norm(embedding1_np)
    norm2 = np.linalg.norm(embedding2_np)
    cosine_similarity = dot_product / (norm1 * norm2)
    return float(cosine_similarity)  # Convert to float

# Function to load and preprocess images from a directory
def load_images_and_extract_embeddings(directory):
    embeddings = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                embedding = extract_face_embeddings(image)
                if embedding is not None:
                    embeddings.append(embedding)
    return embeddings

# Load and preprocess embeddings of stored faces
stored_embeddings = load_images_and_extract_embeddings('D:/extra/Face_detection/ai_detected_images/stored-faces')

# Function to check if there's a matched image and return it
def check_matched_image(frame_embedding, stored_embeddings, threshold=0.6):
    for stored_embedding in stored_embeddings:
        similarity_percentage = cosine_similarity(frame_embedding, stored_embedding)
        print("Similarity:", similarity_percentage)  # Debug statement
        if similarity_percentage > threshold:
            return True, similarity_percentage  # Match found
    return False, 0  # No match found

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Generator function to get frames from the webcam
def webcam_gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detection')
def face_detection():
    return render_template('index.html')

@app.route('/detection/video_feed')
def video_feed():
    return Response(webcam_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection/detect')
def detect():
    global stored_embeddings
    match_found = False
    match_percentage = 0
    while not match_found:
        ret, frame = cap.read()
        if ret:
            # Extract face embeddings for the current frame
            frame_embedding = extract_face_embeddings(frame)
            if frame_embedding is not None:
                match_found, match_percentage = check_matched_image(frame_embedding, stored_embeddings)
                if match_found:
                    break
    return jsonify({'match_found': match_found, 'match_percentage': match_percentage})

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the POST request has the file part
        if "file" not in request.files:
            return "No file part"
        
        file = request.files["file"]
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            return "No selected file"
        
        # If the file is selected and it is an allowed type, save it
        if file:
            # Specify the upload directory
            upload_folder = 'D:/extra/Face_detection/ai_detected_images/stored-faces'
            
            # Save the file to the specified directory
            image_path = os.path.join(upload_folder, file.filename)
            file.save(image_path)
            
            # Check if the uploaded image is AI-generated
            is_generated = is_ai_generated(image_path)
            if is_generated:
                return render_template("result1.html", message="Uploaded image is AI-generated.")
            else:
                return render_template("result1.html", message="Uploaded image is not AI-generated.")
    
    return render_template("upload.html")

if __name__ == '__main__':
    app.run(debug=True)
