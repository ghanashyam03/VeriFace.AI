import requests
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from flask import send_from_directory
from PIL import Image
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
import threading

app = Flask(__name__, static_url_path='/static')

def is_ai_generated(image_path):
    API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
    headers = {"Authorization": "Bearer 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"}

    with open(image_path, "rb") as f:
        data = f.read()
    
    response = requests.post(API_URL, headers=headers, data=data)
    
    try:
        result = response.json()
    except ValueError:
        print("Error parsing JSON response")
        return False  # Handle the error appropriately
    
    print("API Response:", result) 
    
    if isinstance(result, list):  # Check if result is a list
        for item in result:
            if isinstance(item, dict):  # Check if item is a dictionary
                if item.get('label') == 'artificial' and item.get('score', 0) > 0.50:
                    return True
            else:
                print("Invalid item format:", item)
    else:
        print("Invalid result format:", result)
    
    return False



device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

def extract_face_embeddings(image):
    faces = mtcnn(image)
    if faces is not None:
        embeddings = resnet(faces)
        return embeddings
    else:
        return None

def cosine_similarity(embedding1, embedding2):
    embedding1_np = embedding1.detach().numpy().flatten()
    embedding2_np = embedding2.detach().numpy().flatten()
    dot_product = np.dot(embedding1_np, embedding2_np)
    norm1 = np.linalg.norm(embedding1_np)
    norm2 = np.linalg.norm(embedding2_np)
    cosine_similarity = dot_product / (norm1 * norm2)
    return float(cosine_similarity)

def load_images_and_extract_embeddings(directory):
    embeddings = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                embedding = extract_face_embeddings(image)
                if embedding is not None:
                    embeddings.append(embedding)
                    filenames.append(filename)
    return embeddings, filenames

stored_embeddings, stored = load_images_and_extract_embeddings(r'C:\Users\hp\OneDrive\Desktop\detetction\ai_detected_images\stored-faces')



def check_matched_image(frame_embedding, stored_embeddings, stored, threshold=0.6):
    for stored_embedding, filename in zip(stored_embeddings, stored):
        similarity_percentage = cosine_similarity(frame_embedding, stored_embedding)
        if similarity_percentage > threshold:
            # Get the filename without extension
            filename_without_extension = os.path.splitext(filename)[0]
            print(filename_without_extension)
            return True, similarity_percentage, filename_without_extension
    return False, 0, None


cap = cv2.VideoCapture(0)

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
    match_found = False                    
    match_percentage = 0
    matched_filename = None
    while not match_found:
        ret, frame = cap.read()
        if ret:
            frame_embedding = extract_face_embeddings(frame)
            if frame_embedding is not None:
                match_found, match_percentage, matched_filename = check_matched_image(frame_embedding, stored_embeddings, stored)
                if match_found:
                    break
    return jsonify({'match_found': match_found, 'match_percentage': match_percentage, 'matched_filename': matched_filename})

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
            upload_folder = r'C:\Users\hp\OneDrive\Desktop\detetction\ai_detected_images\stored-faces'
            
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