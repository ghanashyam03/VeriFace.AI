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
import firebase_admin
from firebase_admin import credentials, firestore

from transformers import pipeline
import csv
from PIL import Image


pipe = pipeline('image-classification', model=r"C:\Users\DELL\Desktop\Face_detection\model", device=-1)


app = Flask(__name__, static_url_path='/static')

cred = credentials.Certificate(r"C:\Users\DELL\Desktop\Face_detection\facedec-954c6-firebase-adminsdk-gd325-59f0f85612.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def is_ai_generated(image_path):
    


    image = Image.open(image_path)


    image_data_type = type(image)


    print("Data type of the image:", image_data_type)

    a =pipe(image)
    for item in a:

        if item['label'] == 'FAKE':
            fake_score = item['score']
        print(a)

    if fake_score is not None and fake_score < 0.85:
        print("The image is not ai generates")
        return False
    else:
        print("This is ai generated image")
        return True

    
    
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

import numpy as np

def cosine_similarity(embedding1, embedding2):
    embedding1_np = embedding1.detach().numpy().flatten()
    embedding2_np = embedding2.detach().numpy().flatten()
    
    # Reshape embeddings to a common dimension (e.g., 512)
    common_dimension = 512
    embedding1_np = np.resize(embedding1_np, common_dimension)
    embedding2_np = np.resize(embedding2_np, common_dimension)
    
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

stored_embeddings, stored = load_images_and_extract_embeddings(r'C:\Users\DELL\Desktop\Face_detection\ai_detected_images\stored-faces')



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
    return render_template('login.html')

@app.route('/home')
def dashboard():
    return render_template('home.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/detection')
def face_detection():
    return render_template('index.html')

@app.route('/detection/video_feed')
def video_feed():
    return Response(webcam_gen(),
mimetype='multipart/x-mixed-replace; boundary=frame')


csv_file_path = r'C:\Users\DELL\Desktop\Face_detection\ai_detected_images\uploaded_images.csv'
def get_image_details(filename):
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) >= 4 and row[-1] == filename:
                print(row) 
                return {'name': row[0], 'age': row[1], 'profession': row[2], 'gender': row[3]}
    return None

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
                    # Get details of the matched image from CSV file
                    image_details = get_image_details(matched_filename)
                    if image_details:
                        return jsonify({'match_found': match_found, 'match_percentage': match_percentage, 'details': image_details})
                    else:
                        return jsonify({'error': 'Details not found for the matched image.'})



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
            upload_folder = r'C:\Users\DELL\Desktop\Face_detection\ai_detected_images\stored-faces'

            # Save the file to the specified directory
            image_path = os.path.join(upload_folder, file.filename)
            file.save(image_path)

            # Extract additional information from the form
            name = request.form.get('name')
            age = request.form.get('age')
            profession = request.form.get('profession')
            gender = request.form.get('gender')

            # Check if the uploaded image is AI-generated
            is_generated = is_ai_generated(image_path)

            # Save data to CSV file
            with open(csv_file_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                filename_without_ext = os.path.splitext(file.filename)[0]
                # Write a row with the collected data
                csv_writer.writerow([name, age, profession, gender, filename_without_ext])


            if is_generated:
                return render_template("result1.html", message="Uploaded image is AI-generated.")
            else:
                return render_template("result1.html", message="Uploaded image is not AI-generated.")

    return render_template("upload.html")

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')

    # Check login credentials in Firebase (dummy implementation)
    # Replace this with your actual Firebase authentication logic
    user_ref = db.collection('users').document(username)
    user_data = user_ref.get().to_dict()

    if user_data and user_data['password'] == password:
        return redirect(url_for('dashboard'))  # Redirect to home page after successful login
    else:
        return 'Invalid username or password'

@app.route('/signup', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    password = request.form.get('password')

    # Save user data to Firebase (dummy implementation)
    # Replace this with your actual Firebase authentication logic
    user_ref = db.collection('users').document(username)
    user_ref.set({
        'username': username,
        'password': password
    })

    return redirect(url_for('dashboard'))  # Redirect to home page after successful signup

if __name__ == '__main__':
    app.run(debug=True)
