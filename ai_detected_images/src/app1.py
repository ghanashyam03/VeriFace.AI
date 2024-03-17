from flask import Flask, render_template, request
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import os

app = Flask(__name__, static_url_path='/static')

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def is_ai_generated(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=1)[0]
    
    # Check if the prediction indicates AI-generated image
    if decoded_preds[0][1] == 'digital_clock' and decoded_preds[0][2] > 0.5:
        return True
    else:
        return False

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
            upload_folder = "D:/extra/ai_generated_images/stored-faces"
            
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

if __name__ == "__main__":
    app.run(debug=True)
