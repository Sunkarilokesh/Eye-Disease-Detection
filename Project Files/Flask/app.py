from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = load_model('evgg1.h5')

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route to upload images
@app.route('/')
def index():
    return render_template('index.html')

# Route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Load the image and prepare it for prediction
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Rescale image to [0, 1]
        
        # Predict the class
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=-1)
        
        # Map prediction to class label
        index = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        predicted_label = index[pred_class[0]]
        
        # Redirect to result page with prediction
        return render_template('result.html', predicted_label=predicted_label, img_path=file_path)

    return redirect(request.url)

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Run the Flask app
    app.run(debug=True)
