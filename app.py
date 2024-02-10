from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the saved ResNet50 model
loaded_resnet_model = load_model('model\\best_resnet_model.h5')

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array

# Function to predict the class of an image
def predict_class(image_path, model):
    classes=['battery', 'biological','brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper','plastic','shoes','trash','white-glass']
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]
    if predicted_class in {'biological','cardboard','clothes','paper','shoes','trash'}:
        res="Biodegradable"
    else:
        res="Non-Biodegradable"
    return predicted_class, res

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_path = 'static\\uploads\\' + file.filename
        file.save(image_path)
        predicted_class, res = predict_class(image_path, loaded_resnet_model)
        return jsonify({'predicted_class': predicted_class, 'biodegradability': res})
        

if __name__ == '__main__':
    app.run(debug=True)
