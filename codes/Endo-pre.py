from flask import Flask, request, render_template,send_from_directory
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


app = Flask(__name__)

# Load pre-trained model
model =keras.saving.load_model('/path-to-your-model/')

class_names = {
    0: "HGC",
    1: "LGC",
    2: "NST",
    3: "NTL"
}

def preprocess_image(image, target_size):
    image = image.convert('RGB')  # Ensure image has 3 channels
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image



def predict_image(model, image):
    prediction = model.predict(image)
    return prediction

def plot_probabilities(probabilities):
    # Generate a bar plot of the class probabilities
    classes = list(range(probabilities.shape[1]))  # Assuming class indices are 0, 1, 2, ..., n-1
    class_labels = [class_names.get(i, "Unknown") for i in classes]  # Get the class names
                                                                                     
    plt.figure(figsize=(8, 4), facecolor='#BDE8CA')
    plt.bar(class_labels, probabilities[0], width=0.3, color='green')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    plt.xticks(class_labels)  # Set x-ticks to show class names
    plt.tight_layout()
    plt.savefig('static/probabilities.png')  # Save the plot to the static folder
    plt.close()

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

                               
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image = Image.open(image_file)
            processed_image = preprocess_image(image, target_size=(224, 224))
            prediction = predict_image(model, processed_image)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_class = class_names.get(predicted_index, "Unknown")  # Translate index to class name
            # Generate the probability graph
            plot_probabilities(prediction)
            return render_template('result.html', 
                                   prediction=predicted_class,
                                   graph_url='/static/probabilities.png')
    return render_template('uploadPage.html')
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
