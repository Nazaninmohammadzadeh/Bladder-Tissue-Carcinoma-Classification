from flask import Flask, request, render_template,send_from_directory,url_for
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


app = Flask(__name__)

# Load pre-trained model
model =keras.saving.load_model('/The path to the model/')

class_names = {
    0: "HGC",
    1: "LGC",
    2: "NST",
    3: "NTL"
}

'''
Converts the image to RGB format, resizes it to the target dimensions, and adds an extra dimension to prepare it for model prediction. This ensures compatibility with models expecting RGB input with a specific size.
'''
def preprocess_image(image, target_size):
    image = image.convert('RGB')  # Ensure image has 3 channels
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


'''
Takes a pre-trained model and an input image, uses the model to predict the class probabilities for the image, and returns the prediction results.
'''
def predict_image(model, image):
    prediction = model.predict(image)
    return prediction

'''
Generates a bar plot that visualizes the class probabilities predicted by the model. It labels the bars with class names, sets the axis titles, and saves the plot as an image (probabilities.png) in the static folder.
'''
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

'''
The favicon route function serves the favicon.ico file from the static directory. It returns the favicon image when requested by the browser, ensuring the correct icon is displayed in the browser tab.
'''
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


'''
 The upload_predict route processes uploaded images, saves them to static/uploads, and then preprocesses the image for prediction. It uses the model to predict the class, generates a probability graph, and returns the prediction and image URLs to be rendered in result.html. For GET requests, it renders uploadPage.html.
'''                              
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            # Save the image to the static/uploads directory
            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)  # Ensure the directory exists
            image_path = os.path.join(upload_folder, image_file.filename)
            image_file.save(image_path)

            # Process the image
            image = Image.open(image_path)
            processed_image = preprocess_image(image, target_size=(224, 224))
            prediction = predict_image(model, processed_image)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_class = class_names.get(predicted_index, "Unknown")  # Translate index to class name
            
            # Generate the probability graph
            plot_probabilities(prediction)
            
            # Generate the URL for the image
            image_url = url_for('static', filename='uploads/' + image_file.filename)
            
            return render_template('result.html', 
                                   prediction=predicted_class,
                                   graph_url='/static/probabilities.png',
                                   image_url=image_url)
    return render_template('uploadPage.html')



'''
Runs the Flask app on all network interfaces (0.0.0.0) at port 5000 with debugging enabled.
'''
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
