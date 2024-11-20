from django.shortcuts import render
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# Load the trained model
model_path = os.path.join('D:/NeuralNet/best_model.keras')
model = load_model(model_path)

# Define class labels (based on your training data classes)
class_labels = [
    'Achaemenid', 'American Foursquare', 'American craftsman', 'Ancient Egyptian', 'Art Deco',
    'Art Nouveau', 'Baroque', 'Bauhaus', 'Beaux-Arts', 'Byzantine',
    'Chicago school', 'Colonial','Deconstructivism', 'Edwardian', 'Georgian',
    'Gothic', 'Greek Revival', 'International style', 'Novelty', 'Palladian',
    'Postmodern', 'Queen Anne', 'Romanesque', 'Tudor Revival'
]
# Convert class labels list to a dictionary to match indices
class_labels = {i: label for i, label in enumerate(class_labels)}

# Predefined validation metrics (or replace with real values if available)
validation_loss = 0.25
validation_accuracy = 0.87


def upload_and_predict(request):
    predicted_style = None
    prediction_probabilities = []
    image_base64 = None

    if request.method == "POST" and request.FILES.get("image"):
        # Handle the uploaded image
        image_file = request.FILES["image"]

        # Convert the uploaded image to a base64 string
        image_bytes = io.BytesIO(image_file.read())
        img = load_img(image_bytes, target_size=(150, 150))

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Preprocess the image for prediction
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_style = class_labels[predicted_index]
        prediction_probabilities = predictions[0]

        # Generate a horizontal bar chart of prediction probabilities
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(list(class_labels.values()), prediction_probabilities, color=plt.cm.tab20.colors)
        ax.set_xticklabels(list(class_labels.values()), rotation=90, ha="center", fontsize=8)
        ax.set_xlabel("Architectural Styles")
        ax.set_ylabel("Probability")
        ax.set_title(f"Prediction Probabilities for {predicted_style}")

        # Add percentage labels to each bar
        for bar, prob in zip(bars, prediction_probabilities):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{prob:.2%}', ha='center', va='bottom', fontsize=8)

        # Convert plot to base64 string to embed in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()

        # Render the template with prediction results and image as base64 string
        return render(request, "upload.html", {
            "predicted_style": predicted_style,
            "prediction_probabilities": prediction_probabilities,
            "chart_base64": chart_base64,
            "image_base64": image_base64,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })

    # Render the upload page initially
    return render(request, "upload.html")
