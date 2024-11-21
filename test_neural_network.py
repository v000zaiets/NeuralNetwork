import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from data_pipeline import train_generator, validation_generator  # Ensure this import matches your structure

# Load the trained model
model = load_model('best_model.keras')

# Get class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Function to predict image and visualize with prediction probabilities
def predict_image(image_path):
    # Load and preprocess the image
    try:
        img = load_img(image_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        prediction_probs = predictions[0]  # Get the array of prediction probabilities

        # Display the image with prediction
        plt.figure(figsize=(14, 6))

        # Show the image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Predicted: {predicted_label}', fontsize=14)

        # Plot prediction probabilities
        plt.subplot(1, 2, 2)
        colors = plt.cm.get_cmap('tab20', len(class_labels))  # Use 'tab20' colormap with enough colors
        bars = plt.bar(range(len(class_labels)), prediction_probs, tick_label=list(class_labels.values()), color=colors.colors)
        plt.xticks(rotation=90, fontsize=10)
        plt.title('Prediction Probabilities', fontsize=14)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Probability', fontsize=12)

        # Add percentages above bars
        for bar, prob in zip(bars, prediction_probs):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{prob:.2%}', ha='center', va='bottom', fontsize=8)

        # Show plots
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}. Please provide a valid image path.")

# Test the function with an example image
test_image_path = 'D:/NeuralNet/image_for_testing/pyramides.jpg'  # Replace with actual image path
if os.path.exists(test_image_path):
    predict_image(test_image_path)
else:
    print(f"Test image not found at: {test_image_path}")

# Evaluate the model on the validation data (Optional)
val_loss, val_acc = model.evaluate(validation_generator)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)
