import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data directory and parameters
data_dir = "D:/NeuralNet/data"
img_size = (150, 150)
batch_size = 32

# Data augmentation and preprocessing techniques for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Feature scaling (normalize pixel values)
    rotation_range=20,          # Random rotation (up to 20 degrees)
    width_shift_range=0.2,      # Random width shift (up to 20% of image width)
    height_shift_range=0.2,     # Random height shift (up to 20% of image height)
    shear_range=0.2,            # Shear transformation (up to 20%)
    zoom_range=0.2,             # Random zoom (up to 20%)
    horizontal_flip=True,       # Random horizontal flip
    fill_mode='nearest',        # Fill mode for filling in newly created pixels
    validation_split=0.2        # Create a validation split
)

# Load training data with data augmentation
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True  # Shuffle the training data
)

# Data normalization and preprocessing for validation data
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Use validation subset of data
    shuffle=False         # Do not shuffle the validation data
)
