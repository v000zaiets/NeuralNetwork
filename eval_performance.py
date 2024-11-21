import tensorflow as tf
import numpy as np
from data_pipeline import validation_generator
from tensorflow.keras.models import load_model

# Limit TensorFlow to 4 CPU cores to reduce load
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Load the trained model
model = load_model('best_model.keras')

# Evaluate model on validation data
val_loss, val_acc = model.evaluate(validation_generator)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)

# Predictions on validation data
predictions = model.predict(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print("Test Accuracy:", accuracy)
