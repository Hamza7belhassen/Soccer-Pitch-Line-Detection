import cv2
import numpy as np
import os
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('/Desktop/cnn/models/')  # Load the entire directory


# Preprocess function for new images
def preprocess_image(image):
    image = cv2.resize(image, (960, 560))  # Resize to match the input size of the model
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension


# Directory containing the new images
path = '/data/calibration-2023/challenge/challenge'

# Iterate through each file in the directory
for file in os.listdir(path)[:10]:  # Process only the first 10 files for demonstration
    # Read the image using OpenCV
    img = cv2.imread(os.path.join(path, file))

    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Perform prediction
    prediction = model.predict(preprocessed_img)

    # Visualize the original image and the predicted mask
    cv2.imshow('Original Image', img)
    cv2.imshow('Predicted Mask',
               prediction[0] * 255)  # Assuming prediction is a probability map, multiply by 255 for visualization
    cv2.waitKey(0)
    cv2.destroyAllWindows()
