import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Constants
H, W = 256, 256
MODEL_PATH = "unet_model.keras"  # Path to the saved model
RESULTS_DIR = "test_particular_image_results"  # Directory to save test results

# Create directory for test results if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def read_image_for_testing(image_path):
    """Preprocesses an image for testing."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    img = cv2.resize(img, (W, H)) / 255.0  # Resize and normalize
    return np.expand_dims(img.astype(np.float32), axis=(0, -1))  # Add batch and channel dimensions

def save_and_display_prediction(ori_image, y_pred, save_path):
    """Saves and displays the original and predicted images side by side."""
    # Resize and process the prediction
    y_pred = np.squeeze(y_pred)  # Remove batch and channel dimensions
    y_pred_color = np.repeat(y_pred[:, :, np.newaxis], 3, axis=-1) * 255.0  # Convert to 3-channel color for saving

    # Resize original image if necessary and prepare for saving
    ori_image_resized = cv2.resize(ori_image, (W, H)) if ori_image.shape[:2] != (H, W) else ori_image

    # Create a line for separation and concatenate images
    line = np.ones((H, 10, 3)) * 255.0  # Separator line
    result_img = np.concatenate([ori_image_resized, line, y_pred_color], axis=1)
    cv2.imwrite(save_path, result_img)

    # Display the predicted mask
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(ori_image_resized, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(y_pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.show()

# Path to the test image
test_image_path = "564.png"  # Update this path to your test image
save_image_path = os.path.join(RESULTS_DIR, "test_prediction564.jpg")  # Path to save the prediction

# Load and preprocess the image
input_image = read_image_for_testing(test_image_path)

# Predict the mask
predicted_mask = (model.predict(input_image)[0] > 0.5).astype(np.uint8)  # Binary mask

# Load original image for comparison
original_image = cv2.imread(test_image_path)

# Save and display the result with the original and predicted images side by side
save_and_display_prediction(original_image, predicted_mask, save_image_path)

print(f"Prediction saved at: {save_image_path}")
