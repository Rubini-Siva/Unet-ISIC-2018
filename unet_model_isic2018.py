import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

# Set logging level to reduce verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Define constants
H, W = 256, 256
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS =    1
DATASET_PATH = r"F:\Final Year Research Project\DeepMedNet Medical Image Segmentation\Dataset\ISIC 2018 Dataset"  # Updated to your dataset path
MODEL_PATH = "unet_model.keras"
CSV_PATH = "unet_data.csv"
RESULTS_DIR = "unet_results"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "predicted_mask"), exist_ok=True)

def load_data(dataset_path, split=0.2):
    images = sorted(glob(os.path.join(dataset_path, "Training Data", "Preprocessed Images", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "Training Groundtruth", "Preprocessed Label", "*.png")))
    
    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=split, random_state=42)
    return (train_x, train_y), (valid_x, valid_y)

""" def load_data(train_images_path, train_masks_path, valid_images_path, valid_masks_path):
    # Load all training images and masks
    train_images = sorted(glob(os.path.join(train_images_path, "*.jpg")))
    train_masks = sorted(glob(os.path.join(train_masks_path, "*.png")))  # Ensure the correct file pattern for masks
    
    # Load all validation images and masks
    valid_images = sorted(glob(os.path.join(valid_images_path, "*.jpg")))
    valid_masks = sorted(glob(os.path.join(valid_masks_path, "*.png")))  # Ensure the correct file pattern for masks
    
    # Ensure that the number of training images and masks match
    if len(train_images) != len(train_masks):
        raise ValueError(f"Number of training images ({len(train_images)}) does not match number of training masks ({len(train_masks)})")
    
    # Ensure that the number of validation images and masks match
    if len(valid_images) != len(valid_masks):
        raise ValueError(f"Number of validation images ({len(valid_images)}) does not match number of validation masks ({len(valid_masks)})")
    
    return (train_images, train_masks), (valid_images, valid_masks) """

def read_image(path):
    # Directly decode the byte string to a UTF-8 string
    path = path.decode('utf-8')  
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if x is None:
        raise ValueError(f"Image not found at path: {path}")  # Error handling for missing image
    x = cv2.resize(x, (W, H)) / 255.0
    return np.expand_dims(x.astype(np.float32), axis=-1)  # Keep it as a single channel

def read_test_image(path):
    # Directly decode the byte string to a UTF-8 string
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if x is None:
        raise ValueError(f"Image not found at path: {path}")  # Error handling for missing image
    x = cv2.resize(x, (W, H)) / 255.0
    return np.expand_dims(x.astype(np.float32), axis=-1)  # Keep it as a single channel

def read_mask(path):
    # Directly decode the byte string to a UTF-8 string
    path = path.decode('utf-8')  
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if x is None:
        raise ValueError(f"Mask not found at path: {path}")  # Error handling for missing mask
    x = cv2.resize(x, (W, H)) / 255.0
    return np.expand_dims(x.astype(np.float32), axis=-1)  # Keep it as a single channel

def tf_parse(x, y):
    x = tf.numpy_function(read_image, [x], tf.float32)
    y = tf.numpy_function(read_mask, [y], tf.float32)
    x.set_shape([H, W, 1])  # Update shape for grayscale input
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    return dataset

def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = tf.keras.layers.MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = tf.keras.Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    return tf.keras.Model(inputs, outputs)

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return 2. * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def save_result(ori_x, ori_y, y_pred, save_image_path):
    # Squeeze out the batch dimension from y_pred if it exists
    y_pred = np.squeeze(y_pred)  # Remove extra dimensions if present
    y_pred = np.repeat(y_pred[:, :, np.newaxis], 3, axis=-1) * 255.0  # (H, W, 3)
    # Resize ori_x if necessary and repeat ori_y to 3 channels
    ori_x = cv2.resize(ori_x, (W, H)) if ori_x.shape[:2] != (H, W) else ori_x
    ori_y = np.repeat(ori_y[:, :, np.newaxis], 3, axis=-1) * 255.0  # Convert to (H, W, 3)
    # Create a separator line for clarity in saved image
    line = np.ones((H, 10, 3)) * 255.0
    # Concatenate images
    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def save_predicted(y_pred, save_image_path):
    y_pred = np.squeeze(y_pred)  # Remove extra dimensions if present
    y_pred = np.repeat(y_pred[:, :, np.newaxis], 3, axis=-1) * 255.0  # (H, W, 3)
    cv2.imwrite(save_image_path, y_pred * 255)  # Save the predicted mask as an image (scale to 255)


def compute_assd(y_true, y_pred):
    # Find the coordinates of the surface points for both ground truth and prediction
    gt_coords = np.argwhere(y_true > 0.5)  # Surface points for ground truth
    pred_coords = np.argwhere(y_pred > 0.5)  # Surface points for prediction

    if len(gt_coords) == 0 and len(pred_coords) == 0:
        return 0.0  # Both are empty
    elif len(gt_coords) == 0:
        return np.mean([np.linalg.norm(p - gt_coords) for p in pred_coords])  # Only prediction has surface points
    elif len(pred_coords) == 0:
        return np.mean([np.linalg.norm(gt - p) for gt in gt_coords])  # Only ground truth has surface points

    # Compute distances from ground truth to prediction
    distances_gt_to_pred = np.array([np.min([np.linalg.norm(gt - p) for p in pred_coords]) for gt in gt_coords])
    
    # Compute distances from prediction to ground truth
    distances_pred_to_gt = np.array([np.min([np.linalg.norm(p - gt) for gt in gt_coords]) for p in pred_coords])

    # Calculate ASSD
    assd = np.mean(distances_gt_to_pred) + np.mean(distances_pred_to_gt)
    return assd / 2.0  # Average of both distances

# Load data
(train_x, train_y), (valid_x, valid_y) = load_data(DATASET_PATH)

# Create datasets
train_dataset = tf_dataset(train_x, train_y, BATCH_SIZE)
valid_dataset = tf_dataset(valid_x, valid_y, BATCH_SIZE)

# Build and compile model
model = build_unet((H, W, 1))  # Change from (H, W, 3) to (H, W, 1)
model.compile(loss="binary_crossentropy", optimizer=Adam(LEARNING_RATE), metrics=[Recall(), Precision()])

# Callbacks
callbacks = [
    ModelCheckpoint(MODEL_PATH, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
    CSVLogger(CSV_PATH),
    EarlyStopping(monitor='val_loss', patience=20)
]

# Train model
model.fit(train_dataset, validation_data=valid_dataset, epochs=NUM_EPOCHS, callbacks=callbacks)

# Calculate number of parameters and FLOPS
num_params = model.count_params()
print(f"Number of parameters: {num_params}")

# Evaluate model
SCORE = []
for x, y in tqdm(zip(valid_x, valid_y), total=len(valid_x)):
    name = os.path.basename(x)
    ori_x = cv2.imread(x)
    ori_y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    ori_y = cv2.resize(ori_y, (W, H)) / 255.0
    x_input = np.expand_dims(read_test_image(x), axis=0)  # Add batch dimension

    y_pred = (model.predict(x_input)[0] > 0.5).astype(np.int32)  # Convert to binary

    # Save results
    save_image_path = os.path.join(RESULTS_DIR, name)
    save_predicted_path = os.path.join(RESULTS_DIR, "predicted_mask", name)  # Save predicted mask
    save_result(ori_x, ori_y, y_pred, save_image_path)
    save_predicted(y_pred, save_predicted_path)

    # Calculate metrics
    y_flat = ori_y.flatten().astype(np.int32)
    y_pred_flat = y_pred.flatten().astype(np.int32)
    acc_value = accuracy_score(y_flat, y_pred_flat)
    f1_value = f1_score(y_flat, y_pred_flat, average="binary")
    jac_value = jaccard_score(y_flat, y_pred_flat, average="binary")
    recall_value = recall_score(y_flat, y_pred_flat, average="binary")
    precision_value = precision_score(y_flat, y_pred_flat, average="binary")
    dice_value = dice_coefficient(y_flat, y_pred_flat)

    # Calculate ASSD
    #assd_value = compute_assd(ori_y, y_pred)

    SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value, dice_value])

# Metrics values
score = np.mean([s[1:] for s in SCORE], axis=0)
print(f"Accuracy        : {score[0]:0.5f}")
print(f"F1              : {score[1]:0.5f}")
print(f"Jaccard Index   : {score[2]:0.5f}")
print(f"Recall          : {score[3]:0.5f}")
print(f"Precision       : {score[4]:0.5f}")
print(f"Dice Similarity : {score[5]:0.5f}")

# Save all the results
df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision", "Dice"])
df.to_csv("unet_metrics.csv", index=False)