import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from utils import squeeze_last_dim, mean_over_time, expand_last_dim, weighted_multiplication
import logging

def extract_features_from_video(video_path, target_frames=300):
    """Extract features from a video."""
    feature_extractor = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // target_frames)
    
    frame_features = []
    count = 0
    extracted_count = 0
    success = True
    
    while success and extracted_count < target_frames:
        success, frame = cap.read()
        if success and count % frame_interval == 0:
            img = cv2.resize(frame, (224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = feature_extractor.predict(img_array)
            frame_features.append(features[0])
            extracted_count += 1
        count += 1
    
    cap.release()
    
    if not frame_features:
        logging.warning(f"No frames could be extracted from {video_path}.")
        return None
    
    return np.mean(frame_features, axis=0)

def plot_grad_cam(feature_extractor, frame, layer_name='top_conv'):
    """Generate a Grad-CAM heatmap for a given frame."""
    resized_frame = cv2.resize(frame, (224, 224))
    grad_model = tf.keras.models.Model([feature_extractor.inputs], [feature_extractor.get_layer(layer_name).output, feature_extractor.output])
    img_array = preprocess_input(np.expand_dims(resized_frame, axis=0))
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.savefig('grad_cam_output.png')
    plt.close()
    logging.info("Grad-CAM visualization saved as grad_cam_output.png")

def predict_and_visualize(video_path, model_path, custom_objects):
    """Predict video class and generate Grad-CAM visualization."""
    model = load_model(model_path, custom_objects=custom_objects)
    features = extract_features_from_video(video_path)
    if features is None:
        return None, None
    
    predictions = []
    num_predictions = 5
    for _ in range(num_predictions):
        reshaped_features = features.reshape((1, 1, -1))
        prediction = model.predict(reshaped_features)[0][0]
        predictions.append(prediction)
    
    final_prediction = np.mean(predictions)
    predicted_class = "REAL" if final_prediction > 0.8 else "FAKE"
    logging.info(f"The video {os.path.basename(video_path)} is predicted as {predicted_class} with probability {final_prediction:.4f}.")
    
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        plot_grad_cam(EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg'), frame)
    cap.release()
    
    return predicted_class, final_prediction