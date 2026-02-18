import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd
import logging

def extract_features_with_labels(frame_dir, video_labels, batch_size):
    """Extract features from frames using EfficientNetV2B0."""
    model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    features = []
    labels = []
    
    for video_file, label in video_labels.items():
        video_folder = video_file.split('.')[0]
        video_path = os.path.join(frame_dir, video_folder)
        
        if not os.path.exists(video_path):
            logging.warning(f'No frames found for video: {video_folder}')
            continue
        
        video_features = []
        frame_files = sorted(os.listdir(video_path))
        for i in range(0, len(frame_files), batch_size):
            batch_files = frame_files[i:i + batch_size]
            batch_images = []
            
            for frame_file in batch_files:
                frame_path = os.path.join(video_path, frame_file)
                img = image.load_img(frame_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                batch_images.append(img_array)
            
            batch_images = np.vstack(batch_images)
            batch_features = model.predict(batch_images)
            video_features.extend(batch_features)
        
        if video_features:
            mean_features = np.mean(video_features, axis=0)
            features.append(mean_features)
            labels.append(label)
    
    features = np.array(features)
    labels = np.array(labels)
    logging.info(f"Extracted features shape: {features.shape}, labels shape: {labels.shape}")
    return features, labels

def save_features_and_labels(features, labels, feature_path, label_path):
    """Save features and labels to .npy files."""
    np.save(feature_path, features)
    np.save(label_path, labels)
    logging.info(f"Features saved to {feature_path}")
    logging.info(f"Labels saved to {label_path}")

def load_saved_features_and_labels(feature_path, label_path):
    """Load features and labels from .npy files if they exist."""
    if os.path.exists(feature_path) and os.path.exists(label_path):
        features = np.load(feature_path)
        labels = np.load(label_path)
        logging.info("Loaded saved features and labels.")
        return features, labels
    return None, None