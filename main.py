import os
import pandas as pd
from csv_creation import create_csv, shuffle_csv, analyze_label_distribution
from frame_extraction import process_videos_for_frames
from feature_extraction import extract_features_with_labels, save_features_and_labels, load_saved_features_and_labels
from smote_and_data_prep import balance_and_prepare_data
from model import train_model
from grad_cam import predict_and_visualize
from utils import setup_logger, squeeze_last_dim, mean_over_time, expand_last_dim, weighted_multiplication
import logging

def main():
    # Define all paths
    fake_folder = r"..\dataset_avaialable_on_google_drive\Celeb-real"
    real_folder = r"..\dataset_avaialable_on_google_drive\Celeb-synthesis"
    video_dir = r"..\dataset_avaialable_on_google_drive\videos"
    frame_dir = r"..\make_a_new_folder\frames"
    csv_path = "..\avaialable_on_google_drive\video_celeb.csv"
    shuffled_csv_path = "..\avaialable_on_google_driveshuffled_file.csv"
    feature_path = r"..\dataset_avaialable_on_google_drive\features_celeb.npy"
    label_path = r"..\dataset_avaialable_on_google_drive\labels_celeb.npy"
    model_path = "best_bilstm_model_equal.h5"
    log_file = "deepfake_detection.log"
    
    # Setup logger
    setup_logger(log_file)
    
    # Define custom objects for model loading
    custom_objects = {
        'squeeze_last_dim': squeeze_last_dim,
        'mean_over_time': mean_over_time,
        'expand_last_dim': expand_last_dim,
        'weighted_multiplication': weighted_multiplication
    }
    
    # Check if trained model exists
    if os.path.exists(model_path):
        logging.info(f"Trained model found at {model_path}. Skipping to Grad-CAM visualization.")
        test_video_path = input("Enter the path to the test video file: ")
        predict_and_visualize(test_video_path, model_path, custom_objects)
        return
    
    # If no model exists, proceed with full pipeline
    logging.info("No trained model found. Executing full pipeline.")
    
    # Step 1: Create and process CSV
    create_csv(fake_folder, real_folder, csv_path)
    shuffle_csv(csv_path, shuffled_csv_path)
    analyze_label_distribution(shuffled_csv_path)
    
    # Step 2: Extract frames
    process_videos_for_frames(video_dir, frame_dir)
    
    # Step 3: Extract features
    video_labels_df = pd.read_csv(shuffled_csv_path)
    video_labels = dict(zip(video_labels_df['video_file'], video_labels_df['label']))
    features, labels = load_saved_features_and_labels(feature_path, label_path)
    
    if features is None or labels is None:
        logging.info("Extracting features as they were not found in saved files.")
        features, labels = extract_features_with_labels(frame_dir, video_labels, batch_size=32)
        save_features_and_labels(features, labels, feature_path, label_path)
    
    # Step 4: Balance and prepare data
    X_train, X_val, y_train, y_val = balance_and_prepare_data(features, labels)
    
    # Step 5: Train model
    train_model(X_train, y_train, X_val, y_val, model_path)
    
    # Step 6: Test and visualize with Grad-CAM
    test_video_path = input("Enter the path to the test video file: ")
    predict_and_visualize(test_video_path, model_path, custom_objects)

if __name__ == "__main__":
    main()