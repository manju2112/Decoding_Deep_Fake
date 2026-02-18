# Decoding Deepfakes

## Overview

Decoding Deepfakes is a project designed to detect deepfake videos using a deep learning pipeline. It leverages EfficientNetV2B0 for feature extraction, a Bidirectional LSTM with an attention mechanism for classification, and Grad-CAM for visualizing important regions in video frames. The project processes videos from the Celeb-DF dataset, extracts features, balances the dataset, trains a model, and predicts whether a video is real or fake.

## What the Code Does

The codebase is modularized into several Python scripts that form a complete pipeline for deepfake detection:


    1. CSV Creation: Generates a CSV file (video_celeb.csv) with video filenames and labels (0 for fake, 1 for real), shuffles it (shuffled_file.csv), and analyzes label distribution for imbalance.
    2. Frame Extraction: Extracts 300 frames per video, saving them as JPEGs. Skips extraction if frames already exist for all videos.
    3. Feature Extraction: Uses EfficientNetV2B0 to extract features from frames, computes mean features per video, and saves them as .npy files.
    4. Data Preparation: Applies SMOTE to balance the dataset, reshapes features for LSTM input, and performs a stratified train-test split.
    5. Model Training: Trains a Bidirectional LSTM with an attention mechanism, tunes hyperparameters using Keras Tuner, and saves the model.
    6. Grad-CAM Testing: Predicts whether a new video is fake or real and generates a Grad-CAM heatmap to visualize key frame regions.
    7. Utilities: Includes logging and custom TensorFlow functions for the model and Grad-CAM.

## How It’s Done

    1. Data Preparation: Videos from Celeb-real and Celeb-synthesis folders are listed in a CSV with labels. The dataset is shuffled and checked for imbalance.
    2. Frame Extraction: 300 frames are extracted per video at regular intervals, saved in subfolders, and skipped if already extracted.
    3. Feature Extraction: EfficientNetV2B0 processes frames in batches, computing 1280-dimensional mean feature vectors per video.
    4. Data Balancing: SMOTE oversamples the minority class to balance fake and real videos.
    5. Model Training: A Bidirectional LSTM with attention is trained on resampled data, optimized via Keras Tuner, and evaluated with accuracy and loss curves.
    6. Testing and Visualization: A new video’s features are extracted, classified (fake/real), and visualized using Grad-CAM to highlight regions influencing the prediction.

## Dataset

The project uses the Celeb-DF dataset, structured as follows:

    1. Celeb-real: Folder containing real videos.
    2. Celeb-synthesis: Folder containing synthetic (deepfake) videos.
    3. videos: Folder combining all videos from Celeb-real and Celeb-synthesis.
    4. CSV Files: video_celeb.csv (original) and shuffled_file.csv (shuffled).
    4. Saved Model: Trained model (best_bilstm_model_equal.h5) and feature files (features_celeb.npy, labels_celeb.npy).
All dataset files and the saved model are available on Google Drive: [[Decoding Deepfakes dataset].](https://drive.google.com/drive/folders/1E3SQ0XxLcQ4Ur0drg-yqa6XkHPJO98Ly?usp=sharing)

## Results

### Model Performance

The model was trained for 50 epochs, achieving the following performance on the validation set (318 samples):
<img width="335" alt="Classification_report" src="https://github.com/user-attachments/assets/073b6a0f-1b8b-457f-a7e8-f413be44423a" />



### Training Curves

The training and validation accuracy/loss curves are saved as training_curves.png in the project directory (also available on Google Drive):
![training_curves](https://github.com/user-attachments/assets/04a96f8a-072e-48ce-a988-4bddf63442e1)

    1. Training accuracy improved from ~0.70 to ~0.95.
    2. Validation accuracy fluctuated but stabilized around 0.90.
    3. Training loss decreased from ~0.9 to ~0.2.
    4. Validation loss decreased from ~0.8 to ~0.4, with some fluctuations.

### Grad-CAM Visualization

A Grad-CAM heatmap for a test video frame is saved as grad_cam_output.png (also on Google Drive). The heatmap highlights the face region as the most influential area for the model’s prediction, using a jet colormap (red for high importance, blue for low).

![grad_cam_output](https://github.com/user-attachments/assets/1c64fafa-a2ce-426c-844b-697adc850a4f)


## How to Execute the Code

### Prerequisites


    Python 3.8 or higher.
    Install dependencies listed in requirements.txt.

#### Setup

Clone or download the repository.

    Create a virtual environment (optional but recommended):


Install dependencies:

    pip install -r requirements.txt



Update paths in main.py to match your local setup (e.g., fake_folder, video_dir, model_path).

#### Directory Structure

    Decoding_Deepfakes/
    ├── main.py
    ├── csv_creation.py
    ├── frame_extraction.py
    ├── feature_extraction.py
    ├── smote_and_data_prep.py
    ├── model.py
    ├── grad_cam.py
    ├── utils.py
    ├── setup.py
    ├── requirements.txt
    ├── README.md

Running the Pipeline





#### Run the entire pipeline:

    python main.py



#### The pipeline will:

    1. Create and shuffle the CSV.
    2. Extract frames (only for videos missing frame folders).
    3. Extract or load features.
    4. Balance data and train the model.
    5. Prompt for a test video path and generate a Grad-CAM visualization.
    6. Check deepfake_detection.log for logs and training_curves.png or grad_cam_output.png for visualizations.
    

## Notes

    
    Ensure the dataset folders (Celeb-real, Celeb-synthesis, videos) and saved model are accessible.
    For testing a single video, provide its path when prompted by main.py.
