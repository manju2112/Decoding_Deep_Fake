import os
import pandas as pd
import numpy as np
import logging

def create_csv(fake_folder, real_folder, csv_path):
    """Generate a CSV file with video filenames and labels."""
    video_names = []
    labels = []
    
    # Process fake videos
    for video in os.listdir(fake_folder):
        if video.endswith(('.mp4', '.avi', '.mov')):
            video_names.append(video)
            labels.append(0)  # Label for fake videos
    
    # Process real videos
    for video in os.listdir(real_folder):
        if video.endswith(('.mp4', '.avi', '.mov')):
            video_names.append(video)
            labels.append(1)  # Label for real videos
    
    # Create DataFrame
    df = pd.DataFrame({
        'video_file': video_names,
        'label': labels
    })
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    logging.info(f'CSV file created at: {csv_path}')
    return df

def shuffle_csv(csv_path, shuffled_csv_path):
    """Shuffle the CSV file."""
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(shuffled_csv_path, index=False)
    logging.info(f'Shuffled CSV saved at: {shuffled_csv_path}')
    return df

def analyze_label_distribution(csv_path):
    """Analyze label distribution and check for imbalance."""
    df = pd.read_csv(csv_path)
    if 'label' in df.columns:
        label_distribution = df['label'].value_counts()
        logging.info("Label distribution:\n" + str(label_distribution))
        
        min_count = label_distribution.min()
        max_count = label_distribution.max()
        if max_count / min_count > 1.5:
            logging.warning("The dataset is imbalanced.")
        else:
            logging.info("The dataset appears to be balanced.")
    else:
        logging.error("'label' column not found in the file.")