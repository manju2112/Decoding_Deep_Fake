import os
import cv2
import logging

def extract_frames_from_video(video_path, output_dir, target_frames=300):
    """Extract exactly 300 frames from a video and save them to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // target_frames)
    
    count = 0
    success = True
    extracted_count = 0
    
    while success and extracted_count < target_frames:
        success, frame = cap.read()
        if success and count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        count += 1
    
    cap.release()
    logging.info(f"Extracted {extracted_count} frames from {video_path}")

def check_frame_folders(video_dir, frame_dir):
    """Check if frame folders exist for all videos; return list of videos needing extraction."""
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    frame_folders = [f for f in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, f))]
    
    video_names = [os.path.splitext(f)[0] for f in video_files]
    missing_videos = [video for video in video_files if os.path.splitext(video)[0] not in frame_folders]
    
    if len(video_names) == len(frame_folders) and not missing_videos:
        logging.info("All videos have corresponding frame folders. Skipping frame extraction.")
        return []
    else:
        logging.info(f"Found {len(missing_videos)} videos missing frame folders: {missing_videos}")
        return missing_videos

def process_videos_for_frames(video_dir, frame_dir):
    """Process videos in the video directory to extract frames, only for missing frame folders."""
    missing_videos = check_frame_folders(video_dir, frame_dir)
    
    if not missing_videos:
        logging.info("No frame extraction needed.")
        return
    
    for video_file in missing_videos:
        video_path = os.path.join(video_dir, video_file)
        output_folder = os.path.join(frame_dir, video_file.split('.')[0])
        extract_frames_from_video(video_path, output_folder)
    
    logging.info("Frame extraction completed for missing videos.")