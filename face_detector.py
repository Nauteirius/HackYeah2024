import cv2
import argparse
import pickle
import numpy as np
from deepface import DeepFace  # Import the DeepFace emotion recognition library

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in a frame
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces  # Return list of face coordinates if any found

# Function to detect emotions in a frame using DeepFace
def detect_emotion(frame):
    try:
        # Analyze the face and return emotions
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Since analyze returns a list, we take the first result
        dominant_emotion = analysis[0]['dominant_emotion']  # Correctly access the dominant emotion
        emotion_scores = analysis[0]['emotion']  # Get the full emotion scores
        
        return dominant_emotion, emotion_scores  # Return both dominant emotion and scores
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None, None

# Function to process video, detect faces and recognize emotions
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break
        
        faces = detect_face(frame)  # Detect faces in the current frame
        
        if len(faces) > 0:  # If faces are detected
            # Calculate timestamp in seconds for the current frame
            timestamp = frame_number / fps

            # Detect emotion in the frame
            dominant_emotion, emotion_scores = detect_emotion(frame)
            
            # If an emotion is detected, output the information
            if dominant_emotion and emotion_scores:
                print(f"Frame {frame_number}, Timestamp: {timestamp:.2f} seconds, "
                      f"Dominant Emotion: {dominant_emotion}, Emotion Scores: {emotion_scores}")
        
        frame_number += 1

    # Release the video capture
    cap.release()

# Function to process a list of images and detect faces and emotions
def process_images(image_list):
    for idx, image in enumerate(image_list):
        faces = detect_face(image)  # Detect faces in the image
        
        if len(faces) > 0:  # If faces are detected
            dominant_emotion, emotion_scores = detect_emotion(image)  # Detect emotion in the image
            if dominant_emotion and emotion_scores:
                print(f"Image {idx}: Dominant Emotion detected - {dominant_emotion}, "
                      f"Emotion Scores: {emotion_scores}")
            else:
                print(f"Image {idx}: No emotion detected.")
        else:
            print(f"Image {idx}: No face detected.")

if __name__ == "__main__":
    # Argument parser to choose between video mode and image mode
    parser = argparse.ArgumentParser(description="Detect faces and emotions in video or list of images.")
    parser.add_argument("--mode", choices=['video', 'images'], required=True, help="Mode of operation: 'video' or 'images'")
    parser.add_argument("--video_path", type=str, help="Path to the video file (for video mode)")
    parser.add_argument("--image_list_path", type=str, help="Path to the serialized list of images (for images mode, must be a .pkl file)")

    args = parser.parse_args()

    # Process video if mode is 'video'
    if args.mode == 'video':
        if args.video_path:
            process_video(args.video_path)
        else:
            print("Error: --video_path is required when mode is 'video'")

    # Process image list if mode is 'images'
    elif args.mode == 'images':
        if args.image_list_path:
            # Load the serialized NumPy array list (stored in a .pkl file)
            with open(args.image_list_path, 'rb') as f:
                image_list = pickle.load(f)

            # Check if the loaded object is a list of NumPy arrays
            if isinstance(image_list, list) and all(isinstance(img, np.ndarray) for img in image_list):
                process_images(image_list)
            else:
                print("Error: Provided file does not contain a valid list of NumPy arrays.")
        else:
            print("Error: --image_list_path is required when mode is 'images'")