import cv2
import argparse
import pickle
import numpy as np

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in a frame (used in both modes)
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0  # Return True if faces are found

# Function to process video and detect faces
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_number = 0
    frames_with_faces = []

    # Get frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break

        if detect_face(frame):  # Check if the frame contains a face
            # Calculate timestamp in seconds for the current frame
            timestamp = frame_number / fps
            frames_with_faces.append((frame_number, timestamp))  # Save frame number and timestamp

        frame_number += 1

    # Release the video capture
    cap.release()

    # Output: List of frame numbers and timestamps with detected faces
    for frame_num, timestamp in frames_with_faces:
        print(f"Frame {frame_num}, Timestamp: {timestamp:.2f} seconds")

# Function to process a list of images and detect faces
def process_images(image_list):
    images_with_faces = []

    # Loop through each image in the list
    for idx, image in enumerate(image_list):
        if detect_face(image):  # Check if the image contains a face
            images_with_faces.append(idx)  # Save the index of the image if a face is detected

    # Output: List of image indices with detected faces
    if images_with_faces:
        print(f"Images with faces detected at indices: {images_with_faces}")
    else:
        print("No faces detected in the provided images.")

if __name__ == "__main__":
    # Argument parser to choose between video mode and image mode
    parser = argparse.ArgumentParser(description="Detect faces in video or list of images.")
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