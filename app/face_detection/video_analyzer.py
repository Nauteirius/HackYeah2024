import cv2
import argparse
import pickle
import numpy as np
from deepface import DeepFace  # Import the DeepFace emotion recognition library
import mediapipe as mp  # Import Mediapipe for hand detection
import json  # Import the json module for structured output
fps=25
# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initialize Mediapipe hand detector
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Function to detect faces in a frame
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces  # Return list of face coordinates if any found


# Function to detect hands in a frame using Mediapipe
def detect_hand(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Mediapipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        return True  # Return True if at least one hand is detected
    return False  # Return False if no hands are detected

# Function to check if the face is looking at the camera
def is_looking_at_camera(landmarks):
    # Extract key landmarks for eyes and mouth
    left_eye = landmarks[33]  # Example index for left eye
    right_eye = landmarks[263]  # Example index for right eye
    mouth = landmarks[61]  # Example index for mouth center

    # Calculate the eye aspect ratio to determine if the face is frontal
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    mouth_height = np.linalg.norm(np.array(mouth) - np.array([(left_eye[0] + right_eye[0]) / 2, left_eye[1]]))

    # A simple heuristic: if the mouth is below the eye line and the distance between the eyes is reasonable
    return mouth_height < (0.5 * eye_distance)

# Function to detect emotions in a frame using DeepFace
def detect_emotion(frame):
    try:
        # Analyze the face and return emotions
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion_scores = analysis[0]['emotion']  # Get the full emotion scores
        return emotion_scores  # Return the emotion scores
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None

# Function to process video, detect faces and recognize emotions
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Data structure to store results
    results = {}

    # Loop through video frames
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()  # Read a frame
            if not ret:
                break
            print(frame_number)
            # Calculate timestamp in seconds for the current frame
            timestamp = frame_number / fps
            second = int(timestamp)  # Get the current second

            faces = detect_face(frame)  # Detect faces in the current frame
            hand_detected = detect_hand(frame)  # Detect hand in the current frame

            # Initialize results for the current second if not already done
            if second not in results:
                results[second] = {
                    "timestamp": timestamp,
                    "face": {
                        "visible_count": 0,  # Counter for frames with visible faces
                        "emotion": np.zeros(7),  # Assuming 7 emotions; adjust as necessary
                        "total_frames": 0,  # Total frames counted for this second
                        "looking_at_camera_count": 0  # Count for frames looking at the camera
                    },
                    "body": {
                        "gesture_count": 0 # Placeholder for body gesture detection
                    },
                    "environment": {
                        "background_people_count": 0,  # Counter for frames with more than one face
                    }
                }

            # Increment total frames for this second
            results[second]["face"]["total_frames"] += 1

            if len(faces) > 0:  # If faces are detected
                results[second]["face"]["visible_count"] += 1  # Increment visible count


                # Detect emotion in the frame
                emotion_scores = detect_emotion(frame)

                # If emotion scores are detected, accumulate them
                if emotion_scores:
                    results[second]["face"]["emotion"] += np.array(list(emotion_scores.values()))  # Accumulate emotion scores
                
                # If more than one face is detected, increment background people count
                if len(faces) > 1:
                    results[second]["environment"]["background_people_count"] += 1

                
                # Process face mesh for landmark detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_mesh = face_mesh.process(frame_rgb)

                if results_mesh.multi_face_landmarks:
                    for face_landmarks in results_mesh.multi_face_landmarks:
                        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

                        # Check if looking at the camera
                        if is_looking_at_camera(landmarks):
                            print("looking at camera")
                            results[second]["face"]["looking_at_camera_count"] += 1
                    # If hand is detected, increment gesture count
            if hand_detected:
                results[second]["body"]["gesture_count"] += 1
            frame_number += 1

    # After processing all frames, determine visibility and background presence for each second
    for second, data in results.items():
        # Determine if faces were visible in the majority of frames
        data["face"]["visible"] = data["face"]["visible_count"] > (data["face"]["total_frames"] / 2)
        
        # Average the emotion scores if faces were visible
        if data["face"]["visible"]:
            data["face"]["emotion"] /= data["face"]["visible_count"]

        # Determine if background people were present in the majority of frames
        data["environment"]["background_people"] = data["environment"]["background_people_count"] > (data["face"]["total_frames"] / 2)

        # Determine if gestures were present (hand detected in majority of frames)
        data["body"]["gesture"] = data["body"]["gesture_count"] > (data["face"]["total_frames"] / 2)
        data["face"]["looking_at_camera"] = data["face"]["looking_at_camera_count"] > (data["face"]["total_frames"] / 2)

    # Convert results to the desired JSON format
    json_output = []
    for second, data in results.items():
        json_output.append({
            "timestamp": data["timestamp"],
            "face": {
                "visible": data["face"].get("visible", False),  # Use the computed visibility
                "emotion": data["face"]["emotion"].tolist(),  # Convert numpy array to list for JSON serialization
                "looking_at_camera": data["face"]["looking_at_camera"],  # Include looking_at_camera in output
            },
            "body": {
                "gesture": data["body"]["gesture"]
            },
            "environment": {
                "background_people": data["environment"].get("background_people", False)  # Use the computed background people presence
            }
        })

    # Print the structured JSON output
    print(json.dumps(json_output, indent=4))

    # Release the video capture
    cap.release()

# Function to process a list of images and detect faces and emotions
def process_images(image_list,fps):
    # Data structure to store results
    results = {}
    #fps = 25  # Assuming a constant frame rate

    for idx, image in enumerate(image_list):
        # Calculate the current second based on the index (assuming each image represents one frame)
        second = idx // fps  # Assuming fps is constant and each image corresponds to a frame

        faces = detect_face(image)  # Detect faces in the image
        hand_detected = detect_hand(image)  # Detect hand in the image

        # Initialize results for the current second if not already done
        if second not in results:
            results[second] = {
                "timestamp": second,  # Here, timestamp can be set differently if needed
                "face": {
                    "visible_count": 0,  # Counter for images with visible faces
                    "emotion": np.zeros(7),  # Assuming 7 emotions; adjust as necessary
                    "total_frames": 0,  # Total images counted for this second
                    "looking_at_camera_count": 0  # Count for frames looking at the camera
                },
                "body": {
                    "gesture_count": 0 # Placeholder for body gesture detection
                },
                "environment": {
                    "background_people_count": 0,  # Counter for images with more than one face
                }
            }

        # Increment total frames for this second
        results[second]["face"]["total_frames"] += 1

        if len(faces) > 0:  # If faces are detected
            results[second]["face"]["visible_count"] += 1  # Increment visible count

            # Detect emotion in the image
            emotion_scores = detect_emotion(image)

            # If emotion scores are detected, accumulate them
            if emotion_scores:
                results[second]["face"]["emotion"] += np.array(list(emotion_scores.values()))  # Accumulate emotion scores
            
            # If more than one face is detected, increment background people count
            if len(faces) > 1:
                results[second]["environment"]["background_people_count"] += 1

            # Process face mesh for landmark detection
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
                results_mesh = face_mesh.process(frame_rgb)

                if results_mesh.multi_face_landmarks:
                    for face_landmarks in results_mesh.multi_face_landmarks:
                        landmarks = [(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in face_landmarks.landmark]

                        # Check if looking at the camera
                        if is_looking_at_camera(landmarks):
                            results[second]["face"]["looking_at_camera_count"] += 1

        if hand_detected:
            results[second]["body"]["gesture_count"] += 1

    # After processing all images, determine visibility and background presence for each second
    for second, data in results.items():
        # Determine if faces were visible in the majority of images
        data["face"]["visible"] = data["face"]["visible_count"] > (data["face"]["total_frames"] / 2)
        
        # Average the emotion scores if faces were visible
        if data["face"]["visible"]:
            data["face"]["emotion"] /= data["face"]["visible_count"]

        # Determine if background people were present in the majority of images
        data["environment"]["background_people"] = data["environment"]["background_people_count"] > (data["face"]["total_frames"] / 2)

        # Determine if gestures were present (hand detected in majority of frames)
        data["body"]["gesture"] = data["body"]["gesture_count"] > (data["face"]["total_frames"] / 2)

    # Convert results to the desired JSON format
    json_output = []
    emotion_keys = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for second, data in results.items():
        emotion_dict = {key: value for key, value in zip(emotion_keys, data["face"]["emotion"].tolist())}
        json_output.append({
            "timestamp": data["timestamp"],
            "face": {
                "visible": data["face"].get("visible", False),  # Use the computed visibility
                "emotion": emotion_dict,
                "looking_at_camera": data["face"]["looking_at_camera_count"] > (data["face"]["total_frames"] / 2),  # Include looking_at_camera in output
            },
            "body": {
                "gesture": data["body"]["gesture"]
            },
            "environment": {
                "background_people": data["environment"].get("background_people", False)  # Use the computed background people presence
            }
        })

    # Print the structured JSON output
    print(json.dumps(json_output, indent=4))
    return json_output

if __name__ == "__main__":
    # Argument parser to choose between video mode and image mode
    parser = argparse.ArgumentParser(description="Detect faces and emotions in video or list of images.")
    parser.add_argument("--mode", choices=['video', 'images'], required=True, help="Mode of operation: 'video' or 'images'")
    parser.add_argument("--video_path", type=str, help="Path to the video file (for video mode)")
    parser.add_argument("--image_list_path", type=str, help="Path to the serialized list of images (for images mode, must be a .pkl file)")

    args = parser.parse_args()

    # Process video if
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