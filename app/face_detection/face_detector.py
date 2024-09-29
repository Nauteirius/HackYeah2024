import cv2
import argparse
import pickle
import numpy as np
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(frame):
    """
    Wykrywa twarze na podanym obrazie.

    Args:
        frame (numpy.ndarray): Obraz w formacie BGR, na którym mają być wykrywane twarze.

    Returns:
        list: Lista prostokątów (x, y, szerokość, wysokość) reprezentujących wykryte twarze.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detect_emotion(frame):
    """
    Wykrywa emocje na podanym obrazie twarzy.

    Args:
        frame (numpy.ndarray): Obraz w formacie BGR, na którym mają być wykrywane emocje.

    Returns:
        tuple: Krotka zawierająca:
            - dominant_emotion (str): Dominująca emocja wykryta na obrazie.
            - emotion_scores (dict): Słownik z wynikami dla wszystkich wykrytych emocji.

    Przykład:
        >>> dominant_emotion, emotion_scores = detect_emotion(frame)
        >>> print(dominant_emotion)
        'happy'
        >>> print(emotion_scores)
        {'angry': 0.01, 'disgust': 0.0, 'fear': 0.02, 'happy': 0.95, 'sad': 0.01, 'surprise': 0.01, 'neutral': 0.0}
    """
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_scores = analysis[0]['emotion']
        return dominant_emotion, emotion_scores
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return None, None

def process_video(video_path):
    """
    Przetwarza plik wideo, wykrywając twarze i emocje w każdej klatce.

    Args:
        video_path (str): Ścieżka do pliku wideo.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_face(frame)
        if len(faces) > 0:
            timestamp = frame_number / fps
            dominant_emotion, emotion_scores = detect_emotion(frame)
            if dominant_emotion and emotion_scores:
                print(f"Frame {frame_number}, Timestamp: {timestamp:.2f} seconds, "
                      f"Dominant Emotion: {dominant_emotion}, Emotion Scores: {emotion_scores}")
        frame_number += 1
    cap.release()

def process_images(image_list):
    """
    Przetwarza listę obrazów, wykrywając twarze i emocje na każdym obrazie.

    Args:
        image_list (list): Lista obrazów w formacie numpy.ndarray, na których mają być wykrywane twarze i emocje.

    Returns:
        None

    Działanie:
        - Dla każdego obrazu w liście wykrywa twarze za pomocą funkcji detect_face.
        - Jeśli twarze są wykryte, wykrywa emocje na obrazie za pomocą funkcji detect_emotion.
        - Wypisuje na standardowe wyjście dominującą emocję oraz wyniki dla wszystkich wykrytych emocji.
        - Jeśli nie wykryto twarzy lub emocji, wypisuje odpowiedni komunikat.

    Przykład:
        >>> image_list = [image1, image2, image3]
        >>> process_images(image_list)
        Image 0: Dominant Emotion detected - happy, Emotion Scores: {'angry': 0.01, 'disgust': 0.0, 'fear': 0.02, 'happy': 0.95, 'sad': 0.01, 'surprise': 0.01, 'neutral': 0.0}
        Image 1: No face detected.
        Image 2: No emotion detected.
    """
    for idx, image in enumerate(image_list):
        faces = detect_face(image)
        
        if len(faces) > 0:
            dominant_emotion, emotion_scores = detect_emotion(image)
            if dominant_emotion and emotion_scores:
                print(f"Image {idx}: Dominant Emotion detected - {dominant_emotion}, "
                      f"Emotion Scores: {emotion_scores}")
            else:
                print(f"Image {idx}: No emotion detected.")
        else:
            print(f"Image {idx}: No face detected.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect faces and emotions in video or list of images.")
    parser.add_argument("--mode", choices=['video', 'images'], required=True, help="Mode of operation: 'video' or 'images'")
    parser.add_argument("--video_path", type=str, help="Path to the video file (for video mode)")
    parser.add_argument("--image_list_path", type=str, help="Path to the serialized list of images (for images mode, must be a .pkl file)")
    args = parser.parse_args()

    if args.mode == 'video':
        if args.video_path:
            process_video(args.video_path)
        else:
            print("Error: --video_path is required when mode is 'video'")
    elif args.mode == 'images':
        if args.image_list_path:
            with open(args.image_list_path, 'rb') as f:
                image_list = pickle.load(f)

            if isinstance(image_list, list) and all(isinstance(img, np.ndarray) for img in image_list):
                process_images(image_list)
            else:
                print("Error: Provided file does not contain a valid list of NumPy arrays.")
        else:
            print("Error: --image_list_path is required when mode is 'images'")