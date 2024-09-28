import pickle
import numpy as np
import pytesseract
import cv2
import argparse
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\yamin\AppData\Local\Tesseract-OCR\tesseract.exe'
# Ensure that Tesseract is correctly installed and available in PATH or specify its path
# pytesseract.pytesseract.tesseract_cmd = r'path_to_your_tesseract.exe'  # Uncomment and set the path if necessary

def extract_subtitles_from_frames(pkl_path):
    # Step 1: Load the .pkl file containing the list of numpy arrays (frames)
    with open(pkl_path, 'rb') as file:
        frames = pickle.load(file)  # This assumes frames are stored as a list of numpy arrays
    
    subtitles = []
    
    # Step 2: Process each frame to extract text (OCR)
    for i, frame in enumerate(frames):
        # Convert the numpy array (frame) from BGR (OpenCV default) to RGB (required for pytesseract)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 3: Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(img_rgb)
        
        # Add the extracted text to the subtitle list (or empty string if no text)
        subtitles.append(text.strip())
    
    # Step 4: Return the list of subtitles
    return subtitles

def main():
    # Step 1: Set up argument parsing
    parser = argparse.ArgumentParser(description='Extract subtitles from video frames in a .pkl file')
    parser.add_argument('pkl_path', type=str, help='Path to the .pkl file containing video frames')
    
    # Step 2: Parse the arguments
    args = parser.parse_args()
    
    # Step 3: Extract subtitles
    subtitles = extract_subtitles_from_frames(args.pkl_path)
    
    # Step 4: Print the subtitles
    for i, subtitle in enumerate(subtitles):
        print(f"Frame {i+1}: {subtitle}")

if __name__ == '__main__':
    main()