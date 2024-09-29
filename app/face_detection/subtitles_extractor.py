import pickle
import numpy as np
import cv2
import argparse
import easyocr  # EasyOCR library

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)  # 'en' for English, add other languages if needed

def extract_subtitles_from_frames(pkl_path):
    # Step 1: Load the .pkl file containing the list of numpy arrays (frames)
    with open(pkl_path, 'rb') as file:
        frames = pickle.load(file)  # This assumes frames are stored as a list of numpy arrays
    
    subtitles = []
    previous_text = None  # Track the previous subtitle text
    
    # Step 2: Process each frame to extract text (OCR)
    for i, frame in enumerate(frames):
        # Convert the numpy array (frame) from BGR (OpenCV default) to RGB (required for EasyOCR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 3: Use EasyOCR to extract text from the image
        results = reader.readtext(img_rgb, detail=0)  # detail=0 gives just the text as a list
        text = " ".join(results).strip()  # Join the text pieces into one string
        
        # Step 4: Skip empty strings and consecutive duplicate subtitles
        if text and text != previous_text:
            subtitles.append(text)
            previous_text = text  # Update the previous text to the current one
    
    # Step 5: Join all unique subtitles into one block of text
    combined_text = " ".join(subtitles)
    
    # Step 6: Return the combined text
    return combined_text

def main():
    # Step 1: Set up argument parsing
    parser = argparse.ArgumentParser(description='Extract subtitles from video frames in a .pkl file')
    parser.add_argument('pkl_path', type=str, help='Path to the .pkl file containing video frames')
    
    # Step 2: Parse the arguments
    args = parser.parse_args()
    
    # Step 3: Extract subtitles
    combined_subtitles = extract_subtitles_from_frames(args.pkl_path)
    
    # Step 4: Print the combined subtitles
    print("Extracted Subtitles:\n", combined_subtitles)
    return combined_subtitles
if __name__ == '__main__':
    main()
