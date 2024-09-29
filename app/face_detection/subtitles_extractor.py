import pickle
import numpy as np
import cv2
import argparse
import easyocr  # EasyOCR library
from difflib import SequenceMatcher  # For comparing text similarity

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)  # 'en' for English, add other languages if needed

def are_similar(text1, text2, threshold=0.8):
    """Returns True if text1 and text2 are similar above the given threshold."""
    return SequenceMatcher(None, text1, text2).ratio() > threshold

def extract_subtitles_from_frames(pkl_path, frame_skip=10, similarity_threshold=0.8):
    # Step 1: Load the .pkl file containing the list of numpy arrays (frames)
    with open(pkl_path, 'rb') as file:
        frames = pickle.load(file)  # This assumes frames are stored as a list of numpy arrays
    
    subtitles = []
    previous_text = None  # Track the previous subtitle text
    
    # Step 2: Process each frame to extract text (OCR)
    for i, frame in enumerate(frames):
        if i % frame_skip != 0:
            continue  # Skip frames to avoid processing every single frame
        
        # Crop the frame to focus on the bottom part (where subtitles are likely located)
        height, width, _ = frame.shape
        cropped_frame = frame[int(height * 0.75):, :]  # Crop the bottom 25% of the frame
        
        # Convert the numpy array (cropped frame) from BGR to RGB (for EasyOCR)
        img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        
        # Step 3: Filter for white text (we assume the subtitle text is white)
        # Convert to HSV (Hue, Saturation, Value) color space to filter white color
        hsv_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)  # Lower bound for white
        upper_white = np.array([180, 55, 255], dtype=np.uint8)  # Upper bound for white
        white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)  # Mask for white pixels
        
        # Apply the mask to get only the white areas
        white_text_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=white_mask)
        
        # Step 4: Use EasyOCR to extract text from the filtered image
        results = reader.readtext(white_text_frame, detail=0)  # detail=0 gives just the text as a list
        text = " ".join(results).strip()  # Join the text pieces into one string
        
        # Step 5: Skip empty strings and subtitles similar to the previous one
        if text and (previous_text is None or not are_similar(text, previous_text, similarity_threshold)):
            subtitles.append(text)
            previous_text = text  # Update the previous text to the current one
    
    # Step 6: Join all unique subtitles into one block of text
    combined_text = " ".join(subtitles)
    
    # Step 7: Return the combined text
    return combined_text

def main():
    # Step 1: Set up argument parsing
    parser = argparse.ArgumentParser(description='Extract subtitles from video frames in a .pkl file')
    parser.add_argument('pkl_path', type=str, help='Path to the .pkl file containing video frames')
    
    # Step 2: Parse the arguments
    args = parser.parse_args()
    
    # Step 3: Extract subtitles
    combined_subtitles = extract_subtitles_from_frames(args.pkl_path, frame_skip=10, similarity_threshold=0.8)
    
    # Step 4: Print the combined subtitles
    print("Extracted Subtitles:\n", combined_subtitles)

if __name__ == '__main__':
    main()