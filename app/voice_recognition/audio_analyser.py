import librosa
import numpy as np
import argparse

def estimate_words_count(y, sr):
    # Estimate average duration of a spoken word
    average_word_duration = 0.5  # Average duration in seconds (this can vary based on context)
    
    # Calculate total duration of the audio
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Estimate the number of words
    estimated_word_count = total_duration / average_word_duration
    return int(estimated_word_count)

def analyze_audio(audio_bytes):
    # Reset the BytesIO object to the beginning
    audio_bytes.seek(0)
    
    # Load audio from BytesIO object
    y, sr = librosa.load(audio_bytes, sr=None)

    # Calculate pace of speaking
    total_duration = librosa.get_duration(y=y, sr=sr)
    words_count = estimate_words_count(y, sr)  # Estimate words count
    pace = (words_count / total_duration) * 60  # words per minute

    # Loudness detection (RMS)
    rms = np.sqrt(np.mean(y**2))
    loudness = "Too loud" if rms > loud_threshold else "Too quiet" if rms < quiet_threshold else "Acceptable"

    # Pause time and interruptions
    silence_intervals = librosa.effects.split(y, top_db=20)  # Detect silent segments
    pause_time = np.sum(silence_intervals[:, 1] - silence_intervals[:, 0]) / sr  # in seconds

    return {
        'pace': pace,
        'loudness': loudness,
        'pause_time': pause_time
    }

# Define thresholds for loudness
loud_threshold = 0.02
quiet_threshold = 0.005

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze audio file.')
    parser.add_argument('file_path', type=str, help='Path to the audio file to analyze')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Analyze the audio file
    result = analyze_audio(args.file_path)
    
    # Print the result
    print(result)