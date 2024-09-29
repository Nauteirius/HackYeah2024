import librosa
import numpy as np
import argparse

def estimate_words_count(y, sr):
    """
    Szacuje liczbę słów w nagraniu audio.

    Args:
        y (numpy.ndarray): Tablica próbek audio.
        sr (int): Częstotliwość próbkowania audio.

    Returns:
        int: Szacowana liczba słów w nagraniu.
    """
    average_word_duration = 0.5
    total_duration = librosa.get_duration(y=y, sr=sr)
    estimated_word_count = total_duration / average_word_duration
    return int(estimated_word_count)

def analyze_audio(audio_bytes):
    """
    Analizuje plik audio i zwraca statystyki dotyczące tempa mowy, głośności i czasu pauz.

    Args:
        audio_bytes (BytesIO): Obiekt zawierający dane audio.

    Returns:
        dict: Słownik zawierający następujące klucze:
            - 'pace' (float): Tempo mowy w słowach na minutę.
            - 'loudness' (str): Ocena głośności ("Too loud", "Too quiet", "Acceptable").
            - 'pause_time' (float): Całkowity czas pauz w sekundach.
    """
    audio_bytes.seek(0)
    y, sr = librosa.load(audio_bytes, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    words_count = estimate_words_count(y, sr)
    pace = (words_count / total_duration) * 60
    rms = np.sqrt(np.mean(y**2))
    loudness = "Too loud" if rms > loud_threshold else "Too quiet" if rms < quiet_threshold else "Acceptable"
    silence_intervals = librosa.effects.split(y, top_db=20)
    pause_time = np.sum(silence_intervals[:, 1] - silence_intervals[:, 0]) / sr

    return {
        'pace': pace,
        'loudness': loudness,
        'pause_time': pause_time
    }

# Definiowanie progów dla głośności
loud_threshold = 0.02
quiet_threshold = 0.005

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze audio file.')
    parser.add_argument('file_path', type=str, help='Path to the audio file to analyze')
    args = parser.parse_args()
    result = analyze_audio(args.file_path)
    print(result)