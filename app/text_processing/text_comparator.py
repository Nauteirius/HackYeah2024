import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compare_subtitles_and_transcriptions(subtitle: str, transcription: str):
    """
    Compare a single subtitle and transcription and return similarity scores.

    Args:
        subtitle (str): The subtitle text.
        transcription (str): The transcription text.

    Returns:
        tuple: Levenshtein and Cosine similarity scores.
    """
    
    # Helper function for text processing
    def preprocess_text(text):
        """Lowercase and remove punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        return text

    def similarity_levenshtein(str1: str, str2: str):
        """Levenshtein distance similarity ratio (similar to SequenceMatcher)."""
        return SequenceMatcher(None, str1, str2).ratio()

    def similarity_cosine(str1: str, str2: str):
        """Cosine similarity between two strings (bag of words)."""
        vectorizer = CountVectorizer().fit_transform([str1, str2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0, 1]

    # Preprocess the subtitle and transcription
    processed_subtitle = preprocess_text(subtitle)
    processed_transcription = preprocess_text(transcription)

    # Calculate similarities
    lev_similarity = similarity_levenshtein(processed_subtitle, processed_transcription)
    cos_similarity = similarity_cosine(processed_subtitle, processed_transcription)
    
    return lev_similarity, cos_similarity

# Example usage
if __name__ == "__main__":
    subtitle = "This is the first subtitle"
    transcription = "this is the first subtitle, yeah"

    # Call the function to compare a single subtitle and transcription
    lev_similarity, cos_similarity = compare_subtitles_and_transcriptions(subtitle, transcription)

    # Print results
    print(f"Levenshtein Similarity = {lev_similarity:.2f}, Cosine Similarity = {cos_similarity:.2f}")