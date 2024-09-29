import nltk
import yake
import stanza
import json
import nltk
from collections import Counter
from nltk.corpus import stopwords, words
from nltk import word_tokenize
import argparse
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('words')

def tokenize_text(text):
    '''Tokenize string with nltk'''
    tokens = word_tokenize(text, language="polish")
    return tokens

def complex_words_count(tokens):
    '''Count the number of complex words defined as words containing 9 or more characters'''
    complex_words = [token for token in tokens if len(token) >= 9]
    long_token_count = len(complex_words)
    return long_token_count, complex_words

def count_tokens(tokens):
    '''Count the number of tokens'''
    return len(tokens)

# def extract_keywords(text):
#     '''Extract keywords from text using YAKE'''
#     kw_extractor = yake.KeywordExtractor(lan="pl", n=1, top=10)  # Extract top 10 keywords in Polish
#     keywords = kw_extractor.extract_keywords(text.lower())
#     return {kw: score for kw, score in keywords}

def create_stanza_object(text):
    '''Create Stanza doc to be processed further'''
    stanza.download('pl')
    nlp = stanza.Pipeline('pl')
    doc = nlp(text)
    return doc

def count_sentences(doc):
    '''Count the number of sentences in the Stanza doc'''
    return len(doc.sentences)

def get_sentence_lengths(doc):
    '''Return a list of sentence lengths from the Stanza doc'''
    return [len(sentence.words) for sentence in doc.sentences]


def count_pos(doc):
    '''Count parts of speech in a Stanza document'''
    pos_counts = Counter([word.upos for sentence in doc.sentences for word in sentence.words])
    return dict(pos_counts)

def gunning_fog_index(tokens, doc):
    '''Calculate Gunning Fog Index for a text'''
    
    # Calculate counts
    total_words = count_tokens(tokens)
    complex_word_count, complex_word_list = complex_words_count(tokens)
    total_sentences = count_sentences(doc)
    
    if total_sentences == 0 or total_words == 0:
        return 0, "Insufficient data for calculation"
    
    # Calculate Gunning Fog Index
    gfi = 0.4 * ((total_words / total_sentences) + (100 * (complex_word_count / total_words)))

    if gfi < 6:
        reading_level = "5 klasa lub niżej"
    elif gfi < 8:
        reading_level = "6 do 8 klasy"
    elif gfi < 10:
        reading_level = "9 do 10 klasy"
    elif gfi < 12:
        reading_level = "11 do 12 klasy"
    else:
        reading_level = "poziom akademicki" #maybe this indicates jargon?

    return gfi, reading_level, complex_word_list

def detect_non_polish_and_slang(tokens):
    '''Detect potential slang, jargon, or non-Polish words'''
    polish_alphabet = set('aąbcćdeęfghijklłmnńoóprsśtuwyzźż')
    english_words = set(words.words())
    punctuation = set('.,;:!?()[]{}"-\'')
    
    non_polish = []
    for token in tokens:
        if set(token) <= punctuation:
            continue
        
        cleaned_token = ''.join(char for char in token if char not in punctuation)
        if not cleaned_token:
            continue
        
        if set(cleaned_token.lower()) - polish_alphabet:
            non_polish.append(token)
        elif cleaned_token.lower() in english_words:
            non_polish.append(token)
        elif re.match(r'\b(spoko|git|lol|xd|ziomek|siema|masakra|sztos|kumpel|luzik|ziomal|nara|spina|mega|kozacko|klawo|ogarnij|spadaj|kapiszon|chill|czaisz|bekowy|mordo|wporzo|czill|sztosik|epicki|załamka|japa|kminisz|jazda|czad|lajk|fejm|ziomuś|nara|skumać|fajowo|piona|masakryczny|beka|czaderski|ziom)\b', cleaned_token.lower()):
            non_polish.append(token)
    
    return non_polish

def get_repeated_words(tokens):
    '''Get words that occur more than once in the text'''
    word_counts = Counter(token.lower() for token in tokens if token.isalnum())
    repeated_words = {word: count for word, count in word_counts.items() if count > 1}
    return repeated_words

def detect_pause_words(tokens):
    '''Detect pause words or filler words in the text'''
    pause_words = set(['hm', 'em', 'uh', 'um', 'eh', 'mm', 'hmm', 'erm', 'eee', 'yyyy'])
    detected_pause_words = [token.lower() for token in tokens if token.lower() in pause_words]
    return detected_pause_words

def analyze_word_breaks(words):
    """
    Analyze breaks between words in a list of TextSlice objects.
    
    Args:
    words (list): A list of TextSlice objects with word, time_start_s, and time_end_s attributes.
    
    Returns:
    tuple: (average_break, longest_break, longest_break_start)
        - average_break (float): The average break duration between words in seconds.
        - longest_break (float): The duration of the longest break between words in seconds.
        - longest_break_start (float): The start time of the longest break in seconds.
    """
    if len(words) < 2:
        return 0, 0, None

    breaks = []
    longest_break = 0
    longest_break_start = None

    for i in range(1, len(words)):
        break_start = words[i-1].time_end_s
        break_duration = words[i].time_start_s - break_start
        breaks.append(break_duration)
        
        if break_duration > longest_break:
            longest_break = break_duration
            longest_break_start = break_start

    average_break = sum(breaks) / len(breaks)

    return average_break, longest_break, longest_break_start

def detect_hate_speech(doc):
    '''Detect hate speech in a tokenized string using predefined hate speech lemmas'''
    
    # Lemmas commonly found in hate speech
    hate_speech_keywords = [
        "nienawidz", "głupi", "głupek", "głupk", "idiot", "kretyn", "kretyń", "debil", "rasist", "seksist",
        "homofob", "zdrajc", "śmieć", "zabij", "zabić", "nazi", "terroryst",
        "pedofil", "bandyt", "gwałciciel", "brudas", "morderc", "kanali"
    ]
    
    # Extract lemmas from the Stanza document
    lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words]

    # Check if any lemma contains one of the hate speech keywords (even as a part of the lemma)
    detected_hate_speech = [lemma for lemma in lemmas if any(keyword in lemma for keyword in hate_speech_keywords)]

    # If hate speech is detected, return the list of offensive lemmas
    # if detected_hate_speech:
    #     return {
    #         "Hate Speech Detected": True,
    #         "Potentially Offensive Words": detected_hate_speech
    #     }
    # else:
    #     return {
    #         "Hate Speech Detected": False,
    #         "Potentially Offensive Words": []
    #     }
    # if detect_hate_speech:
    #     return True, detected_hate_speech
    # else:


def text_analyzer(text):
    '''Main function to process text and write results to output file'''

    
    # Tokenize text once
    tokens = tokenize_text(text)
    
    # Process text with Stanza
    doc = create_stanza_object(text)
    
    # Extract information
    token_count = count_tokens(tokens)
    sentence_count = count_sentences(doc)
    sentence_lengths = get_sentence_lengths(doc)
    pos_count = count_pos(doc)
    gfi_value, reading_level, complex_words = gunning_fog_index(tokens, doc)
    non_polish_words = detect_non_polish_and_slang(tokens)
    pause_words = detect_pause_words(tokens)
    repeated_words = get_repeated_words(tokens)
    hate_speech_detection = detect_hate_speech(doc)

    # Prepare the result dictionary
    results = {
        "Ilośc słów": token_count,
        "Ilośc zdań": sentence_count,
        "Długość zdań": sentence_lengths,
        "Trudne słowa": complex_words,
        "Części mowy": pos_count,
        "Indeks Gunning Fog": gfi_value,
        "Poziom słuchacza": reading_level,
        "Słowa slangowe/nie Polskie": non_polish_words,
        "Słowa przerwy": pause_words,
        "Słowa powtórzone": repeated_words,
        "Analiza sentymentu": hate_speech_detection
    }

    return results
