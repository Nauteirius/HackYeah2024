import nltk
import yake
import stanza
import json
from collections import Counter
from nltk.corpus import stopwords, words
from nltk import word_tokenize
import argparse
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

def tokenize_text(text):
    '''Tokenize string with nltk'''
    tokens = word_tokenize(text, language="polish")
    return tokens

def complex_words_count(tokens):
    '''Count the number of complex words defined as words containing 9 or more characters'''
    long_token_count = sum(1 for token in tokens if len(token) >= 9)
    return long_token_count

def count_tokens(tokens):
    '''Count the number of tokens'''
    return len(tokens)

def extract_keywords(text):
    '''Extract keywords from text using YAKE'''
    kw_extractor = yake.KeywordExtractor(lan="pl", n=1, top=10)  # Extract top 10 keywords in Polish
    keywords = kw_extractor.extract_keywords(text.lower())
    return {kw: score for kw, score in keywords}

def create_stanza_object(text):
    '''Create Stanza doc to be processed further'''
    stanza.download('pl')
    nlp = stanza.Pipeline('pl')
    doc = nlp(text)
    return doc

def count_sentences(doc):
    '''Count the number of sentences in the Stanza doc'''
    return len(doc.sentences)

def count_pos(doc):
    '''Count parts of speech in a Stanza document'''
    pos_counts = Counter([word.upos for sentence in doc.sentences for word in sentence.words])
    return dict(pos_counts)

def gunning_fog_index(tokens, doc):
    '''Calculate Gunning Fog Index for a text'''
    
    # Calculate counts
    total_words = count_tokens(tokens)
    complex_words = complex_words_count(tokens)
    total_sentences = count_sentences(doc)
    
    if total_sentences == 0 or total_words == 0:
        return 0, "Insufficient data for calculation"
    
    # Calculate Gunning Fog Index
    gfi = 0.4 * ((total_words / total_sentences) + (100 * (complex_words / total_words)))

    if gfi < 6:
        reading_level = "5th Grade and below"
    elif gfi < 8:
        reading_level = "6th to 8th Grade"
    elif gfi < 10:
        reading_level = "9th to 10th Grade"
    elif gfi < 12:
        reading_level = "11th to 12th Grade"
    else:
        reading_level = "College Level and above" #maybe this indicates jargon?

    return gfi, reading_level

def detect_non_polish_and_slang(tokens):
    '''Detect potential slang, jargon, or non-Polish words'''
    polish_alphabet = set('aąbcćdeęfghijklłmnńoóprsśtuwyzźż')
    english_words = set(words.words())
    
    non_polish = []
    for token in tokens:
        if set(token.lower()) - polish_alphabet:
            non_polish.append(token)
        elif token.lower() in english_words:
            non_polish.append(token)
        #string of polish slang words
        elif re.match(r'\b(spoko|git|lol|xd|ziomek|siema|masakra|sztos|kumpel|luzik|ziomal|nara|spina|mega|kozacko|klawo|ogarnij|spadaj|kapiszon|chill|czaisz|bekowy|mordo|wporzo|czill|sztosik|epicki|załamka|japa|kminisz|jazda|czad|lajk|fejm|ziomuś|nara|skumać|fajowo|piona|masakryczny|beka|czaderski|ziom)\b', token.lower()):
            non_polish.append(token)
    
    return non_polish

def text_analyzer(text):
    '''Main function to process text and write results to output file'''

    
    # Tokenize text once
    tokens = tokenize_text(text)
    
    # Process text with Stanza
    doc = create_stanza_object(text)
    
    # Extract information
    token_count = count_tokens(tokens)
    sentence_count = count_sentences(doc)
    pos_count = count_pos(doc)
    keywords = extract_keywords(text)
    gfi_value, reading_level = gunning_fog_index(tokens, doc)
    non_polish_words = detect_non_polish_and_slang(tokens)

    # Prepare the result dictionary
    results = {
        "Token Count": token_count,
        "Sentence Count": sentence_count,
        "POS Count": pos_count,
        "Keywords": keywords,
        "Gunning Fog Index": gfi_value,
        "Reading Level": reading_level,
        "Non-Polish/Slang Words": non_polish_words
    }

    print(results)
