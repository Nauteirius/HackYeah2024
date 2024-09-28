import nltk
import yake
import stanza
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
import argparse

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def tokenize_text(text):
    '''Tokenize string with nltk'''
    tokens = word_tokenize(text, language="polish")
    return tokens

def complex_words_count(tokens):
    '''Count the number of complex words defined as words containing 9 or more characters'''
    long_token_count = sum(1 for token in tokens if len(token) >= 9)
    return long_token_count

def count_tokens(text):
    '''Count the number of tokens in text'''
    tokens = word_tokenize(text, language="polish")
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

def gunning_fog_index(text, doc):
    '''Calculate Gunning Fog Index for a text'''
    tokens = word_tokenize(text, language="polish")
    
    # Calculate counts
    total_words = count_tokens(text)
    complex_words = complex_words_count(tokens)
    total_sentences = count_sentences(doc)
    
    # Avoid division by zero
    if total_sentences == 0 or total_words == 0:
        return 0, "Insufficient data for calculation"
    
    # Calculate Gunning Fog Index
    gfi = 0.4 * ((total_words / total_sentences) + (100 * (complex_words / total_words)))

    # Determine the reading level
    if gfi < 6:
        reading_level = "5th Grade and below"
    elif gfi < 8:
        reading_level = "6th to 8th Grade"
    elif gfi < 10:
        reading_level = "9th to 10th Grade"
    elif gfi < 12:
        reading_level = "11th to 12th Grade"
    else:
        reading_level = "College Level and above"

    return gfi, reading_level

def main(input_file, output_file):
    '''Main function to process text and write results to output file'''
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process text with Stanza
    doc = create_stanza_object(text)
    
    # Extract information
    token_count = count_tokens(text)
    sentence_count = count_sentences(doc)
    pos_count = count_pos(doc)
    keywords = extract_keywords(text)
    gfi_value, reading_level = gunning_fog_index(text, doc)
    
    # Prepare the result dictionary
    results = {
        "Token Count": token_count,
        "Sentence Count": sentence_count,
        "POS Count": pos_count,
        "Keywords": keywords,
        "Gunning Fog Index": gfi_value,
        "Reading Level": reading_level
    }

    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Text processing script with Gunning Fog Index and POS tagging")
    parser.add_argument('input_file', type=str, help="Input text file")
    parser.add_argument('output_file', type=str, help="Output JSON file with results")
    
    args = parser.parse_args()
    
    # Call the main function
    main(args.input_file, args.output_file)