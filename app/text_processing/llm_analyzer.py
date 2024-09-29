from openai import OpenAI
import typing
import re

from app.text_processing.speech_to_text import API_KEY

def split_numbered_list(text: str) -> list[str]:
    # Split the text on newlines and numbered points
    items = re.split(r'\n\d+\.\s*', text)
    items = [item.strip() for item in items if item.strip()]
    
    return items

def call_openai_api(system_content: str, user_content: str) -> str:
    client = OpenAI(api_key=API_KEY)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    )
    return completion.choices[0].message.content

def generate_ten_questions(transcription: str) -> str:
    return call_openai_api(
        "Jesteś personalnym trenerem wypowiadania się publicznie",  #"Zachowuj się jak bardzo zaciekawiony uczestnik wystąpienia"
        f"{transcription}. Czy możesz zaproponować 10 pytań dotyczących tego tekstu? zwróć je każdy w kolejnej linii w formacie np. 1. pytanie?"
    )

def find_false_words(transcription: str) -> str:
    return call_openai_api(
        "Jesteś personalnym trenerem wypowiadania się publicznie",  #"Zachowuj się jak wybitny polonista"
        f"{transcription}. Czy znajdujesz w tym tekście jakieś zwroty, które nie mają sensu? (jak np. kwaśna wata cukrowa, czy kwadratowa rocznica zamiast okrągła rocznica)"
    )

def generate_tags(transcription: str) -> str:
    return call_openai_api(
        "Jesteś personalnym trenerem wypowiadania się publicznie",  #"Zachowuj się jak bardzo uważny czytelnik"
        f"{transcription}. Wygeneruj pięć tagów opisujących ten tekst. zwróć je każdy w kolejnej linii w formacie np. 1. tag"
    )

def generate_summary(transcription: str) -> str:
    return call_openai_api(
        "Zachowuj się jak bardzo uważny czytelnik",
        f"{transcription}. Napisz w 2 zdaniach co zrozumiałeś z tego tekstu."
    )


# TODO: secure the prompt injection
def llm_output(transcription):
    questions_text = generate_ten_questions(transcription)
    false_words = find_false_words(transcription)
    tags_text = generate_tags(transcription)
    text_summary = generate_summary(transcription)

    questions = split_numbered_list(questions_text)
    tags = split_numbered_list(tags_text)

    return false_words, questions, tags, text_summary

