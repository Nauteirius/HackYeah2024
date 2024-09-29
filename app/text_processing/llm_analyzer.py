from openai import OpenAI
import typing

from app.text_processing.speech_to_text import API_KEY


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
        "Jesteś personalnym trenerem wypowiadania się publicznie",
        f"{transcription}. Czy możesz zaproponować 10 pytań dotyczących tego tekstu?"
    )

def find_false_words(transcription: str) -> str:
    return call_openai_api(
        "Jesteś personalnym trenerem wypowiadania się publicznie",
        f"{transcription}. Czy znajdujesz w tym tekście jakieś zwroty, które nie mają sensu? (jak np. kwaśna wata cukrowa, czy kwadratowa rocznica zamiast okrągła rocznica)"
    )

def generate_tags(transcription: str) -> str:
    return call_openai_api(
        "Jesteś personalnym trenerem wypowiadania się publicznie",
        f"{transcription}. Wygeneruj pięć tagów opisujących ten tekst"
    )

# TODO: secure the prompt injection
def llm_output(transcription):
    
    print("Testing generate_ten_questions:")
    questions = generate_ten_questions(transcription)
    print(questions)
    print("\n" + "-"*50 + "\n")
    
    print("Testing find_false_words:")
    false_words = find_false_words(transcription)
    print(false_words)
    print("\n" + "-"*50 + "\n")
    
    print("Testing generate_tags:")
    tags = generate_tags(transcription)
    print(tags)
