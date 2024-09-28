from openai import OpenAI
import typing

client = OpenAI()

def call_openai_api(system_content: str, user_content: str) -> str:
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