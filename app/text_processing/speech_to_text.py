from pathlib import Path
from typing import NamedTuple

from openai import OpenAI

from app.upload_pipeline.split_data import AudioVideo

API_KEY = Path("./app/keys").read_text()


class TextSlice(NamedTuple):
    string: str
    time_start_s: float
    time_end_s: float


class AnnotatedText(NamedTuple):
    text: str
    words: list[TextSlice]


def annotate(audio_video: AudioVideo) -> AnnotatedText:

    client = OpenAI(api_key=API_KEY)

    transcript = client.audio.transcriptions.create(
        file=audio_video.audio,
        model="whisper-1",
        language="pl",
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

    words = [
        TextSlice(w.word, time_start_s=w.start, time_end_s=w.end)
        for w in transcript.words
    ]

    return AnnotatedText(text=transcript.text, words=words)
