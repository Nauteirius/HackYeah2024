from typing import NamedTuple

import app.text_processing.openai_client as openai_client
from app.upload_pipeline.split_data import AudioVideo


class TextSlice(NamedTuple):
    string: str
    time_start_s: float
    time_end_s: float


class AnnotatedText(NamedTuple):
    text: str
    words: list[TextSlice]


def annotate(audio_video: AudioVideo) -> AnnotatedText:
    transcript = openai_client.instance.audio.transcriptions.create(
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
