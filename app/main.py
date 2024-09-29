import streamlit as st

import app.upload_pipeline.split_data as split_data
import app.text_processing.speech_to_text as speech_to_text
from app.text_processing.llm_analyzer import llm_output
from app.text_processing.processor import text_analyzer
import app.face_detection.video_analyzer as video_analyzer


def main():
    st.file_uploader
    uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

    if uploaded_file is not None:
        # metadata
        file_details = {
            "Filename": uploaded_file.name,
            "File type": uploaded_file.type,
            "File size": f"{uploaded_file.size} bytes",
        }

        st.json(file_details)

        audio_video = split_data.split(uploaded_file)
        text, words = speech_to_text.annotate(audio_video)

        # text processors
        # llm_output(text)
        # text_analyzer(text)

        print("Full Text:\n", text)

        print("Individual Words:")
        video_analyzer_output = video_analyzer.process_images(audio_video.frames,audio_video.fps)
        for word in words:
            print(word)


if __name__ == "__main__":
    main()
