import streamlit as st

import app.upload_pipeline.split_data as split_data
import app.text_processing.speech_to_text as speech_to_text


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
        # print(speech_to_text.annotate(audio_video))


if __name__ == "__main__":
    main()
